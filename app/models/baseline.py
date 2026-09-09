"""CPU-friendly transfer-learning baseline for breed classification."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0", "resnet18")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(
    num_classes: int,
    architecture: str = "mobilenet_v3_small",
    pretrained: bool = True,
) -> nn.Module:
    """Build a transfer-learning classifier with a new output head."""
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def image_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    """Return augmentation for training and deterministic preprocessing for validation."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_loader(
    root: Path,
    train: bool,
    image_size: int,
    batch_size: int,
    seed: int,
    workers: int,
) -> tuple[datasets.ImageFolder, DataLoader]:
    dataset = datasets.ImageFolder(root, transform=image_transforms(train, image_size))
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes)).astype(np.float32)
    counts[counts == 0] = 1
    return torch.tensor(len(dataset) / (len(dataset.classes) * counts), dtype=torch.float32)


def calculate_metrics(labels: list[int], predictions: list[int], num_classes: int) -> dict[str, Any]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[label, prediction] += 1
    true_positives = np.diag(matrix).astype(np.float64)
    precision = true_positives / np.maximum(matrix.sum(axis=0), 1)
    recall = true_positives / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(true_positives.sum() / max(matrix.sum(), 1)),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for images, batch_labels in loader:
        images, batch_labels = images.to(device), batch_labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(images)
            loss = criterion(output, batch_labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(output.argmax(dim=1).cpu().tolist())
    result = calculate_metrics(labels, predictions, num_classes)
    result["loss"] = total_loss / max(len(loader.dataset), 1)
    return result


def save_confusion_matrix(matrix: list[list[int]], classes: list[str], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *classes])
        writer.writerows([classes[index], *row] for index, row in enumerate(matrix))
    (output / "confusion_matrix.json").write_text(
        json.dumps({"classes": classes, "matrix": matrix}, indent=2), encoding="utf-8"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    architecture: str,
    class_to_idx: dict[str, int],
    image_size: int,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "architecture": architecture,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
    }, path)