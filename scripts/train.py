"""Train the baseline breed classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.baseline import (
    SUPPORTED_ARCHITECTURES,
    build_model,
    class_weights,
    make_loader,
    run_epoch,
    save_checkpoint,
    save_confusion_matrix,
    seed_everything,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a transfer-learning breed classifier.")
    parser.add_argument("--data", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline"))
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet weights.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    train_dataset, train_loader = make_loader(args.data / "train", True, args.image_size, args.batch_size, args.seed, args.workers)
    validation_dataset, validation_loader = make_loader(args.data / "validation", False, args.image_size, args.batch_size, args.seed, args.workers)
    if train_dataset.classes != validation_dataset.classes:
        raise ValueError("Training and validation class labels do not match")

    model = build_model(len(train_dataset.classes), args.architecture, not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset).to(device))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "class_labels.json").write_text(json.dumps(train_dataset.class_to_idx, indent=2), encoding="utf-8")

    history = []
    best_validation_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, criterion, device, len(train_dataset.classes), optimizer)
        validation = run_epoch(model, validation_loader, criterion, device, len(train_dataset.classes))
        row = {"epoch": epoch, "train": training, "validation": validation}
        history.append(row)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(args.output / "last.pt", model, optimizer, epoch, args.architecture, train_dataset.class_to_idx, args.image_size, history)
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            save_checkpoint(args.output / "best.pt", model, optimizer, epoch, args.architecture, train_dataset.class_to_idx, args.image_size, history)
            save_confusion_matrix(validation["confusion_matrix"], train_dataset.classes, args.output)
        print(f"epoch={epoch} train_loss={training['loss']:.4f} val_loss={validation['loss']:.4f} val_f1={validation['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())