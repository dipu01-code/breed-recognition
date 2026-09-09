"""Inference boundary for the exported baseline model."""

import json
from pathlib import Path

import torch
from PIL import Image

from app.core.errors import ApplicationError
from app.models.baseline import image_transforms


def load_model(model_dir: Path):
    """Load the exported TorchScript model and its class-label mapping."""
    model_path = model_dir / "best_model.ts"
    labels_path = model_dir / "classes.json"
    if not model_path.exists() or not labels_path.exists():
        raise ApplicationError(f"No trained model is available in {model_dir}.")
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return model, {int(index): name for name, index in labels.items()}


def predict(image_path: Path, model_dir: Path) -> str:
    """Predict one image using an exported model, without inventing fallback output."""
    if not image_path.exists():
        raise ApplicationError(f"Image not found: {image_path}")
    model, labels = load_model(model_dir)
    with Image.open(image_path) as image:
        tensor = image_transforms(False)(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        class_index = int(model(tensor).argmax(dim=1).item())
    if class_index not in labels:
        raise ApplicationError(f"Model output class {class_index} has no label mapping.")
    return labels[class_index]