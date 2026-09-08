"""Inference boundary for the future ML model."""

from pathlib import Path

from app.core.errors import ApplicationError


def predict(image_path: Path, model_dir: Path) -> None:
    """Validate prediction inputs and report that model inference is pending."""
    if not image_path.exists():
        raise ApplicationError(f"Image not found: {image_path}")
    raise ApplicationError(
        f"No trained model is available in {model_dir}. "
        "Prediction will be enabled in a later phase."
    )