"""Prediction service boundary."""

from app.core.exceptions import ApplicationError


def predict(image_path: str) -> None:
    """Do not fabricate predictions before a trained model is installed."""
    raise ApplicationError(f"Prediction is unavailable until a trained model is installed: {image_path}")