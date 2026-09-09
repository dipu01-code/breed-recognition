"""Image preprocessing helpers reserved for the trained model."""


def prepare_image(image_path: str) -> str:
    """Return a validated image path for the future inference pipeline."""
    return image_path