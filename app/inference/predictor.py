"""Framework-independent inference service for the exported baseline model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.core.errors import ApplicationError

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class InferenceError(ApplicationError):
    """Raised when an image or model cannot be used for inference."""


@dataclass(frozen=True)
class TopPrediction:
    breed: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    animal_type: str
    predicted_breed: str
    confidence: float
    top_predictions: tuple[TopPrediction, ...]
    is_mock: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["top_predictions"] = [asdict(item) for item in self.top_predictions]
        return result


class ModelLike(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...


def _load_labels(labels_path: Path) -> dict[int, str]:
    try:
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = {int(index): str(name) for name, index in raw.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise InferenceError(f"Invalid class-label mapping: {labels_path}") from error
    if not labels or sorted(labels) != list(range(len(labels))):
        raise InferenceError("Class-label mapping must contain contiguous indexes starting at zero.")
    return labels


def _load_preprocessing(config_path: Path) -> tuple[int, tuple[float, ...], tuple[float, ...]]:
    if not config_path.exists():
        return 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        image_size = int(config["image_size"])
        mean = tuple(float(value) for value in config["mean"])
        std = tuple(float(value) for value in config["std"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise InferenceError(f"Invalid preprocessing configuration: {config_path}") from error
    if image_size <= 0 or len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise InferenceError("Preprocessing configuration has invalid image size, mean, or standard deviation.")
    return image_size, mean, std


class Predictor:
    """Load an exported model once and return ranked, structured predictions."""

    def __init__(self, model_dir: Path, mock: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.mock = mock
        self._model: ModelLike | None = None
        self._labels: dict[int, str] = {}
        self._image_size = 224
        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)
        if not mock:
            self._load()

    def _load(self) -> None:
        model_path = self.model_dir / "best_model.ts"
        labels_path = self.model_dir / "classes.json"
        if not model_path.exists():
            raise InferenceError(f"No trained model is available: {model_path}")
        if not labels_path.exists():
            raise InferenceError(f"Class-label mapping is missing: {labels_path}")
        try:
            self._model = torch.jit.load(str(model_path), map_location="cpu")
            self._model.eval()
        except (OSError, RuntimeError) as error:
            raise InferenceError(f"Unable to load model: {model_path}") from error
        self._labels = _load_labels(labels_path)
        self._image_size, self._mean, self._std = _load_preprocessing(self.model_dir / "preprocessing.json")

    def _preprocess(self, image_path: Path) -> torch.Tensor:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                transform = transforms.Compose([
                    transforms.Resize((self._image_size, self._image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self._mean, self._std),
                ])
                return transform(image.convert("RGB")).unsqueeze(0)
        except (OSError, UnidentifiedImageError) as error:
            raise InferenceError(f"Invalid or corrupt image: {image_path}") from error

    def predict(self, image_path: str | Path, top_k: int = 3) -> dict[str, Any]:
        """Return top-k breed probabilities; mock results are explicitly marked."""
        path = Path(image_path)
        if not path.exists():
            raise InferenceError(f"Image not found: {path}")
        if not path.is_file():
            raise InferenceError(f"Image path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise InferenceError(f"Unsupported image format: {path.suffix or '<none>'}")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        inputs = self._preprocess(path)
        if self.mock:
            return PredictionResult("unknown", "mock-unavailable", 0.0, (), is_mock=True).as_dict()
        if self._model is None:
            raise InferenceError("Model is not loaded.")
        with torch.no_grad():
            logits = self._model(inputs)
            if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(self._labels):
                raise InferenceError("Model output does not match the class-label mapping.")
            probabilities = torch.softmax(logits[0], dim=0)
        values, indexes = torch.topk(probabilities, k=min(top_k, len(self._labels)))
        predictions = tuple(TopPrediction(self._labels[int(index)], float(value)) for value, index in zip(values, indexes))
        top = predictions[0]
        buffaloes = {"Murrah", "Jaffarabadi", "Surti", "Mehsana", "Nili-Ravi", "Bhadawari", "Pandharpuri", "Toda"}
        animal_type = "buffalo" if top.breed in buffaloes else "cattle"
        return PredictionResult(animal_type, top.breed, top.confidence, predictions).as_dict()


def load_model(model_dir: Path):
    """Backward-compatible model loader."""
    predictor = Predictor(model_dir)
    return predictor._model, predictor._labels


def predict(image_path: Path, model_dir: Path) -> str:
    """Backward-compatible single-label prediction helper."""
    return Predictor(model_dir).predict(image_path)["predicted_breed"]
