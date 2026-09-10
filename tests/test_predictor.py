"""Unit tests for the independent inference service."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from app.inference.predictor import InferenceError, Predictor


def write_image(path: Path) -> None:
    Image.new("RGB", (16, 16), (120, 80, 40)).save(path)


def write_model(directory: Path, outputs: int = 2) -> None:
    class FixedModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[3.0, 1.0]], dtype=torch.float32).expand(inputs.shape[0], -1)

    directory.mkdir(parents=True)
    torch.jit.trace(FixedModel(), torch.zeros(1, 3, 16, 16)).save(directory / "best_model.ts")
    (directory / "classes.json").write_text(json.dumps({"Gir": 0, "Sahiwal": 1}), encoding="utf-8")
    (directory / "preprocessing.json").write_text(
        json.dumps({"image_size": 16, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
        encoding="utf-8",
    )


def test_predictor_returns_ranked_structured_probabilities(tmp_path: Path):
    model_dir = tmp_path / "model"
    image = tmp_path / "cow.jpg"
    write_model(model_dir)
    write_image(image)

    result = Predictor(model_dir).predict(image, top_k=2)

    assert result["animal_type"] == "cattle"
    assert result["predicted_breed"] == "Gir"
    assert result["top_predictions"][0]["breed"] == "Gir"
    assert result["top_predictions"][0]["confidence"] > result["top_predictions"][1]["confidence"]
    assert result["is_mock"] is False


def test_predictor_handles_missing_and_invalid_inputs(tmp_path: Path):
    with pytest.raises(InferenceError, match="No trained model"):
        Predictor(tmp_path / "missing")

    model_dir = tmp_path / "model"
    write_model(model_dir)
    with pytest.raises(InferenceError, match="Image not found"):
        Predictor(model_dir).predict(tmp_path / "missing.jpg")
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(InferenceError, match="Invalid or corrupt image"):
        Predictor(model_dir).predict(corrupt)
    unsupported = tmp_path / "cow.gif"
    write_image(unsupported)
    with pytest.raises(InferenceError, match="Unsupported image format"):
        Predictor(model_dir).predict(unsupported)


def test_mock_mode_is_explicit_and_validates_images(tmp_path: Path):
    image = tmp_path / "cow.jpg"
    write_image(image)

    result = Predictor(tmp_path / "missing", mock=True).predict(image)

    assert result["is_mock"] is True
    assert result["predicted_breed"] == "mock-unavailable"


def test_predictor_rejects_invalid_preprocessing_configuration(tmp_path: Path):
    model_dir = tmp_path / "model"
    write_model(model_dir)
    (model_dir / "preprocessing.json").write_text("{\"image_size\": 0}", encoding="utf-8")

    with pytest.raises(InferenceError, match="Invalid preprocessing configuration"):
        Predictor(model_dir)


def test_predictor_marks_low_confidence_and_recommends_verification(tmp_path: Path):
    model_dir = tmp_path / "model"
    image = tmp_path / "cow.jpg"
    write_model(model_dir)
    write_image(image)

    result = Predictor(model_dir, confidence_threshold=0.95).predict(image)

    assert result["confidence_status"] == "LOW"
    assert result["manual_verification_recommended"] is True
    assert "not a scientifically calibrated probability" in result["confidence_note"]


def test_predictor_rejects_invalid_confidence_threshold(tmp_path: Path):
    with pytest.raises(ValueError, match="between 0 and 1"):
        Predictor(tmp_path, confidence_threshold=1.1)