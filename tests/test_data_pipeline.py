"""Tests for dataset validation, preprocessing, and leakage prevention."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.data_pipeline.pipeline import DatasetPipeline, load_normalized


def write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_pipeline_reports_bad_files_duplicates_and_dimensions(tmp_path: Path):
    raw = tmp_path / "raw"
    write_image(raw / "cattle" / "gir" / "animal-a" / "front.png", (255, 0, 0))
    write_image(raw / "cattle" / "gir" / "animal-a" / "side.png", (0, 255, 0))
    duplicate = raw / "cattle" / "gir" / "animal-b" / "copy.png"
    write_image(duplicate, (255, 0, 0))
    (raw / "buffalo" / "murrah" / "animal-c" / "broken.jpg").parent.mkdir(parents=True)
    (raw / "buffalo" / "murrah" / "animal-c" / "broken.jpg").write_bytes(b"not an image")

    report = DatasetPipeline(raw, tmp_path / "processed", image_size=(20, 20), seed=7).run()

    assert report.number_of_images == 2
    assert report.number_of_classes == 1
    assert report.images_per_class == {"cattle/gir": 2}
    assert report.image_dimensions == {"32x16": 2}
    assert len(report.corrupted_files) == 1
    assert len(report.duplicate_files) == 1
    assert (tmp_path / "processed" / "dataset_report.json").exists()
    assert json.loads((tmp_path / "processed" / "dataset_report.json").read_text())["number_of_images"] == 2


def test_split_keeps_animal_images_together_and_resizes(tmp_path: Path):
    raw = tmp_path / "raw"
    for animal_number in range(12):
        animal = raw / "cattle" / "gir" / f"animal-{animal_number}"
        write_image(animal / "front.png", (animal_number, 10, 20), (30, 40))
        write_image(animal / "side.png", (animal_number, 20, 30), (30, 40))

    output = tmp_path / "processed"
    report = DatasetPipeline(raw, output, image_size=(24, 18), seed=3).run()
    locations: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for path in (output / split).rglob("*.jpg"):
            animal_id = path.stem.split("_", 1)[0]
            assert animal_id not in locations or locations[animal_id] == split
            locations[animal_id] = split
            with Image.open(path) as image:
                assert image.size == (24, 18)
    assert len(locations) == 12
    assert sum(report.split_distribution.values()) == 24


def test_load_normalized_returns_rgb_float_values(tmp_path: Path):
    image_path = tmp_path / "image.png"
    write_image(image_path, (128, 64, 0), (3, 2))

    result = load_normalized(image_path, image_size=(2, 2))

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32
    assert np.all(result >= 0) and np.all(result <= 1)