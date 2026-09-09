"""Reproducible preparation of animal image datasets."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    source: Path
    animal_type: str
    breed: str
    animal_id: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class DatasetReport:
    number_of_images: int
    number_of_classes: int
    images_per_class: dict[str, int]
    animal_type_distribution: dict[str, int]
    image_dimensions: dict[str, int]
    corrupted_files: list[str]
    duplicate_files: list[str]
    split_distribution: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DatasetPipeline:
    """Prepare images from ``raw/<type>/<breed>/<animal_id>/image``.

    The animal directory is the grouping key. Consequently all images from one
    animal are assigned to exactly one split, preventing animal-level leakage.
    """

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        image_size: tuple[int, int] = (224, 224),
        seed: int = 42,
    ) -> None:
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size values must be positive")
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.seed = seed

    def _iter_images(self) -> Iterator[Path]:
        if not self.raw_dir.exists():
            return
        yield from sorted(
            path
            for path in self.raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _inspect(self, path: Path) -> ImageRecord:
        relative = path.relative_to(self.raw_dir)
        if len(relative.parts) < 4:
            raise ValueError(
                f"Expected raw/<animal_type>/<breed>/<animal_id>/<image>: {path}"
            )
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return ImageRecord(
            source=path,
            animal_type=relative.parts[0].lower(),
            breed=relative.parts[1],
            animal_id=relative.parts[2],
            sha256=digest,
            width=width,
            height=height,
        )

    def discover(self) -> tuple[list[ImageRecord], list[str], list[str]]:
        """Return valid records, corrupt paths, and exact duplicate paths."""
        records: list[ImageRecord] = []
        corrupted: list[str] = []
        seen_hashes: dict[str, Path] = {}
        duplicates: list[str] = []
        for path in self._iter_images():
            try:
                record = self._inspect(path)
            except (OSError, UnidentifiedImageError, ValueError):
                corrupted.append(str(path))
                continue
            if record.sha256 in seen_hashes:
                duplicates.append(str(path))
                continue
            seen_hashes[record.sha256] = path
            records.append(record)
        return records, corrupted, duplicates

    def _split_records(self, records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
        groups: dict[tuple[str, str, str], list[ImageRecord]] = defaultdict(list)
        for record in records:
            groups[(record.animal_type, record.breed, record.animal_id)].append(record)
        group_values = list(groups.values())
        random.Random(self.seed).shuffle(group_values)
        total = len(records)
        targets = {"train": total * 0.70, "validation": total * 0.15, "test": total * 0.15}
        result = {name: [] for name in targets}
        counts = Counter()
        for group in group_values:
            split = min(targets, key=lambda name: counts[name] / max(targets[name], 1))
            result[split].extend(group)
            counts[split] += len(group)
        return result

    def _write_image(self, record: ImageRecord, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record.source) as image:
            image.convert("RGB").resize(self.image_size, Image.Resampling.LANCZOS).save(
                destination, format="JPEG", quality=95
            )

    def run(self) -> DatasetReport:
        records, corrupted, duplicates = self.discover()
        if self.output_dir.exists():
            for split in ("train", "validation", "test"):
                shutil.rmtree(self.output_dir / split, ignore_errors=True)
        splits = self._split_records(records)
        for split, split_records in splits.items():
            for record in split_records:
                destination = (
                    self.output_dir / split / record.animal_type / record.breed
                    / f"{record.animal_id}_{record.source.stem}.jpg"
                )
                self._write_image(record, destination)

        dimensions = Counter(f"{record.width}x{record.height}" for record in records)
        report = DatasetReport(
            number_of_images=len(records),
            number_of_classes=len({(r.animal_type, r.breed) for r in records}),
            images_per_class=dict(sorted(Counter(f"{r.animal_type}/{r.breed}" for r in records).items())),
            animal_type_distribution=dict(sorted(Counter(r.animal_type for r in records).items())),
            image_dimensions=dict(sorted(dimensions.items())),
            corrupted_files=sorted(corrupted),
            duplicate_files=sorted(duplicates),
            split_distribution={name: len(items) for name, items in splits.items()},
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "dataset_report.json").write_text(
            json.dumps(report.as_dict(), indent=2), encoding="utf-8"
        )
        self._write_markdown_report(report)
        return report

    def _write_markdown_report(self, report: DatasetReport) -> None:
        lines = [
            "# Dataset Report",
            "",
            f"- Images: {report.number_of_images}",
            f"- Classes: {report.number_of_classes}",
            f"- Corrupted files: {len(report.corrupted_files)}",
            f"- Duplicate files: {len(report.duplicate_files)}",
            "",
            "## Splits",
            "",
        ]
        lines.extend(f"- {name}: {count}" for name, count in report.split_distribution.items())
        lines.extend(["", "## Images per class", ""])
        lines.extend(f"- {name}: {count}" for name, count in report.images_per_class.items())
        (self.output_dir / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_normalized(image_path: Path, image_size: tuple[int, int] = (224, 224)):
    """Load an image as RGB float32 values normalized to the [0, 1] range."""
    with Image.open(image_path) as image:
        resized = image.convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
        # Keep numpy optional for callers that only need dataset preparation.
        import numpy as np

        return np.asarray(resized, dtype=np.float32) / 255.0