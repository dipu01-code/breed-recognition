"""Generate or display reproducible dataset statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_pipeline import DatasetPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Report image dataset statistics without inventing data.")
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON.")
    args = parser.parse_args()

    pipeline = DatasetPipeline(args.raw, args.output)
    records, corrupted, duplicates = pipeline.discover()
    if args.output.joinpath("dataset_report.json").exists() and not records and not corrupted:
        report = json.loads(args.output.joinpath("dataset_report.json").read_text(encoding="utf-8"))
    else:
        report = pipeline.run().as_dict()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Total images: {report['number_of_images']}")
        print(f"Number of classes: {report['number_of_classes']}")
        print(f"Images per class: {report['images_per_class']}")
        print(f"Image dimensions: {report['image_dimensions']}")
        print(f"Invalid images: {len(report['corrupted_files'])}")
        print(f"Duplicate images: {len(report['duplicate_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())