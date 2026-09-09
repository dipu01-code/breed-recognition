"""Run dataset preparation from the repository root."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data_pipeline import DatasetPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and prepare the breed image dataset.")
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = DatasetPipeline(args.raw, args.output, (args.width, args.height), args.seed).run()
    print(f"Prepared {report.number_of_images} images across {report.number_of_classes} classes.")
    print(f"Reports: {args.output / 'dataset_report.json'} and {args.output / 'dataset_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())