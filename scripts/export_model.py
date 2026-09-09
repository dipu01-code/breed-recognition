"""Export a trained checkpoint for CPU or edge inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.baseline import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a trained breed classifier.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/baseline/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline/model.pt"))
    parser.add_argument("--format", choices=("torchscript", "onnx"), default="torchscript")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run scripts/train.py first.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(len(checkpoint["class_to_idx"]), checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    example = torch.randn(1, 3, checkpoint["image_size"], checkpoint["image_size"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "torchscript":
        torch.jit.trace(model, example).save(str(args.output))
    else:
        torch.onnx.export(model, example, args.output, input_names=["image"], output_names=["logits"], opset_version=17)
    labels_path = args.output.with_suffix(args.output.suffix + ".labels.json")
    labels_path.write_text(json.dumps(checkpoint["class_to_idx"], indent=2), encoding="utf-8")
    print(f"Exported {args.format} model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())