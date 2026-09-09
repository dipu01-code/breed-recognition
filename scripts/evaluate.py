"""Evaluate a saved baseline checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.baseline import build_model, make_loader, run_epoch, save_confusion_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained breed classifier.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/baseline/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("models/baseline/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Run scripts/train.py first.")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    dataset, loader = make_loader(args.data / args.split, False, checkpoint["image_size"], args.batch_size, 42, args.workers)
    if dataset.class_to_idx != class_to_idx:
        raise ValueError("Dataset class labels do not match the checkpoint")
    model = build_model(len(class_to_idx), checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device)
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device, len(class_to_idx))
    args.output.mkdir(parents=True, exist_ok=True)
    metrics["split"] = args.split
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    save_confusion_matrix(metrics["confusion_matrix"], classes, args.output)
    print(json.dumps({key: metrics[key] for key in ("loss", "accuracy", "precision", "recall", "f1")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())