# Model

## Baseline

The baseline is MobileNetV3-Small with ImageNet transfer learning by default. EfficientNet-B0 and ResNet18 are available for comparison. The design favors a model that can later be adapted to CPU, mobile, or edge deployment.

## Training

```bash
python scripts/prepare_dataset.py
python scripts/train.py --data datasets/processed --output models --epochs 10
```

Training uses deterministic seeds, train-only augmentation, class-weighted cross-entropy, validation loss, checkpointing, and best-checkpoint selection. It records loss, accuracy, macro precision, macro recall, macro F1, and a confusion matrix.

## Evaluation and export

```bash
python scripts/evaluate.py --checkpoint models/best_model.pt --data datasets/processed --split test
python scripts/export_model.py --checkpoint models/best_model.pt --output models/best_model.ts --format torchscript
```

The export writes `best_model.ts`, `classes.json`, and preprocessing metadata. `Predictor` loads those artifacts and returns ranked model scores.

## Evidence policy

No real dataset or production checkpoint is committed. Therefore this repository makes no claim about real-world accuracy or robustness. Synthetic smoke tests verify code paths only. Scores are not scientifically calibrated probabilities unless a future calibration study demonstrates that property.
