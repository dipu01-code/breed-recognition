# Indian Breed Recognizer

CLI and knowledge database for image-based breed recognition of Indian cattle
and buffaloes. The breed metadata is a research-oriented reference and is not
itself an ML classifier label list.

## Installation

Use Python 3.10 or newer. A virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
```

The CLI has no runtime dependencies at this stage.

## Usage

```bash
python main.py --help
python main.py --version
python main.py breeds
python main.py breeds --type cattle
python main.py breeds --type buffalo
python main.py breeds --search gir
python main.py info
python main.py info --breed gir
python main.py predict path/to/animal.jpg
python scripts/prepare_dataset.py
python scripts/train.py --data datasets/processed --output models
python scripts/evaluate.py --checkpoint models/best_model.pt --split test
python scripts/export_model.py --checkpoint models/best_model.pt --output models/best_model.ts
```

## Terminal application

Install dependencies, then launch the full-screen Textual interface:

```bash
python -m pip install -r requirements.txt
python main.py
```

Use the arrow keys to navigate HOME, PREDICT, BREEDS, HISTORY, MODEL, and
ABOUT. Press Enter to select, Escape to return home, `?` for help, and `Q` to
quit. Prediction remains unavailable until a trained model is installed.

The database is stored in `app/data/breeds.json`. Each record includes a
confidence level and an `uncertainties` list so regional or variable traits are
not presented as universal facts.

`predict` validates the image path and returns a clear message until a trained
model is installed. Expected application errors use a non-zero exit code.

## Baseline model

The baseline uses transfer learning with MobileNetV3-Small by default. It
supports EfficientNet-B0 and ResNet18, applies training augmentation, uses
class-weighted loss, and writes reproducible checkpoints and metrics under
`models/`. Pretrained ImageNet weights are downloaded on the first
training run; use `--no-pretrained` when working offline.

No real training accuracy is included in this repository because the actual
dataset is not committed. The pipeline was smoke-tested with a tiny synthetic
dataset only; those metrics must not be interpreted as breed-model performance.

Predictions use a configurable confidence threshold of `70%` by default. Set
`BREED_CONFIDENCE_THRESHOLD` to a value between `0` and `1` to change it. The
displayed score is a model score, not a scientifically calibrated probability;
scores below the threshold are marked low-confidence and recommend manual
verification.

Train:

```bash
python scripts/train.py --data datasets/processed --output models --epochs 10
```

Evaluate and generate metrics plus a confusion matrix:

```bash
python scripts/evaluate.py --checkpoint models/best_model.pt --data datasets/processed --split test
```

Export a CPU-friendly TorchScript model and its class-label mapping:

```bash
python scripts/export_model.py --checkpoint models/best_model.pt --output models/best_model.ts --format torchscript
```

Use `--format onnx` to export ONNX instead. Training writes `best.pt`,
`last_model.pt`, `classes.json`, `history.json`, and confusion-matrix files.

## Layout

- `app/`: application package and CLI implementation
- `tests/`: automated tests
- `configs/`: future training and deployment configuration
- `datasets/`: local dataset workspace
- `models/`: model artifacts
- `scripts/`: development and data utilities
- `docs/`: project documentation

## Dataset preparation

Place source images under `datasets/raw/<animal_type>/<breed>/<animal_id>/`.
The preparation script writes resized train, validation, and test images under
`datasets/processed/` and generates a JSON and Markdown report. The animal ID
directory is used as the split group to prevent data leakage. See
`datasets/README.md` for the complete layout and options.

## Starting the terminal application

From the repository root, activate the environment and run:

```bash
source .venv/bin/activate
python main.py --help
```
