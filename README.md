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
# Indian Breed AI

Indian Breed AI is a terminal-first MVP for exploring Indian cattle and buffalo
breed recognition. It combines a local breed reference database, a reproducible
image pipeline, an optional transfer-learning classifier, and a confidence-aware
Textual interface.

## Problem

Bharat Pashudhan and related livestock workflows need practical breed
identification support across varied images, regions, lighting, backgrounds, and
animal poses. Similar indigenous breeds can share visual traits, so an image
classifier should assist field or research workflows without replacing expert
verification.

## Solution

```text
image -> AI model -> breed score -> confidence status -> breed information -> human verification
```

The application keeps model output, confidence policy, and reference metadata
separate. Reference information supports investigation; it does not prove that
the image belongs to the predicted breed.

## Features

- AI image classification pipeline
- Cattle and buffalo breed recognition support
- Top-3 model scores
- Configurable confidence threshold and low-confidence warnings
- Structured breed database with aliases and uncertainty notes
- Full-screen Textual terminal UI
- Local prediction history
- Manual verification recommendations
- Dataset validation, grouped splitting, reports, and optional augmentation

## Architecture

```text
				+----------------------+
				|      main.py         |
				+----------+-----------+
						 |
				+----------v-----------+
				|   Textual TUI / CLI  |
				+----+-------------+---+
					|             |
		    +----------v--+     +----v-------------+
		    | Predictor   |     | BreedRepository  |
		    +------+------+     +----+-------------+
				 |                   |
		+----------v----------+   +---v-------------+
		| TorchScript model   |   | app/data/breeds |
		| labels + preprocessing|  | reference data  |
		+----------+-----------+   +-----------------+
				 |
		+----------v-----------+
		| JSON prediction     |
		| history and reports |
		+----------------------+
```

See [docs/architecture.md](docs/architecture.md) for component boundaries.

## Installation

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
```

## Usage

Launch the terminal application:

```bash
python main.py
```

Use Up/Down to navigate, Enter to select, Left/Right to switch file panels,
Escape to return home, `R` to refresh, `?` for help, and `Q` to quit. The
PREDICT screen opens in `datasets/raw`, shows directories, supported images, and
metadata preview, then calls the inference service when an image is selected.

The BREEDS screen supports search and details. HISTORY displays successful
predictions saved in `prediction_history.json`. Without a trained model, the UI
reports that prediction is unavailable and does not fabricate a result.

## Model

The baseline uses MobileNetV3-Small transfer learning by default, with optional
EfficientNet-B0 and ResNet18 choices. Training applies augmentation, class
weighted loss, deterministic seeds, validation, checkpointing, metrics, and a
confusion matrix.

```bash
python scripts/prepare_dataset.py
python scripts/train.py --data datasets/processed --output models --epochs 10
python scripts/evaluate.py --checkpoint models/best_model.pt --data datasets/processed --split test
python scripts/export_model.py --checkpoint models/best_model.pt --output models/best_model.ts
```

The repository contains no real dataset or trained production checkpoint, so it
does not claim accuracy, precision, recall, or F1 results. The pipeline has only
been smoke-tested with tiny synthetic fixtures. Model scores are not presented
as scientifically calibrated probabilities.

See [docs/model.md](docs/model.md) for artifact and inference details.

## Dataset

Supply real images locally using this structure:

```text
datasets/raw/<animal_type>/<breed>/<animal_id>/<image>
datasets/processed/{train,validation,test}/<animal_type>/<breed>/<image>.jpg
```

The animal ID is the split group, keeping views of one animal together where
possible. Raw and processed datasets are ignored by Git. See
[docs/dataset.md](docs/dataset.md) and [datasets/README.md](datasets/README.md).

## Demo

The TUI can be demonstrated with `python main.py` and a local dataset. No
screenshots or model results are checked into this repository, so no fabricated
demo evidence is presented here.

## Limitations

- Supported breeds depend on the classes represented in training data.
- Image quality, blur, lighting, background, and viewpoint affect prediction.
- Visually similar breeds can be difficult to distinguish.
- Human verification remains important, especially for low-confidence scores.
- The current implementation is an MVP.
- The breed database is broader than any model-supported label mapping.

## Future Work

- More breeds and a larger, field-representative dataset
- Animal detection before breed classification
- Mobile deployment and offline inference
- BPA API integration where appropriate and authorized
- Model quantization for edge hardware
- Improved confidence calibration

## Development

```bash
pytest -q
python scripts/dataset_stats.py --raw datasets/raw --output datasets/processed
```

See [docs/development.md](docs/development.md) for the workflow and test
guidance.
directory is used as the split group to prevent data leakage. See
`datasets/README.md` for the complete layout and options.

## Starting the terminal application

From the repository root, activate the environment and run:

```bash
source .venv/bin/activate
python main.py --help
```
