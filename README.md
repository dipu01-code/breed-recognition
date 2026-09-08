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
```

The database is stored in `app/data/breeds.json`. Each record includes a
confidence level and an `uncertainties` list so regional or variable traits are
not presented as universal facts.

`predict` validates the image path and returns a clear message until a trained
model is installed. Expected application errors use a non-zero exit code.

## Layout

- `app/`: application package and CLI implementation
- `tests/`: automated tests
- `configs/`: future training and deployment configuration
- `datasets/`: local dataset workspace
- `models/`: model artifacts
- `scripts/`: development and data utilities
- `docs/`: project documentation
