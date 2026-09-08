# Indian Breed Recognizer

Initial CLI foundation for image-based breed recognition of Indian cattle and
buffaloes. The project currently provides configuration, logging, taxonomy
placeholders, and a model-independent command interface. ML inference will be
added in a later phase.

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
python main.py info
python main.py predict path/to/animal.jpg
```

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
