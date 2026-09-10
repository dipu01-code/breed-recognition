# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
```

## Test

Run the complete suite from the repository root:

```bash
pytest -q
```

The tests cover CLI behavior, breed repository queries, dataset validation and splitting, predictor errors and confidence policy, history persistence, and headless Textual navigation. Fixture images and tiny TorchScript models are synthetic test inputs; they are not production performance evidence.

## Common commands

```bash
python main.py
python scripts/dataset_stats.py --raw datasets/raw --output datasets/processed
python scripts/train.py --data datasets/processed --output models
python scripts/evaluate.py --checkpoint models/best_model.pt --split test
python scripts/export_model.py --checkpoint models/best_model.pt --output models/best_model.ts
```

Keep real datasets, checkpoints, prediction history, and workspace settings out of commits. Review `git status` before committing. Changes should preserve the separation between Textual presentation, inference, breed reference data, and persistence services.
