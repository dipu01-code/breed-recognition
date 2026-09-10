# Dataset

The dataset is intentionally local and is never committed to Git.

## Raw layout

```text
datasets/raw/
  cattle/<breed>/<animal_id>/<image>
  buffalo/<breed>/<animal_id>/<image>
```

Each animal should have a stable ID directory. Multiple views of one animal belong in that directory.

## Preparation

```bash
python scripts/prepare_dataset.py --raw datasets/raw --output datasets/processed --seed 42
python scripts/dataset_stats.py --raw datasets/raw --output datasets/processed
```

The pipeline validates images, detects corrupt files and exact duplicates, converts valid images to RGB JPEGs, resizes them, reports dimensions and class counts, and creates best-effort train/validation/test splits. Animal directories are the split groups to reduce leakage between views of one animal.

Use `--augment` for horizontal-flip copies in the training split only. Validation and test data are not augmented.

```text
datasets/processed/
  train/<animal_type>/<breed>/image.jpg
  validation/<animal_type>/<breed>/image.jpg
  test/<animal_type>/<breed>/image.jpg
```

No dataset is present in the repository, and empty dataset runs must be treated as setup status rather than model evidence.
