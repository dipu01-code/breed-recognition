# Dataset Workspace

The dataset is intentionally local and ignored by Git. Use this structure:

```text
datasets/
  raw/
    cattle/<breed>/<animal_id>/<image>
    buffalo/<breed>/<animal_id>/<image>
  processed/
    train/<animal_type>/<breed>/<image>.jpg
    validation/<animal_type>/<breed>/<image>.jpg
    test/<animal_type>/<breed>/<image>.jpg
```

Each animal must have a stable unique directory name. The pipeline uses that
directory as the split group, so images of the same animal cannot be copied
between train, validation, and test. Processed images and generated reports are
also ignored and should not be committed.

Run preprocessing from the repository root:

```bash
python scripts/prepare_dataset.py
python scripts/prepare_dataset.py --raw datasets/raw --output datasets/processed --seed 42
python scripts/dataset_stats.py --raw datasets/raw --output datasets/processed
```

The command validates images, detects corrupt files and exact duplicates,
resizes images to 224x224 RGB JPEGs, creates best-effort 70/15/15 splits, and
writes `dataset_report.json` plus `dataset_report.md`.
Use `--augment` to add deterministic horizontal-flip copies to the training
split only; validation and test images are never augmented. No dataset is
included in the repository. Supply real images in the documented `raw`
structure before running these commands.