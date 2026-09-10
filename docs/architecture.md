# Architecture

Indian Breed AI is organized around a small set of boundaries so the terminal UI does not own model logic.

```text
main.py
  |
  +-- app.main -> BreedRecognizerApp
  |       |
  |       +-- file selector and screen state
  |       +-- Predictor service call
  |       +-- BreedRepository lookup
  |       +-- HistoryStore persistence
  |
  +-- app.cli.main -> scriptable CLI compatibility

Dataset pipeline -> processed image folders -> training scripts
Training scripts -> checkpoint + classes.json + preprocessing metadata
Export script -> models/best_model.ts -> Predictor
```

## Boundaries

- `app/ui/`: Textual presentation, navigation, file selection, and display.
- `app/inference/`: model loading, image validation, preprocessing, scores, and top-k results.
- `app/data/`: packaged breed reference data and repository queries.
- `app/services/history.py`: local JSON persistence for successful predictions.
- `app/data_pipeline/`: validation, duplicate detection, resizing, splitting, reports, and normalization.
- `scripts/`: reproducible dataset, training, evaluation, and export entry points.

The UI calls `Predictor`; it does not duplicate preprocessing or inference. Breed metadata is supporting reference information and is never treated as proof of visual identity.
