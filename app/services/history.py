"""Local JSON persistence for successful prediction history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HistoryStore:
    """Persist prediction records locally without requiring a database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return records if isinstance(records, list) else []

    def append(
        self,
        image_filename: str,
        animal_type: str,
        predicted_breed: str,
        confidence: float,
        model_version: str,
    ) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_filename": image_filename,
            "animal_type": animal_type,
            "predicted_breed": predicted_breed,
            "confidence": confidence,
            "model_version": model_version,
        }
        records = [record, *self.load()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return record
