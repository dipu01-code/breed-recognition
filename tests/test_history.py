"""Tests for local prediction history persistence."""

import json

from app.services.history import HistoryStore


def test_history_store_persists_newest_prediction_first(tmp_path):
    store = HistoryStore(tmp_path / "history.json")

    first = store.append("cow.jpg", "cattle", "Gir", 0.94, "best_model.ts")
    second = store.append("buffalo.png", "buffalo", "Murrah", 0.88, "v2")

    assert first["timestamp"]
    assert store.load() == [second, first]
    assert json.loads((tmp_path / "history.json").read_text()) == [second, first]


def test_history_store_recovers_from_missing_or_malformed_file(tmp_path):
    path = tmp_path / "history.json"
    store = HistoryStore(path)

    assert store.load() == []
    path.write_text("not json", encoding="utf-8")
    assert store.load() == []
    path.write_text(json.dumps({"wrong": "shape"}), encoding="utf-8")
    assert store.load() == []