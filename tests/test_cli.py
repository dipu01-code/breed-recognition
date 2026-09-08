"""Tests for the initial CLI skeleton."""

from pathlib import Path

from app.cli.main import main


def test_help(capsys):
    assert main(["--help"]) == 0
    assert "predict" in capsys.readouterr().out


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"


def test_breeds(capsys):
    assert main(["breeds"]) == 0
    assert "cattle:" in capsys.readouterr().out


def test_info(capsys):
    assert main(["info"]) == 0
    assert "inference: unavailable" in capsys.readouterr().out


def test_predict_without_model(caplog, tmp_path: Path):
    image = tmp_path / "animal.jpg"
    image.write_bytes(b"placeholder")
    assert main(["predict", str(image)]) == 2
    assert "No trained model" in caplog.text