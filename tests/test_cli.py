"""Tests for the CLI and breed knowledge database."""

import json

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
    output = capsys.readouterr().out
    assert "cattle: Gir" in output
    assert "buffalo: Murrah" in output


def test_breeds_filter_and_search(capsys):
    assert main(["breeds", "--type", "buffalo"]) == 0
    output = capsys.readouterr().out
    assert "buffalo: Murrah" in output
    assert "cattle:" not in output

    assert main(["breeds", "--search", "gir"]) == 0
    assert capsys.readouterr().out.strip() == "cattle: Gir"


def test_database_contains_requested_breeds():
    from app.core.breeds import BREEDS

    expected = {
        "Gir", "Sahiwal", "Red Sindhi", "Rathi", "Tharparkar", "Kankrej",
        "Ongole", "Hariana", "Deoni", "Hallikar", "Kangayam", "Krishna Valley",
        "Amritmahal", "Dangi", "Malnad Gidda", "Murrah", "Jaffarabadi", "Surti",
        "Mehsana", "Nili-Ravi", "Bhadawari", "Pandharpuri", "Toda",
    }
    assert {breed.breed_name for breed in BREEDS} == expected
    assert all(breed.uncertainties for breed in BREEDS)


def test_info_breed_outputs_structured_metadata(capsys):
    assert main(["info", "--breed", "gyr"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["breed_name"] == "Gir"
    assert result["animal_type"] == "cattle"
    assert "identification_notes" in result


def test_unknown_breed_is_an_application_error(caplog):
    assert main(["info", "--breed", "not-a-breed"]) == 2
    assert "Unknown breed" in caplog.text


def test_info(capsys):
    assert main(["info"]) == 0
    assert "inference: unavailable" in capsys.readouterr().out


def test_predict_without_model(caplog, tmp_path: Path):
    image = tmp_path / "animal.jpg"
    image.write_bytes(b"placeholder")
    assert main(["predict", str(image)]) == 2
    assert "No trained model" in caplog.text


def test_breed_repository_supports_phase_two_queries():
    from app.data.breed_repository import BreedRepository, MODEL_SUPPORTED_BREEDS

    repository = BreedRepository()
    assert len(repository.get_all_breeds()) >= 19
    assert {breed.animal_type for breed in repository.get_by_animal_type("buffalo")} == {"buffalo"}
    assert repository.get_breed("gyr").breed_name == "Gir"
    assert [breed.breed_name for breed in repository.search("murrah")] == ["Murrah"]
    assert not MODEL_SUPPORTED_BREEDS
    assert not repository.is_model_supported("Gir")