"""Structured breed metadata and lookup helpers."""

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from app.core.errors import ApplicationError


@dataclass(frozen=True)
class Breed:
    """A canonical breed metadata record."""

    breed_name: str
    animal_type: str
    origin_state: str
    origin_region: str
    purpose: tuple[str, ...]
    physical_characteristics: str
    color: str
    horn_characteristics: str
    body_characteristics: str
    dairy_characteristics: str
    aliases: tuple[str, ...]
    identification_notes: str
    confidence: str
    uncertainties: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Breed":
        return cls(
            breed_name=data["breed_name"],
            animal_type=data["animal_type"],
            origin_state=data["origin_state"],
            origin_region=data["origin_region"],
            purpose=tuple(data["purpose"]),
            physical_characteristics=data["physical_characteristics"],
            color=data["color"],
            horn_characteristics=data["horn_characteristics"],
            body_characteristics=data["body_characteristics"],
            dairy_characteristics=data["dairy_characteristics"],
            aliases=tuple(data["aliases"]),
            identification_notes=data["identification_notes"],
            confidence=data["confidence"],
            uncertainties=tuple(data["uncertainties"]),
        )

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["purpose"] = list(self.purpose)
        result["aliases"] = list(self.aliases)
        result["uncertainties"] = list(self.uncertainties)
        return result


def _load_breeds() -> tuple[Breed, ...]:
    data_file = resources.files("app.data").joinpath("breeds.json")
    records = json.loads(data_file.read_text(encoding="utf-8"))
    return tuple(Breed.from_dict(record) for record in records)


BREEDS: tuple[Breed, ...] = _load_breeds()


def list_breeds(
    animal_type: str | None = None,
    search: str | None = None,
) -> tuple[Breed, ...]:
    """Return breeds filtered by animal type and name or alias search."""

    normalized_type = animal_type.lower() if animal_type else None
    normalized_search = search.casefold().strip() if search else None

    matches = []
    for breed in BREEDS:
        if normalized_type and breed.animal_type != normalized_type:
            continue
        searchable_names = (breed.breed_name, *breed.aliases)
        if normalized_search and not any(
            normalized_search == name.casefold()
            or any(
                word.startswith(normalized_search)
                for word in name.casefold().split()
            )
            for name in searchable_names
        ):
            continue
        matches.append(breed)
    return tuple(matches)


def get_breed(name: str) -> Breed:
    """Find a breed by canonical name or alias, case-insensitively."""

    normalized_name = name.casefold().strip()
    for breed in BREEDS:
        if normalized_name in {
            breed.breed_name.casefold(),
            *(alias.casefold() for alias in breed.aliases),
        }:
            return breed
    raise ApplicationError(f"Unknown breed: {name}")