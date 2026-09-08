"""Initial breed taxonomy placeholder.

The taxonomy is intentionally small until authoritative labels and dataset
governance are agreed with domain experts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Breed:
    """A canonical breed entry."""

    name: str
    species: str


BREEDS: tuple[Breed, ...] = (
    Breed(name="Cattle taxonomy pending", species="cattle"),
    Breed(name="Buffalo taxonomy pending", species="buffalo"),
)