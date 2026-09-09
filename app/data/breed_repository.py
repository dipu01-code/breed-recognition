"""Repository access to the structured breed database."""

from app.core.breeds import Breed, get_breed, list_breeds

MODEL_SUPPORTED_BREEDS: frozenset[str] = frozenset()


class BreedRepository:
	"""Provide the Phase 2 database contract to application consumers."""

	def get_all_breeds(self) -> tuple[Breed, ...]:
		return list_breeds()

	def get_breed(self, name: str) -> Breed:
		return get_breed(name)

	def get_by_animal_type(self, animal_type: str) -> tuple[Breed, ...]:
		return list_breeds(animal_type=animal_type)

	def search(self, query: str) -> tuple[Breed, ...]:
		return list_breeds(search=query)

	def is_model_supported(self, name: str) -> bool:
		"""Return whether a trained model currently exposes this breed label."""
		return self.get_breed(name).breed_name in MODEL_SUPPORTED_BREEDS


__all__ = ["BreedRepository", "MODEL_SUPPORTED_BREEDS", "get_breed", "list_breeds"]