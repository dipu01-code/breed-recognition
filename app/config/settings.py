"""Runtime configuration for the command-line application."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application paths and runtime options."""

    project_root: Path
    model_dir: Path
    log_level: str = "INFO"
    confidence_threshold: float = 0.70

    @classmethod
    def from_project_root(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        raw_threshold = os.getenv("BREED_CONFIDENCE_THRESHOLD", "0.70")
        try:
            confidence_threshold = float(raw_threshold)
        except ValueError as error:
            raise ValueError("BREED_CONFIDENCE_THRESHOLD must be a number between 0 and 1") from error
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("BREED_CONFIDENCE_THRESHOLD must be between 0 and 1")
        return cls(project_root=root, model_dir=root / "models", confidence_threshold=confidence_threshold)