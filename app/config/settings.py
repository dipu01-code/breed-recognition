"""Runtime configuration for the command-line application."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application paths and runtime options."""

    project_root: Path
    model_dir: Path
    log_level: str = "INFO"

    @classmethod
    def from_project_root(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        return cls(project_root=root, model_dir=root / "models")