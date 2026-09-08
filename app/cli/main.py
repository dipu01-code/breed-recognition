"""Command-line parser and command handlers."""

import argparse
import logging
from pathlib import Path

from app import __version__
from app.config.settings import Settings
from app.core.breeds import BREEDS
from app.core.errors import ApplicationError
from app.inference.predictor import predict

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breed-recognizer",
        description="Image-based breed recognition for Indian cattle and buffaloes.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Set logging verbosity (default: WARNING).",
    )

    commands = parser.add_subparsers(dest="command")

    predict_parser = commands.add_parser("predict", help="Predict a breed from an image.")
    predict_parser.add_argument("image", type=Path, help="Path to an animal image.")

    commands.add_parser("breeds", help="List the currently configured breed taxonomy.")
    commands.add_parser("info", help="Show application and runtime information.")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s: %(message)s")


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "breeds":
        for breed in BREEDS:
            print(f"{breed.species}: {breed.name}")
        return 0

    if args.command == "info":
        print(f"version: {__version__}")
        print(f"project_root: {settings.project_root}")
        print(f"model_dir: {settings.model_dir}")
        print("inference: unavailable (model not installed)")
        return 0

    if args.command == "predict":
        predict(args.image, settings.model_dir)
        return 0

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    configure_logging(args.log_level)
    try:
        return run(args, Settings.from_project_root())
    except ApplicationError as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())