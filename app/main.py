"""Application entry point."""

import sys

from app.core.errors import ApplicationError
from app.ui.app import BreedRecognizerApp


def main() -> int:
    try:
        app = BreedRecognizerApp()
    except ApplicationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
