"""Application entry point."""

from app.ui.app import BreedRecognizerApp


def main() -> None:
    BreedRecognizerApp().run()


if __name__ == "__main__":
    main()