"""Headless Textual checks for navigation and terminal-sized layout."""

import asyncio

from app.ui.app import BreedRecognizerApp


def test_navigation_file_panel_and_breed_search():
    async def check() -> None:
        async with BreedRecognizerApp().run_test(size=(120, 40)) as pilot:
            assert pilot.app.active_section == "HOME"
            await pilot.press("down", "enter")
            assert pilot.app.active_section == "PREDICT"
            await pilot.press("right")
            assert pilot.app.active_panel == "files"
            await pilot.press("escape")
            assert pilot.app.active_section == "HOME"
            await pilot.press("down", "down", "enter")
            assert pilot.app.active_section == "BREEDS"
            pilot.app.query_one("#breed-search").value = "gir"
            await pilot.pause()
            assert pilot.app.query_one("#breed-table").row_count == 1

    asyncio.run(check())


def test_ui_supports_resize_quit_and_history_screen():
    async def check() -> None:
        async with BreedRecognizerApp().run_test(size=(80, 24)) as pilot:
            await pilot.resize_terminal(140, 45)
            await pilot.press("down", "down", "down", "enter")
            assert pilot.app.active_section == "HISTORY"
            await pilot.press("escape")
            assert pilot.app.active_section == "HOME"
            await pilot.press("q")
            assert pilot.app.return_value is None

    import asyncio

    asyncio.run(check())



def test_enter_on_file_predicts_once_and_empty_folder_does_not_crash(tmp_path):
    from PIL import Image

    async def check(folder, expected_calls) -> None:
        app = BreedRecognizerApp()
        app.dataset_root = app.current_path = folder
        calls = []
        app.run_prediction = calls.append
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("down", "enter", "right", "down", "enter")
            await pilot.pause()
        assert len(calls) == expected_calls

    with_image = tmp_path / "images"
    with_image.mkdir()
    Image.new("RGB", (8, 8)).save(with_image / "cow.jpg")
    empty = tmp_path / "empty"
    empty.mkdir()
    asyncio.run(check(with_image, 1))
    asyncio.run(check(empty, 0))


def test_left_steps_back_through_panels_and_hidden_folders_are_skipped(tmp_path):
    from PIL import Image

    (tmp_path / ".venv").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / ".venv" / "icon.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "cow.jpg")

    async def check() -> None:
        app = BreedRecognizerApp()
        app.dataset_root = app.current_path = tmp_path
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("down", "enter", "right")
            assert [str(path.name) for path in app.file_targets.values()] == ["cow.jpg"]
            await pilot.press("left")
            assert app.focused.id == "directory-list"
            await pilot.press("left")
            assert app.focused.id == "menu"

    asyncio.run(check())


def test_invalid_threshold_shows_clean_error(monkeypatch, capsys):
    from app.main import main

    monkeypatch.setenv("BREED_CONFIDENCE_THRESHOLD", "abc")
    assert main() == 2
    assert "BREED_CONFIDENCE_THRESHOLD" in capsys.readouterr().err
