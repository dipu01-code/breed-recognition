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
