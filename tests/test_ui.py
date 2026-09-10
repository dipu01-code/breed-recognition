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
