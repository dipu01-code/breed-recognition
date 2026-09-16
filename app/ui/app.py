"""Professional terminal workspace for the breed recognizer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Label, ListItem, ListView, Static

from app.config.settings import Settings
from app.core.breeds import Breed
from app.data.breed_repository import BreedRepository
from app.inference.predictor import InferenceError, Predictor, SUPPORTED_IMAGE_EXTENSIONS
from app.services.history import HistoryStore


class BreedRecognizerApp(App[None]):
    """Keyboard-first full-screen terminal application."""

    CSS = """
    Screen { background: #101820; color: #e7edf2; }
    Header { background: #17324d; color: #f4c95d; }
    Footer { background: #0a1118; }
    #layout { height: 1fr; }
    #navigation { width: 23; background: #142637; border-right: solid #2b5878; padding: 1; }
    #content { width: 1fr; padding: 1 2; }
    #title { color: #f4c95d; text-style: bold; margin-bottom: 1; }
    #screen-title { color: #f4c95d; text-style: bold; height: 2; }
    #screen-body { color: #b9c8d3; height: auto; margin-bottom: 1; }
    #status { dock: bottom; height: 1; background: #17324d; color: #b9c8d3; padding: 0 1; }
    ListItem { padding: 1; }
    ListItem.--highlight { background: #2b5878; color: #ffffff; }
    DataTable { height: 1fr; border: round #2b5878; }
    Input { border: round #2b5878; margin-bottom: 1; }
    #file-selector { height: 1fr; }
    .panel { border: round #2b5878; padding: 1; width: 1fr; }
    .panel-title { color: #f4c95d; text-style: bold; height: 1; }
    #directory-list, #file-list { height: 1fr; }
    #preview { height: 1fr; color: #b9c8d3; }
    #breed-detail, #prediction, #settings, #about { border: round #2b5878; padding: 1; height: auto; margin-top: 1; }
    #history-detail { border: round #2b5878; padding: 1; height: auto; margin-top: 1; }
    .hidden { display: none; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "back", "Back"),
        ("?", "help", "Help"),
        ("r", "refresh", "Refresh"),
        ("left", "focus_menu", "Menu"),
        ("right", "focus_files", "Files"),
    ]
    sections = ("HOME", "PREDICT", "BREEDS", "HISTORY", "MODEL", "SETTINGS", "ABOUT")
    repository = BreedRepository()
    settings = Settings.from_project_root()
    dataset_root = settings.project_root
    current_path = dataset_root
    active_section = "HOME"
    active_panel = "menu"
    _predictor: Predictor | None = None
    history: list[dict[str, Any]] = []
    history_store = HistoryStore(Settings.from_project_root().project_root / "prediction_history.json")
    history_targets: dict[str, dict[str, Any]] = {}
    directory_targets: dict[str, Path] = {}
    file_targets: dict[str, Path] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Vertical(id="navigation"):
                yield Label("INDIAN BREED\nAI", id="title")
                yield ListView(*(ListItem(Label(section), id=f"menu-{section.lower()}") for section in self.sections), id="menu")
            with Container(id="content"):
                yield Static(id="screen-title")
                yield Static(id="screen-body")
                yield Input(placeholder="Search breed name or alias...", id="breed-search", classes="hidden")
                yield DataTable(id="breed-table", classes="hidden")
                yield Static(id="breed-detail", classes="hidden")
                yield DataTable(id="history-table", classes="hidden")
                yield Static(id="history-detail", classes="hidden")
                with Horizontal(id="file-selector", classes="hidden"):
                    with Vertical(classes="panel"):
                        yield Static("DIRECTORIES", classes="panel-title")
                        yield ListView(id="directory-list")
                    with Vertical(classes="panel"):
                        yield Static("FILES", classes="panel-title")
                        yield ListView(id="file-list")
                    with Vertical(classes="panel"):
                        yield Static("PREVIEW", classes="panel-title")
                        yield Static(id="preview")
                yield Static(id="prediction", classes="hidden")
                yield Static(id="settings", classes="hidden")
                yield Static(id="about", classes="hidden")
        yield Static("Up/Down Navigate   Left/Right Panels   Enter Select   Esc Back   R Refresh   ? Help   Q Quit", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#menu", ListView).focus()
        self.history = self.history_store.load()
        await self.show_section("HOME")

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id.startswith("menu-"):
            await self.show_section(item_id.removeprefix("menu-").upper())
        elif item_id.startswith("dir-"):
            self.current_path = self.directory_targets[item_id]
            await self.populate_files()
        elif item_id.startswith("file-"):
            image_path = self.file_targets[item_id]
            self.show_preview(image_path)
            self.run_prediction(image_path)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "breed-search":
            self.populate_breeds(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row = event.data_table.get_row(event.row_key)
        if event.data_table.id == "breed-table":
            breed = self.repository.get_breed(str(row[1]))
            self.show_breed_detail(breed)
        elif event.data_table.id == "history-table":
            self.show_history_detail(self.history_targets[str(event.row_key.value)])

    async def show_section(self, section: str) -> None:
        self.active_section = section
        self.active_panel = "menu"
        menu = self.query_one("#menu", ListView)
        if section == "HOME":
            menu.index = 0
        menu.focus()
        self.query_one("#screen-title", Static).update(section)
        self._hide_content_widgets()
        if section == "HOME":
            self.query_one("#screen-body", Static).update("Breed recognition workspace\n\nSelect a section to begin.")
        elif section == "PREDICT":
            self.query_one("#screen-body", Static).update("Select an image to preview it and run prediction.")
            self.query_one("#file-selector").remove_class("hidden")
            await self.populate_files()
        elif section == "BREEDS":
            self.query_one("#screen-body", Static).update("Database reference. Model-supported breeds are tracked separately.")
            self.query_one("#breed-search").remove_class("hidden")
            self.query_one("#breed-table").remove_class("hidden")
            self.query_one("#breed-detail").remove_class("hidden")
            self.populate_breeds()
        elif section == "HISTORY":
            self.show_history()
        elif section == "MODEL":
            self.show_model()
        elif section == "SETTINGS":
            self.query_one("#settings", Static).remove_class("hidden")
            self.query_one("#settings", Static).update(self._settings_text())
        elif section == "ABOUT":
            self.query_one("#about", Static).remove_class("hidden")
            self.query_one("#about", Static).update("Indian Breed AI\n\nA terminal-first research tool for Indian cattle and buffalo breed recognition.\n\nPredictions are only shown when a trained model is available.")

    def _hide_content_widgets(self) -> None:
        for widget_id in ("#breed-search", "#breed-table", "#breed-detail", "#history-table", "#history-detail", "#file-selector", "#prediction", "#settings", "#about"):
            self.query_one(widget_id).add_class("hidden")
        self.query_one("#screen-body", Static).update("")

    async def populate_directories(self) -> None:
        directory_list = self.query_one("#directory-list", ListView)
        await directory_list.remove_children()
        self.directory_targets.clear()
        self.current_path.mkdir(parents=True, exist_ok=True)
        parent = self.current_path.parent if self.current_path != self.dataset_root else self.dataset_root
        parent_id = "dir-parent"
        self.directory_targets[parent_id] = parent
        directory_list.append(ListItem(Label(".."), id=parent_id))
        try:
            directories = sorted(path for path in self.current_path.iterdir() if path.is_dir() and not path.name.startswith("."))
        except OSError:
            directories = []
        for index, directory in enumerate(directories):
            directory_id = f"dir-{index}"
            self.directory_targets[directory_id] = directory
            directory_list.append(ListItem(Label(f"[DIR] {directory.name}"), id=directory_id))

    async def populate_files(self) -> None:
        await self.populate_directories()
        file_list = self.query_one("#file-list", ListView)
        await file_list.remove_children()
        self.file_targets.clear()
        try:
            files = sorted(
                path for path in self.current_path.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
        except OSError:
            files = []
        if not files:
            file_list.append(ListItem(Label("No supported images found"), id="file-empty"))
        for index, image in enumerate(files):
            file_id = f"file-{index}"
            self.file_targets[file_id] = image
            relative_path = image.relative_to(self.current_path)
            file_list.append(ListItem(Label(str(relative_path)), id=file_id))
        self.query_one("#preview", Static).update(
            f"{self.current_path}\n\nSelect an image to inspect it and run prediction."
        )

    def show_preview(self, image_path: Path) -> None:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format or "unknown"
            self.query_one("#preview", Static).update(
                f"{image_path.name}\n\nFormat: {image_format}\nDimensions: {width} x {height}\nSize: {image_path.stat().st_size:,} bytes\n\nTerminal image rendering is unavailable; metadata preview shown."
            )
        except (OSError, UnidentifiedImageError) as error:
            self.query_one("#preview", Static).update(f"Unable to preview {image_path.name}: {error}")

    def on_key(self, event) -> None:
        if event.key == "enter" and self.active_section == "PREDICT" and self.active_panel == "files":
            files = self.query_one("#file-list", ListView)
            if files.highlighted_child and (files.highlighted_child.id or "").startswith("file-"):
                self.run_prediction(self.file_targets[files.highlighted_child.id])
        elif event.key == "right" and self.active_section == "PREDICT":
            self.active_panel = "files"
            self.query_one("#file-list", ListView).focus()
        elif event.key == "left" and self.active_section == "PREDICT":
            self.active_panel = "directories"
            self.query_one("#directory-list", ListView).focus()

    def run_prediction(self, image_path: Path) -> None:
        try:
            if self._predictor is None:
                self._predictor = Predictor(
                    self.settings.model_dir,
                    confidence_threshold=self.settings.confidence_threshold,
                )
            result = self._predictor.predict(image_path, top_k=3)
            record = self.history_store.append(
                image_path.name,
                result["animal_type"],
                result["predicted_breed"],
                result["confidence"],
                self._predictor.model_version,
            )
            self.history.insert(0, record)
            self.show_prediction(result, image_path.name)
            self.show_breed_support(result["predicted_breed"])
        except InferenceError as error:
            self.query_one("#prediction", Static).remove_class("hidden")
            self.query_one("#prediction", Static).update(f"PREDICTION UNAVAILABLE\n\n{error}")

    def show_prediction(self, result: dict[str, Any], filename: str) -> None:
        status = result["confidence_status"]
        lines = [
            f"Image: {filename}",
            f"Animal: {result['animal_type'].title()}",
            f"Breed: {result['predicted_breed']}",
            f"Confidence score: {result['confidence']:.1%}",
            f"Threshold: {result['confidence_threshold']:.1%}",
            f"Status: {'✓ HIGH CONFIDENCE' if status == 'HIGH' else '⚠ LOW CONFIDENCE'}",
        ]
        if result["manual_verification_recommended"]:
            lines.append("Manual verification recommended.")
        lines.extend(["", "TOP PREDICTIONS"])
        for item in result["top_predictions"]:
            bar = "█" * max(1, int(item["confidence"] * 24)) + "░" * max(0, 24 - int(item["confidence"] * 24))
            lines.append(f"{item['breed']:<16} {bar} {item['confidence']:.1%}")
        self.query_one("#prediction", Static).remove_class("hidden")
        self.query_one("#prediction", Static).update("\n".join(lines))

    def show_breed_support(self, breed_name: str) -> None:
        try:
            breed = self.repository.get_breed(breed_name)
        except Exception:
            return
        self.query_one("#breed-detail", Static).remove_class("hidden")
        self.query_one("#breed-detail", Static).update(
            f"SUPPORTING BREED INFORMATION\n"
            f"Breed: {breed.breed_name}\n"
            f"Origin: {breed.origin_region}\n"
            f"Animal type: {breed.animal_type}\n"
            f"Characteristics: {breed.physical_characteristics}\n"
            f"Identification notes: {breed.identification_notes}\n\n"
            "Reference metadata only; it does not prove the image belongs to this breed."
        )

    def populate_breeds(self, query: str = "") -> None:
        table = self.query_one("#breed-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Type", "Breed", "Confidence")
        breeds = self.repository.search(query) if query.strip() else self.repository.get_all_breeds()
        for breed in breeds:
            table.add_row(breed.animal_type, breed.breed_name, breed.confidence)

    def show_breed_detail(self, breed: Breed) -> None:
        self.query_one("#breed-detail", Static).update(
            f"{breed.breed_name} ({breed.animal_type})\nOrigin: {breed.origin_region}\nPurpose: {', '.join(breed.purpose)}\nColor: {breed.color}\nHorns: {breed.horn_characteristics}\nNotes: {breed.identification_notes}"
        )

    def show_history(self) -> None:
        self.query_one("#history-table").remove_class("hidden")
        self.query_one("#history-detail").remove_class("hidden")
        self.query_one("#history-table", DataTable).clear(columns=True)
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Time", "Animal", "Breed", "Confidence", "Model")
        self.history_targets.clear()
        for index, entry in enumerate(self.history):
            key = str(index)
            self.history_targets[key] = entry
            timestamp = entry.get("timestamp", "")
            time_text = timestamp[11:16] if len(timestamp) >= 16 else timestamp
            table.add_row(time_text, entry.get("animal_type", "").upper(), entry.get("predicted_breed", ""), f"{entry.get('confidence', 0):.1%}", entry.get("model_version", "unknown"), key=key)
        self.query_one("#screen-body", Static).update("Select a history entry to view its details." if self.history else "No prediction history is available yet.")

    def show_history_detail(self, entry: dict[str, Any]) -> None:
        self.query_one("#history-detail", Static).update(
            f"Image: {entry.get('image_filename', 'unknown')}\n"
            f"Breed: {entry.get('predicted_breed', 'unknown')}\n"
            f"Animal type: {entry.get('animal_type', 'unknown')}\n"
            f"Confidence score: {entry.get('confidence', 0):.1%}\n"
            f"Model version: {entry.get('model_version', 'unknown')}\n\n"
            "This record is a previous model output, not independent verification."
        )

    def show_model(self) -> None:
        model = self.settings.model_dir / "best_model.ts"
        labels = self.settings.model_dir / "classes.json"
        self.query_one("#screen-body", Static).update(f"Model: {'installed' if model.exists() else 'not installed'}\nLabels: {'available' if labels.exists() else 'not available'}\n\nNo performance claims are shown without real evaluation data.")

    def _settings_text(self) -> str:
        return f"Project root: {self.settings.project_root}\nModel directory: {self.settings.model_dir}\nCurrent file directory: {self.current_path}\nConfidence threshold: {self.settings.confidence_threshold:.1%}\nOverride with BREED_CONFIDENCE_THRESHOLD.\nSupported images: JPG, JPEG, PNG, WEBP"

    async def action_refresh(self) -> None:
        if self.active_section == "PREDICT":
            await self.populate_files()
        elif self.active_section == "BREEDS":
            self.populate_breeds()

    async def action_back(self) -> None:
        await self.show_section("HOME")

    def action_focus_menu(self) -> None:
        self.active_panel = "menu"
        self.query_one("#menu", ListView).focus()

    def action_focus_files(self) -> None:
        if self.active_section == "PREDICT":
            self.active_panel = "files"
            self.query_one("#file-list", ListView).focus()

    def action_help(self) -> None:
        self.query_one("#screen-body", Static).update("Keys\n\nUp/Down  Navigate\nLeft/Right  Switch panels\nEnter  Select / predict\nEsc  Home\nR  Refresh\nQ  Quit")
