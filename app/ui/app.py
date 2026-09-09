"""Professional terminal interface for the breed recognizer."""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Label, ListItem, ListView, Static

from app.data.breed_repository import BreedRepository


class BreedRecognizerApp(App[None]):
    """Full-screen keyboard-navigable application shell."""

    CSS = """
    Screen { background: #101820; color: #e7edf2; }
    Header { background: #17324d; color: #f4c95d; }
    Footer { background: #0a1118; }
    #layout { height: 1fr; }
    #navigation { width: 24; background: #142637; border: round #2b5878; padding: 1; }
    #content { width: 1fr; border: round #2b5878; padding: 2; }
    #title { color: #f4c95d; text-style: bold; margin-bottom: 1; }
    #status { dock: bottom; height: 1; background: #17324d; color: #b9c8d3; padding: 0 1; }
    #breed-search { margin: 1 0; display: none; }
    #breed-detail { margin-top: 1; border: round #2b5878; padding: 1; display: none; }
    ListItem { padding: 1; }
    ListItem.--highlight { background: #2b5878; color: #ffffff; }
    DataTable { height: 1fr; }
    .muted { color: #9db0bd; }
    """

    BINDINGS = [("q", "quit", "Quit"), ("escape", "back", "Back"), ("?", "help", "Help")]
    sections = ("HOME", "PREDICT", "BREEDS", "HISTORY", "MODEL", "ABOUT")
    repository = BreedRepository()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Vertical(id="navigation"):
                yield Label("INDIAN BREED\nRECOGNIZER", id="title")
                yield ListView(*(ListItem(Label(section), id=section.lower()) for section in self.sections), id="menu")
            with Container(id="content"):
                yield Static(id="screen-title")
                yield Static(id="screen-body")
                yield Input(placeholder="Search breed name or alias...", id="breed-search")
                yield DataTable(id="breed-table")
                yield Static(id="breed-detail")
        yield Static("Ready | Up/Down navigate  Enter select  Esc back  ? help  Q quit", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#menu", ListView).focus()
        self.show_section("HOME")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.show_section(event.item.id.upper())

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "breed-search":
            self.populate_breeds(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        breed = event.data_table.get_row(event.row_key)
        detail = self.repository.get_breed(str(breed[1]))
        self.query_one("#breed-detail", Static).update(
            f"{detail.breed_name} ({detail.animal_type})\n"
            f"Origin: {detail.origin_region}\n"
            f"Purpose: {', '.join(detail.purpose)}\n"
            f"Color: {detail.color}\n"
            f"Horns: {detail.horn_characteristics}\n"
            f"Notes: {detail.identification_notes}\n"
            "\nDatabase record only; model support is tracked separately."
        )

    def show_section(self, section: str) -> None:
        self.query_one("#screen-title", Static).update(section)
        body = {
            "HOME": "Breed recognition workspace\n\nSelect a section to begin.",
            "PREDICT": "Prediction is unavailable until a trained model is installed.\nNo prediction is fabricated.",
            "BREEDS": "Browse the Indian cattle and buffalo breed reference database.",
            "HISTORY": "No prediction history is available yet.",
            "MODEL": "Model status: baseline model not installed.\nTraining and export are available through the project scripts.",
            "ABOUT": "Indian Breed Recognizer\nA research-oriented tool for Indian cattle and buffalo breed recognition.",
        }[section]
        self.query_one("#screen-body", Static).update(body)
        search = self.query_one("#breed-search", Input)
        detail = self.query_one("#breed-detail", Static)
        search.display = section == "BREEDS"
        detail.display = section == "BREEDS"
        if section == "BREEDS":
            self.populate_breeds()

    def populate_breeds(self, query: str = "") -> None:
        table = self.query_one("#breed-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Type", "Breed", "Confidence")
        breeds = self.repository.search(query) if query.strip() else self.repository.get_all_breeds()
        for breed in breeds:
            table.add_row(breed.animal_type, breed.breed_name, breed.confidence)

    def action_back(self) -> None:
        self.show_section("HOME")

    def action_help(self) -> None:
        self.query_one("#screen-body", Static).update("Keyboard navigation\n\nUp/Down: navigate\nEnter: select\nEsc: home\nQ: quit")