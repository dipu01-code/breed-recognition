"""Professional terminal interface for the breed recognizer."""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static, TabbedContent, TabPane, DataTable

from app.core.breeds import list_breeds


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
    ListItem { padding: 1; }
    ListItem.--highlight { background: #2b5878; color: #ffffff; }
    DataTable { height: 1fr; }
    .muted { color: #9db0bd; }
    """

    BINDINGS = [("q", "quit", "Quit"), ("escape", "back", "Back"), ("?", "help", "Help")]
    sections = ("HOME", "PREDICT", "BREEDS", "HISTORY", "MODEL", "ABOUT")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Vertical(id="navigation"):
                yield Label("INDIAN BREED\nRECOGNIZER", id="title")
                yield ListView(*(ListItem(Label(section), id=section.lower()) for section in self.sections), id="menu")
            with Container(id="content"):
                yield Static(id="screen-title")
                yield Static(id="screen-body")
                yield DataTable(id="breed-table")
        yield Static("Ready | Up/Down navigate  Enter select  Esc back  ? help  Q quit", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#menu", ListView).focus()
        self.show_section("HOME")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.show_section(event.item.id.upper())

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
        table = self.query_one("#breed-table", DataTable)
        table.clear(columns=True)
        table.display = section == "BREEDS"
        if section == "BREEDS":
            table.add_columns("Type", "Breed", "Confidence")
            for breed in list_breeds():
                table.add_row(breed.animal_type, breed.breed_name, breed.confidence)

    def action_back(self) -> None:
        self.show_section("HOME")

    def action_help(self) -> None:
        self.query_one("#screen-body", Static).update("Keyboard navigation\n\nUp/Down: navigate\nEnter: select\nEsc: home\nQ: quit")