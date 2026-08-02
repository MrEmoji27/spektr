"""Filterable picker overlay with live preview.

Cycling blind through a list with a toast notification is fine for three
options and unusable for thirty. This is the pattern cliamp uses and it's the
right one: a narrow panel down one side so the visualiser stays visible,
arrow keys preview as you move, Enter keeps, Escape puts back what you had.
"""

from __future__ import annotations

from typing import Callable, Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList


def _fuzzy(query: str, candidate: str) -> bool:
    """Subsequence match — 'ctp' finds 'catppuccin'."""
    if not query:
        return True
    q = query.lower()
    c = candidate.lower()
    if q in c:
        return True
    i = 0
    for ch in c:
        if ch == q[i]:
            i += 1
            if i == len(q):
                return True
    return False


class Picker(ModalScreen[str | None]):
    """Returns the chosen item, or None if cancelled."""

    CSS = """
    Picker {
        align: right top;
        background: transparent;
    }
    Picker > #panel {
        width: 32;
        height: 100%;
        background: $surface;
        border-left: tall $accent;
        padding: 0 1;
    }
    Picker #title {
        color: $accent;
        text-style: bold;
        padding: 1 0 0 0;
    }
    Picker Input {
        border: none;
        background: $surface;
        padding: 0;
        margin: 0 0 1 0;
    }
    Picker OptionList {
        background: $surface;
        border: none;
        height: 1fr;
        scrollbar-size-vertical: 1;
    }
    Picker #hint {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "move(-8)", "Page up", show=False),
        Binding("pagedown", "move(8)", "Page down", show=False),
        Binding("enter", "choose", "Choose", show=False),
    ]

    def __init__(
        self,
        title: str,
        items: Sequence[str],
        current: str | None = None,
        on_preview: Callable[[str], None] | None = None,
        labels: dict[str, str] | None = None,
    ):
        super().__init__()
        self._title = title
        self._items = list(items)
        self._current = current
        self._on_preview = on_preview
        self._labels = labels or {}
        self._shown: list[str] = list(items)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Label(self._title, id="title")
            yield Input(placeholder="filter…", id="filter")
            yield OptionList(id="list")
            yield Label("↑↓ preview · ⏎ keep · esc cancel", id="hint")

    def on_mount(self) -> None:
        self._repopulate()
        ol = self.query_one("#list", OptionList)
        if self._current in self._shown:
            ol.highlighted = self._shown.index(self._current)
        self.query_one("#filter", Input).focus()

    # ── list management ──
    def _label_for(self, name: str) -> str:
        extra = self._labels.get(name)
        mark = "▸ " if name == self._current else "  "
        return f"{mark}{name}" + (f"  {extra}" if extra else "")

    def _repopulate(self, query: str = "") -> None:
        self._shown = [i for i in self._items if _fuzzy(query, i)]
        ol = self.query_one("#list", OptionList)
        ol.clear_options()
        if self._shown:
            ol.add_options([self._label_for(i) for i in self._shown])
            ol.highlighted = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        self._repopulate(event.value)
        self._preview_current()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_choose()

    def on_option_list_option_highlighted(self, event) -> None:
        self._preview_current()

    # ── actions ──
    def _selected(self) -> str | None:
        ol = self.query_one("#list", OptionList)
        i = ol.highlighted
        if i is None or not self._shown or i >= len(self._shown):
            return None
        return self._shown[i]

    def _preview_current(self) -> None:
        name = self._selected()
        if name and self._on_preview:
            self._on_preview(name)

    def action_move(self, delta: int) -> None:
        ol = self.query_one("#list", OptionList)
        if not self._shown:
            return
        cur = ol.highlighted or 0
        ol.highlighted = max(0, min(len(self._shown) - 1, cur + delta))
        self._preview_current()

    def action_choose(self) -> None:
        self.dismiss(self._selected())

    def action_cancel(self) -> None:
        self.dismiss(None)


class Setting:
    """One row of the settings panel: a label, a value, and how to change it.

    Values are picked from a fixed list of stops rather than typed. Every
    setting here has a small set of sensible values and a live preview, so
    stepping through them with the arrow keys is both faster than typing and
    impossible to get wrong — there is no invalid state to validate.
    """

    __slots__ = ("key", "label", "choices", "render", "apply", "note")

    def __init__(self, key, label, choices, render, apply, note=""):
        self.key = key
        self.label = label
        self.choices = list(choices)
        self.render = render          # value -> str
        self.apply = apply            # value -> None
        self.note = note

    def index_of(self, value) -> int:
        """Nearest stop to the current value, so an --fps 37 still lands somewhere."""
        if value in self.choices:
            return self.choices.index(value)
        numeric = [c for c in self.choices if isinstance(c, (int, float))]
        if numeric and isinstance(value, (int, float)):
            nearest = min(numeric, key=lambda c: abs(c - value))
            return self.choices.index(nearest)
        return 0


class SettingsPanel(ModalScreen[None]):
    """Live settings, in the same shape as the pickers.

    Everything applies as you move — the visualiser is right there behind the
    panel, and a settings screen you have to close to see the effect of is a
    settings screen you fight with. There is no OK button for the same reason.
    """

    CSS = """
    SettingsPanel {
        align: right top;
        background: transparent;
    }
    SettingsPanel > #panel {
        width: 42;
        height: 100%;
        background: $surface;
        border-left: tall $accent;
        padding: 0 1;
    }
    SettingsPanel #title {
        color: $accent;
        text-style: bold;
        padding: 1 0 0 0;
    }
    SettingsPanel OptionList {
        background: $surface;
        border: none;
        height: 1fr;
        scrollbar-size-vertical: 1;
    }
    SettingsPanel #hint {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape,enter", "close", "Close", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("left", "step(-1)", "Lower", show=False),
        Binding("right", "step(1)", "Raise", show=False),
        Binding("h", "step(-1)", "Lower", show=False),
        Binding("l", "step(1)", "Raise", show=False),
    ]

    def __init__(self, settings: Sequence[Setting], values: dict):
        super().__init__()
        self._settings = list(settings)
        self._values = dict(values)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Label("settings", id="title")
            yield OptionList(id="rows")
            yield Label("↑↓ row · ←→ change · esc done", id="hint")

    def on_mount(self) -> None:
        self._repaint()
        self.query_one("#rows", OptionList).focus()

    def _row_text(self, s: Setting) -> str:
        value = s.render(self._values[s.key])
        pad = " " * max(1, 14 - len(s.label))
        line = f"  {s.label}{pad}{value}"
        return f"{line}\n    [dim]{s.note}[/dim]" if s.note else line

    def _repaint(self) -> None:
        rows = self.query_one("#rows", OptionList)
        keep = rows.highlighted or 0
        rows.clear_options()
        rows.add_options([self._row_text(s) for s in self._settings])
        rows.highlighted = min(keep, len(self._settings) - 1)

    def _current(self) -> Setting | None:
        i = self.query_one("#rows", OptionList).highlighted
        if i is None or i >= len(self._settings):
            return None
        return self._settings[i]

    def action_move(self, delta: int) -> None:
        rows = self.query_one("#rows", OptionList)
        cur = rows.highlighted or 0
        rows.highlighted = max(0, min(len(self._settings) - 1, cur + delta))

    def action_step(self, delta: int) -> None:
        s = self._current()
        if s is None:
            return
        i = s.index_of(self._values[s.key])
        i = max(0, min(len(s.choices) - 1, i + delta))
        value = s.choices[i]
        self._values[s.key] = value
        s.apply(value)
        self._repaint()

    def action_close(self) -> None:
        self.dismiss(None)
