"""The live settings panel and its row model."""

from __future__ import annotations

from typing import Callable, Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Label, OptionList, Static

from .pickers import markup_safe


class Setting:
    """One row of the settings panel: a label, a value, and how to change it.

    Values are picked from a fixed list of stops rather than typed. Every
    setting here has a small set of sensible values and a live preview, so
    stepping through them with the arrow keys is both faster than typing and
    impossible to get wrong — there is no invalid state to validate.

    Audio source doesn't fit that shape — there's no fixed list to pick a
    stop from, "next" is a one-way cycle spektr already does its own way (see
    app.py's ``next_source``/``default_source``), and the value worth showing
    is a live status string, not something read back out of a values dict.
    ``step``/``live`` are the escape hatch for exactly that row without
    forcing every other row through the same generality: when ``step`` is
    set, the panel calls it directly instead of walking ``choices``; when
    ``live`` is set, the displayed value comes from calling it fresh instead
    of from the values dict, so the row reflects device switching actually
    settling rather than the instant the key was pressed.
    """

    def __init__(
        self,
        key: str,
        label: str,
        choices: list,
        render: Callable[[object], str] | None = None,
        apply: Callable[[object], None] | None = None,
        note: "str | Callable[[], str]" = "",
        step: Callable[[int], None] | None = None,
        live: Callable[[], object] | None = None,
    ):
        self.key = key
        self.label = label
        self.choices = choices
        self.note = note
        self._apply = apply
        self._render = render or str
        self.step = step
        self.live = live

    def render(self, value) -> str:
        return self._render(value)

    def about(self) -> str:
        """What this setting does, said when its row is the one you are on.

        A note can be a callable, for a row whose explanation depends on the
        state it is in: shuffle's says whether shuffle is on at all, and the
        frame rate's says what display rate was detected.
        """
        return self.note() if callable(self.note) else self.note

    def apply(self, value) -> None:
        if self._apply:
            self._apply(value)

    def index_of(self, value) -> int:
        for i, choice in enumerate(self.choices):
            if choice == value:
                return i
        if isinstance(value, (int, float)):
            nearest = min(self.choices, key=lambda c: abs(c - value))
            return self.choices.index(nearest)
        return 0


class SettingsPanel(Widget):
    """Live settings, in the same docked-overlay shape as the pickers.

    Everything applies as you move — the visualiser is right there behind the
    panel, and a settings screen you have to close to see the effect of is a
    settings screen you fight with. There is no OK button for the same reason.

    Each row is one line, its name and its value, and the values line up in
    one column. What a setting does is said once, in the box under the list,
    for the row you are on. Every row used to carry its note underneath it,
    wrapped to the panel's edge rather than to the row's indent, so the panel
    was a column of half-sentences that the rows had to be picked out of.
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

    #: Characters a row has for its name and value: the panel's width, less
    #: its border, padding and scrollbar, and the list's own margin.
    ROW_WIDTH = 42

    def __init__(
        self,
        settings: Sequence[Setting],
        values: dict,
        on_done: Callable[[], None] | None = None,
    ):
        super().__init__()
        self._settings = list(settings)
        self._values = dict(values)
        self._on_done = on_done
        self._on: int | None = None
        self._label_width = max((len(s.label) for s in self._settings), default=0)

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Label("settings", id="title")
            yield OptionList(id="rows")
            yield Static("", id="about")
            yield Label("↑↓ choose · ←→ change · esc done", id="hint")

    def on_mount(self) -> None:
        rows = self.query_one("#rows", OptionList)
        rows.add_options([self._row_text(i) for i in range(len(self._settings))])
        rows.styles.max_height = len(self._settings)
        rows.highlighted = 0 if self._settings else None
        rows.focus()
        self._show(0)
        # a `live` row (source) can keep changing after the key that
        # triggered it — the capture thread settles on its own schedule, not
        # on the next keypress — so it needs its own refresh rather than
        # waiting on user input to notice.
        if any(s.live is not None for s in self._settings):
            self.set_interval(1.0, self._repaint)

    def _value(self, s: Setting) -> str:
        return s.render(s.live() if s.live is not None else self._values[s.key])

    def _row_text(self, i: int) -> str:
        s = self._settings[i]
        label = s.label.ljust(self._label_width)
        room = max(4, self.ROW_WIDTH - self._label_width - 2)
        value = self._value(s)
        if i == self._on:
            # the row you are on shows it can be changed, left and right
            value = f"‹ {_fit(value, room - 4)} ›"
            return f"{markup_safe(label)}  [b]{markup_safe(value)}[/b]"
        return f"{markup_safe(label)}  [dim]{markup_safe(_fit(value, room))}[/dim]"

    def _show(self, i: int | None) -> None:
        """Make row ``i`` the one you are on: mark it, and say what it does."""
        rows = self.query_one("#rows", OptionList)
        old, self._on = self._on, i
        for j in {old, i} - {None}:
            if j < len(self._settings):
                rows.replace_option_prompt_at_index(j, self._row_text(j))
        about = self._settings[i].about() if i is not None and i < len(self._settings) else ""
        self.query_one("#about", Static).update(markup_safe(about))

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id == "rows":
            self._show(event.option_index)

    def _repaint(self) -> None:
        rows = self.query_one("#rows", OptionList)
        for i in range(len(self._settings)):
            rows.replace_option_prompt_at_index(i, self._row_text(i))
        self._show(self._on)

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
        if s.step is not None:
            s.step(delta)
            self._repaint()
            return
        i = s.index_of(self._values[s.key])
        i = max(0, min(len(s.choices) - 1, i + delta))
        value = s.choices[i]
        self._values[s.key] = value
        s.apply(value)
        self._repaint()

    def action_close(self) -> None:
        cb = self._on_done
        if cb is not None:
            self._on_done = None
            cb()


def _fit(text: str, room: int) -> str:
    """``text`` cut to ``room`` characters, with an ellipsis when it was cut."""
    return text if len(text) <= room else text[: max(0, room - 1)] + "…"
