"""Filterable picker overlays with live preview.

Cycling blind through a list with a toast notification is fine for three
options and unusable for thirty. This is the pattern cliamp uses and it's the
right one: a narrow panel down one side, arrow keys preview as you move, Enter
keeps, Escape puts back what you had.

The panels are *overlay widgets on the same screen* rather than ``ModalScreen``
layers. A pushed modal covers the whole terminal and hides whatever is beneath
it, so the visualiser disappeared the moment you opened a picker — you could
see nothing previewed live, because there was nothing to see. As a docked,
upper-layer widget the visualiser stays mounted and repainting behind the
panel, so the theme (or mode) you arrow through really shows up on the bands.
"""

from __future__ import annotations

from typing import Callable, Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Input, Label, OptionList

from .palette import resolve_colour


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


class Picker(Widget):
    """A docked overlay panel returning the chosen item, or None if cancelled.

    ``on_done(value)`` is called exactly once, when the user chooses (with a
    string) or cancels (with None). The caller is expected to remove this
    widget afterwards.

    Styling lives in the App CSS (see Spektr.CSS) because a widget's own CSS
    attribute is not applied to widgets mounted after the app starts.
    """

    BINDINGS = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("pageup", "move(-8)", "Page up", show=False),
        Binding("pagedown", "move(8)", "Page down", show=False),
        Binding("enter", "choose", "Choose", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        title: str,
        items: Sequence[str],
        current: str | None = None,
        on_preview: Callable[[str], None] | None = None,
        labels: dict[str, str] | None = None,
        on_done: Callable[[str | None], None] | None = None,
        display: dict[str, str] | None = None,
    ):
        super().__init__()
        self._title = title
        self._items = list(items)
        self._current = current
        self._on_preview = on_preview
        self._labels = labels or {}
        #: item -> the name to *show*, where that differs from the name the
        #: item is selected and stored by. The subcell variants are registered
        #: as ``(o)`` and shown as ``(q)`` while the cell setting says
        #: quadrant, because the suffix is there to report the geometry the
        #: mode is actually drawing with.
        self._display = display or {}
        self._shown: list[str] = list(items)
        self._on_done = on_done

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
        """One option's text: the name, plus its blurb on its own dim line.

        Blurbs run well past what fits beside the name in a docked side
        panel — some past sixty characters against a forty-column panel —
        and appending one inline just made every option wrap with no hanging
        indent, which reads as a run-on paragraph instead of a list. A
        second line, styled dim and indented under the name, is how
        SettingsPanel already shows a row's note; this just matches it.
        """
        extra = self._labels.get(name)
        mark = "▸ " if name == self._current else "  "
        line = f"{mark}{self._display.get(name, name)}"
        return f"{line}\n    [dim]{extra}[/dim]" if extra else line

    def _match(self, query: str, item: str) -> bool:
        """Whether an item survives the filter. ColourPicker overrides this
        to also match a colour's display name."""
        return _fuzzy(query, item)

    def _repopulate(self, query: str = "") -> None:
        self._shown = [i for i in self._items if self._match(query, i)]
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
        self._finish(self._selected())

    def action_cancel(self) -> None:
        self._finish(None)

    def _finish(self, value: str | None) -> None:
        cb = self._on_done
        if cb is not None:
            self._on_done = None
            cb(value)


class ColourPicker(Picker):
    """A :class:`Picker` over hex colours, shown with names and a swatch.

    The items are the hex strings themselves, so ``on_done`` hands back the
    exact colour to apply; ``names`` maps each hex to a display name, which
    the filter also searches — typing "red" finds ``#ff0000``. The swatch is
    a marked-up inline block that OptionList renders as a real colour patch:
    the point of a colour picker is to see the colour before choosing it.
    """

    def __init__(
        self,
        title: str,
        colours: Sequence[str],
        names: dict[str, str],
        current: str | None = None,
        on_preview: Callable[[str], None] | None = None,
        on_done: Callable[[str | None], None] | None = None,
    ):
        super().__init__(
            title, list(colours), current=current,
            on_preview=on_preview, on_done=on_done,
        )
        self._names = dict(names)

    def _match(self, query: str, item: str) -> bool:
        return _fuzzy(query, item) or _fuzzy(query, self._names.get(item, ""))

    def _label_for(self, colour: str) -> str:
        name = self._names.get(colour, colour)
        mark = "▸ " if colour == self._current else "  "
        return f"{mark}{name:<10} [on {colour}]      [/on] {colour}"


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
        note: str = "",
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

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Label("settings", id="title")
            yield OptionList(id="rows")
            yield Label("↑↓ row · ←→ change · esc done", id="hint")

    def on_mount(self) -> None:
        self._repaint()
        self.query_one("#rows", OptionList).focus()
        # a `live` row (source) can keep changing after the key that
        # triggered it — the capture thread settles on its own schedule, not
        # on the next keypress — so it needs its own refresh rather than
        # waiting on user input to notice.
        if any(s.live is not None for s in self._settings):
            self.set_interval(1.0, self._repaint)

    def _row_text(self, s: Setting) -> str:
        value = s.render(s.live() if s.live is not None else self._values[s.key])
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


#: Sentinel: ``NamePrompt._resolve`` returns it to keep the prompt open
#: rather than committing anything.
_KEEP_OPEN = object()


class NamePrompt(Widget):
    """A single text field for naming something new — save-as, not pick-one.

    Picker and SettingsPanel both choose from something that already exists;
    saving a preset needs a name typed in, which neither of them does. Same
    docked-overlay shape and the same ``on_done`` contract (called once, with
    the typed name or ``None`` on cancel) so it drops into ``_open_overlay``
    unchanged.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self,
        title: str,
        placeholder: str = "",
        on_done: Callable[[str | None], None] | None = None,
    ):
        super().__init__()
        self._title = title
        self._placeholder = placeholder
        self._on_done = on_done

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield Label(self._title, id="title")
            yield Input(placeholder=self._placeholder, id="name")
            yield Label("⏎ save · esc cancel", id="hint")

    def on_mount(self) -> None:
        self.query_one("#name", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = self._resolve(event.value)
        if value is _KEEP_OPEN:
            return
        self._finish(value)

    def _resolve(self, text: str) -> str | None | object:
        """Hook: what to commit for the typed text.

        Returns the value to commit, ``None`` for commit-nothing (enter on an
        empty field), or :data:`_KEEP_OPEN` to keep the prompt open. Textual
        runs a message handler on every MRO class that defines it, so the
        validation lives here instead of in a subclass ``on_input_submitted``
        — otherwise the parent's handler commits whatever the subclass
        rejected.
        """
        return text.strip() or None

    def action_cancel(self) -> None:
        self._finish(None)

    def _finish(self, value: str | None) -> None:
        cb = self._on_done
        if cb is not None:
            self._on_done = None
            cb(value)


class HexPrompt(NamePrompt):
    """A colour entry field — #rrggbb, #rgb, or a colour name.

    NamePrompt's shape and ``on_done`` contract, with validation via the
    ``_resolve`` hook: the input is resolved to a canonical hex before it is
    handed back, and text that is not a colour keeps the prompt open with an
    error rather than committing nothing. Same docked-overlay shape, so it
    drops into ``_open_overlay`` unchanged.
    """

    def __init__(
        self,
        title: str,
        current: str = "",
        on_done: Callable[[str | None], None] | None = None,
    ):
        placeholder = f"#rrggbb — e.g. {current}" if current else "#rrggbb or a colour name"
        super().__init__(title, placeholder=placeholder, on_done=on_done)

    def _resolve(self, text: str) -> str | None | object:
        colour = resolve_colour(text)
        if colour is None and text.strip():
            self.notify(
                f"{text.strip()!r} is not a colour — try #rrggbb or a name",
                severity="error",
                timeout=3,
            )
            return _KEEP_OPEN
        return colour
