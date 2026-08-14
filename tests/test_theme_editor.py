"""Theme editor colour selection tests — the swatches picker and hex entry.

The editor grew two selection paths next to the HSL nudge rows: a named
colour picker ("swatches" row) and a free-text hex entry ("hex" row). Both
write into the draft slot the editor's colour row is on, then live-preview —
and anything picked must still satisfy the same visibility rule the audit
suite applies to built-in themes, or the editor's check row warns about it.

Redirects palette.config_dir the same way test_app.py does: the app writes
settings and saved themes on exit, and a test that typed a theme name into
the real config tree would be a repeat of the preset-leak bug this project
already had once.

The integration test drives the real app headlessly the way test_app.py does,
so it gets the same two concessions: a slow frame rate (the pilot waits for
the app to go idle between keys, and 60 fps never is) and a stubbed notify.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.palette as _palette  # noqa: E402

_scratch_dir = Path(tempfile.mkdtemp(prefix="spektr-theme-test-"))
_palette.config_dir = lambda: _scratch_dir  # type: ignore[attr-defined]

from spektr import config  # noqa: E402
from spektr.app import Spektr  # noqa: E402
from spektr.palette import (  # noqa: E402
    BUILTIN,
    NAMED_COLOURS,
    ThemeDraft,
    resolve_colour,
    theme_visibility_problems,
)
from spektr.pickers import ColourPicker, HexPrompt, SettingsPanel  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── resolve_colour ───────────────────────────────────────────────────────────

def test_resolve_colour() -> None:
    cases = {
        "#ff4800": "#ff4800",
        "  #F80 ": "#ff8800",
        "ff0000": "#ff0000",
        "cyan": "#00ffff",
        "CYAN": "#00ffff",
        "hot pink": "#ff1493",
    }
    for given, want in cases.items():
        got = resolve_colour(given)
        assert got == want, f"resolve_colour({given!r}) = {got!r}, want {want!r}"

    for given in ("", "  ", "#fff0", "#fffff", "notacolour", "##f80", "ff", "zzzz"):
        assert resolve_colour(given) is None, f"resolve_colour({given!r}) should be None"


def test_named_colours_are_real_hexes() -> None:
    for name, colour in NAMED_COLOURS.items():
        assert resolve_colour(colour) == colour, f"{colour} ({name}) does not round-trip"
        assert resolve_colour(name) == colour, f"name {name!r} does not resolve to {colour}"


# ── picking into a slot ──────────────────────────────────────────────────────

def test_pick_lands_in_the_slot() -> None:
    draft = ThemeDraft(BUILTIN["gruvbox"])
    before = draft.hex_of("low")
    draft.set_slot("mid", "#ff4800")
    assert draft.hex_of("mid") == "#ff4800"
    assert draft.hex_of("low") == before, "picking one slot must not move another"

    draft.nudge("mid", "h", 0.5)
    assert draft.hex_of("mid") != "#ff4800", "nudge rows still fine-tune after a pick"


def test_picked_colour_still_passes_audit() -> None:
    draft = ThemeDraft(BUILTIN["gruvbox"])
    draft.set_slot("low", "#00ff41")
    assert theme_visibility_problems(draft.to_theme()) == []

    draft.set_slot("low", "#050000")
    assert theme_visibility_problems(draft.to_theme()), "near-black anchor must warn"


def test_advanced_slot_can_be_picked() -> None:
    draft = ThemeDraft(BUILTIN["gruvbox"])
    draft.set_advanced(True)
    draft.set_slot("bg", "#101820")
    assert draft.hex_of("bg") == "#101820"


# ── the editor itself, driven headless ───────────────────────────────────────

async def _row_index(app: Spektr, label: str) -> int:
    panel = app._overlay
    rows = panel.query_one("#rows")
    for i, option in enumerate(rows._options):
        if str(option.prompt).lstrip().startswith(label):
            return i
    raise AssertionError(f"no {label!r} row")


async def _goto(app: Spektr, pilot, label: str) -> None:
    target = await _row_index(app, label)
    for _ in range(target):
        await pilot.press("down")
    await pilot.pause()


async def _open_editor(app: Spektr, pilot) -> None:
    await pilot.press("c")
    await pilot.pause()
    await _goto(app, pilot, "theme editor")
    await pilot.press("right")
    await pilot.pause()
    await pilot.pause()


def test_editor_picker_and_hex_mutate_the_slot() -> None:
    async def run() -> None:
        app = Spektr(settings=config.Settings(fps=15))
        app.notify = lambda *a, **k: None  # type: ignore[method-assign]
        async with app.run_test(size=(120, 32)) as pilot:
            viz = app.viz
            viz.apply_theme("gruvbox")
            viz.commit_theme()

            await _open_editor(app, pilot)
            assert isinstance(app._overlay, SettingsPanel)

            # swatches row → named colour picker. "red" matches one entry,
            # enter commits it into the selected slot (low).
            await _goto(app, pilot, "swatches")
            await pilot.press("right")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app._overlay, ColourPicker)
            for ch in "red":
                await pilot.press(ch)
            await pilot.pause()
            assert app._overlay._shown == ["#ff0000"], "filter must narrow to red"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app._overlay, SettingsPanel)
            assert viz.palette.theme.low == "#ff0000"

            # the picker acts on the colour row's slot: move it to mid and
            # pick cyan there — low must not move with it.
            await _goto(app, pilot, "colour")
            await pilot.press("right")
            await pilot.pause()
            await _goto(app, pilot, "swatches")
            await pilot.press("right")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app._overlay, ColourPicker)
            for ch in "cyan":
                await pilot.press(ch)
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert viz.palette.theme.low == "#ff0000"
            assert viz.palette.theme.mid == "#00ffff"

            # hex row → free text. Garbage keeps the prompt open with the
            # draft untouched; a valid value commits to the selected slot.
            await _goto(app, pilot, "hex")
            await pilot.press("right")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app._overlay, HexPrompt)
            for ch in "zzzz":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app._overlay, HexPrompt), "garbage must keep the prompt open"
            assert viz.palette.theme.mid == "#00ffff", "garbage must not touch the draft"
            for _ in range(4):
                await pilot.press("backspace")
            for ch in "00ff41":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app._overlay, SettingsPanel)
            assert viz.palette.theme.mid == "#00ff41"
            assert viz.palette.theme.low == "#ff0000"

            # the whole edited theme still passes the audit rule the editor's
            # check row runs.
            assert theme_visibility_problems(viz.palette.theme) == []

    asyncio.run(run())