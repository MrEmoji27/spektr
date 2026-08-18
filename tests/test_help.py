"""The `h` help panel, and the promise that it cannot go stale.

The panel is generated from the live ``BINDINGS`` list and the live mode
registry rather than written out. That is the whole design: a help screen is
read at the moment someone is least able to tell that it is wrong, and this
project's own README had drifted twice in a day before anything checked it.

So these do not check the wording. They check that every key the app answers
to is in there, that nothing in there is a key the app does not have, and that
the panel survives being built from a junk config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr.app import Spektr  # noqa: E402
from spektr.config import Settings  # noqa: E402
from spektr.pickers import HelpPanel, key_label  # noqa: E402
from spektr.widget import AudioVisualizer  # noqa: E402


def _panel(monkeypatch, **settings):
    app = Spektr(settings=Settings(**settings))
    viz = AudioVisualizer(settings=app.settings)
    monkeypatch.setattr(Spektr, "viz", property(lambda self: viz))
    grabbed = {}
    monkeypatch.setattr(Spektr, "_open_overlay",
                        lambda self, w, cb: grabbed.setdefault("w", w))
    app.action_help()
    return app, grabbed["w"]


def _rows(panel):
    return [row for _, entries in panel._sections for row in entries]


def test_h_is_bound_and_shown_in_the_footer():
    keys = {k.strip() for b in Spektr.BINDINGS for k in b.key.split(",")}
    assert "h" in keys, "no help key"
    binding = next(b for b in Spektr.BINDINGS if "h" in b.key.split(","))
    assert binding.action == "help"
    assert binding.show, "help is the one binding that must be discoverable"


def test_every_binding_appears_in_the_help(monkeypatch):
    """Including the ones hidden from the footer — those are the point."""
    _, panel = _panel(monkeypatch)
    shown = {left for left, _ in _rows(panel)}
    missing = [
        b.key for b in Spektr.BINDINGS
        if b.description and key_label(b.key) not in shown
    ]
    assert not missing, f"bindings the help never mentions: {missing}"


def test_the_help_lists_no_key_the_app_does_not_have(monkeypatch):
    """The other direction: a key removed from BINDINGS must vanish here too."""
    _, panel = _panel(monkeypatch)
    keys_section = dict(panel._sections)["keys"]
    real = {key_label(b.key) for b in Spektr.BINDINGS if b.description}
    assert {left for left, _ in keys_section} == real


def test_punctuation_keys_are_spelled_the_way_a_keyboard_has_them():
    assert key_label("left_square_bracket") == "["
    assert key_label("right_square_bracket") == "]"
    assert key_label("h,question_mark") == "h / ?"
    assert key_label("m,space") == "m / space"


def test_the_help_reports_the_live_mode_counts(monkeypatch):
    _, panel = _panel(monkeypatch)
    now = dict(panel._sections)["now"]
    modes = dict(now)["modes"]
    assert str(len(M.listed())) in modes
    assert str(len([m for m in M.MODES if m.hidden])) in modes


def test_the_help_reports_the_geometry_actually_set(monkeypatch):
    for cells, suffix in (("octant", "(o)"), ("quadrant", "(q)")):
        _, panel = _panel(monkeypatch, cells=cells)
        row = dict(dict(panel._sections)["now"])["subcells"]
        assert cells in row and suffix in row


def test_the_panel_builds_from_a_junk_config(monkeypatch):
    """Help is what someone reaches for when things are already wrong."""
    _, panel = _panel(monkeypatch, fps="soon", bands=None, cells="sextants",
                      mode=None, theme=42)
    assert _rows(panel)


def test_closing_is_bound_to_the_obvious_keys():
    keys = {k.strip() for b in HelpPanel.BINDINGS for k in b.key.split(",")}
    assert {"escape", "q", "h"} <= keys, "the key that opened it should close it"


def test_no_row_is_blank(monkeypatch):
    _, panel = _panel(monkeypatch)
    for left, right in _rows(panel):
        assert right, f"row {left!r} has nothing to say"


# ── the markup the panel actually hands to Textual ───────────────────────────

def test_every_help_row_is_markup_textual_can_parse(monkeypatch):
    """Pressing `h` crashed the app, and no test noticed.

    The rows are built as Textual markup, and two bindings are labelled with
    the characters `[` and `]` — the sensitivity keys. Textual reads `[` as
    the start of a tag, so the row came out as
    `"  [           [dim]Sens -[/dim]"`: one unterminated tag, then a
    `[/dim]` closing nothing. That raises MarkupError inside the compositor's
    reflow, so it is not a row that looks wrong, it is the whole app going
    down on layout.

    Every other test here checks the *content* of the rows — which binding
    appears, which section it is in — and content was never the problem. This
    parses them, which is what the app actually does with them.
    """
    from textual.content import Content

    _, panel = _panel(monkeypatch)
    bad = []
    for line in panel._lines():
        try:
            Content.from_markup(line)
        except Exception as exc:                     # noqa: BLE001 — reporting
            bad.append(f"{line!r}: {type(exc).__name__}: {exc}")
    assert not bad, "help rows Textual cannot parse:\n" + "\n".join(bad)



def test_the_bracket_keys_survive_being_put_in_markup():
    """The two rows that broke it, named so a refactor cannot quietly drop them."""
    from textual.content import Content

    from spektr.pickers import key_label, markup_safe

    for key in ("left_square_bracket", "right_square_bracket"):
        label = key_label(key)
        assert label in "[]", f"{key} should read as a bracket, got {label!r}"
        row = f"  {markup_safe(label)}    [dim]Sens[/dim]"
        assert Content.from_markup(row).plain == f"  {label}    Sens"


def test_markup_safe_handles_what_escape_does_not():
    """`textual.markup.escape` only escapes tag-shaped text.

    `escape("[b]")` is escaped and `escape("[")` is returned unchanged, which
    is precisely the case that crashed. Pinned because reaching for the
    library helper is the obvious fix and it is not sufficient.
    """
    from textual.content import Content

    from spektr.pickers import markup_safe

    for raw in ("[", "]", "[b]", "[/dim]", "no brackets here", "a [ b ] c"):
        assert Content.from_markup(markup_safe(raw) + " [dim]x[/dim]").plain == raw + " x"

def test_pressing_h_shows_the_panel_without_eating_the_chrome():
    """The end-to-end version, because both failures were in layout, not content.

    Twice now this panel has been broken in a way no unit test could see. The
    first was markup that would not parse, which raised out of the
    compositor. The second was worse to diagnose from a description: the
    panel was in none of the app's CSS rules, so it had no ``layer: overlay``
    and no width, mounted into the screen's own vertical layout at full size,
    pushed the Header and Footer off the screen, and painted nothing — its
    background is transparent and only ``#panel`` carries a surface. Pressing
    ``h`` looked exactly like a key that hides the chrome.

    So this checks the three things a mounted-and-populated assertion missed:
    the chrome is still there, the panel is docked on the overlay layer rather
    than sitting in the layout, and the rows are actually painted.
    """
    import asyncio

    from textual.geometry import Region
    from textual.widgets import Footer, Header

    async def drive():
        app = Spektr(settings=Settings(mode="Bars", theme="gruvbox"))
        async with app.run_test(size=(120, 40)) as pilot:
            header, footer = app.query(Header).first(), app.query(Footer).first()
            before = (header.region, footer.region)

            await pilot.press("h")
            await pilot.pause()

            panels = [w for w in app.screen.walk_children() if isinstance(w, HelpPanel)]
            assert panels, "h did not open the help panel"
            panel = panels[0]

            assert (header.region, footer.region) == before, (
                "the help panel displaced the header/footer instead of overlaying them"
            )
            assert panel.styles.layer == "overlay", (
                f"the panel is on layer {panel.styles.layer!r}, so it is in the layout"
            )
            assert 0 < panel.region.width < app.screen.region.width, (
                f"the panel spans {panel.region.width} of "
                f"{app.screen.region.width} columns — it is not docked"
            )

            rows = app.query_one("#rows")
            assert rows.size.width > 0 and rows.size.height > 0, "the rows have no room"
            painted = []
            for strip in rows.render_lines(
                Region(0, 0, rows.size.width, rows.size.height)
            ):
                painted.append("".join(seg.text for seg in strip))
            screen_text = " ".join(painted)
            assert "Mode" in screen_text, "the panel is mounted but paints nothing"
            # The two rows whose labels are brackets — the markup bug.
            assert "Sens" in screen_text, "the sensitivity rows are missing"

            await pilot.press("escape")
            await pilot.pause()
            assert app.is_running

    asyncio.run(drive())
