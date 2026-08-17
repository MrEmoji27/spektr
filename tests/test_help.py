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
