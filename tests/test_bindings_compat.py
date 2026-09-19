"""Every key does what it did in 0.5.0. The help panel is generated from
these bindings (see test_help.py); this pins the bindings themselves."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.app import Spektr  # noqa: E402

EXPECTED = [
    ("m,space", "cycle_mode"),
    ("M", "cycle_mode(-1)"),
    ("v", "pick_mode"),
    ("l", "loadout"),
    ("t", "pick_theme"),
    ("c", "settings"),
    ("T", "cycle_theme"),
    ("f", "toggle_chrome"),
    ("d", "next_source"),
    ("D", "default_source"),
    ("r", "reload"),
    ("s", "toggle_shuffle"),
    ("left_square_bracket", "gain(-1)"),
    ("right_square_bracket", "gain(1)"),
    ("g", "gate(-1)"),
    ("G", "gate(1)"),
    ("h,question_mark", "help"),
    ("p", "show_perf"),
    ("q", "quit"),
]


def test_the_bindings_are_the_0_5_0_bindings():
    got = [(b.key, b.action) for b in Spektr.BINDINGS]
    assert got == EXPECTED
