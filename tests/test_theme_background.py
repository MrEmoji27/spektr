"""Every visualizer cell has to name the theme's background colour.

Textual draws the lines a Line-API widget returns exactly as they are handed
over; the widget's own ``styles.background`` only fills padding. So a segment
whose style has no bgcolor goes to the terminal as "default background", and
the terminal paints its own scheme there. On Windows Terminal running
Catppuccin Mocha at 80% opacity, gruvbox rendered on #1e1e2e and flexoki-light
on a dark ground instead of cream — logged at the byte level as foreground-only
SGR sequences for every cell below the header.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import SPACE, make_strips  # noqa: E402


@pytest.mark.parametrize("theme", sorted(BUILTIN))
def test_every_cell_carries_the_theme_background(theme):
    pal = Palette(BUILTIN[theme])
    codes = np.full((6, 30), SPACE, dtype=np.int32)
    codes[2, 3:9] = ord("█")
    codes[4, 10] = ord("·")
    cidx = np.tile(np.arange(30, dtype=np.int32) * 2, (6, 1))
    want = BUILTIN[theme].bg.lower()
    for strip in make_strips(codes, cidx, pal):
        for seg in strip:
            assert seg.style is not None and seg.style.bgcolor is not None, "a cell with no background"
            assert seg.style.bgcolor.triplet.hex == want


def test_an_animated_theme_keeps_its_background_as_the_ramp_turns():
    pal = Palette(BUILTIN["rainbow"])
    codes = np.full((2, 8), ord("█"), dtype=np.int32)
    cidx = np.tile(np.arange(8, dtype=np.int32) * 8, (2, 1))
    for phase in (0.0, 0.37, 0.81):
        pal.set_phase(phase)
        for strip in make_strips(codes, cidx, pal):
            for seg in strip:
                assert seg.style.bgcolor.triplet.hex == BUILTIN["rainbow"].bg.lower()
