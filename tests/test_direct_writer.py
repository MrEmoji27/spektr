"""The direct writer draws the same cells, and only the cells that moved.

The writer exists to skip Textual's compositor, not to change the picture. The
check that it has not is here rather than in a golden file: every case walks
the strips ``make_strips`` builds, reads back the glyphs and colours the writer
emitted, and compares them cell for cell.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden  # noqa: E402

from spektr import modes as M  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import direct, make_strips  # noqa: E402

MOVE = re.compile(r"\[(\d+);(\d+)H")


def _cells_of_strips(strips) -> list[list[tuple]]:
    """``(glyph, fg, bg)`` per cell, read out of the strips themselves."""
    grid = []
    for strip in strips:
        row = []
        for segment in strip:
            style = segment.style
            fg = tuple(segment.style.color.get_truecolor()[:3]) if style.color else None
            bg = tuple(style.bgcolor.get_truecolor()[:3]) if style.bgcolor else None
            row.extend((ch, fg, bg) for ch in segment.text)
        grid.append(row)
    return grid


def _cells_of_ansi(text: str, cols: int, rows: int) -> list[list[tuple]]:
    """Read the writer's escapes back into the same shape.

    Longhand on purpose: this is also the only place the wire format — a cursor
    move, then a run — is written down.
    """
    grid: list[list[tuple]] = [[None] * cols for _ in range(rows)]
    y = x = 0
    fg = bg = None
    for chunk in text.split("\x1b")[1:]:
        if "H" in chunk.split("m")[0]:
            move, _, chunk = chunk.partition("H")
            y, x = (int(v) - 1 for v in move.lstrip("[").split(";"))
        else:
            params, _, chunk = chunk.partition("m")
            fg = bg = None
            values = [int(v) for v in params.lstrip("[").split(";") if v]
            i = 0
            while i < len(values):
                if values[i] == 38 and values[i + 1] == 2:
                    fg = tuple(values[i + 2:i + 5])
                    i += 5
                elif values[i] == 48 and values[i + 1] == 2:
                    bg = tuple(values[i + 2:i + 5])
                    i += 5
                else:
                    i += 1
        for ch in chunk:
            if 0 <= y < rows and x < cols:
                grid[y][x] = (ch, fg, bg)
            x += 1
    return grid


def _frame(mode_name: str, w: int, h: int) -> tuple:
    case = golden.Case("beat", w, h)
    mode = M.get(mode_name)
    state: dict = {}
    onset = 0
    out = None
    palette = Palette(BUILTIN["gruvbox"])
    for i in range(8):
        ctx, onset = golden._ctx(case, i, palette, state, onset)
        out = mode.fn(ctx)
    return out


def _ramp(palette) -> list[tuple[int, int, int]]:
    return [
        (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in palette.hexes
    ]


def _slack(palette, ramp) -> int:
    """How far a colour may move and still be the same colour on this theme.

    A run may absorb a step of ``rle_tol`` ramp steps, so the furthest one of
    those can carry a channel is the ramp's steepest step times that. Measured
    in bytes rather than in ramp steps on purpose: a gentle ramp rounds many
    steps onto the same byte triple, and counting steps off a hex cannot tell
    them apart.
    """
    step = max(
        abs(a - b)
        for one, other in zip(ramp, ramp[1:])
        for a, b in zip(one, other)
    )
    return step * int(palette.rle_budget.max())


@pytest.mark.parametrize("mode", ["Bars", "Swell", "Chladni", "Scope"])
def test_the_writer_draws_the_cells_the_strips_do(mode):
    """Same glyphs and colours, to the tolerances the strips themselves keep.

    Two of those, both of them the strip builder's own and neither of them a
    difference on screen:

    * a run keeps the colour it started on and absorbs cells that drift within
      ``rle_tol`` of it — a segment count decision, invisible by construction;
    * a blank cell in a foreground-only frame inherits its neighbour's colour,
      because a space paints no foreground for it to be seen in.

    The writer has no segments to save and no runs to watch drift, so it draws
    each cell's own colour: never further from the mode's than the strip's is.
    """
    w, h = 60, 20
    palette = Palette(BUILTIN["gruvbox"])
    ramp = _ramp(palette)
    slack = _slack(palette, ramp)
    frame = _frame(mode, w, h)
    codes, cidx = frame[0], frame[1]
    bidx = frame[2] if len(frame) == 3 else None

    strips = make_strips(codes, cidx, palette, bidx, None)
    screen = direct.Screen(w, h)
    ansi = screen.frame(codes, cidx, bidx, palette, None)
    assert ansi, "a first frame drew nothing"

    expected = _cells_of_strips(strips)
    got = _cells_of_ansi(ansi, w, h)
    for y in range(h):
        for x in range(w):
            g_glyph, g_fg, g_bg = got[y][x]
            w_glyph, w_fg, w_bg = expected[y][x]
            assert g_glyph == w_glyph, f"{mode} {x},{y}: {g_glyph!r} != {w_glyph!r}"
            for what, mine, theirs in (("bg", g_bg, w_bg), ("fg", g_fg, w_fg)):
                if what == "fg" and bidx is None and g_glyph in (" ", "\u2800"):
                    continue  # nothing draws it, so it is nobody's business
                assert (mine is None) == (theirs is None), (
                    f"{mode} {x},{y}: {what} {mine} against {theirs} — one names a "
                    "colour and the other does not"
                )
                if mine is None:
                    continue
                assert all(abs(a - b) <= slack for a, b in zip(mine, theirs)), (
                    f"{mode} {x},{y}: {what} {mine} against {theirs} is further "
                    f"than rle_tol reaches on this theme"
                )


def test_an_unchanged_frame_writes_nothing_and_a_changed_one_writes_that_row():
    w, h = 16, 6
    palette = Palette(BUILTIN["gruvbox"])
    codes = np.full((h, w), 0x1CD00, dtype=np.int32)
    cidx = np.full((h, w), 20, dtype=np.int32)
    screen = direct.Screen(w, h)
    first = screen.frame(codes, cidx, None, palette)
    assert MOVE.search(first), "the first frame has to position and draw"
    assert screen.frame(codes, cidx, None, palette) == "", "a still frame redrew"

    moved = codes.copy()
    moved[3] = 0x1CD10 + 1
    again = screen.frame(moved, cidx, None, palette)
    rows = [int(m.group(1)) for m in MOVE.finditer(again)]
    assert rows == [4], f"a one-row change touched rows {rows}"


def test_the_writer_starts_where_the_widget_is():
    """Cells land on the widget's own rows and columns, not the screen's."""
    w, h = 8, 4
    palette = Palette(BUILTIN["gruvbox"])
    codes = np.full((h, w), 0x1CD00, dtype=np.int32)
    cidx = np.full((h, w), 5, dtype=np.int32)
    screen = direct.Screen(w + 4, h + 2, x=4, y=2)
    ansi = screen.frame(codes, cidx, None, palette)
    move = MOVE.search(ansi)
    assert (int(move.group(1)), int(move.group(2))) == (3, 5)


def test_every_mode_draws_one_glyph_a_cell():
    """The writer maps a cell to a column, which a wide glyph would break.

    Not a rule the renderer can enforce at speed — checking widths per frame
    would cost as much as drawing — so it is a rule about the modes instead,
    checked here over every built-in one at every recorded case size.
    """
    bad = []
    for mode in golden.builtin_modes():
        for case in golden.CASES:
            for code in np.unique(_frame(mode.name, case.w, case.h)[0]):
                ch = chr(int(code))
                # Ambiguous-width glyphs are the ones every mode here draws —
                # box drawing, block elements — and a terminal gives them one
                # column. Full or wide ones would shift the rest of the row.
                if unicodedata.combining(ch) or unicodedata.east_asian_width(ch) in ("F", "W"):
                    bad.append(f"{mode.name} {case.id}: {ch!r} (U+{int(code):04X})")
    assert not bad, "a mode draws something wider than a cell: " + "; ".join(bad[:5])
