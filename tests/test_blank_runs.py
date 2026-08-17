"""Blank cells inherit their neighbour's colour so they stop splitting runs.

A space and a blank braille cell paint no foreground, so their colour index is
not visible — but ``make_strips`` compared it anyway, and a sparse mode is
mostly blanks, so every gap between two lit cells cost two extra Segments for
colours nobody can see. At 400x100 that was 61% of Vinyl's colour runs and
48% of Ember's.

The thing that has to stay true is that no *lit* cell's colour moves. These
pin that, because "it got faster and still looks right" is not a property a
future change can be checked against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import BRAILLE_BASE, SPACE, make_strips  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])


def _colours_per_cell(strips, w):
    """Read back the colour actually painted at each cell, from the Segments."""
    out = []
    for strip in strips:
        row, x = [None] * w, 0
        for seg in strip._segments:
            for _ in seg.text:
                if x < w:
                    row[x] = seg.style
                x += 1
        out.append(row)
    return out


def _grid(codes, cidx):
    return make_strips(np.asarray(codes, dtype=np.int32),
                       np.asarray(cidx, dtype=np.int32), PAL)


def test_a_lit_cell_keeps_its_own_colour_across_a_blank_gap():
    """The invariant the whole optimisation rests on."""
    lit = ord("#")
    codes = [[lit, SPACE, SPACE, lit, BRAILLE_BASE, lit]]
    cidx = [[10, 0, 0, 40, 0, 63]]
    painted = _colours_per_cell(_grid(codes, cidx), 6)[0]

    assert painted[0] == PAL.styles[10]
    assert painted[3] == PAL.styles[40], "a lit cell took the blank's colour"
    assert painted[5] == PAL.styles[63], "a lit cell took the blank's colour"


def test_a_blank_gap_does_not_split_a_run():
    """Two lit cells of one colour with blanks between them are one Segment."""
    lit = ord("#")
    codes = [[lit, SPACE, SPACE, SPACE, lit]]
    same = _grid(codes, [[25, 0, 0, 0, 25]])
    assert len(same[0]._segments) == 1, "blanks still cost Segments"

    # ...and a real colour change still does split one.
    diff = _grid(codes, [[25, 0, 0, 0, 60]])
    assert len(diff[0]._segments) > 1, "a real colour change stopped splitting"


def test_a_row_that_starts_blank_is_fine():
    """Nothing to inherit from — the fill must not read off the left edge."""
    lit = ord("#")
    strips = _grid([[SPACE, SPACE, lit]], [[0, 0, 33]])
    assert _colours_per_cell(strips, 3)[0][2] == PAL.styles[33]


def test_an_all_blank_row_is_a_single_segment():
    strips = _grid([[SPACE] * 8], [[0, 9, 18, 27, 36, 45, 54, 63]])
    assert len(strips[0]._segments) == 1


def test_a_grid_with_no_blanks_is_untouched():
    """The fill is skipped entirely, and the result is what it always was.

    Colours far apart on purpose: neighbouring indices merge anyway under
    ``palette.rle_tol``, which is a different feature and would hide this one.
    """
    codes = [[ord("#")] * 5]
    cidx = [[1, 1, 32, 32, 63]]
    assert len(_grid(codes, cidx)[0]._segments) == 3


def _ctx(w, h, frame, state, bands):
    return Ctx(
        w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
        wave=np.sin(np.arange(512, dtype=np.float32) * 0.1),
        stereo=np.zeros((512, 2), dtype=np.float32),
        frame=frame, t=frame / 60.0, dt=1 / 60.0, energy=0.6,
        silent=False, palette=PAL, state=state, bars=len(bands),
    )


def test_no_mode_has_a_lit_cell_recoloured():
    """The same invariant, over every mode that draws on the terminal ground.

    A unit grid proves the arithmetic; this proves no real mode produces a
    shape the fill mishandles.
    """
    bands = np.linspace(0.25, 0.95, N_BANDS).astype(np.float32)
    for m in M.MODES:
        state: dict = {}
        for frame in range(6):
            out = m.fn(_ctx(90, 26, frame, state, bands))
        if len(out) > 2:
            continue                     # fg+bg pairs: a blank's ground shows
        codes, cidx = out[0], out[1]
        h, w = codes.shape
        blank = (codes == SPACE) | (codes == BRAILLE_BASE)
        if not blank.any():
            continue
        src = np.where(blank, np.int32(0), np.arange(w, dtype=np.int32)[None, :])
        np.maximum.accumulate(src, axis=1, out=src)
        filled = np.take_along_axis(cidx, src, axis=1)
        assert np.array_equal(filled[~blank], cidx[~blank]), (
            f"{m.name}: the fill moved a lit cell's colour"
        )


def test_the_goniometer_is_actually_circular():
    """Its geometry *is* the reading, so a stretched display is a wrong one.

    A goniometer is read by shape: uncorrelated stereo of equal level draws a
    circle, mono draws a line. The mode scaled each axis by its own half-extent
    and drew the trace into whatever rectangle the terminal was — measured at
    400x100, a circle came out 684 x 344 dots — while the comment above it
    claimed a squash that was never applied.
    """
    n = 2048
    t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    stereo = (np.stack([np.sin(t), np.cos(t)], axis=1) * 0.9).astype(np.float32)
    bands = np.full(N_BANDS, 0.5, dtype=np.float32)

    for w, h in ((400, 100), (120, 30)):
        state: dict = {}
        for frame in range(3):
            codes, _ = M.get("Gonio").fn(Ctx(
                w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
                wave=stereo[:, 0].copy(), stereo=stereo, frame=frame,
                t=frame / 60.0, dt=1 / 60.0, energy=0.6, silent=False,
                palette=PAL, state=state, bars=N_BANDS,
            ))
        ys, xs = np.nonzero(codes != BRAILLE_BASE)
        # cells -> dots: 2 wide and 4 tall per cell, and a terminal cell is
        # about twice as tall as wide, so a braille dot is square.
        wide = (xs.max() - xs.min() + 1) * 2
        tall = (ys.max() - ys.min() + 1) * 4
        assert abs(wide / tall - 1.0) < 0.10, (
            f"{w}x{h}: a circle drew {wide}x{tall} dots, aspect {wide / tall:.2f}"
        )
