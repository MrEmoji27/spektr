"""The run-merge tolerance, and the colour error it is allowed to cause.

``make_strips`` may let a colour run drift before it splits, which is how a
smooth field costs a few hundred Segments instead of ten thousand. What bounds
it is ``_RLE_MAX_RGB``: no cell may end up more than that far, per channel,
from the colour it asked for.

That budget is now per ramp index rather than one number for the whole ramp,
because one number is decided by the ramp's steepest segment — on `classic`
that made it 0, so the merge never fired on the default theme.

The test that matters is the last one. The first version of the per-index
budget checked only that rgb[i+t] was close to rgb[i], but a run absorbs
values *below* its start as well as above, and `classic` drifted 16/255
against a bound of 10. Constructing it correctly is not the same as it being
correct, so this measures the colour actually emitted for every inked cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
import spektr.render as R  # noqa: E402
from spektr import palette as P  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402

THEMES = ["classic", "gruvbox", "nord", "rainbow"]
BLANK = {R.SPACE, R.BRAILLE_BASE}


def _ctx(w, h, frame, state, pal):
    bands = np.linspace(0.2, 0.95, N_BANDS).astype(np.float32)
    t = np.arange(512, dtype=np.float32) / 512.0
    return Ctx(
        w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
        wave=np.sin(t * 40.0), stereo=np.stack([np.sin(t * 40.0)] * 2, axis=1),
        frame=frame, t=frame / 60.0, dt=1 / 60.0, energy=0.6,
        silent=False, palette=pal, state=state, bars=N_BANDS,
    )


@pytest.mark.parametrize("theme", THEMES)
def test_the_budget_is_symmetric(theme):
    """A run absorbs colours below its start as well as above.

    The asymmetric version passed every structural check and still broke the
    bound, because it only ever compared upward.
    """
    pal = Palette(BUILTIN[theme])
    rgb = np.clip(np.rint(pal.rgb), 0, 255).astype(int)
    for i, t in enumerate(pal.rle_budget.tolist()):
        for k in range(1, t + 1):
            for j in (i - k, i + k):
                if 0 <= j < len(rgb):
                    err = int(np.abs(rgb[j] - rgb[i]).max())
                    assert err <= P._RLE_MAX_RGB, (
                        f"{theme}: index {i} may drift {t}, but {j} is {err} away"
                    )


@pytest.mark.parametrize("theme", THEMES)
def test_the_budget_never_exceeds_the_cap(theme):
    b = Palette(BUILTIN[theme]).rle_budget
    assert b.min() >= 0 and b.max() <= P._RLE_MAX_TOL
    assert len(b) == P.RAMP_STEPS


@pytest.mark.parametrize("theme", THEMES)
def test_no_inked_cell_drifts_past_the_bound(theme):
    """The property the whole feature is allowed to exist under.

    Blank cells are excluded on purpose: they carry no ink, and the blank fold
    deliberately gives them a neighbour's colour so they stop splitting runs.
    """
    pal = Palette(BUILTIN[theme])
    rgb = np.clip(np.rint(pal.rgb), 0, 255).astype(int)
    R.set_cell_mode("octant")

    worst = 0
    for name in ("Plasma", "Auroras", "Arcs", "Chladni", "Vinyl", "Bars", "Tunnel"):
        state: dict = {}
        for frame in range(8):
            out = M.get(name).fn(_ctx(160, 40, frame, state, pal))
        codes, cidx = out[0], out[1]
        bidx = out[2] if len(out) > 2 else None
        for row, strip in enumerate(R.make_strips(codes, cidx, pal, bidx)):
            x = 0
            for seg in strip._segments:
                got = seg.style.color.triplet
                got = np.array([got.red, got.green, got.blue])
                for _ in seg.text:
                    if x < cidx.shape[1] and int(codes[row, x]) not in BLANK:
                        err = int(np.abs(got - rgb[int(cidx[row, x])]).max())
                        worst = max(worst, err)
                    x += 1
    assert worst <= P._RLE_MAX_RGB, f"{theme}: a cell drifted {worst}/255"


def test_the_merge_actually_fires_on_the_default_theme():
    """The point of going per-index: `classic` had a global tolerance of 0."""
    pal = Palette(BUILTIN["classic"])
    assert pal.rle_tol == 0, "if this theme got a global tolerance, retune the test"
    assert pal.rle_budget.max() > 0, "the per-index budget is doing nothing"
    assert pal.rle_budget.mean() > 0.5


def test_a_smooth_field_costs_far_fewer_segments_than_cells():
    """Plasma is the mode this was measured on: ~9,400 Segments down to ~2,300."""
    pal = Palette(BUILTIN["classic"])
    R.set_cell_mode("octant")
    state: dict = {}
    for frame in range(10):
        codes, cidx, bidx = M.get("Plasma").fn(_ctx(400, 100, frame, state, pal))
    segs = sum(len(s._segments) for s in R.make_strips(codes, cidx, pal, bidx))
    assert segs < codes.size // 8, f"{segs} Segments for {codes.size} cells"
