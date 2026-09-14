"""Tunnel's spokes are straight lines into the vanishing point.

The corridor is sixteen radial lines and a train of rings. In the live
terminal the lines broke up exactly where they should read cleanest — the
inner third, converging on the centre — into a curve and then a scatter of
dots. Three separate terms did it, and none of them was intended curvature:

* a ``depth * 0.03`` twist on the spoke angle, with depth ``max_r / dist``, so
  the sideways offset grew without bound toward the centre (about 5 degrees
  ten dots out, against a spoke two degrees wide);
* a fixed angular width, under a dot wide inside about 28 dots of the centre;
* the depth dither, applied to the spokes as well as the rings.

The window-ratio stretch of the shared polar grid is linear, so it makes the
corridor elliptical and cannot bend a line; it is not part of this.

These are measured on the rendered braille, unpacked back to dots, because the
picture is what went wrong.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import BRAILLE_BASE  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])
_BITS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1)]


def _dots(codes: np.ndarray) -> np.ndarray:
    h, w = codes.shape
    d = np.zeros((h * 4, w * 2), dtype=bool)
    b = np.where((codes >= BRAILLE_BASE) & (codes < BRAILLE_BASE + 256), codes - BRAILLE_BASE, 0)
    for i, (ry, rx) in enumerate(_BITS):
        d[ry::4, rx::2] = ((b >> i) & 1).astype(bool)
    return d


def _frames(name, w, h, frames=12, energy=0.0):
    fn = M.get(name).fn
    state: dict = {}
    bands = np.full(N_BANDS, energy)
    for f in range(frames):
        ctx = Ctx(w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
                  wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=f, t=f / 60, dt=1 / 60,
                  energy=energy, silent=energy == 0.0, palette=PAL, state=state)
        yield _dots(fn(ctx)[0])


def _spoke_inner_third(d: np.ndarray):
    """Per spoke, over the inner third: continuity and the offset of the line."""
    dr, dc = d.shape
    cx, cy = dc / 2.0, dr / 2.0
    xs_ = cy / max(cx, 1.0)
    max_r = min(cy, cx * xs_)
    cont, offsets = [], []
    for k in range(16):
        th = k / 16 * 2 * math.pi
        ux, uy = math.cos(th), math.sin(th)
        hits = steps = 0
        for r in np.arange(0.07 * max_r + 1.0, 0.35 * max_r, 1.0):
            gx, gy = cx + r * ux / xs_, cy + r * uy
            ys, xs = np.mgrid[int(gy) - 2:int(gy) + 3, int(gx) - 3:int(gx) + 4]
            ok = (ys >= 0) & (ys < dr) & (xs >= 0) & (xs < dc)
            ys, xs = ys[ok], xs[ok]
            lit = d[ys, xs]
            steps += 1
            if lit.any():
                perp = np.abs((xs[lit] - cx) * xs_ * uy - (ys[lit] - cy) * ux)
                hits += int((perp <= 1.2).any())
                offsets.append(float(perp.min()))
        cont.append(hits / max(steps, 1))
    return float(np.mean(cont)), float(np.percentile(offsets, 90))


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
@pytest.mark.parametrize("name", ["Tunnel", "Tunnel In"])
def test_the_spokes_stay_straight_and_unbroken_into_the_centre(name, size):
    for d in _frames(name, *size, frames=6):
        continuity, offset = _spoke_inner_third(d)
        assert continuity > 0.95, f"{name} spokes are broken near the centre ({continuity:.2f})"
        assert offset < 0.3, f"{name} spokes wander off their line near the centre ({offset:.2f} dots)"


@pytest.mark.parametrize("name", ["Tunnel", "Tunnel In"])
def test_the_spokes_do_not_shimmer(name):
    """The spokes are geometry: frame to frame they must not change at all.

    The dither used to be applied to them, which re-rolled which spoke dots
    were drawn every frame; only the rings may thin out with depth.
    """
    fn = M.get(name).fn
    state: dict = {}
    z = np.zeros(N_BANDS)
    walls = None
    for f in range(8):
        ctx = Ctx(w=120, h=40, bands=z, peaks=z, bands_l=z, bands_r=z, wave=np.zeros(512),
                  stereo=np.zeros((512, 2)), frame=f, t=f / 60, dt=1 / 60, energy=0.0,
                  silent=True, palette=PAL, state=state)
        d = _dots(fn(ctx)[0])
        geo = next(v for k, v in state.items() if k[0] == "tunnel_geo")
        spokes = d & geo["walls"]
        if walls is not None:
            assert np.array_equal(spokes, walls), "spoke dots changed between frames"
        walls = spokes
        assert np.array_equal(spokes, geo["walls"]), "a spoke dot was dropped"
