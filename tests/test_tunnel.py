"""Tunnel is a wireframe: thin, even strokes into a vanishing point.

Two rounds of this, both found by looking at the terminal rather than at the
arithmetic.

The first was straightness. The spokes bent and scattered in the inner third:
a ``depth * 0.03`` twist on the angle grew without bound toward the centre, a
fixed angular width fell under a dot, and the depth dither thinned the lines.

The second was line quality, measured on the rendered braille at 188x50 (the
size of the screenshot that raised it). The corridor was built from bands of
the stretched polar coordinates, so:

* a spoke's width was quantised from a one-dot hairline to a three- or
  four-dot band at a single radius, and the window's x stretch made
  horizontal strokes heavier than vertical ones;
* a ring was a fixed fraction of one depth period — tens of dots near the rim,
  so a loud ring was a filled annulus (79% of lit dots in solid 5x5 blocks)
  with a per-frame dither punching holes in it;
* near the centre that fraction was under a dot, and the dither left 60-odd
  specks of fewer than six dots.

These are measured on the braille unpacked back to dots, because the picture
is what went wrong, and a geometry test that passes is not the same thing as
lines that read.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import BRAILLE_BASE  # noqa: E402

PAL = Palette(BUILTIN["hackerman"])
MODES = ["Tunnel", "Tunnel In"]
SIZES = [(188, 50), (120, 40), (60, 20)]
_BITS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1)]
_X = np.linspace(0.0, 1.0, N_BANDS)


def _dots(codes: np.ndarray) -> np.ndarray:
    h, w = codes.shape
    d = np.zeros((h * 4, w * 2), dtype=bool)
    b = np.where((codes >= BRAILLE_BASE) & (codes < BRAILLE_BASE + 256), codes - BRAILLE_BASE, 0)
    for i, (ry, rx) in enumerate(_BITS):
        d[ry::4, rx::2] = ((b >> i) & 1).astype(bool)
    return d


def _play(name, w, h, frames=30, energy=0.0):
    """Render a passage; return the state and the dot grids of every frame."""
    fn = M.get(name).fn
    state: dict = {}
    out = []
    for f in range(frames):
        bands = np.clip(energy * (0.7 + 0.3 * np.sin(f * 0.2 + _X * 6)), 0, 1)
        ctx = Ctx(w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
                  wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=f, t=f / 60, dt=1 / 60,
                  energy=float(bands.mean()), silent=energy == 0.0, palette=PAL, state=state)
        out.append(_dots(fn(ctx)[0]))
    return state, out


def _geo(state):
    return next(v for k, v in state.items() if k[0] == "tunnel_geo")


def _components(d: np.ndarray) -> np.ndarray:
    dr, dc = d.shape
    seen = np.zeros(d.shape, dtype=bool)
    sizes = []
    for y0, x0 in zip(*np.nonzero(d)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        q, n = deque([(y0, x0)]), 0
        while q:
            y, x = q.popleft()
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < dr and 0 <= xx < dc and d[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        q.append((yy, xx))
        sizes.append(n)
    return np.array(sizes)


@pytest.mark.parametrize("energy", [0.0, 0.7])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", MODES)
def test_no_scattered_fragments_and_no_heavy_blocks(name, size, energy):
    _, frames = _play(name, *size, frames=24, energy=energy)
    for d in frames[-6:]:
        sizes = _components(d)
        assert int((sizes < 6).sum()) == 0, f"{name} {size}: {int((sizes < 6).sum())} specks of under six dots"
        win = sliding_window_view(np.pad(d, 2), (5, 5)).sum(axis=(-1, -2))
        heavy = float((win[d] >= 18).mean())
        # 37-79% before, depending on size and level; the wireframe's worst
        # measured case is under 9%, a loud ring crossing spoke stubs at 60x20
        assert heavy < 0.10, f"{name} {size}: {100 * heavy:.0f}% of lit dots sit in solid blocks"


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", MODES)
def test_spokes_are_thin_even_strokes_with_a_gentle_taper(name, size):
    state, _ = _play(name, *size, frames=2)
    walls = _geo(state)["walls"]
    dr, dc = walls.shape
    cx, cy = dc / 2.0, dr / 2.0
    s = cy / cx
    for k in range(16):
        th = k / 16 * 2 * math.pi
        ux, uy = math.cos(th) / s, math.sin(th)
        norm = math.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        px, py = -uy, ux
        widths = []
        for r in np.arange(0.0, 2.0 * max(cx, cy), 1.0):
            x0, y0 = cx + ux * r, cy + uy * r
            if not (2 <= x0 < dc - 2 and 2 <= y0 < dr - 2):
                break
            cells = {(int(round(y0 + py * t)), int(round(x0 + px * t))) for t in np.arange(-4, 4.01, 0.25)}
            wd = sum(1 for y, x in cells if 0 <= y < dr and 0 <= x < dc and walls[y, x])
            # Near the vanishing point the neighbouring spokes are closer than
            # the +-4 dot window measures across, so their dots would count as
            # width; judge each spoke where its neighbours are clear of it.
            if wd and r * 2 * math.pi / 16 > 10:
                widths.append(wd)
        widths = np.array(widths)
        assert widths.size > 5, f"spoke {k} is missing"
        assert widths.max() <= 4, f"spoke {k} swells to {widths.max()} dots"
        # consecutive samples a dot apart never jump by more than a dot
        assert np.all(np.abs(np.diff(widths)) <= 2), f"spoke {k} changes weight abruptly: {widths.tolist()}"


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", MODES)
def test_rings_are_even_all_the_way_round_and_none_is_too_small(name, size):
    state, frames = _play(name, *size, frames=20, energy=0.3)
    geo = _geo(state)
    d = frames[-1]
    rings = d & ~geo["walls"]
    dr, dc = d.shape
    cx, cy = dc / 2.0, dr / 2.0
    s = cy / cx
    ys, xs = np.nonzero(rings)
    if ys.size == 0:
        pytest.skip("no ring on screen this frame")
    dist = np.hypot((xs - cx) * s, ys - cy)
    max_r = max(1.0, cy - 1.0)
    cut = math.sqrt(3.5 * 0.55 * max_r)
    assert dist.min() >= cut - 1.0, "a ring was drawn inside the radius where rings cannot be resolved"
    # stroke thickness of the ring, measured along vertical and horizontal cuts
    thick = []
    col = rings[:, int(cx) + 3]
    row = rings[int(cy) + 3, :]
    for line in (col, row):
        run = 0
        for v in line:
            if v:
                run += 1
            elif run:
                thick.append(run)
                run = 0
    thick = np.array(thick)
    assert thick.max() <= 4, f"{name} {size}: a ring stroke is {thick.max()} dots thick"


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
@pytest.mark.parametrize("name", MODES)
def test_the_spokes_stay_straight_and_unbroken(name, size):
    state, frames = _play(name, *size, frames=6)
    walls = _geo(state)["walls"]
    d = frames[-1]
    dr, dc = d.shape
    cx, cy = dc / 2.0, dr / 2.0
    s = cy / cx
    for k in range(16):
        th = k / 16 * 2 * math.pi
        ux, uy = math.cos(th) / s, math.sin(th)
        norm = math.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        radii = [r for r in np.arange(0.0, 2.0 * max(cx, cy), 1.0)
                 if 0 <= cx + ux * r < dc and 0 <= cy + uy * r < dr]
        on = [r for r in radii if walls[int(cy + uy * r), int(cx + ux * r)]
              or walls[min(dr - 1, int(round(cy + uy * r))), min(dc - 1, int(round(cx + ux * r)))]]
        assert on, f"spoke {k} missing"
        first = min(on)
        span = [r for r in radii if r >= first]
        # every step outward from where the spoke starts lands on or beside it
        gaps = 0
        for r in span:
            y, x = cy + uy * r, cx + ux * r
            ys_, xs_ = np.mgrid[int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2]
            ok = (ys_ >= 0) & (ys_ < dr) & (xs_ >= 0) & (xs_ < dc)
            gaps += not d[ys_[ok], xs_[ok]].any()
        assert gaps == 0, f"spoke {k} of {name} has {gaps} gaps along its length"


@pytest.mark.parametrize("name", MODES)
def test_the_spokes_do_not_shimmer(name):
    """The spokes are geometry: frame to frame they must not change at all."""
    state, frames = _play(name, 120, 40, frames=8)
    walls = _geo(state)["walls"]
    for d in frames:
        assert np.array_equal(d & walls, walls), "a spoke dot was dropped"


def _frame_at_phase(name, w, h, level, phase=0.37):
    """One frame with the rings pinned at a phase, so only the level differs."""
    fn = M.get(name).fn
    state: dict = {}
    bands = np.full(N_BANDS, level)
    ctx = Ctx(w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
              wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=0, t=0.0, dt=0.0,
              energy=level, silent=level == 0.0, palette=PAL, state=state)
    fn(ctx)
    next(v for k, v in state.items() if k[0].endswith("phase"))["v"] = phase
    return _dots(fn(ctx)[0]), _geo(state)["walls"]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", MODES)
def test_loud_bands_draw_heavier_rings(name, size):
    """The audio still shows: at the same ring position a loud band's ring is
    visibly heavier. (Speed follows the level too, so comparing two passages
    would compare rings in different places.)"""
    quiet, walls = _frame_at_phase(name, *size, 0.1)
    loud, _ = _frame_at_phase(name, *size, 0.9)
    q = int((quiet & ~walls).sum())
    lo = int((loud & ~walls).sum())
    assert lo > 1.5 * q, f"{name} {size}: loud rings carry {lo} dots against {q} when quiet"


@pytest.mark.parametrize("name", MODES)
def test_a_loud_small_tunnel_stays_a_line_drawing(name):
    """At full level on a small terminal, loud rings near the vanishing point
    are capped to a fraction of the gap to the next ring. Without the cap the
    mean share of lit dots in solid 5x5 blocks measured 14-16% here; with it,
    10-12%. (The same measure was 64% before the wireframe.)"""
    _, frames = _play(name, 60, 20, frames=40, energy=1.0)
    heavy = []
    for d in frames[-20:]:
        win = sliding_window_view(np.pad(d, 2), (5, 5)).sum(axis=(-1, -2))
        heavy.append(float((win[d] >= 18).mean()))
    assert np.mean(heavy) < 0.135, f"{name}: {100 * np.mean(heavy):.1f}% of lit dots in solid blocks at full level"


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", MODES)
def test_spokes_converge_on_the_vanishing_point_without_a_blob(name, size):
    """The spokes run on into the centre and fade out there, with no blob.

    They first stopped at a hole several dots wide, then at staggered radii
    that read live as a ring of stubs round a void. Now each continues to
    where it would fuse with its neighbour, measured in real dots so the
    horizontal axis reaches in as far as the vertical one, fading to nearly
    the background on the way, so the ends dissolve rather than cut off.
    """
    from spektr.modes import bg_contrast

    fn = M.get(name).fn
    state: dict = {}
    bands = np.full(N_BANDS, 0.5)
    for f in range(4):
        ctx = Ctx(w=size[0], h=size[1], bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
                  wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=f, t=f / 60, dt=1 / 60,
                  energy=0.5, silent=False, palette=PAL, state=state)
        codes, cidx = fn(ctx)[:2]
    geo = _geo(state)
    d = _dots(codes)
    dr, dc = d.shape
    cx, cy = dc / 2.0, dr / 2.0
    yy, xx = np.mgrid[0:dr, 0:dc]
    dots = np.hypot(xx - cx, yy - cy)
    # the axes reach within four real dots of the centre, both of them
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ray = [(int(round(cy + dy * r)), int(round(cx + dx * r))) for r in np.arange(4.0, 9.0, 0.5)]
        assert all(geo["walls"][y, x] or geo["walls"][y, min(dc - 1, x + 1)] or geo["walls"][min(dr - 1, y + 1), x]
                   for y, x in ray), f"{name} {size}: an axis spoke stops short of the centre"
    # no blob: sparse, and what is lit there is dim
    core = dots <= 6.0
    assert d[core].mean() < 0.45, f"{name} {size}: {100 * d[core].mean():.0f}% of the centre is lit"
    cr = bg_contrast(PAL)
    cell = lambda m: np.unique(np.stack(np.nonzero(m), 1) // [4, 2], axis=0)  # noqa: E731
    core_cells = cell(core & d)
    rim_cells = cell((dots > 40.0) & (dots < 60.0) & geo["walls"] & d)
    core_c = float(np.mean(cr[cidx[core_cells[:, 0], core_cells[:, 1]]]))
    rim_c = float(np.mean(cr[cidx[rim_cells[:, 0], rim_cells[:, 1]]]))
    assert core_c < 0.6 * rim_c, f"{name} {size}: the centre ({core_c:.2f}) is not dimmer than the spokes ({rim_c:.2f})"
    # and the fade only ever dims inward
    fade = geo["spoke_fade"]
    inner = geo["walls"] & (dots < 8.0) & (fade > 0)
    mid = geo["walls"] & (dots > 12.0) & (fade > 0)
    if inner.any() and mid.any():
        assert fade[inner].mean() < fade[mid].mean(), "the spoke ends brighten toward the centre"
