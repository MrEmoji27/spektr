"""Shared geometry and layout helpers for render modes."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..palette import Palette, _to_linear, hex_to_rgb
from ..render import SPACE
from . import Ctx


def band_columns(w: int, n: int) -> tuple:
    """Map each terminal column to a band index, with gutters that fit.

    Returns ``(col_band, active)`` where ``col_band`` is an int array of length
    w and ``active`` is False on gutter columns. Cached, because this only
    changes when the terminal is resized.

    Gutters are what make neighbouring bars read as separate bars, and every
    gutter column is a column of background. A hard one-column gutter between
    each pair of bands therefore only works while the bands are wide enough to
    pay for it: past roughly one band per four columns the bands shrink onto
    single cells and the gutters grow into half the picture, which reads as
    thin bright bars ruled apart by dark lines — exactly what a band-count
    control set past what the screen affords produces. So the gutter count
    scales with what the bands can spare, stepping down while any band would
    get fewer than two columns:

    * one column between every pair of bands — the classic look, and what
      every default layout still gets;
    * half of those (after every second band), which groups bars into pairs;
    * none at all, which reads as the gapless wall :meth:`Columns` draws;
    * and when even one column per band does not fit, a nearest-band
      downsample that keeps every band on screen and never leaves a column
      unassigned.
    """
    def layout(gutters_after: int) -> tuple | None:
        """Spread n bands over w columns, gutters after every k-th band.

        A gutter follows band ``b`` when ``(b + 1)`` is a multiple of
        ``gutters_after`` — so 1 means between every pair of bands, 2 groups
        bars into pairs, and anything ≥ n means none. Returns None when some
        band would draw less than two columns wide, which is the signal to
        step down to a sparser gutter plan rather than thin the bands out.
        """
        gaps = (n - 1) // gutters_after
        usable = w - gaps
        if usable < 2 * n:
            return None
        base, extra = divmod(usable, n)

        col_band = np.zeros(w, dtype=np.int32)
        active = np.zeros(w, dtype=bool)
        x = 0
        for b in range(n):
            width = base + (1 if b < extra else 0)
            end = min(w, x + width)
            col_band[x:end] = b
            active[x:end] = True
            x = end
            if b < n - 1 and (b + 1) % gutters_after == 0:
                x += 1          # leave a gutter
            if x >= w:
                break
        return col_band, active

    # One gutter per gap first, then every other gap, then none: the classic
    # layouts take the first branch and look exactly as they always did.
    for step in (1, 2, n):
        got = layout(step)
        if got is not None:
            return got

    # More bands than columns. Every band still appears: column i shows the
    # band nearest i's position across the count, so the spectrum's shape
    # survives whole instead of losing its treble off the right edge.
    if w > 1:
        col_band = np.rint(np.arange(w) * ((n - 1) / (w - 1))).astype(np.int32)
    else:
        col_band = np.zeros(w, dtype=np.int32)
    return col_band, np.ones(w, dtype=bool)


@lru_cache(maxsize=64)
def smooth_columns(w: int, n: int) -> tuple:
    """Continuous column -> band position, for gapless interpolated modes."""
    pos = np.linspace(0.0, n - 1, w)
    idx = np.floor(pos).astype(np.int32)
    idx = np.clip(idx, 0, n - 1)
    nxt = np.clip(idx + 1, 0, n - 1)
    frac = pos - idx
    # cosine blend reads smoother than linear across a coarse band set
    frac = (1.0 - np.cos(frac * np.pi)) * 0.5
    return idx, nxt, frac


def spread(levels: np.ndarray, w: int) -> np.ndarray:
    """Band levels -> one interpolated level per terminal column."""
    n = len(levels)
    idx, nxt, frac = smooth_columns(w, n)
    return levels[idx] * (1.0 - frac) + levels[nxt] * frac


def empty(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((h, w), SPACE, dtype=np.int32),
        np.zeros((h, w), dtype=np.int32),
    )


# ── shared polar geometry ────────────────────────────────────────────────────
#
# These two lived in ``particles.py`` and were imported out of it by
# ``scenes.py`` and ``lofi.py``, which made a leaf module the de facto home of
# a shared primitive: adding a polar mode to a third file meant importing from
# a sibling that has nothing to do with it. They belong here beside
# ``band_columns`` and ``spread``, which are the same kind of thing.
#
# Not every polar mode uses these, and that is deliberate rather than an
# oversight to clean up later. ``Needle`` pivots at the bottom edge of a
# *cell* grid rather than the centre of a dot grid; ``Locket`` normalises each
# axis independently so the heart stretches to fill whatever aspect it is
# given; ``Kaleidoscope`` evaluates on a half-width ``|x|`` fold so that a dot
# and its mirror compute bit-identical values. Those are different geometries
# with different reasons, and forcing them through one signature would cost
# more in parameters than it saves in lines.
#
# What *was* worth removing is two derivations of the same grid inside one
# function, which is how ``Kaleidoscope`` ended up with ``max(1.0, cy - 1.0)``
# in one place and ``max(cy - 1.0, 1.0)`` in another — the same number by luck
# rather than by construction.

_LUMA = np.array([0.2126, 0.7152, 0.0722])


def bg_contrast(palette: Palette) -> np.ndarray:
    """WCAG contrast of every ramp entry against the theme's own background.

    A theme's ramp is a hue gradient, not a brightness one: on gruvbox the low
    end is the most visible colour on it and on plasma the low end is nearly
    the background. A mode that wants something to *recede* or to *stand out*
    has to ask this rather than assume a low index is dim. Computed per call,
    because an animated theme's ramp moves every frame; it is 64 entries.
    """
    lum = _to_linear(np.asarray(palette.rgb, dtype=np.float64)) @ _LUMA
    bg = hex_to_rgb(palette.theme.bg or "#000000")
    lb = float(_to_linear(np.array(bg, dtype=np.float64)) @ _LUMA)
    return (np.maximum(lum, lb) + 0.05) / (np.minimum(lum, lb) + 0.05)


def contrast_ramp(palette: Palette, values, faint: float = 1.8) -> np.ndarray:
    """Ramp indices for 0..1 *brightness*, measured against the background.

    0 lands on the entry whose contrast is closest to ``faint`` — visible, but
    the quietest thing the theme can show — and 1 on the highest-contrast
    entry; values between walk the ramp from one to the other. Walking the
    ramp keeps neighbouring values neighbouring colours, and across the 55
    built-ins contrast never steps backwards along that walk.
    """
    cr = bg_contrast(palette)
    lo = int(np.argmin(np.abs(cr - faint)))
    hi = int(np.argmax(cr))
    v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.rint(lo + (hi - lo) * v).astype(np.int32)


def polar_grid(ctx: Ctx):
    """``(dist, turn, max_r)`` over the dot grid, cached per size.

    ``dist`` is in dots from the centre, aspect-corrected so a circle comes
    out round rather than as an ellipse — a braille cell is about twice as
    tall as it is wide. ``turn`` is the angle as a fraction of a full turn
    (0..1), which is the form :func:`angular_bands` wants and which keeps a
    divide out of the per-frame path. ``max_r`` is the inscribed radius.

    Rebuilt only when the terminal resizes.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols

    def build():
        cx, cy = dc / 2.0, dr / 2.0
        x_scale = cy / max(cx, 1.0)          # braille cells are ~2x taller than wide
        xs = (np.arange(dc, dtype=np.float32) - cx) * x_scale
        ys = np.arange(dr, dtype=np.float32) - cy
        dx = xs[None, :]
        dy = ys[:, None]
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        ang = np.arctan2(dy + np.zeros_like(dx), dx + np.zeros_like(dy))
        ang = np.where(ang < 0, ang + 2 * math.pi, ang).astype(np.float32)
        turn = (ang / np.float32(2 * math.pi)).astype(np.float32)
        return dist, turn, max(1.0, cy - 1.0)

    return ctx.scratch("polar", build)


def angular_lut(ctx: Ctx, turn: np.ndarray, n: int, spin: float):
    """The pieces behind :func:`angular_bands`: ``(lut, idx)``.

    ``lut`` is 512 blended band levels around the circle; ``idx`` says which
    entry each dot reads. :func:`angular_bands` is ``lut[idx]`` and that is
    what most modes want.

    This exists for the ones that then do arithmetic on the result. A mode
    computing, say, ``max_r * (0.1 + 0.9 * nrg * nrg)`` is doing three passes
    over every dot on the grid to produce a value that can only take 512
    distinct forms — the same arithmetic on ``lut`` is 512 elements wide and
    the gather is unchanged. At 400x100 that is 320,000 elements against 512,
    per operation, for exactly the same numbers: ``f(lut)[idx]`` and
    ``f(lut[idx])`` are the same float operations on the same inputs.

    Transform the table, then gather. Not the other way round.
    """
    steps = 512
    bands = ctx.display_bands(n).astype(np.float32)
    pos = np.linspace(0.0, n, steps, endpoint=False, dtype=np.float32)
    bi = pos.astype(np.int32) % n
    f = pos - np.floor(pos)
    tm = (1.0 - np.cos(f * np.float32(math.pi))) * np.float32(0.5)
    lut = bands[bi] * (1.0 - tm) + bands[(bi + 1) % n] * tm

    # keep the spin bounded so float32 precision doesn't drift over a long session
    offset = np.float32(float(spin) % 1.0)
    idx = ((turn + offset) * np.float32(steps)).astype(np.int32) & (steps - 1)
    return lut, idx


def angular_bands(ctx: Ctx, turn: np.ndarray, n: int, spin: float) -> np.ndarray:
    """Map every dot's angle onto the band set, blended between neighbours.

    ``turn`` is the angle as a fraction of a full turn. Doing the lookup as a
    single table index into a pre-blended ramp costs one gather instead of the
    two gathers, a cosine and three multiplies the per-dot blend needed — worth
    it when this runs over 100k dots a frame.
    """
    lut, idx = angular_lut(ctx, turn, n, spin)
    return lut[idx]
