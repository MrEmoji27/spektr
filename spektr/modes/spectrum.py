"""Block-character spectrum modes."""

from __future__ import annotations

import numpy as np

from ..palette import RAMP_STEPS
from ..render import SPACE, blocks_from_levels
from . import Ctx, band_columns, empty, mode, spread

_PEAK_CHAR = ord("─")
_FULL = ord("█")
_HALF_DOWN = ord("▄")
_LED = ord("▆")
_TICK = ord("│")


def _vgrad(ctx: Ctx, top_hot: bool = True) -> np.ndarray:
    """Vertical colour ramp broadcast across the width."""
    v = np.linspace(1.0, 0.0, ctx.h) if top_hot else np.linspace(0.0, 1.0, ctx.h)
    return np.repeat(ctx.ramp(v)[:, None], ctx.w, axis=1)


def _draw_peaks(ctx: Ctx, codes, cidx, levels_per_col, peaks_per_col) -> None:
    """Stamp peak markers wherever the cell underneath is empty."""
    h = ctx.h
    rows = h - 1 - np.floor(np.clip(peaks_per_col, 0.0, 0.999) * h).astype(np.int32)
    valid = (peaks_per_col > 0.02) & (rows >= 0) & (rows < h)
    cols = np.flatnonzero(valid)
    if cols.size == 0:
        return
    rws = rows[cols]
    free = codes[rws, cols] == SPACE
    rws, cols = rws[free], cols[free]
    codes[rws, cols] = _PEAK_CHAR
    cidx[rws, cols] = RAMP_STEPS - 1


@mode("Bars", blurb="the classic — ten-ish bars with peak markers")
def bars(ctx: Ctx):
    n = ctx.n_display
    col_band, active = band_columns(ctx.w, n)
    lv = ctx.display_bands(n)
    pk = ctx.display_peaks(n)

    levels = np.where(active, lv[col_band], 0.0)
    codes = blocks_from_levels(levels, ctx.h)
    cidx = _vgrad(ctx)
    _draw_peaks(ctx, codes, cidx, levels, np.where(active, pk[col_band], 0.0))
    return codes, cidx


@mode("Bricks", blurb="chunky, no partial cells")
def bricks(ctx: Ctx):
    n = ctx.n_display
    col_band, active = band_columns(ctx.w, n)
    lv = ctx.display_bands(n)
    levels = np.where(active, lv[col_band], 0.0)

    thresh = (np.arange(ctx.h - 1, -1, -1, dtype=np.float64) / ctx.h)[:, None]
    lit = levels[None, :] > thresh
    codes = np.where(lit, _HALF_DOWN, SPACE).astype(np.int32)
    cidx = _vgrad(ctx)
    _draw_peaks(ctx, codes, cidx, levels, np.where(active, ctx.display_peaks(n)[col_band], 0.0))
    return codes, cidx


@mode("Columns", blurb="gapless, interpolated across the full width")
def columns(ctx: Ctx):
    n = ctx.n_display
    levels = spread(ctx.display_bands(n), ctx.w)
    codes = blocks_from_levels(levels, ctx.h)
    cidx = _vgrad(ctx)
    return codes, cidx


@mode("Ladder", blurb="segmented LED stack")
def ladder(ctx: Ctx):
    n = ctx.n_display
    col_band, active = band_columns(ctx.w, n)
    lv = ctx.display_bands(n)
    levels = np.where(active, lv[col_band], 0.0)

    thresh = (np.arange(ctx.h - 1, -1, -1, dtype=np.float64) / ctx.h)[:, None]
    lit = levels[None, :] > thresh
    codes = np.where(lit, _LED, SPACE).astype(np.int32)
    cidx = _vgrad(ctx)
    _draw_peaks(ctx, codes, cidx, levels, np.where(active, ctx.display_peaks(n)[col_band], 0.0))
    return codes, cidx


@mode("Mirror", blurb="grows out from the centre line")
def mirror(ctx: Ctx):
    h, w = ctx.h, ctx.w
    if h < 2:
        return empty(w, h)

    half = h // 2
    levels = spread(ctx.display_bands(), w)

    top = blocks_from_levels(levels, half)            # (half, w), row 0 = top
    bottom = blocks_from_levels(levels, h - half)     # grows upward…
    bottom = bottom[::-1]                             # …so flip it to grow down

    codes = np.concatenate((top, bottom), axis=0)

    # hottest at the centre, cooling outward
    dist = np.abs(np.arange(h) - (h - 1) / 2.0) / max(1.0, (h - 1) / 2.0)
    cidx = np.repeat(ctx.ramp(1.0 - dist)[:, None], w, axis=1)
    return codes, cidx


@mode("Stereo", group="stereo", blurb="per-band L/R meters, mirrored from centre")
def stereo(ctx: Ctx):
    """cliamp's stereo meters, done as a mirrored pair.

    Each row is one band: the left channel grows leftward from the centre
    column, the right channel grows rightward. LED gutters every third column
    keep it reading as segments rather than a solid slab.
    """
    h, w = ctx.h, ctx.w
    if w < 8 or h < 2:
        return empty(w, h)

    from ..analysis import resample_bands

    n = min(h, 24)
    left = resample_bands(ctx.bands_l, n)[::-1]     # low frequencies at the bottom
    right = resample_bands(ctx.bands_r, n)[::-1]

    codes, cidx = empty(w, h)
    mid = w // 2
    span_l = mid
    span_r = w - mid - 1

    # one band per row, centred vertically if the widget is taller than n
    top_pad = max(0, (h - n) // 2)
    rows = min(n, h - top_pad)
    xs = np.arange(w)
    gutter = (np.abs(xs - mid) % 3) == 2

    # colour by distance from centre: quiet is cool, loud is hot. Identical for
    # every row, so it's computed once rather than per band.
    heat = ctx.ramp(np.abs(xs - mid) / max(1.0, max(span_l, span_r)))
    cidx[top_pad : top_pad + rows] = heat

    # solve the whole block at once: how far from the centre each row reaches
    lit_l = (np.rint(left[:rows] * span_l).astype(np.int32))[:, None]
    lit_r = (np.rint(right[:rows] * span_r).astype(np.int32))[:, None]
    left_side = (xs[None, :] >= mid - lit_l) & (xs[None, :] < mid)
    right_side = (xs[None, :] > mid) & (xs[None, :] <= mid + lit_r)
    block = (left_side | right_side) & ~gutter[None, :]

    target = codes[top_pad : top_pad + rows]
    np.copyto(target, _FULL, where=block)

    codes[:, mid] = _TICK
    cidx[:, mid] = RAMP_STEPS // 3
    return codes, cidx
