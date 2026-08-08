"""Grid primitives and the Strip pipeline.

Every render mode produces the same thing: a ``(h, w)`` array of Unicode
codepoints and a ``(h, w)`` array of ramp indices. This module turns that pair
into Textual ``Strip`` objects.

That uniformity is the point. The old design had each mode assemble a Rich
``Text`` by appending styled runs, which meant Textual re-rendered the whole
widget through the Rich console every frame — 2-4 ms at fullscreen regardless
of what was being drawn. Emitting Strips straight from ``render_line`` skips
the console pass entirely, and run-length encoding the colour indices means a
smooth field costs a handful of Segments per row instead of one per cell.
"""

from __future__ import annotations

import numpy as np
from rich.segment import Segment
from textual.strip import Strip

from .palette import RAMP_STEPS, Palette

BRAILLE_BASE = 0x2800
#: braille dot -> bit, indexed [row][col] over the 4x2 cell
BRAILLE_BITS = np.array(
    [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ],
    dtype=np.uint32,
)

BLOCKS_UP = " ▁▂▃▄▅▆▇█"
BLOCKS_LEFT = " ▏▎▍▌▋▊▉█"
SHADES = " ░▒▓█"

SPACE = ord(" ")


# ── dot grid ─────────────────────────────────────────────────────────────────

def pack_braille(dots: np.ndarray) -> np.ndarray:
    """``(h*4, w*2)`` bool dot grid -> ``(h, w)`` braille codepoints.

    Written as eight strided slices OR-ed together rather than the obvious
    ``reshape(h, 4, w, 2)`` followed by a two-axis reduction. Both touch the
    same number of elements, but reducing across non-contiguous axes is
    dramatically slower and builds a large temporary: at 240x60 this is 1.49 ms
    the reshape way against 0.18 ms this way, and the dot-field modes call it
    every frame. Output is bit-identical.
    """
    h4, w2 = dots.shape
    h, w = h4 // 4, w2 // 2
    d = dots[: h * 4, : w * 2]

    out = d[0::4, 0::2].astype(np.int32)      # 0x01
    out |= d[1::4, 0::2] << 1                 # 0x02
    out |= d[2::4, 0::2] << 2                 # 0x04
    out |= d[3::4, 0::2] << 6                 # 0x40
    out |= d[0::4, 1::2] << 3                 # 0x08
    out |= d[1::4, 1::2] << 4                 # 0x10
    out |= d[2::4, 1::2] << 5                 # 0x20
    out |= d[3::4, 1::2] << 7                 # 0x80
    out += BRAILLE_BASE
    return out


def cell_max(field: np.ndarray) -> np.ndarray:
    """Reduce a dot-resolution float field to one value per text cell.

    Same story as :func:`pack_braille` — seven pairwise maxima over strided
    views beat one reduction over two non-contiguous axes by roughly 16x.
    """
    h4, w2 = field.shape
    h, w = h4 // 4, w2 // 2
    g = field[: h * 4, : w * 2]

    m = np.maximum(g[0::4, 0::2], g[1::4, 0::2])
    m = np.maximum(m, g[2::4, 0::2])
    m = np.maximum(m, g[3::4, 0::2])
    m = np.maximum(m, g[0::4, 1::2])
    m = np.maximum(m, g[1::4, 1::2])
    m = np.maximum(m, g[2::4, 1::2])
    m = np.maximum(m, g[3::4, 1::2])
    return m


def cell_mean(field: np.ndarray) -> np.ndarray:
    h4, w2 = field.shape
    h, w = h4 // 4, w2 // 2
    g = field[: h * 4, : w * 2]

    total = g[0::4, 0::2] + g[1::4, 0::2] + g[2::4, 0::2] + g[3::4, 0::2]
    total = total + g[0::4, 1::2] + g[1::4, 1::2] + g[2::4, 1::2] + g[3::4, 1::2]
    return total * 0.125


_NOISE_BASE: dict[tuple[int, int], np.ndarray] = {}
_INV24 = np.float32(1.0 / 0x1000000)


def noise(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Deterministic hash noise in [0, 1).

    Cheaper than an RNG and stable across frames for a given seed, which is
    what the sparkle modes want. The positional part of the hash only depends
    on the grid size, so it's computed once per terminal size and reused —
    at 240x60 that's two 115k-element broadcasts saved per call, and the
    particle modes call this two or three times a frame.

    uint32 is deliberate: unsigned overflow wraps, which is exactly the mixing
    behaviour wanted, and it halves the memory traffic against int64.
    """
    base = _NOISE_BASE.get(shape)
    if base is None:
        if len(_NOISE_BASE) > 4:
            _NOISE_BASE.clear()
        rows = np.arange(shape[0], dtype=np.uint32)[:, None]
        cols = np.arange(shape[1], dtype=np.uint32)[None, :]
        base = rows * np.uint32(6271) + cols * np.uint32(3037)
        _NOISE_BASE[shape] = base

    with np.errstate(over="ignore"):
        h = base + np.uint32((int(seed) * 104729) & 0xFFFFFFFF)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x45D9F3B)
        h ^= h >> np.uint32(16)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) * _INV24


def frac(x: np.ndarray) -> np.ndarray:
    """Fractional part — ``x - floor(x)``, equivalent to ``np.mod(x, 1.0)``.

    Not a rewrite for style: ``np.mod`` on floats is dramatically slower than
    this, likely because it has to handle an arbitrary divisor rather than
    the fixed 1.0 every caller here actually wants. Measured on a 400x800
    dot grid — Tunnel and Radial's size at a 400x100 terminal — np.mod cost
    6.7 ms against this function's 0.5 ms, a 13x difference that was most of
    Tunnel's entire frame budget. Bit-identical output, including for
    negative input: ``floor`` rounds toward -infinity, which is the same
    convention Python's ``%`` and ``np.mod`` use.
    """
    return x - np.floor(x)


def blocks_from_levels(levels: np.ndarray, h: int, chars: str = BLOCKS_UP) -> np.ndarray:
    """Column heights in 0..1 -> ``(h, len(levels))`` partial-block codepoints.

    Row 0 is the top of the widget, so the comparison is inverted here once
    rather than in every caller.
    """
    n = len(chars) - 1
    rows = np.arange(h - 1, -1, -1, dtype=np.float64)[:, None]  # bottom row = 0
    lo = rows / h
    hi = (rows + 1) / h
    lv = np.asarray(levels, dtype=np.float64)[None, :]

    frac = np.clip((lv - lo) / (hi - lo), 0.0, 1.0)
    idx = np.rint(frac * n).astype(np.int32)
    lut = np.array([ord(c) for c in chars], dtype=np.int32)
    return lut[idx]


# ── colour ───────────────────────────────────────────────────────────────────

def row_gradient(h: int, top_is_hot: bool = True) -> np.ndarray:
    """A vertical 0..1 ramp, one value per text row."""
    v = np.linspace(1.0, 0.0, h) if top_is_hot else np.linspace(0.0, 1.0, h)
    return v


def broadcast_rows(values: np.ndarray, w: int) -> np.ndarray:
    """``(h,)`` per-row values -> ``(h, w)``, for modes coloured by height."""
    return np.repeat(np.asarray(values)[:, None], w, axis=1)


# ── strips ───────────────────────────────────────────────────────────────────

def make_strips(
    codes: np.ndarray,
    cidx: np.ndarray,
    palette: Palette,
    bidx: np.ndarray | None = None,
) -> list[Strip]:
    """``(h, w)`` codepoints + ramp indices -> one Strip per row.

    Colour indices are run-length encoded, so a row of constant colour becomes
    a single Segment and a smooth gradient becomes a few dozen. Runs tolerate
    a small drift from their start colour before splitting, and every merged
    boundary is one Segment fewer. How much drift is ``palette.rle_tol``, which
    each theme derives from its own ramp: "two ramp steps" is not a fixed
    amount of colour, and a tolerance that is invisible on a gentle ramp is
    visible banding on ``rainbow``, which walks a hue wheel. Blank cells are
    folded into whichever run they land in — a space has no visible
    foreground, so letting it inherit avoids splitting runs for nothing.
    """
    h, w = codes.shape
    styles = palette.styles
    strips: list[Strip] = []
    # Per-palette, not a module constant: how many ramp steps a run may drift
    # is a question about *this* theme's colours. See Palette.rle_tol.
    tol = palette.rle_tol

    # One C-level decode for the whole grid beats h*w calls to chr(). At 240x60
    # that is 14,400 interpreter round trips replaced by a single memcpy and a
    # codec pass — worth more than every other optimisation in this function.
    #
    # errors="replace" is a safety net, not an expectation: a codepoint in the
    # UTF-16 surrogate range or past U+10FFFF is undecodable and would
    # otherwise raise, taking the whole render down. Replacement substitutes
    # one character per four-byte unit, so the grid stays aligned. It costs
    # nothing on the normal path.
    text_all = codes.astype("<u4", copy=False).tobytes().decode("utf-32-le", errors="replace")

    if h == 0:
        return strips

    # Whole-grid run starts: column 0 of every row, plus every cell that
    # differs from its left neighbour. One flatnonzero over the whole grid
    # replaces ~five numpy calls per row — at h=50 that is roughly 200 calls
    # a frame, each paying microseconds of dispatch on arrays of only a few
    # hundred elements. Runs are row-ordered and cannot cross a row boundary;
    # the merge loops below enforce that by breaking at every row end.
    chg = np.empty((h, w), dtype=bool)
    chg[:, 0] = True
    np.not_equal(cidx[:, 1:], cidx[:, :-1], out=chg[:, 1:])
    if bidx is not None:
        np.bitwise_or(chg[:, 1:], np.not_equal(bidx[:, 1:], bidx[:, :-1]), out=chg[:, 1:])
    flat = np.flatnonzero(chg.ravel())
    starts = flat.tolist()

    if bidx is None:
        arr = cidx.ravel()[flat]
        vals = arr.tolist()
        # If no two adjacent runs are within tol of each other, the drift
        # merge below is a no-op — every adjacent pair more than T apart
        # forces a boundary immediately, by induction — and the whole Python
        # merge pass can be skipped. Rows of sharply distinct colours (a
        # Chladni grid, say) stay exactly as cheap as before this tolerance.
        if np.any(np.abs(np.diff(arr)) <= tol):
            ms, mv, ends = _rle_merge(starts, vals, w, tol)
        else:
            ms, mv, ends = starts, vals, _row_clamped_ends(flat, h, w)
        row_end = w
        segs: list[Segment] = []
        for s, e, c in zip(ms, ends, mv):
            if s >= row_end:
                strips.append(Strip(segs, w))
                segs = []
                row_end += w
            segs.append(Segment(text_all[s:e], styles[c]))
        strips.append(Strip(segs, w))
        return strips

    # foreground + background: run-length encode on the pair.
    #
    # The style cache lives on the Palette, not here. A per-call dict meant
    # every distinct fg/bg pair was re-parsed from a hex string on every frame
    # — the one per-frame style cost left in the pipeline. It now survives
    # until the theme changes, which is when the colours actually change.
    cache = palette.pair_styles
    pair_style = palette.pair_style
    f_arr = cidx.ravel()[flat]
    b_arr = bidx.ravel()[flat]
    # Same skip-check and drift merge as the foreground-only path, applied to
    # both indices — a step of background colour is as invisible as a step of
    # foreground.
    # AND, not OR: a merge needs *both* channels inside the tolerance, so a
    # pair that qualifies on only one of them can never merge and must not
    # drag the grid onto the Python path.
    if np.any(
        (np.abs(np.diff(f_arr)) <= tol) & (np.abs(np.diff(b_arr)) <= tol)
    ):
        ms, mv, ends = _rle_merge_pair(
            starts, f_arr.tolist(), b_arr.tolist(), w, tol
        )
    else:
        ms = starts
        mv = (f_arr.astype(np.int32) * RAMP_STEPS + b_arr).tolist()
        ends = _row_clamped_ends(flat, h, w)
    row_end = w
    segs = []
    for s, e, key in zip(ms, ends, mv):
        if s >= row_end:
            strips.append(Strip(segs, w))
            segs = []
            row_end += w
        st = cache.get(key)
        if st is None:
            st = pair_style(key)
        segs.append(Segment(text_all[s:e], st))
    strips.append(Strip(segs, w))
    return strips


def _rle_merge(starts, vals, w, tol):
    """Merge runs whose colour stays within ``tol`` of the current run's start.

    Sequential by nature — whether a run merges depends on the value the
    current run started with, which is what the previous merges left behind —
    so this is one flat Python scan over the whole grid's runs rather than one
    scan per row. ``p >= row_end`` is the row-boundary test: run starts are
    contiguous within a row, so the first run of the next row sits exactly at
    the previous row's end. Returns ``(starts, values, ends)`` of the merged
    runs; each end is its successor in the same row, or the row end — merged
    runs never cross a row boundary.
    """
    ms = [0]
    mv = [vals[0]]
    me: list[int] = []
    v0 = vals[0]
    row_end = w
    for k in range(1, len(starts)):
        p = starts[k]
        v = vals[k]
        if p >= row_end or v > v0 + tol or v < v0 - tol:
            me.append(row_end if p >= row_end else p)
            ms.append(p)
            mv.append(v)
            v0 = v
            if p >= row_end:
                row_end += w
    me.append(row_end)
    return ms, mv, me


def _rle_merge_pair(starts, fvals, bvals, w, tol):
    """Pair version of :func:`_rle_merge`.

    A run merges only if *both* channels stay within ``tol`` of the current
    run's start pair — AND, not OR: a pair that qualifies on one channel
    alone can never merge.
    """
    ms = [0]
    mv = [fvals[0] * RAMP_STEPS + bvals[0]]
    me: list[int] = []
    f0, b0 = fvals[0], bvals[0]
    row_end = w
    for k in range(1, len(starts)):
        p = starts[k]
        f, b = fvals[k], bvals[k]
        if (
            p >= row_end
            or f > f0 + tol
            or f < f0 - tol
            or b > b0 + tol
            or b < b0 - tol
        ):
            me.append(row_end if p >= row_end else p)
            ms.append(p)
            mv.append(f * RAMP_STEPS + b)
            f0, b0 = f, b
            if p >= row_end:
                row_end += w
    me.append(row_end)
    return ms, mv, me


def _row_clamped_ends(flat, h, w):
    """Exclusive end of every run, clamped to its start's row end.

    Only the fast path needs this: a run's end is the next run's start unless
    that lies in a later row. The merge loops build the same list for free
    while they scan, so it is not computed twice.
    """
    ends = np.empty_like(flat)
    ends[:-1] = flat[1:]
    ends[-1] = h * w
    np.minimum(ends, (flat // w + 1) * w, out=ends)
    return ends.tolist()


def blank(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((h, w), SPACE, dtype=np.int32),
        np.zeros((h, w), dtype=np.int32),
    )


__all__ = [
    "BLOCKS_LEFT",
    "BLOCKS_UP",
    "BRAILLE_BASE",
    "RAMP_STEPS",
    "SHADES",
    "SPACE",
    "blank",
    "blocks_from_levels",
    "broadcast_rows",
    "cell_max",
    "cell_mean",
    "frac",
    "make_strips",
    "noise",
    "pack_braille",
    "row_gradient",
]
