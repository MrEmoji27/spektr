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

    if bidx is None:
        for y in range(h):
            base = y * w
            idx = cidx[y]
            change = np.flatnonzero(idx[1:] != idx[:-1]) + 1
            if change.size == 0:
                strips.append(
                    Strip([Segment(text_all[base : base + w], styles[int(idx[0])])], w)
                )
                continue
            starts = [0, *change.tolist()]
            pick = idx[starts].tolist()
            # If no two adjacent runs are within tol of each other, the
            # drift merge below is a no-op — every adjacent pair more than T
            # apart forces a boundary immediately, by induction. Skipping the
            # Python merge loop keeps rows of sharply distinct colours (a
            # Chladni grid, say) exactly as cheap as before this tolerance.
            if np.any(np.abs(np.diff(idx[starts])) <= tol):
                # Merge runs while the colour stays within tol of the
                # current run's start. Adjacent runs a step or two apart are
                # perceptually identical; this is what keeps a smooth field
                # from turning into a Segment for every other cell.
                ms = [0]
                mv = [pick[0]]
                v0 = pick[0]
                for k in range(1, len(starts)):
                    v = pick[k]
                    if v > v0 + tol or v < v0 - tol:
                        ms.append(starts[k])
                        mv.append(v)
                        v0 = v
                segs = [
                    Segment(text_all[base + s : base + e], styles[c])
                    for s, e, c in zip(ms, [*ms[1:], w], mv)
                ]
            else:
                segs = [
                    Segment(text_all[base + s : base + e], styles[c])
                    for s, e, c in zip(starts, [*starts[1:], w], pick)
                ]
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
    for y in range(h):
        base = y * w
        fi = cidx[y]
        bi = bidx[y]
        change = np.flatnonzero((fi[1:] != fi[:-1]) | (bi[1:] != bi[:-1])) + 1
        starts = [0, *change.tolist()]
        fvals = fi[starts].tolist()
        bvals = bi[starts].tolist()
        # Same skip-check and drift merge as the foreground-only path, applied
        # to both indices — a step of background colour is as invisible as a
        # step of foreground.
        # AND, not OR: a merge needs *both* channels inside the tolerance, so
        # a pair that qualifies on only one of them can never merge and must
        # not drag the row onto the Python path.
        if np.any(
            (np.abs(np.diff(fi[starts])) <= tol)
            & (np.abs(np.diff(bi[starts])) <= tol)
        ):
            ms = [0]
            mv = [fvals[0] * RAMP_STEPS + bvals[0]]
            f0, b0 = fvals[0], bvals[0]
            for k in range(1, len(starts)):
                f, b = fvals[k], bvals[k]
                if (
                    f > f0 + tol
                    or f < f0 - tol
                    or b > b0 + tol
                    or b < b0 - tol
                ):
                    ms.append(starts[k])
                    mv.append(f * RAMP_STEPS + b)
                    f0, b0 = f, b
            starts = ms
            keys = mv
        else:
            keys = (fi[starts].astype(np.int32) * RAMP_STEPS + bi[starts]).tolist()
        segs = []
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else w
            st = cache.get(keys[k])
            if st is None:
                st = pair_style(keys[k])
            segs.append(Segment(text_all[base + s : base + e], st))
        strips.append(Strip(segs, w))
    return strips


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
