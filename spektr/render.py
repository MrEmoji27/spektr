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

#: The two-colour half-block: foreground paints the top half of the cell,
#: background the bottom. What every smooth field in the app is drawn with,
#: and what :func:`shade_cells` falls back to for a cell holding no edge.
UPPER_HALF = ord("▀")

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


# ── octant cells ─────────────────────────────────────────────────────────────

OCTANT_BASE = 0x1CD00

#: octant -> bit, indexed [row][col] over the 4x2 cell. Row-major, matching the
#: numbering Unicode uses in the character names:
#:
#:      1 2
#:      3 4
#:      5 6
#:      7 8
#:
#: Deliberately *not* the braille bit order — braille numbers its dots down the
#: left column then down the right, and reusing that table here silently
#: transposes the picture.
OCTANT_BITS = np.array(
    [
        [0x01, 0x02],
        [0x04, 0x08],
        [0x10, 0x20],
        [0x40, 0x80],
    ],
    dtype=np.int32,
)

#: The 26 patterns the octant block does not contain, because characters for
#: them already existed when it was encoded (Unicode 16). Everything else lives
#: contiguously in U+1CD00..U+1CDE5 in ascending mask order, which is what
#: :func:`_octant_lut` relies on.
#:
#: The four single-subcell patterns are the ones worth pointing at: they are
#: *not* in the octant block at all but in the half-by-quarter set at U+1CEA0,
#: and U+1FBE6/U+1FBE7 — the middle-left and middle-right quarter blocks — were
#: added by Unicode 16 specifically so this set could be completed. Every entry
#: was checked against the real character names rather than derived by eye.
_OCTANT_ELSEWHERE = {
    0b00000000: 0x0020,   # SPACE
    0b00000001: 0x1CEA8,  # LEFT HALF UPPER ONE QUARTER BLOCK
    0b00000010: 0x1CEAB,  # RIGHT HALF UPPER ONE QUARTER BLOCK
    0b00000011: 0x1FB82,  # UPPER ONE QUARTER BLOCK
    0b00000101: 0x2598,   # QUADRANT UPPER LEFT
    0b00001010: 0x259D,   # QUADRANT UPPER RIGHT
    0b00001111: 0x2580,   # UPPER HALF BLOCK
    0b00010100: 0x1FBE6,  # MIDDLE LEFT ONE QUARTER BLOCK
    0b00101000: 0x1FBE7,  # MIDDLE RIGHT ONE QUARTER BLOCK
    0b00111111: 0x1FB85,  # UPPER THREE QUARTERS BLOCK
    0b01000000: 0x1CEA3,  # LEFT HALF LOWER ONE QUARTER BLOCK
    0b01010000: 0x2596,   # QUADRANT LOWER LEFT
    0b01010101: 0x258C,   # LEFT HALF BLOCK
    0b01011010: 0x259E,   # QUADRANT UPPER RIGHT AND LOWER LEFT
    0b01011111: 0x259B,   # QUADRANT UPPER LEFT AND UPPER RIGHT AND LOWER LEFT
    0b10000000: 0x1CEA0,  # RIGHT HALF LOWER ONE QUARTER BLOCK
    0b10100000: 0x2597,   # QUADRANT LOWER RIGHT
    0b10100101: 0x259A,   # QUADRANT UPPER LEFT AND LOWER RIGHT
    0b10101010: 0x2590,   # RIGHT HALF BLOCK
    0b10101111: 0x259C,   # QUADRANT UPPER LEFT AND UPPER RIGHT AND LOWER RIGHT
    0b11000000: 0x2582,   # LOWER ONE QUARTER BLOCK
    0b11110000: 0x2584,   # LOWER HALF BLOCK
    0b11110101: 0x2599,   # QUADRANT UPPER LEFT AND LOWER LEFT AND LOWER RIGHT
    0b11111010: 0x259F,   # QUADRANT UPPER RIGHT AND LOWER LEFT AND LOWER RIGHT
    0b11111100: 0x2586,   # LOWER THREE QUARTERS BLOCK
    0b11111111: 0x2588,   # FULL BLOCK
}


def _octant_lut() -> np.ndarray:
    lut = np.zeros(256, dtype=np.int32)
    n = 0
    for mask in range(256):
        cp = _OCTANT_ELSEWHERE.get(mask)
        if cp is None:
            cp = OCTANT_BASE + n
            n += 1
        lut[mask] = cp
    return lut


#: mask -> codepoint, all 256. Built once; 230 octants plus the 26 above.
OCTANT_LUT = _octant_lut()

#: Eight patterns rendered as their nearest neighbour instead of exactly.
#:
#: Of the 26 patterns held outside the octant block, eighteen are Block
#: Elements — quadrants, halves, quarters, the full block — which every
#: terminal font has had for decades. The other eight are not: the four
#: single-subcell patterns live in the Symbols for Legacy Computing
#: *Supplement* at U+1CEA0, and the middle quarters at U+1FBE6/U+1FBE7 were
#: added by Unicode 16 alongside the octants themselves. A font can ship the
#: whole octant block and still miss all eight, which is exactly what happens
#: in practice — and the patterns are not exotic: an isolated lit subcell is
#: what the *rim of any shape* is made of, so a mode like Valentine produces
#: them constantly while a continuous field like Kaleidoscope almost never
#: does. One mode looks perfect and the next is speckled with tofu.
#:
#: Each one is widened to the nearest pattern that is either a Block Element
#: or inside the octant block: an isolated subcell grows to its quadrant, the
#: top row grows to the top half, a middle quarter picks up the subcell above
#: it, and three-quarters fills. Growing rather than dropping, so a lit
#: subcell is never silently erased — at a 4x4 pixel subcell the difference is
#: not visible, where a hole in an outline is.
#:
#: Every substitution keeps its mirror partner's substitution its own mirror
#: (1 and 2 to the two upper quadrants, 64 and 128 to the two lower ones,
#: 20 and 40 to left and right, 3 and 63 both self-mirrored). Kaleidoscope's
#: bilateral symmetry is a property of the glyphs, not just of the field, and
#: an asymmetric substitution table would break it.
_OCTANT_WIDEN = {
    0b00000001: 0b00000101,   # subcell 1      -> quadrant upper left
    0b00000010: 0b00001010,   # subcell 2      -> quadrant upper right
    0b00000011: 0b00001111,   # top row        -> upper half
    0b00010100: 0b00010101,   # middle left    -> left column, rows 1-3
    0b00101000: 0b00101010,   # middle right   -> right column, rows 1-3
    0b00111111: 0b11111111,   # upper 3/4      -> full block
    0b01000000: 0b01010000,   # subcell 7      -> quadrant lower left
    0b10000000: 0b10100000,   # subcell 8      -> quadrant lower right
}

#: What the packers actually emit: :data:`OCTANT_LUT` with the eight patterns
#: above widened. Keep :data:`OCTANT_LUT` for anything that needs the exact
#: mapping — it is the one the character names verify against.
OCTANT_LUT_WIDE = OCTANT_LUT[[_OCTANT_WIDEN.get(m, m) for m in range(256)]]


def _octant_subcells(field: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    h4, w2 = field.shape
    h, w = h4 // 4, w2 // 2
    g = field[: h * 4, : w * 2]
    return [g[r::4, c::2] for r in range(4) for c in range(2)], h, w


def cell_hilo(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(4h, 2w)`` dot field -> per-cell ``(lo, hi)``.

    The two-colour companion to :func:`cell_max`: an octant cell carries a
    background and a foreground, so it needs both ends of the cell's range
    rather than one. Seven pairwise min/max over strided views, for the same
    reason :func:`cell_max` is written that way.
    """
    sub, _, _ = _octant_subcells(field)
    lo = sub[0].copy()
    hi = sub[0].copy()
    for s in sub[1:]:
        np.minimum(lo, s, out=lo)
        np.maximum(hi, s, out=hi)
    return lo, hi


#: mask -> codepoint over a 2x2 cell, bits row-major (1 = upper left).
#:
#: All sixteen are Block Elements, which is the entire point: the quadrant set
#: is *complete* where the octant set is not, and it has been in every terminal
#: font for decades. Half the linear resolution of octants and twice that of
#: the half-block trick.
QUADRANT_LUT = np.array(
    [
        0x0020,  # ....
        0x2598,  # UL
        0x259D,  # UR
        0x2580,  # UL UR      upper half
        0x2596,  # LL
        0x258C,  # UL LL      left half
        0x259E,  # UR LL
        0x259B,  # UL UR LL
        0x2597,  # LR
        0x259A,  # UL LR
        0x2590,  # UR LR      right half
        0x259C,  # UL UR LR
        0x2584,  # LL LR      lower half
        0x2599,  # UL LL LR
        0x259F,  # UR LL LR
        0x2588,  # all        full block
    ],
    dtype=np.int32,
)

#: Ordered thresholds down a 2x2 cell's two rows. Both columns equal, for the
#: mirror reason in :data:`OCTANT_DITHER`.
QUADRANT_DITHER = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32)

#: Which cell geometry the packers emit: ``"octant"`` (2x4 subcells, Unicode
#: 16) or ``"quadrant"`` (2x2, Block Elements only).
#:
#: There is no way to ask a terminal which it can draw — a missing glyph comes
#: back as a replacement box, not an error — so this is a setting, not a probe.
#: It lives here rather than in each mode because it is a property of the
#: *terminal*, and because a mode that has already built its field at subcell
#: resolution should not have to know: the packers reduce.
CELL_MODE = "octant"


def set_cell_mode(mode: str) -> None:
    """Choose octant or quadrant cells for every mode that draws with them."""
    global CELL_MODE
    if mode not in ("octant", "quadrant"):
        raise ValueError(f"cell mode must be octant or quadrant, not {mode!r}")
    CELL_MODE = mode


def subcell_rows() -> int:
    """Subcell rows per text cell in the current mode: 4 octant, 2 quadrant.

    A mode that builds its own field should ask, and build ``h * subcell_rows()``
    rows. Building four and letting the packer reduce is correct but wasteful,
    and the waste is the whole frame at a large terminal: Plasma's field is
    1.28 million points at 800x200 with four rows a cell, and half of them are
    averaged away again before anything is drawn.
    """
    return 2 if CELL_MODE == "quadrant" else 4


def _quadrant_direct(field: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    """The four quadrants of each cell, from a field already at ``(2h, 2w)``."""
    h2, w2 = field.shape
    h, w = h2 // 2, w2 // 2
    g = field[: h * 2, : w * 2]
    return [g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2]], h, w


def _quadrant_subcells(field: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    """A ``(4h, 2w)`` subcell field reduced to the four quadrants of each cell.

    Pairs of octant rows collapse by maximum rather than by mean: these fields
    carry thin features — a nodal line, a trace, the rim of a shape — and a
    mean loses a one-row-thick one to its neighbour, where a maximum keeps it
    and only thickens it.
    """
    h4, w2 = field.shape
    h, w = h4 // 4, w2 // 2
    g = field[: h * 4, : w * 2]
    return (
        [
            np.maximum(g[0::4, 0::2], g[1::4, 0::2]),
            np.maximum(g[0::4, 1::2], g[1::4, 1::2]),
            np.maximum(g[2::4, 0::2], g[3::4, 0::2]),
            np.maximum(g[2::4, 1::2], g[3::4, 1::2]),
        ],
        h,
        w,
    )


def _pack_quadrant(field, lo, hi, dither: bool) -> np.ndarray:
    sub, h, w = _quadrant_subcells(field)
    span = hi - lo
    mask = np.zeros((h, w), dtype=np.int32)
    k = 0
    for r in range(2):
        for c in range(2):
            thr = lo + span * (QUADRANT_DITHER[r, c] if dither else np.float32(0.5))
            np.add(mask, np.where(sub[k] >= thr, 1 << k, 0), out=mask)
            k += 1
    return QUADRANT_LUT[mask]


def pack_octant_bits(lit: np.ndarray) -> np.ndarray:
    """``(4h, 2w)`` bool subcell grid -> ``(h, w)`` octant codepoints.

    The octant twin of :func:`pack_braille`, and the right one for a mode that
    already knows which subcells are lit — a silhouette, an outline, a shape
    with an inside and an outside. Same dot resolution as braille, drawn as
    solid block mosaic rather than as separated dots, which is the difference
    between a filled region reading as a surface and reading as stipple.

    Colour it with a foreground alone and the unlit subcells stay the
    terminal's background; give it a background index too and the cell becomes
    opaque, which is what the two-colour modes want and what a shape drawn
    against empty space does not.
    """
    if CELL_MODE == "quadrant":
        # OR, not maximum: a subcell pair is lit if either half is, so a
        # one-row-thick outline survives the reduction as a thicker one rather
        # than vanishing.
        f = lit.astype(np.float32)
        return _pack_quadrant(f, np.float32(0.0), np.float32(1.0), dither=False)

    sub, h, w = _octant_subcells(lit)
    mask = np.zeros((h, w), dtype=np.int32)
    k = 0
    for r in range(4):
        for c in range(2):
            np.add(mask, np.where(sub[k], OCTANT_BITS[r, c], 0), out=mask)
            k += 1
    return OCTANT_LUT_WIDE[mask]


#: Ordered thresholds over the 4x2 cell with all eight distinct — a proper 2-D
#: Bayer spread rather than the mirror-safe one below.
#:
#: This is the matrix for *shading*, and the distinction is worth stating
#: because using the wrong one silently costs half the resolution. When the two
#: columns share a threshold a cell can only ever show an even number of lit
#: subcells: five coverages out of a possible nine, so the shading resolves
#: half of what it could. Measured on a smooth ramp, that came out at exactly
#: 1.00 ramp steps per level — the same granularity as no dithering at all,
#: which is what "still pixelated" looks like.
#:
#: Kaleidoscope needs the symmetric one because a mirror swaps a cell's subcell
#: columns and its whole promise is bilateral symmetry. Nothing else does.
OCTANT_DITHER_2D = (
    np.array([[0, 4], [6, 2], [1, 5], [7, 3]], dtype=np.float32) + 0.5
) / 8.0

#: The same for a 2x2 cell: four distinct thresholds, five coverages.
QUADRANT_DITHER_2D = (np.array([[0, 2], [3, 1]], dtype=np.float32) + 0.5) / 4.0

#: Ordered threshold offsets over the 4x2 cell, in (0, 1).
#:
#: A Bayer spread down the rows — neighbouring rows get thresholds far apart
#: in the sequence, which breaks an edge into a stipple instead of stepping it
#: a row at a time. Fixed relative to the cell rather than to the grid, so a
#: shape crossing the screen does not drag a crawling pattern behind it.
#:
#: **The two columns are deliberately identical.** Mirroring a cell swaps its
#: subcell columns, so a matrix whose columns differ makes a symmetric field
#: come out asymmetric — measured, on the first version of this table, as
#: Kaleidoscope losing the bilateral symmetry that is the whole point of a
#: mirror tube. Equal columns are the only form that survives the mirror, and
#: the loss is nothing: a cell is four rows tall and two columns wide, so the
#: staircase worth breaking up is the vertical one.
OCTANT_DITHER = np.array(
    [[0.125, 0.125], [0.625, 0.625], [0.375, 0.375], [0.875, 0.875]],
    dtype=np.float32,
)


def shade_block_for(cells: int, subcells: int) -> int:
    """How wide the colour pair should be, in ramp steps.

    The pair spans ``block`` steps and the dither resolves ``subcells + 1``
    positions inside it, so a level costs ``block / subcells`` ramp steps. The
    whole point is to land *below* one — a block equal to the subcell count
    reproduces the ramp's own granularity exactly and buys nothing, which is
    the trap the first version fell into: 4 steps across 4 quadrant subcells
    measured 1.00 steps a level, indistinguishable from not dithering.

    Half the subcell count is the default, so a level is half a ramp step and
    the field carries roughly twice what the palette can name.

    Above that the strip builder starts to bind — colour costs per *run* and
    runs scale with cells — so a very large grid trades back toward the coarse
    end. It can afford to: the cells are smaller, so a step between two of
    them subtends less.
    """
    fine = max(2, subcells // 2)
    if cells <= 56_000:       # up to about 400x140, a maximised window
        return fine
    if cells <= 200_000:      # up to about 800x250
        return fine * 2
    return fine * 3


def shade_cells(
    field: np.ndarray, steps: int = RAMP_STEPS, block: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(4h, 2w)`` field -> ``(codes, fg_index, bg_index)``, shaded not masked.

    The other packers treat a cell's two colours as its extremes and the glyph
    as a shape cut between them. That is right for an edge and wrong for a
    gradient: a smooth field has no shape to cut, so thresholding it turns
    colour precision into texture and the picture comes out *more* pixelated
    than the half-block renderer it replaced. Plasma is the mode that shows it.

    Here the two colours are **adjacent ramp steps** bracketing what is in the
    cell, and the glyph says what fraction of the way between them the cell
    sits. Neighbouring steps are nearly the same colour, so the pattern does
    not read as texture — it reads as a shade between them, and the field gets
    roughly eight times the levels the 64-step ramp can name. That is ordinary
    two-colour dithering, and it is how you draw a gradient smoother than your
    palette rather than coarser.

    Where the cell *does* straddle an edge — its range covering several ramp
    steps — the same arithmetic degrades gracefully back into a shape cut
    between the two ends, because that is what the coverage then means.

    The field must be ``(h * subcell_rows(), w * 2)`` — four rows a cell in
    octant mode, two in quadrant. A mode that builds its own field should ask
    :func:`subcell_rows` rather than always building four, because in quadrant
    mode half of those samples are averaged away again and at a large terminal
    that half is most of the frame.

    Returns palette indices directly rather than floats: the quantisation is
    the point here, so doing it once inside is both cheaper and the only way
    the two colours are guaranteed to land a single step apart.
    """
    if CELL_MODE == "quadrant":
        sub, h, w = _quadrant_direct(field)
        dither, bits = QUADRANT_DITHER_2D, None
        rows, cols = 2, 2
    else:
        sub, h, w = _octant_subcells(field)
        dither, bits = OCTANT_DITHER_2D, OCTANT_BITS
        rows, cols = 4, 2

    lo = sub[0].copy()
    hi = sub[0].copy()
    for s in sub[1:]:
        np.minimum(lo, s, out=lo)
        np.maximum(hi, s, out=hi)

    top = np.float32(steps - 1)
    nsub = rows * cols
    b = np.float32(shade_block_for(h * w, nsub) if block is None else block)

    # Both ends snapped to a block of ramp steps, and the pair is *always* one
    # block wide unless the cell genuinely spans more.
    #
    # Without this the two colours track the cell's own range, which is fine on
    # a smooth field and ruinous on a sharp one: every edge cell picks its own
    # pair and the strip builder pays a run boundary for each. Measured on
    # Chladni at 400x100 — 11,039 colour runs against 7,300, and Chladni
    # Extreme at 17,621. Snapping to blocks leaves ``steps / block`` distinct
    # pairs for the whole frame, so neighbouring cells share one and merge,
    # while the dither still resolves the value *inside* the block: sixteen
    # blocks of eight subcell gradations is 128 levels where the ramp names 64.
    lo_i = np.floor(np.clip(lo, 0.0, 1.0) * top / b) * b
    hi_i = np.maximum(np.ceil(np.clip(hi, 0.0, 1.0) * top / b) * b, lo_i + b)
    hi_i = np.minimum(hi_i, top)
    lo_i = np.minimum(lo_i, hi_i - np.float32(1.0))

    # Where a cell holds no edge, do not dither it at all.
    #
    # This is the correction to the first version, and the screenshot that
    # forced it showed why: a smooth field has an edge in almost no cells, so
    # thresholding every one of them covered the whole screen in a 1-bit
    # stipple. Dithering buys sub-cell *position*; a cell with nothing to
    # position gets texture in exchange for nothing, and the half-block
    # renderer it replaced — two exact colours, no pattern — was plainly
    # better there.
    #
    # So a flat cell is drawn the half-block way: upper half foreground, lower
    # half background, straight from the means of its two halves. A cell that
    # does straddle an edge keeps the octant mask, which is where the extra
    # resolution was ever worth having. The two agree at the boundary because
    # a barely-flat cell's halves are nearly its extremes.
    span = hi_i - lo_i
    # The *raw* range, not the snapped one: snapping forces a pair at least a
    # block wide, so testing that would call every cell an edge and dither the
    # whole screen — which is exactly the bug this guard exists to undo.
    edge = (hi - lo) * top >= np.float32(1.5)

    half = len(sub) // 2
    top_mean = sub[0].copy()
    for s in sub[1:half]:
        top_mean += s
    top_mean *= np.float32(1.0 / half)
    bot_mean = sub[half].copy()
    for s in sub[half + 1:]:
        bot_mean += s
    bot_mean *= np.float32(1.0 / half)

    mask = np.zeros((h, w), dtype=np.int32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            on = sub[k] * top >= lo_i + span * dither[r, c]
            weight = (1 << k) if bits is None else bits[r, c]
            np.add(mask, np.where(on, weight, 0), out=mask)
            k += 1

    lut = QUADRANT_LUT if CELL_MODE == "quadrant" else OCTANT_LUT_WIDE
    codes = np.where(edge, lut[mask], np.int32(UPPER_HALF))

    # A flat cell's colours are its two halves, exact and unquantised beyond
    # the ramp itself — no block snapping, because there is no coverage to
    # resolve inside the block and snapping would only band a smooth gradient.
    flat_fg = np.clip(top_mean * top, 0.0, top)
    flat_bg = np.clip(bot_mean * top, 0.0, top)
    fg = np.where(edge, hi_i, flat_fg).astype(np.int32)
    bg = np.where(edge, lo_i, flat_bg).astype(np.int32)
    return codes, fg, bg


def pack_octant_smooth(
    field: np.ndarray, lo: np.ndarray | None = None, hi: np.ndarray | None = None
) -> np.ndarray:
    """``(4h, 2w)`` field -> ``(h, w)`` codepoints, with the edge antialiased.

    :func:`pack_octant` decides each subcell against the cell's midpoint, so a
    boundary crossing a cell lands on a subcell edge — the staircase that
    reads as "pixelated" however fine the grid is, and one that no amount of
    resolution removes because it is a quantisation of *position*, not of
    detail.

    Here each subcell is judged against its own ordered threshold instead, so
    how many subcells light follows how far through the cell the boundary
    actually sits. The step becomes a stipple whose density tracks coverage,
    and the eye integrates that back into a boundary falling *between*
    subcells.

    This only does something when the field has a gradient across the cell. A
    field that is flat on each side of its edges — one drawn from a table of
    flat values, say — hands every cell exactly two values, and for two values
    every threshold in (0, 1) picks the same subcells. Interpolate the source
    first or this is an expensive no-op; Kaleidoscope Ultra's docstring has
    the measured version of that mistake.
    """
    if lo is None or hi is None:
        lo, hi = cell_hilo(field)
    if CELL_MODE == "quadrant":
        return _pack_quadrant(field, lo, hi, dither=True)

    sub, h, w = _octant_subcells(field)
    span = hi - lo

    mask = np.zeros((h, w), dtype=np.int32)
    k = 0
    for r in range(4):
        for c in range(2):
            on = sub[k] >= lo + span * OCTANT_DITHER[r, c]
            np.add(mask, np.where(on, OCTANT_BITS[r, c], 0), out=mask)
            k += 1
    return OCTANT_LUT_WIDE[mask]


def pack_octant(field: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``(4h, 2w)`` float field -> ``(h, w)`` octant codepoints.

    The dot-grid resolution of :func:`pack_braille` with two colours a cell
    instead of one, which is the whole point: ``make_strips`` bills per colour
    *run*, so shape carried by the glyph is shape that costs nothing. A subcell
    lights when it is at or above its own cell's midpoint, leaving ``lo`` and
    ``hi`` to be ramped into the background and foreground.

    Octants are Unicode 16 (2024). Terminals that custom-draw block glyphs
    (kitty, ghostty, Windows Terminal 1.22+) and fonts that ship them (Cascadia
    Code 2404.03+) render them; older setups show tofu, so a mode using this
    should be reachable but not the default.
    """
    if CELL_MODE == "quadrant":
        return _pack_quadrant(field, lo, hi, dither=False)

    sub, h, w = _octant_subcells(field)
    thr = (lo + hi) * np.float32(0.5)
    mask = np.zeros((h, w), dtype=np.int32)
    k = 0
    for r in range(4):
        for c in range(2):
            np.add(mask, np.where(sub[k] >= thr, OCTANT_BITS[r, c], 0), out=mask)
            k += 1
    return OCTANT_LUT_WIDE[mask]


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
    return _noise_bits(shape, seed).astype(np.float32) * _INV24


def _noise_bits(shape: tuple[int, int], seed: int) -> np.ndarray:
    """The hash itself, as 24-bit integers. See :func:`noise`."""
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
        np.bitwise_and(h, np.uint32(0xFFFFFF), out=h)
    return h


def noise_level(level: np.ndarray | float) -> np.ndarray:
    """A 0..1 dither threshold, pre-scaled into :func:`noise_below`'s space.

    Fixed per terminal size in every mode that dithers by radius or depth, so
    it belongs in scratch rather than in the frame.

    ``ceil``, not truncation, and that is the whole reason this is a function
    rather than a multiply at the call site: ``h < ceil(t * 2**24)`` is exactly
    ``h * 2**-24 < t`` for integer ``h``, where truncating flips the comparison
    for every sample that lands on the bucket the threshold sits inside.
    """
    scaled = np.ceil(np.clip(np.asarray(level, dtype=np.float64), 0.0, 1.0) * 0x1000000)
    return scaled.astype(np.uint32)


def noise_below(shape: tuple[int, int], seed: int, level: np.ndarray) -> np.ndarray:
    """``noise(shape, seed) < level``, without building the float field.

    Same answer, bit for bit, for a threshold prepared by :func:`noise_level`
    — the dither is a comparison, and the conversion to float existed only to
    make the comparison readable. At 400x800 dots this measured 1.27 ms
    against 2.02 ms for the float route, which on the tunnel modes is most of
    the difference between holding 60 fps and being reused on alternate
    frames.
    """
    return _noise_bits(shape, seed) < level


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

    if h == 0 or w == 0:
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
    "OCTANT_BASE",
    "OCTANT_BITS",
    "OCTANT_DITHER",
    "OCTANT_DITHER_2D",
    "QUADRANT_DITHER_2D",
    "OCTANT_LUT",
    "OCTANT_LUT_WIDE",
    "QUADRANT_DITHER",
    "QUADRANT_LUT",
    "set_cell_mode",
    "RAMP_STEPS",
    "SHADES",
    "SPACE",
    "UPPER_HALF",
    "blank",
    "blocks_from_levels",
    "broadcast_rows",
    "cell_hilo",
    "cell_max",
    "cell_mean",
    "frac",
    "make_strips",
    "noise",
    "noise_below",
    "noise_level",
    "pack_braille",
    "pack_octant",
    "pack_octant_bits",
    "pack_octant_smooth",
    "shade_cells",
    "subcell_rows",
    "row_gradient",
]




