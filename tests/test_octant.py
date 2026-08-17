"""Octant cells: the lookup table, the packing, and the mode that uses them.

The table is the risky part. 230 of the 256 patterns live contiguously in
U+1CD00..U+1CDE5 and the other 26 are scattered characters that existed before
Unicode 16 encoded the block, so a wrong entry is a mirrored or transposed
picture — invisible in review, obvious on screen, and impossible to spot from
a benchmark. These tests pin the geometry rather than the numbers: they render
a known pattern and assert which character comes out.
"""

from __future__ import annotations

import numpy as np
import pytest

from spektr.render import (
    BRAILLE_BASE,
    OCTANT_BASE,
    OCTANT_LUT,
    SPACE,
    cell_hilo,
    pack_braille,
    pack_octant,
    pack_octant_bits,
)

# ── the table ────────────────────────────────────────────────────────────────


def test_lut_covers_every_mask():
    assert OCTANT_LUT.shape == (256,)
    assert len(set(OCTANT_LUT.tolist())) == 256, "two masks share a codepoint"


def test_octant_block_is_contiguous_and_the_right_size():
    """230 in the block, ascending, no gaps — the rule the LUT is built on."""
    in_block = sorted(c for c in OCTANT_LUT.tolist() if OCTANT_BASE <= c <= 0x1CDE5)
    assert len(in_block) == 230
    assert in_block[0] == OCTANT_BASE
    assert in_block[-1] == 0x1CDE5
    assert in_block == list(range(OCTANT_BASE, OCTANT_BASE + 230))


def test_masks_ascend_through_the_block():
    """The n-th mask that is not an exception maps to the n-th codepoint."""
    seen = [
        (m, int(c))
        for m, c in enumerate(OCTANT_LUT)
        if OCTANT_BASE <= int(c) <= 0x1CDE5
    ]
    for i in range(1, len(seen)):
        assert seen[i][0] > seen[i - 1][0]
        assert seen[i][1] == seen[i - 1][1] + 1


# ── the geometry ─────────────────────────────────────────────────────────────

def _tiled(pattern):
    """A 4x2 subcell pattern, tiled over a 3x3 grid of cells."""
    return np.tile(np.array(pattern, dtype=np.float32), (3, 3))


@pytest.mark.parametrize(
    "pattern, want, label",
    [
        ([[1, 1], [1, 1], [0, 0], [0, 0]], 0x2580, "UPPER HALF BLOCK"),
        ([[0, 0], [0, 0], [1, 1], [1, 1]], 0x2584, "LOWER HALF BLOCK"),
        ([[1, 0], [1, 0], [1, 0], [1, 0]], 0x258C, "LEFT HALF BLOCK"),
        ([[0, 1], [0, 1], [0, 1], [0, 1]], 0x2590, "RIGHT HALF BLOCK"),
        ([[0, 0], [0, 0], [0, 0], [1, 1]], 0x2582, "LOWER ONE QUARTER BLOCK"),
        ([[1, 0], [1, 0], [0, 0], [0, 0]], 0x2598, "QUADRANT UPPER LEFT"),
        ([[0, 0], [0, 0], [0, 1], [0, 1]], 0x2597, "QUADRANT LOWER RIGHT"),
        ([[1, 1], [1, 1], [1, 1], [0, 1]], 0x1CDAB, "BLOCK OCTANT-1234568"),
    ],
)
def test_known_patterns_render_as_the_right_character(pattern, want, label):
    field = _tiled(pattern)
    lo, hi = cell_hilo(field)
    codes = pack_octant(field, lo, hi)
    got = set(codes.ravel().tolist())
    assert got == {want}, f"{label}: got U+{got.pop():04X}, wanted U+{want:04X}"


def test_row_major_bit_order():
    """A single lit subcell walks the mask through 1, 2, 4, ... row-major.

    Against the exact table, not through a packer: four of the eight
    single-subcell patterns are widened before they are drawn (see
    :data:`_OCTANT_WIDEN`), and this is the property that decides whether the
    picture is transposed, which has to be checked where it is still exact.
    """
    from spektr.render import _OCTANT_ELSEWHERE

    for r in range(4):
        for c in range(2):
            mask = 1 << (r * 2 + c)
            code = int(OCTANT_LUT[mask])
            if mask in _OCTANT_ELSEWHERE:
                assert code == _OCTANT_ELSEWHERE[mask]
            else:
                assert OCTANT_BASE <= code <= 0x1CDE5


def test_flat_cell_is_solid():
    """A cell with no internal contrast is every subcell at the midpoint."""
    field = np.full((8, 4), 0.42, dtype=np.float32)
    lo, hi = cell_hilo(field)
    assert np.array_equal(lo, hi)
    assert set(pack_octant(field, lo, hi).ravel().tolist()) == {0x2588}


#: bit weight per subcell for each packer, indexed [row][col]. Braille numbers
#: its dots down the left column then down the right; octants are row-major.
#: Reusing one table for the other is the transposition these tests exist for.
_BRAILLE_BITS = np.array([[1, 8], [2, 16], [4, 32], [64, 128]], dtype=np.int32)
_OCTANT_BITS = np.array([[1, 2], [4, 8], [16, 32], [64, 128]], dtype=np.int32)


def _unpack(codes, bits, base=None):
    """Codepoints -> the ``(4h, 2w)`` bool subcell grid that produced them."""
    if base is not None:
        mask = codes - base
    else:
        inverse = {int(c): i for i, c in enumerate(OCTANT_LUT)}
        mask = np.array([[inverse[int(c)] for c in row] for row in codes], dtype=np.int32)
    h, w = codes.shape
    out = np.zeros((h * 4, w * 2), dtype=bool)
    for r in range(4):
        for c in range(2):
            out[r::4, c::2] = (mask & bits[r, c]) != 0
    return out


@pytest.mark.parametrize(
    "pattern, want",
    [
        ([[1, 1], [1, 1], [0, 0], [0, 0]], 0x2580),
        ([[1, 1], [1, 1], [1, 1], [1, 1]], 0x2588),
        ([[0, 0], [0, 0], [0, 0], [0, 0]], SPACE),
        # widened: a lone subcell has no widely drawn glyph of its own
        ([[1, 0], [0, 0], [0, 0], [0, 0]], 0x2598),
        ([[0, 0], [0, 0], [0, 0], [0, 1]], 0x2597),
    ],
)
def test_pack_octant_bits_uses_the_same_geometry(pattern, want):
    grid = np.tile(np.array(pattern, dtype=bool), (3, 3))
    got = set(pack_octant_bits(grid).ravel().tolist())
    assert got == {want}, f"got U+{got.pop():04X}, wanted U+{want:04X}"


def test_widening_only_ever_adds_subcells():
    """A lit subcell is never dropped, and few are added.

    Widening exists so the eight patterns without a widely drawn glyph get
    one; it must not turn into a licence to redraw the picture. Growing is the
    safe direction — a hole in an outline is visible where a subcell of extra
    thickness at 4x4 pixels is not.
    """
    rng = np.random.default_rng(11)
    lit = rng.random((4 * 9, 2 * 13)) > 0.5
    back = _unpack(pack_octant_bits(lit), _OCTANT_BITS)

    assert np.array_equal(back & lit, lit), "a lit subcell was dropped"
    added = int((back & ~lit).sum())
    assert added <= lit.size * 0.05, f"{added} subcells added of {lit.size}"


def test_no_packer_emits_a_glyph_from_the_thinly_supported_blocks():
    """The whole point of the widening table.

    Fonts that ship the entire octant block routinely miss the four
    single-subcell characters at U+1CEA0 and the two middle quarters at
    U+1FBE6/7 — and those are exactly the patterns the rim of any shape is
    made of, so the mode that hits them is the one drawing a silhouette.
    Nothing reaching the terminal may come from there.
    """
    thin = {0x1CEA0, 0x1CEA3, 0x1CEA8, 0x1CEAB, 0x1FB82, 0x1FB85, 0x1FBE6, 0x1FBE7}

    for mask in range(256):
        pattern = [[(mask >> (r * 2 + c)) & 1 for c in range(2)] for r in range(4)]
        grid = np.array(pattern, dtype=bool)
        got = int(pack_octant_bits(grid)[0, 0])
        assert got not in thin, f"mask {mask:08b} emitted U+{got:04X}"
        assert got == SPACE or got == 0x2588 or 0x2580 <= got <= 0x259F or (
            OCTANT_BASE <= got <= 0x1CDE5
        ), f"mask {mask:08b} emitted U+{got:04X}, outside Block Elements and the octant block"


def test_widening_keeps_its_mirror_partner():
    """Kaleidoscope's symmetry is a property of the glyphs, not just the field.

    Mirroring a cell swaps its subcell columns. If a widened pattern's
    substitute were not the mirror of its mirror's substitute, a symmetric
    field would come out asymmetric — which is the one thing that mode
    promises.
    """
    def swap(mask):
        out = 0
        for r in range(4):
            left, right = 1 << (2 * r), 1 << (2 * r + 1)
            if mask & left:
                out |= right
            if mask & right:
                out |= left
        return out

    for mask in range(256):
        pattern = [[(mask >> (r * 2 + c)) & 1 for c in range(2)] for r in range(4)]
        grid = np.array(pattern, dtype=bool)
        mirrored = np.array(
            [[(swap(mask) >> (r * 2 + c)) & 1 for c in range(2)] for r in range(4)],
            dtype=bool,
        )
        a = _unpack(pack_octant_bits(grid), _OCTANT_BITS)
        b = _unpack(pack_octant_bits(mirrored), _OCTANT_BITS)
        assert np.array_equal(a[:, ::-1], b), f"mask {mask:08b} widens asymmetrically"


def test_the_two_packers_agree_on_a_thresholded_field():
    """``pack_octant`` is ``pack_octant_bits`` of its own threshold."""
    rng = np.random.default_rng(5)
    field = rng.random((4 * 6, 2 * 6), dtype=np.float32)
    lo, hi = cell_hilo(field)
    thr = np.repeat(np.repeat((lo + hi) * 0.5, 4, axis=0), 2, axis=1)
    assert np.array_equal(pack_octant(field, lo, hi), pack_octant_bits(field >= thr))


def test_cell_hilo_is_the_range_of_the_cell():
    rng = np.random.default_rng(3)
    field = rng.random((4 * 5, 2 * 7), dtype=np.float32)
    lo, hi = cell_hilo(field)
    assert lo.shape == hi.shape == (5, 7)
    for y in range(5):
        for x in range(7):
            block = field[y * 4 : y * 4 + 4, x * 2 : x * 2 + 2]
            assert lo[y, x] == pytest.approx(block.min())
            assert hi[y, x] == pytest.approx(block.max())


# ── the mode ─────────────────────────────────────────────────────────────────

def _ctx(w, h, frame, state, palette):
    from spektr.analysis import N_BANDS
    from spektr.modes import Ctx

    n = np.linspace(0.15, 0.9, N_BANDS).astype(np.float32)
    return Ctx(
        w=w, h=h, bands=n, peaks=n, bands_l=n, bands_r=n,
        wave=np.zeros(512, dtype=np.float32), stereo=np.zeros((512, 2), dtype=np.float32),
        frame=frame, t=frame / 60.0, dt=1 / 60.0, energy=float(n.mean()),
        silent=False, palette=palette, state=state, bars=len(n),
    )


def _kaleido_fine():
    """The octant variant, or a skip.

    The primitive above stands on its own — a plugin can use it without any
    built-in doing so — and these two tests are about the first mode that
    does. Skipping rather than failing keeps the table's own tests meaningful
    if the mode is ever renamed or pulled.
    """
    import spektr.modes as M

    reg = {m.name: m for m in M.MODES}
    if "Kaleidoscope Fine" not in reg:
        pytest.skip("no built-in mode uses octant cells")
    return reg["Kaleidoscope Fine"]


def test_kaleidoscope_fine_draws_octants():
    from spektr.palette import BUILTIN, Palette

    pal = Palette(BUILTIN["gruvbox"])
    m = _kaleido_fine()
    st: dict = {}
    for f in range(6):
        codes, cidx, bidx = m.fn(_ctx(120, 30, f, st, pal))

    assert codes.shape == cidx.shape == bidx.shape == (30, 120)
    octants = ((codes >= OCTANT_BASE) & (codes <= 0x1CDE5)).sum()
    assert octants > 0, "no octant glyph anywhere — the mode is not using them"
    assert len(np.unique(codes)) > 8, "the glyph is barely varying; is the mask stuck?"


def _shaded(field, cells):
    from spektr import render

    before = render.CELL_MODE
    render.set_cell_mode(cells)
    try:
        return render.shade_cells(field)
    finally:
        render.set_cell_mode(before)


@pytest.mark.parametrize("cells, rows", [("quadrant", 2), ("octant", 4)])
def test_a_smooth_field_is_not_dithered_at_all(cells, rows):
    """The bug a screenshot caught, and the one arithmetic could not.

    Dithering buys sub-cell *position*. A cell with no edge in it has no
    position to resolve, so thresholding it trades colour precision for
    texture and gets nothing back — and since a smooth field has an edge in
    almost no cells, the first version covered the entire screen in stipple.
    It measured perfectly: right picture, right cost, and visibly worse than
    the half-block renderer it replaced.

    A flat cell is now drawn the half-block way, which is what those modes
    used before any of this.
    """
    from spektr.render import UPPER_HALF

    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
    codes, fg, bg = _shaded(np.repeat(ramp, rows * 64, axis=0), cells)

    assert set(codes.ravel().tolist()) == {UPPER_HALF}, (
        "a smooth gradient came out dithered"
    )
    assert len(np.unique(np.stack([fg, bg]))) > 32, (
        "the gradient lost its colour resolution as well"
    )


def _boundary(rows, nx, ny, edge):
    """One cell holding a soft straight boundary with normal ``(nx, ny)``."""
    yc = (np.arange(rows, dtype=np.float32)[:, None] + 0.5) / rows
    xc = (np.arange(2, dtype=np.float32)[None, :] + 0.5) / 2.0
    return np.clip((xc * nx + yc * ny - edge) / np.float32(0.35) + 0.5, 0.0, 1.0)


def test_a_horizontal_boundary_travels_down_an_octant_cell():
    """What the mask is for, now that flat cells opt out of it.

    Tone comes from the flat path, which has the whole 64-step ramp to spend.
    What is left for the mask is *position*: as a boundary sweeps down a cell,
    the number of lit subcells has to follow it, so the edge appears to land
    between subcell rows rather than snapping to the cell border. Four rows is
    the entire reason these modes have octant variants.
    """
    from spektr.render import OCTANT_LUT, UPPER_HALF

    popcount = {int(c): bin(i).count("1") for i, c in enumerate(OCTANT_LUT)}
    seen = []
    for edge in np.linspace(0.1, 0.9, 9, dtype=np.float32):
        codes, _, _ = _shaded(_boundary(4, 0.0, 1.0, edge), "octant")
        code = int(codes[0, 0])
        seen.append(8 - popcount[code] if code != UPPER_HALF else None)

    lit = [s for s in seen if s is not None]
    assert len(lit) >= 6, f"the edge path fired only {len(lit)} times of 9"
    assert lit == sorted(lit), f"coverage is not monotonic in the edge: {lit}"
    assert max(lit) - min(lit) >= 2, f"the mask barely moved: {lit}"


def test_two_rows_cannot_place_a_horizontal_boundary_and_do_not_pretend_to():
    """The limit of quadrant mode, pinned so nobody re-derives it by eye.

    Cut against a single midpoint, a cell of two rows holding a monotonic
    gradient always lights exactly the upper one — there is no threshold that
    puts a horizontal boundary anywhere but the middle. So for a *horizontal*
    edge, quadrant mode is the half-block renderer and cannot be better.

    Trying to beat it is what produced both stipple bugs: an ordered threshold
    does vary the count, but it does so by lighting subcells away from where
    the field is high, which is a halftone screen rather than a boundary.
    Quadrant mode earns its keep on the other axis — see the test below — and
    on colour, not here.
    """
    from spektr.render import QUADRANT_LUT

    popcount = {int(c): bin(i).count("1") for i, c in enumerate(QUADRANT_LUT)}
    for edge in np.linspace(0.1, 0.9, 9, dtype=np.float32):
        codes, _, _ = _shaded(_boundary(2, 0.0, 1.0, edge), "quadrant")
        assert popcount[int(codes[0, 0])] in (0, 2, 4)


def test_an_edge_cell_is_coloured_by_each_side_s_mean_not_its_extremes():
    """The third and last thing that made these modes read as pixels.

    A cell painted with its extremes is more contrasty than the field it
    stands for. Around a Chladni nodal line that turned the halo into flat
    slabs with a hard step between them, which is what "the coloured streaks
    are what are kinda pixelated" was pointing at — every cell in the halo was
    reporting the brightest and darkest thing in it rather than what was there.

    Here the cut lights the top three rows, holding 1.0, 0.9 and 0.6, so the
    foreground should be their mean of 0.833. The extremes version paints it
    1.0 — a sixth of the ramp too bright, on every cell of the halo at once.
    """
    from spektr.render import RAMP_STEPS, UPPER_HALF

    top = RAMP_STEPS - 1
    col = np.array([1.0, 0.9, 0.6, 0.0], dtype=np.float32)[:, None]
    codes, fg, bg = _shaded(np.repeat(col, 2, axis=1), "octant")

    assert codes[0, 0] != UPPER_HALF, "three lit rows is not the half-block glyph"
    want = (1.0 + 0.9 + 0.6) / 3.0 * top
    assert abs(int(fg[0, 0]) - want) <= 4, (
        f"foreground {fg[0, 0]} is not the mean of the lit subcells (~{want:.0f})"
    )
    assert int(fg[0, 0]) < 0.95 * top, "foreground is still tracking the cell's maximum"
    assert int(bg[0, 0]) <= 4, f"background {bg[0, 0]} is not the mean of the dark side"


@pytest.mark.parametrize("cells, rows", [("quadrant", 2), ("octant", 4)])
def test_the_mask_follows_the_boundary_angle(cells, rows):
    """Both geometries are two columns wide, so both carry a boundary's slope.

    This is the half of the gain quadrant mode keeps: a diagonal edge lights a
    different set of subcells than a flat one, so the glyph reports which way
    the boundary runs. A half-block cell has one column and cannot.
    """
    seen = set()
    for nx, ny in ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (1.0, -1.0)):
        for edge in np.linspace(0.0, 1.0, 7, dtype=np.float32):
            codes, _, _ = _shaded(_boundary(rows, nx, ny, edge), cells)
            seen.add(int(codes[0, 0]))
    assert len(seen) >= 4, f"{cells}: the mask ignores the boundary angle: {seen}"


def test_the_dither_matrix_survives_a_mirror():
    """Its two columns must be identical, or symmetric modes come out crooked.

    Mirroring a cell swaps its subcell columns. The first version of this
    table was a 2-D Bayer spread with different values left and right, and
    Kaleidoscope Ultra lost the bilateral symmetry that is the entire point of
    a mirror tube — caught by the symmetry test below, not by eye.
    """
    from spektr.render import OCTANT_DITHER

    assert np.array_equal(OCTANT_DITHER[:, 0], OCTANT_DITHER[:, 1])
    assert ((OCTANT_DITHER > 0.0) & (OCTANT_DITHER < 1.0)).all()
    assert len(set(OCTANT_DITHER[:, 0].tolist())) == 4, "rows must not share a threshold"


def test_smoothing_is_a_no_op_on_a_two_valued_cell():
    """The trap worth pinning: a flat-sided edge cannot be antialiased.

    A cell holding exactly two values — one per side of a boundary between
    two flat regions — lights the same subcells for *every* threshold in
    (0, 1), so the ordered version cannot differ from the midpoint one. A mode
    with flat fragments has to interpolate its source before this does
    anything, and finding that out by measurement cost real time.
    """
    from spektr.render import pack_octant_smooth

    field = np.array([[0.2, 0.9], [0.2, 0.9], [0.2, 0.9], [0.2, 0.9]], dtype=np.float32)
    field = np.tile(field, (3, 3))
    lo, hi = cell_hilo(field)
    assert np.array_equal(pack_octant_smooth(field), pack_octant(field, lo, hi))


def test_smoothing_tracks_coverage_on_a_gradient():
    """Across a ramp, how many subcells light follows how far the edge is in."""
    from spektr.render import OCTANT_LUT, pack_octant_smooth

    inverse = {int(c): i for i, c in enumerate(OCTANT_LUT)}
    centres = np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float32)[:, None]

    counts = []
    # Kept inside the cell: a boundary swept past either end leaves the cell
    # uniform, and a uniform cell has no range to threshold against, so it
    # comes out solid — see test_flat_cell_is_solid. A mode that draws against
    # empty space must use pack_octant_bits for exactly that reason.
    for edge in np.linspace(0.05, 0.95, 10, dtype=np.float32):
        # a soft boundary crossing the cell at `edge`, the shape interpolating
        # a source produces — dark below it, bright above, blending across
        rows = np.clip((centres - edge) / np.float32(0.4) + 0.5, 0.0, 1.0)
        field = np.tile(np.repeat(rows, 2, axis=1), (2, 2))
        code = int(pack_octant_smooth(field)[0, 0])
        counts.append(bin(inverse[code]).count("1"))

    assert counts == sorted(counts, reverse=True), (
        f"coverage is not monotonic as the boundary sweeps through: {counts}"
    )
    assert max(counts) - min(counts) >= 3, (
        f"the mask barely moved as the boundary swept the whole cell: {counts}"
    )


BLOCK_ELEMENTS = set(range(0x2580, 0x25A0)) | {SPACE}


@pytest.fixture
def quadrant_cells():
    """Run a test with the packers in Block-Elements-only mode."""
    from spektr import render

    before = render.CELL_MODE
    render.set_cell_mode("quadrant")
    try:
        yield
    finally:
        render.set_cell_mode(before)


def test_quadrant_lut_is_entirely_block_elements():
    """The reason the fallback exists: this set is complete, octants are not."""
    from spektr.render import QUADRANT_LUT

    assert len(QUADRANT_LUT) == 16
    assert len(set(QUADRANT_LUT.tolist())) == 16
    assert set(QUADRANT_LUT.tolist()) <= BLOCK_ELEMENTS


@pytest.mark.parametrize(
    "pattern, want, label",
    [
        ([[1, 0], [0, 0]], 0x2598, "QUADRANT UPPER LEFT"),
        ([[0, 1], [0, 0]], 0x259D, "QUADRANT UPPER RIGHT"),
        ([[0, 0], [1, 0]], 0x2596, "QUADRANT LOWER LEFT"),
        ([[0, 0], [0, 1]], 0x2597, "QUADRANT LOWER RIGHT"),
        ([[1, 1], [0, 0]], 0x2580, "UPPER HALF BLOCK"),
        ([[0, 0], [1, 1]], 0x2584, "LOWER HALF BLOCK"),
        ([[1, 0], [1, 0]], 0x258C, "LEFT HALF BLOCK"),
        ([[0, 1], [0, 1]], 0x2590, "RIGHT HALF BLOCK"),
        ([[1, 1], [1, 1]], 0x2588, "FULL BLOCK"),
        ([[0, 0], [0, 0]], SPACE, "SPACE"),
    ],
)
def test_quadrant_geometry(quadrant_cells, pattern, want, label):
    """Same geometric pinning as the octant table, on the reduced grid.

    Each quadrant covers two octant rows, so the pattern is written at octant
    resolution and doubled down — which is also exactly what the reduction in
    the packer has to undo.
    """
    grid = np.repeat(np.array(pattern, dtype=np.float32), 2, axis=0)
    grid = np.tile(grid, (3, 3))
    got = set(pack_octant_bits(grid.astype(bool)).ravel().tolist())
    assert got == {want}, f"{label}: got U+{got.pop():04X}, wanted U+{want:04X}"


def test_quadrant_mode_never_leaves_block_elements(quadrant_cells):
    """Every one of the 256 subcell patterns, through all three packers."""
    from spektr.render import pack_octant_smooth

    for mask in range(256):
        pattern = [[(mask >> (r * 2 + c)) & 1 for c in range(2)] for r in range(4)]
        bits = np.array(pattern, dtype=bool)
        field = np.tile(bits.astype(np.float32) * 0.8 + 0.1, (2, 2))
        lo, hi = cell_hilo(field)
        for got in (
            int(pack_octant_bits(np.tile(bits, (2, 2)))[0, 0]),
            int(pack_octant(field, lo, hi)[0, 0]),
            int(pack_octant_smooth(field, lo, hi)[0, 0]),
        ):
            assert got in BLOCK_ELEMENTS, f"mask {mask:08b} emitted U+{got:04X}"


def test_quadrant_keeps_a_thin_line(quadrant_cells):
    """A one-row outline must thicken, never vanish.

    Pairs of octant rows collapse by maximum for exactly this reason: a mean
    would average a single lit row against its dark neighbour and drop it
    below any threshold, which is how an outline mode loses its outline.
    """
    lit = np.zeros((8, 4), dtype=bool)
    lit[0::4] = True   # the top octant row of every cell, and nothing else
    codes = pack_octant_bits(lit)
    assert set(codes.ravel().tolist()) == {0x2580}, "the top row was lost"


def test_cell_mode_rejects_nonsense():
    from spektr import render

    with pytest.raises(ValueError):
        render.set_cell_mode("sextant")
    assert render.CELL_MODE in ("octant", "quadrant")


def _mode(name):
    import spektr.modes as M

    reg = {m.name: m for m in M.MODES}
    if name not in reg:
        pytest.skip(f"{name} is not registered")
    return reg[name]


def test_valentine_fine_draws_the_same_heart_solid():
    """The port's whole promise: identical shape, drawn as a surface.

    Valentine is a silhouette, so the octant version is not a different
    picture — it is the same lit dots packed into block mosaic instead of
    braille. If the dot sets ever diverge, the variant has stopped being a
    rendering of the same mode.
    """
    from spektr.palette import BUILTIN, Palette

    pal = Palette(BUILTIN["gruvbox"])
    plain, fine = _mode("Valentine"), _mode("Valentine Fine")
    sa: dict = {}
    sb: dict = {}
    for f in range(8):
        codes_a, idx_a = plain.fn(_ctx(120, 30, f, sa, pal))
        codes_b, idx_b = fine.fn(_ctx(120, 30, f, sb, pal))

    braille_dots = _unpack(codes_a, _BRAILLE_BITS, base=BRAILLE_BASE)
    octant_dots = _unpack(codes_b, _OCTANT_BITS)
    assert np.array_equal(octant_dots & braille_dots, braille_dots), (
        "the octant variant dropped dots the braille one draws"
    )
    # The rim picks up a few subcells where a lone lit one is widened to its
    # quadrant, because the exact glyph for a lone subcell is one most fonts
    # do not have. Anything beyond a couple of percent means the shape moved.
    extra = int((octant_dots & ~braille_dots).sum())
    assert extra <= braille_dots.size * 0.02, f"{extra} extra dots"
    assert np.array_equal(idx_a, idx_b), "the colour path is meant to be untouched"

    solid = int((codes_b == 0x2588).sum())
    partial = int(((codes_b != SPACE) & (codes_b != 0x2588)).sum())
    assert solid > 0, "nothing came out solid — the heart is still stipple"
    assert partial > 0, "no partial cells — the rim has no detail"


def test_valentine_itself_still_draws_braille():
    """The original must not have been quietly ported underneath the variant."""
    from spektr.palette import BUILTIN, Palette

    pal = Palette(BUILTIN["gruvbox"])
    st: dict = {}
    for f in range(6):
        codes, _ = _mode("Valentine").fn(_ctx(120, 30, f, st, pal))
    braille = (codes >= BRAILLE_BASE) & (codes <= BRAILLE_BASE + 0xFF)
    assert bool((braille | (codes == SPACE)).all())


def test_kaleidoscope_ultra_is_smoother_and_still_symmetric():
    """Ultra must place more boundaries inside cells, and stay a mirror tube."""
    from spektr.palette import BUILTIN, Palette

    pal = Palette(BUILTIN["gruvbox"])
    fine, ultra = _mode("Kaleidoscope Fine"), _mode("Kaleidoscope Ultra")
    sa: dict = {}
    sb: dict = {}
    for f in range(10):
        codes_f, cidx_f, bidx_f = fine.fn(_ctx(160, 40, f, sa, pal))
        codes_u, cidx_u, bidx_u = ultra.fn(_ctx(160, 40, f, sb, pal))

    def partial(codes):
        return float(((codes != SPACE) & (codes != 0x2588)).mean())

    assert partial(codes_u) > partial(codes_f), (
        "Ultra draws no more partial cells than Fine — the smoothing is a no-op"
    )

    inverse = {int(c): i for i, c in enumerate(OCTANT_LUT)}
    mask = np.array([[inverse[int(c)] for c in row] for row in codes_u], dtype=np.int32)
    swapped = np.zeros_like(mask)
    for r in range(4):
        left, right = 1 << (2 * r), 1 << (2 * r + 1)
        swapped |= np.where(mask & left, right, 0) | np.where(mask & right, left, 0)
    assert np.array_equal(swapped[:, ::-1], mask), "Ultra is not bilaterally symmetric"
    assert np.array_equal(cidx_u, cidx_u[:, ::-1])
    assert np.array_equal(bidx_u, bidx_u[:, ::-1])


def test_kaleidoscope_fine_stays_mirror_symmetric():
    """The mode's whole promise. A mirrored cell swaps its subcell columns."""
    from spektr.palette import BUILTIN, Palette

    pal = Palette(BUILTIN["gruvbox"])
    m = _kaleido_fine()
    st: dict = {}
    for f in range(6):
        codes, cidx, bidx = m.fn(_ctx(120, 30, f, st, pal))

    assert np.array_equal(cidx, cidx[:, ::-1])
    assert np.array_equal(bidx, bidx[:, ::-1])

    inverse = {int(c): i for i, c in enumerate(OCTANT_LUT)}
    mask = np.array([[inverse[int(c)] for c in row] for row in codes], dtype=np.int32)
    swapped = np.zeros_like(mask)
    for r in range(4):
        left, right = 1 << (2 * r), 1 << (2 * r + 1)
        swapped |= np.where(mask & left, right, 0) | np.where(mask & right, left, 0)
    assert np.array_equal(swapped[:, ::-1], mask)
