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
        ([[1, 1], [0, 0], [0, 0], [0, 0]], 0x1FB82, "UPPER ONE QUARTER BLOCK"),
        ([[0, 0], [0, 0], [0, 0], [1, 1]], 0x2582, "LOWER ONE QUARTER BLOCK"),
        ([[1, 0], [1, 0], [0, 0], [0, 0]], 0x2598, "QUADRANT UPPER LEFT"),
        ([[0, 0], [0, 0], [0, 1], [0, 1]], 0x2597, "QUADRANT LOWER RIGHT"),
        ([[1, 0], [0, 0], [0, 0], [0, 0]], 0x1CEA8, "LEFT HALF UPPER ONE QUARTER"),
        ([[0, 0], [0, 0], [0, 0], [0, 1]], 0x1CEA0, "RIGHT HALF LOWER ONE QUARTER"),
        ([[0, 0], [1, 0], [1, 0], [0, 0]], 0x1FBE6, "MIDDLE LEFT ONE QUARTER"),
        ([[0, 0], [0, 1], [0, 1], [0, 0]], 0x1FBE7, "MIDDLE RIGHT ONE QUARTER"),
    ],
)
def test_known_patterns_render_as_the_right_character(pattern, want, label):
    field = _tiled(pattern)
    lo, hi = cell_hilo(field)
    codes = pack_octant(field, lo, hi)
    got = set(codes.ravel().tolist())
    assert got == {want}, f"{label}: got U+{got.pop():04X}, wanted U+{want:04X}"


def test_row_major_bit_order():
    """A single lit subcell walks the mask through 1, 2, 4, ... row-major."""
    for r in range(4):
        for c in range(2):
            pat = [[0, 0] for _ in range(4)]
            pat[r][c] = 1
            field = _tiled(pat)
            lo, hi = cell_hilo(field)
            code = int(pack_octant(field, lo, hi)[0, 0])
            mask = int(np.flatnonzero(OCTANT_LUT == code)[0])
            assert mask == 1 << (r * 2 + c)


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
        ([[1, 0], [0, 0], [0, 0], [0, 0]], 0x1CEA8),
        ([[0, 0], [0, 0], [0, 0], [0, 1]], 0x1CEA0),
        ([[1, 1], [1, 1], [1, 1], [1, 1]], 0x2588),
        ([[0, 0], [0, 0], [0, 0], [0, 0]], SPACE),
    ],
)
def test_pack_octant_bits_uses_the_same_geometry(pattern, want):
    grid = np.tile(np.array(pattern, dtype=bool), (3, 3))
    got = set(pack_octant_bits(grid).ravel().tolist())
    assert got == {want}, f"got U+{got.pop():04X}, wanted U+{want:04X}"


def test_pack_octant_bits_round_trips():
    rng = np.random.default_rng(11)
    lit = rng.random((4 * 9, 2 * 13)) > 0.5
    assert np.array_equal(_unpack(pack_octant_bits(lit), _OCTANT_BITS), lit)


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

    assert np.array_equal(
        _unpack(codes_a, _BRAILLE_BITS, base=BRAILLE_BASE),
        _unpack(codes_b, _OCTANT_BITS),
    ), "the octant variant lights a different set of dots"
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
