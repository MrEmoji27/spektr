"""The octant variants: the same modes at 2x4 subcells, as separate modes.

``Sonar Fine``, ``Scope Fine`` and ``ECG Fine`` pack the identical lit dot
set into Unicode 16 octant glyphs (:func:`render.pack_octant_bits`) — a
shape drawn against empty space wants one colour and a foreground only,
because a background index would paint the whole cell opaque. ``Radial
Fine`` and ``Maelstrom Fine`` threshold a continuous field at each cell's
own midpoint (:func:`render.pack_octant` + :func:`render.cell_hilo`) — a
mode that fills its viewport has both a background and a foreground.

Every test that compares the two versions renders them with separate
scratch dicts: the two modes share one body, and sharing one dict would
make both advance the same phase accumulators, which makes every frame
come out "different" for no reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.palette import BUILTIN, Palette
from spektr.render import (
    BRAILLE_BASE,
    OCTANT_BASE,
    OCTANT_LUT,
    SPACE,
)

PAL = Palette(BUILTIN["gruvbox"])

#: (original, variant) pairs — the whole point of this file. A variant must
#: register right behind its original in the mode cycle.
VARIANT_OF = [
    ("Sonar", "Sonar Fine"),
    ("Scope", "Scope Fine"),
    ("ECG", "ECG Fine"),
    ("Radial", "Radial Fine"),
]

#: bit weight per subcell for each packer, indexed [row][col]. Braille numbers
#: its dots down the left column then down the right; octants are row-major.
_BRAILLE_BITS = np.array([[1, 8], [2, 16], [4, 32], [64, 128]], dtype=np.int32)
_OCTANT_BITS = np.array([[1, 2], [4, 8], [16, 32], [64, 128]], dtype=np.int32)


def _ctx(w, h, frame, state, palette):
    """A moving spectrum and a moving waveform, so trace modes draw real curves.

    test_octant's helper feeds a zero waveform, which is exactly what a scope
    test must not do: a flat line renders one glyph and "the glyph varies"
    fails for the wrong reason.
    """
    n = np.linspace(0.15, 0.9, N_BANDS).astype(np.float32)
    wave = np.sin(np.linspace(0, 40, 512) + frame * 0.4) * 0.7
    stereo = np.stack((wave, np.roll(wave, 7)), axis=1)
    return Ctx(
        w=w, h=h, bands=n, peaks=n, bands_l=n * 0.9, bands_r=n,
        wave=wave, stereo=stereo,
        frame=frame, t=frame / 60.0, dt=1 / 60.0, energy=float(n.mean()),
        silent=False, palette=palette, state=state,
    )


def _mode(name):
    import spektr.modes as M

    reg = {m.name: m for m in M.MODES}
    if name not in reg:
        pytest.skip(f"{name} is not registered")
    return reg[name]


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


# ── registration ─────────────────────────────────────────────────────────────

def test_fine_modes_registered_after_their_originals():
    import spektr.modes as M

    order = [m.name for m in M.MODES]
    for original, variant in VARIANT_OF:
        if M.get(original) is None:
            pytest.skip(f"{original} is not registered")
        assert M.get(variant) is not None, f"{variant} is not registered"
        assert order.index(variant) == order.index(original) + 1, (
            f"{variant} must sit directly after {original} in the mode cycle"
        )
        blurb = M.get(variant).blurb.lower()
        assert "octant" in blurb, f"{variant} blurb does not warn about octants"


# ── the pack_octant_bits ports: same dots, drawn solid ───────────────────────

def _assert_solid_superset(plain, fine, frames=8, w=120, h=30):
    """The port's whole promise: identical lit dots, drawn as a surface.

    Renders both versions on separate scratch dicts and checks, frame by
    frame, that the octant dots are a superset of the braille dots — the
    eight rim patterns are widened to the nearest glyph most fonts actually
    have, which only ever adds subcells — and that the colour path is
    untouched.
    """
    sa: dict = {}
    sb: dict = {}
    for f in range(frames):
        codes_a, idx_a = plain.fn(_ctx(w, h, f, sa, PAL))
        codes_b, idx_b = fine.fn(_ctx(w, h, f, sb, PAL))
        braille_dots = _unpack(codes_a, _BRAILLE_BITS, base=BRAILLE_BASE)
        octant_dots = _unpack(codes_b, _OCTANT_BITS)
        assert np.array_equal(octant_dots & braille_dots, braille_dots), (
            "the octant variant dropped dots the braille one draws"
        )
        extra = int((octant_dots & ~braille_dots).sum())
        assert extra <= braille_dots.size * 0.02, f"{extra} extra dots"
        assert np.array_equal(idx_a, idx_b), "the colour path is meant to be untouched"

    assert (codes_b >= OCTANT_BASE).any(), "no octant-block glyph anywhere"
    assert len(np.unique(codes_b)) > 8, "the glyph is barely varying; is the mask stuck?"


def test_sonar_fine_is_a_solid_superset_of_sonar():
    _assert_solid_superset(_mode("Sonar"), _mode("Sonar Fine"))


def test_scope_fine_is_a_solid_superset_of_scope():
    _assert_solid_superset(_mode("Scope"), _mode("Scope Fine"))


def test_ecg_fine_is_a_solid_superset_of_ecg():
    _assert_solid_superset(_mode("ECG"), _mode("ECG Fine"))


# ── the pack_octant ports: two colours a cell ────────────────────────────────

def _assert_two_colour(plain, fine, frames=8, w=120, h=30):
    """The two-colour ports: a background and a foreground per cell.

    These modes fill their whole viewport with a continuous field, so the
    variant carries the cell's range — ``cell_hilo`` — into both ramp ends
    and draws the mask with ``pack_octant``: the glyph, not a second colour
    run, carries which subcells are on which side of the cell's midpoint.
    """
    sa: dict = {}
    sb: dict = {}
    for f in range(frames):
        out_a = plain.fn(_ctx(w, h, f, sa, PAL))
        out_b = fine.fn(_ctx(w, h, f, sb, PAL))
        assert len(out_a) == 2, "the original must stay a single-colour mode"
        codes, cidx, bidx = out_b
        assert codes.shape == cidx.shape == bidx.shape == (h, w)
        assert (codes >= OCTANT_BASE).any(), "no octant-block glyph anywhere"
        assert (bidx != cidx).any(), "background and foreground never differ"
        assert len(np.unique(codes)) > 8, "the glyph is barely varying; is the mask stuck?"


def test_radial_fine_draws_two_colours():
    _assert_two_colour(_mode("Radial"), _mode("Radial Fine"))


def test_radial_itself_still_draws_braille():
    st: dict = {}
    for f in range(6):
        codes, _ = _mode("Radial").fn(_ctx(120, 30, f, st, PAL))
    braille = (codes >= BRAILLE_BASE) & (codes <= BRAILLE_BASE + 0xFF)
    assert bool((braille | (codes == SPACE)).all())


def test_sonar_itself_still_draws_braille():
    """The original must not have been quietly ported underneath the variant."""
    st: dict = {}
    for f in range(6):
        codes, _ = _mode("Sonar").fn(_ctx(120, 30, f, st, PAL))
    braille = (codes >= BRAILLE_BASE) & (codes <= BRAILLE_BASE + 0xFF)
    assert bool((braille | (codes == SPACE)).all())


def test_scope_itself_still_draws_braille():
    st: dict = {}
    for f in range(6):
        codes, _ = _mode("Scope").fn(_ctx(120, 30, f, st, PAL))
    braille = (codes >= BRAILLE_BASE) & (codes <= BRAILLE_BASE + 0xFF)
    assert bool((braille | (codes == SPACE)).all())


def test_ecg_itself_still_draws_braille():
    st: dict = {}
    for f in range(6):
        codes, _ = _mode("ECG").fn(_ctx(120, 30, f, st, PAL))
    braille = (codes >= BRAILLE_BASE) & (codes <= BRAILLE_BASE + 0xFF)
    assert bool((braille | (codes == SPACE)).all())
