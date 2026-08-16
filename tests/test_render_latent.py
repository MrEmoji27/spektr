"""Latent-crash fixes from the 2026-08-16 audit.

Three defects that only crash outside the app's guard rails: ``make_strips``
on a zero-width grid (``IndexError``), 1-D stereo buffers through the four
scope/stereo modes (``AxisError``/``IndexError``), and Flame's per-frame
divide by a flame width that is legitimately zero on dead cells
(``FloatingPointError`` under ``np.seterr``).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import BRAILLE_BASE, blank, make_strips  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])

#: The scope-family modes that read ``ctx.stereo`` as L/R pairs.
STEREO_MODES = ("Scope", "ECG", "Helix", "Gonio")


def ctx_for(w, h, frame, state, t, bands=None, stereo=None):
    if bands is None:
        bands = np.clip(np.abs(np.sin(np.linspace(0, 3, N_BANDS) + t * 2)) * 0.8, 0, 1)
    wave = np.sin(np.linspace(0, 40, 512) + t * 10) * 0.7
    if stereo is None:
        stereo = np.stack((wave, np.roll(wave, 7)), axis=1)
    return Ctx(
        w=w, h=h, bands=bands, peaks=np.clip(bands * 1.05, 0, 1),
        bands_l=bands * 0.9, bands_r=bands, wave=wave, stereo=stereo,
        frame=frame, t=t, dt=1 / 60, energy=float(bands.mean()),
        silent=False, palette=PAL, state=state,
    )


def test_make_strips_zero_width_returns_empty():
    for w, h in ((0, 5), (5, 0), (0, 0)):
        codes, cidx = blank(w, h)
        assert make_strips(codes, cidx, PAL) == []
    # the foreground+background path is guarded the same way
    codes, cidx = blank(0, 5)
    assert make_strips(codes, cidx, PAL, np.zeros_like(cidx)) == []


def test_scope_modes_render_1d_stereo_as_mono():
    stereo_1d = np.sin(np.linspace(0, 40, 512)) * 0.7
    for name in STEREO_MODES:
        state: dict = {}
        for f in range(3):
            codes, cidx = M.get(name).fn(ctx_for(80, 24, f, state, f / 60, stereo=stereo_1d))
        assert codes.shape == (24, 80), name
        assert cidx.shape == (24, 80), name
        # the mono trace must actually draw — a blank screen would mean the
        # signal was silently dropped rather than rendered as mono
        assert (codes != BRAILLE_BASE).any(), name


def test_scope_modes_tolerate_empty_stereo():
    for name in STEREO_MODES:
        state: dict = {}
        codes, cidx = M.get(name).fn(ctx_for(80, 24, 5, state, 0.5, stereo=np.zeros((0, 2))))
        assert codes.shape == (24, 80), name
        assert cidx.shape == (24, 80), name


def test_flame_never_divides_by_zero():
    # ``tip`` goes negative on dead cells below quiet bands, driving ``fw``
    # through exactly zero there — a FloatingPointError under strict numpy
    # error handling. The saw spectrum puts zero-fw cells on the 400x100 grid.
    saw = np.linspace(0, 1, N_BANDS, endpoint=False)
    old = np.seterr(all="raise")
    try:
        for w, h in ((400, 100), (200, 100), (80, 24)):
            state: dict = {}
            for f in range(3):
                codes, cidx = M.get("Flame").fn(ctx_for(w, h, f, state, f / 60, bands=saw))
                assert codes.shape == (h, w), (w, h)
                assert cidx.shape == (h, w), (w, h)
    finally:
        np.seterr(**old)


def test_flame_emits_no_warnings():
    # the only mode of 52 that warned under default settings must be silent
    saw = np.linspace(0, 1, N_BANDS, endpoint=False)
    state: dict = {}
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        M.get("Flame").fn(ctx_for(400, 100, 3, state, 0.0, bands=saw))
    rw = [w for w in wlist if issubclass(w.category, RuntimeWarning)]
    assert not rw