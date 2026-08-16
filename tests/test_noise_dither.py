"""The integer dither path has to agree with the float one, exactly.

``noise_below`` exists because the float field it replaces was 2.0 ms of a
tunnel frame at 400x100 and the comparison never needed floats. That is only
worth having if the answer is identical, and the interesting part is the
rounding: the threshold has to be taken to the *ceiling* of the integer
bucket, because truncating flips the comparison for every sample landing
inside the bucket the threshold sits in. These tests hunt exactly that.
"""

from __future__ import annotations

import numpy as np
import pytest

from spektr.render import noise, noise_below, noise_level

SHAPES = [(4, 8), (40, 80), (97, 31)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_matches_the_float_comparison_on_a_varying_threshold(shape, seed):
    rng = np.random.default_rng(shape[0] * 31 + seed)
    level = rng.random(shape, dtype=np.float32)
    assert np.array_equal(
        noise_below(shape, seed, noise_level(level)),
        noise(shape, seed) < level,
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_on_thresholds_that_land_exactly_on_a_bucket(shape):
    """The case truncation gets wrong.

    Every sample of the hash is ``k / 2**24`` for an integer k, so a threshold
    of exactly that value must exclude k and include k-1. Build thresholds
    straight out of the hash's own values to guarantee the boundary is hit.
    """
    field = noise(shape, 3)
    for level in (field, field + np.float32(2.0**-24), field - np.float32(2.0**-24)):
        assert np.array_equal(
            noise_below(shape, 3, noise_level(level)),
            noise(shape, 3) < level,
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_scalar_and_out_of_range_thresholds(shape):
    for level in (0.0, 0.25, 1.0, -0.5, 1.7):
        want = noise(shape, 5) < min(max(level, 0.0), 1.0)
        assert np.array_equal(noise_below(shape, 5, noise_level(level)), want)


def test_nothing_passes_at_zero_and_everything_at_one():
    shape = (32, 64)
    assert not noise_below(shape, 9, noise_level(0.0)).any()
    assert noise_below(shape, 9, noise_level(1.0)).all()


def test_the_dither_still_changes_frame_to_frame():
    """A cached threshold must not accidentally freeze the pattern."""
    shape = (32, 64)
    level = noise_level(np.full(shape, 0.5, dtype=np.float32))
    a = noise_below(shape, 1, level)
    b = noise_below(shape, 2, level)
    assert not np.array_equal(a, b)
    assert 0.3 < a.mean() < 0.7


def test_tunnel_geometry_is_cached_not_rebuilt():
    """The mode keeps its fixed corridor in one scratch entry, and reuses it."""
    import spektr.modes as M
    from spektr.analysis import N_BANDS
    from spektr.modes import Ctx
    from spektr.palette import BUILTIN, Palette

    pal = Palette(BUILTIN["gruvbox"])
    reg = {m.name: m for m in M.MODES}
    state: dict = {}
    bands = np.linspace(0.2, 0.8, N_BANDS).astype(np.float32)

    def ctx(frame):
        return Ctx(
            w=60, h=20, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
            wave=np.zeros(512, dtype=np.float32),
            stereo=np.zeros((512, 2), dtype=np.float32),
            frame=frame, t=frame / 60.0, dt=1 / 60.0, energy=0.5,
            silent=False, palette=pal, state=state, bars=len(bands),
        )

    key = ("tunnel_geo", 60, 20)   # scratch keys carry the size they were built at
    reg["Tunnel"].fn(ctx(0))
    first = state[key]
    reg["Tunnel"].fn(ctx(1))
    assert state[key] is first, "the corridor was rebuilt on a plain frame"
    assert first["depth055"].dtype == np.float32
    assert first["dither"].dtype == np.uint32
