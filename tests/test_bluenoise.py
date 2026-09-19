"""The dissolve mask is a real blue-noise rank mask, and stays the same."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import bluenoise  # noqa: E402

N = 32  # small enough to build quickly, large enough to measure


@pytest.fixture(scope="module")
def m():
    return bluenoise.mask(N)


def test_every_rank_appears_exactly_once(m):
    assert sorted(np.unique(m)) == pytest.approx(sorted(np.arange(N * N) / (N * N)))


def test_ranks_run_from_zero_to_just_under_one(m):
    assert m.min() == 0.0
    assert m.max() < 1.0


def test_the_same_size_gives_the_same_mask():
    assert np.array_equal(bluenoise.mask(N), bluenoise.mask(N))


def test_it_refuses_a_size_that_is_not_a_power_of_two():
    with pytest.raises(ValueError):
        bluenoise.mask(24)


def _longest_run(revealed: np.ndarray) -> int:
    longest = 0
    for row in revealed:
        run = 0
        for cell in row:
            run = run + 1 if cell else 0
            longest = max(longest, run)
    return longest


def test_half_way_through_the_dissolve_is_spread_not_clumped(m):
    """The whole point of blue noise, measured against white noise.

    At half progress, the longest run of revealed dots along a row says how
    clumped the reveal looks. Random ranks on this size reach runs of about
    13; an even spread stays far shorter. Compared rather than fixed, because
    the number a run reaches depends on the mask size.
    """
    rng = np.random.default_rng(1)
    white = [
        _longest_run(rng.permutation(N * N).reshape(N, N) / (N * N) < 0.5)
        for _ in range(20)
    ]
    blue = _longest_run(m < 0.5)
    # Measured at 32x32: 5 against white noise's 9.1, a 45% shorter run. The
    # bar is set below that, so a real regression in the mask fails while the
    # exact figure is free to drift a little.
    assert blue <= 0.7 * np.mean(white), (
        f"reveal clumps: longest run {blue}, white noise averages {np.mean(white):.1f}"
    )


def test_neighbouring_dots_are_revealed_at_different_times(m):
    """No two side-by-side cells share nearly the same rank."""
    gaps = np.abs(np.diff(m, axis=1))
    assert gaps.mean() > 0.2, f"neighbouring ranks too close: mean gap {gaps.mean():.3f}"


def test_it_tiles_without_a_seam(m):
    big = bluenoise.tile(m, N * 2, N * 2)
    assert big.shape == (N * 2, N * 2)
    assert np.array_equal(big[:N, :N], m)
    assert np.array_equal(big[N:, N:], m)


def test_the_cached_mask_cannot_be_edited_by_a_caller(m):
    with pytest.raises(ValueError):
        m[0, 0] = 0.5
