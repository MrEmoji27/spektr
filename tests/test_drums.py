"""Kick, snare and hat likelihoods, from where the spectrum moved."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.audio import drums  # noqa: E402

HZ = drums.band_freqs(32)


def burst(lo: float, hi: float, broad: bool = True, peak: float = 1.0) -> np.ndarray:
    """A rise across a range of bands: broad like noise, or narrow like a note."""
    f = np.zeros(HZ.size, dtype=np.float32)
    inside = np.flatnonzero((HZ >= lo) & (HZ < hi))
    f[inside if broad else inside[:2]] = peak
    return f


def test_a_narrow_low_burst_reads_as_a_kick():
    r = drums.classify(burst(20, 160, broad=False), HZ)
    assert r["kick"] > 0.6
    assert r["snare"] == 0.0 and r["hat"] == 0.0


def test_a_broad_high_burst_reads_as_a_hat():
    r = drums.classify(burst(3500, 16000), HZ)
    assert r["hat"] > 0.6
    assert r["kick"] == 0.0


def test_a_body_with_noise_above_it_reads_as_a_snare():
    r = drums.classify(burst(160, 1200) + 0.5 * burst(3500, 16000), HZ)
    assert r["snare"] > 0.5
    assert r["snare"] > r["hat"], "a snare is not mostly its own noise"


def test_two_drums_at_once_both_show():
    """They do not compete: a kick and a hat together is a normal bar."""
    r = drums.classify(burst(20, 160, broad=False) + burst(3500, 16000), HZ)
    assert r["kick"] > 0.2 and r["hat"] > 0.2


def test_music_without_percussion_claims_nothing():
    assert drums.classify(burst(300, 1200, broad=False), HZ) == {
        "kick": 0.0, "snare": 0.0, "hat": 0.0}


def test_silence_and_nonsense_claim_nothing():
    zero = {"kick": 0.0, "snare": 0.0, "hat": 0.0}
    assert drums.classify(np.zeros(32), HZ) == zero
    assert drums.classify(np.zeros(0), np.zeros(0)) == zero
    assert drums.classify(np.ones(8), HZ) == zero        # mismatched lengths


def test_loudness_does_not_change_the_answer():
    quiet = drums.classify(burst(20, 160, broad=False, peak=0.01), HZ)
    loud = drums.classify(burst(20, 160, broad=False, peak=100.0), HZ)
    assert quiet == loud


def test_a_bass_note_is_known_to_look_like_a_kick():
    """A limitation, pinned rather than hidden.

    A bass note and a kick occupy the same bands with the same shape, and a
    spectrum cannot separate them. This is why these are likelihoods: a mode
    should lean on them, not branch on them.
    """
    note = burst(20, 160, broad=False) + 0.2 * burst(160, 400, broad=False)
    assert drums.classify(note, HZ)["kick"] > 0.5


@pytest.mark.parametrize("n", [12, 16, 32, 64])
def test_band_frequencies_cover_the_range_in_order(n):
    hz = drums.band_freqs(n)
    assert hz.size == n
    assert np.all(np.diff(hz) > 0)
    assert 50 <= hz[0] < hz[-1] <= 10000
