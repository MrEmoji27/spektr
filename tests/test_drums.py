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


# ── against real material, not rectangles ────────────────────────────────────
#
# Everything above feeds ``classify`` idealised bursts: a range of bands either
# fully lit or not. Real audio is nothing like that, and the first version of
# this module passed every test above while calling all fifteen kicks of a
# four-on-the-floor a snare. The analyser hands it thirty-two log-spaced band
# sums, the mid range covers more of them than the low range, and a kick's
# click spreads upward -- so share alone said "snare" on every hit.
#
# These drive the real analyser over the real corpus, which is the only thing
# that would have caught it.

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onset_eval import SCENARIOS  # noqa: E402

from spektr.analysis import HOP, Analyser  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402


def named_hits(scenario: str) -> list[str]:
    """What each onset in ``scenario`` was called, in order.

    ``"-"`` where nothing cleared the floor, which is a legitimate answer.
    """
    signal, rate, _truth = SCENARIOS[scenario]()
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: rate, clock=lambda: now[0])
    an._ensure_plan(rate)

    out: list[str] = []
    last = 0
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / rate
        an._analyse_once()
        frame = an._frame
        if frame.onset_seq != last:
            last = frame.onset_seq
            likely = dict(frame.drums)
            out.append(max(likely, key=likely.get) if max(likely.values()) > 0 else "-")
    return out


@pytest.mark.parametrize("scenario", ["four_on_floor", "pad_under_kick"])
def test_a_track_of_nothing_but_kicks_reads_as_kicks(scenario):
    named = named_hits(scenario)
    assert named, "the corpus produced no onsets to name"
    kicks = named.count("kick")
    assert kicks >= len(named) - 1, f"{scenario} named {named}"


def test_kicks_and_snares_alternate_the_way_they_were_played():
    """The corpus puts a kick on 1 and 3 and a snare on 2 and 4."""
    named = named_hits("kick_snare")
    assert len(named) >= 12, f"too few onsets to judge: {named}"
    assert set(named) == {"kick", "snare"}, f"unexpected names: {set(named)}"
    swapped = [a != b for a, b in zip(named, named[1:])]
    assert all(swapped), f"the two drums did not alternate: {named}"


def test_nothing_is_claimed_where_nothing_was_played():
    for scenario in ("silence", "note_stream"):
        assert named_hits(scenario) == [], f"{scenario} produced onsets"
