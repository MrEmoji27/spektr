"""Build, drop and breakdown: the mechanics, not the musicology.

What is pinned here is what can honestly be pinned without real tracks: that
the features survive the analyser's auto-sensitivity, that level is read as a
loudness envelope rather than an average, that gated frames are ignored, and
that a synthetic build followed by the bass returning is reported as a build
followed by a drop, in that order.

What is deliberately *not* pinned is where the boundaries fall on real music.
The thresholds in :mod:`spektr.audio.sections` are provisional and were set
against one synthetic track; calling a passage a build is a perceptual
judgement and a synthetic corpus cannot settle it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onset_eval as corpus  # noqa: E402

from spektr.analysis import HOP, Analyser  # noqa: E402
from spektr.audio.sections import (  # noqa: E402
    STATES,
    SectionTracker,
    _envelope,
    features,
)
from spektr.capture import RingBuffer  # noqa: E402


def test_the_shares_do_not_move_when_everything_is_scaled():
    """The analyser's auto-sensitivity multiplies every band together, so
    anything read as a share of the frame has to be blind to it. This is the
    property the whole module rests on: reading absolute band levels instead
    had a drop come back at 1.0x its own average."""
    bands = np.array([0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    _, low, bright = features(bands, 3)
    for gain in (0.001, 0.5, 1000.0):
        _, low_g, bright_g = features(bands * gain, 3)
        assert low_g == pytest.approx(low)
        assert bright_g == pytest.approx(bright)


def test_brightness_says_where_the_energy_is():
    low_heavy = features(np.array([1.0, 0.8, 0.1, 0.05, 0.0, 0.0]), 2)[2]
    high_heavy = features(np.array([0.0, 0.0, 0.05, 0.1, 0.8, 1.0]), 2)[2]
    assert low_heavy < 0.3 < 0.7 < high_heavy


def test_level_comes_from_rms_when_it_is_offered():
    bands = np.array([0.5, 0.5, 0.5, 0.5])
    assert features(bands, 2)[0] == pytest.approx(0.5)
    assert features(bands, 2, rms=0.02)[0] == pytest.approx(0.02)


def test_the_level_envelope_rises_at_once_and_falls_slowly():
    """Between two drum hits the signal really is near silence. An average
    would read that as the track getting quieter; a meter does not."""
    assert _envelope(0.1, 0.9, 0.01, 1.0) == 0.9        # straight up
    fallen = _envelope(0.9, 0.0, 0.01, 1.0)
    assert 0.85 < fallen < 0.9                           # barely down
    assert _envelope(None, 0.4, 0.01, 1.0) == 0.4


def test_a_gated_frame_is_not_evidence_of_anything():
    st = SectionTracker()
    st.feed(np.array([0.8, 0.6, 0.2, 0.1]), 0.1, rms=0.5)
    before = st._low_fast
    for _ in range(50):
        st.feed(np.zeros(4), 0.1, rms=0.0)
    assert st._low_fast == before, "silence dragged the bass share down"


def test_it_starts_and_resets_steady():
    st = SectionTracker()
    assert st.state == "steady" and st.confidence == 0.0
    for _ in range(200):
        st.feed(np.array([0.5, 0.4, 0.3, 0.6, 0.7]), 0.05, rms=0.3)
    st.reset()
    assert st.state == "steady" and st.confidence == 0.0


def test_an_unchanging_track_is_steady():
    st = SectionTracker()
    rng = np.random.default_rng(5)
    for _ in range(600):
        bands = np.array([0.6, 0.5, 0.4, 0.3, 0.25, 0.2]) * (1 + rng.normal(0, 0.02))
        st.feed(bands, 0.05, rms=0.3)
    assert st.state == "steady"


def test_every_state_it_can_report_is_named():
    st = SectionTracker()
    assert st.state in STATES


# ── end to end, on a synthetic build and drop ────────────────────────────────

def _bar(bar_s, rng, bass=True, bright=0.0, level=1.0):
    n = round(bar_s * corpus.SR)
    out = np.zeros(n)
    kick = corpus._struck(55.0, 0.9, 0.4, 0.003, 0.08)
    snare = corpus._noise_burst(0.7, 0.15, 0.04, rng)
    quarter = bar_s / 4
    for beat in range(4):
        i = round(beat * quarter * corpus.SR)
        if bass and beat % 2 == 0:
            out[i:i + len(kick)] += kick
        if beat % 2 == 1:
            out[i:i + len(snare)] += snare
    if bright > 0:
        # Differentiating noise tilts it upward, which is what a build's
        # filter sweep and rising hats do to the spectrum.
        out += np.diff(np.concatenate([[0], rng.normal(size=n) * bright])) * 3
    return out * level


def build_then_drop():
    rng = np.random.default_rng(3)
    bars = [_bar(2.0, rng) for _ in range(6)]
    bars += [_bar(2.0, rng, bass=False, bright=0.05 + 0.05 * k) for k in range(6)]
    bars += [_bar(2.0, rng) for _ in range(6)]
    return corpus._stereo(np.concatenate(bars) * 0.5)


def states_over(signal):
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: corpus.SR, clock=lambda: now[0])
    an._ensure_plan(corpus.SR)
    st = SectionTracker()
    seen, previous = [], None
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / corpus.SR
        an._analyse_once()
        frame = an._frame
        st.feed(frame.bands, HOP / corpus.SR, rms=frame.rms)
        if st.state != previous:
            seen.append((now[0], st.state))
            previous = st.state
    return seen


def test_a_build_then_a_drop_is_read_in_that_order():
    seen = states_over(build_then_drop())
    order = [state for _, state in seen]
    assert order[0] == "steady"
    assert "build" in order, f"no build found: {seen}"
    assert "drop" in order, f"no drop found: {seen}"
    assert order.index("build") < order.index("drop")


def test_the_build_is_not_called_during_the_steady_opening():
    """A bar with a snare in it is brighter and thinner in the bass than the
    bar before it — over one bar, a backbeat has a build's shape. Nothing in
    the first six bars may be called one."""
    for when, state in states_over(build_then_drop()):
        if when < 11.0:
            assert state == "steady", f"called {state} at {when:.2f}s"


def test_the_drop_lands_near_where_the_bass_returns():
    seen = states_over(build_then_drop())
    when = next(t for t, state in seen if state == "drop")
    # The bass returns at 24 s. The fast average needs a moment to see it, and
    # a drop reported early would be worse than one reported a beat late.
    assert 24.0 <= when < 26.0, f"drop reported at {when:.2f}s"
