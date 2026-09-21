"""The tempo the analyser reports, and the octave it reports it in.

``tempo_bpm`` is not just a readout. ``beat_phase`` is derived from it, and
``ctx.pulse`` -- the field every mode is pointed at for beat-locked motion --
is derived from that. A track reported at half its tempo makes every mode in
the app swell on every other beat, which looks like the visualiser ignoring
the music rather than like a number being wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onset_eval as corpus  # noqa: E402

from spektr.analysis import (  # noqa: E402
    HOP,
    TEMPO_MAX_S,
    TEMPO_MIN_S,
    Analyser,
)
from spektr.capture import RingBuffer  # noqa: E402


def backbeat(bpm: float, bars: int = 14, seed: int = 9):
    """Kick on one, snare on two and four, at ``bpm``."""
    rate = corpus.SR
    rng = np.random.default_rng(seed)
    bar_s = 4 * 60.0 / bpm
    out = np.zeros(round(bar_s * bars * rate))
    kick = corpus._struck(55.0, 0.9, 0.4, 0.003, 0.08)
    snare = corpus._noise_burst(0.7, 0.15, 0.04, rng)
    quarter = bar_s / 4
    for b in range(bars):
        i = round(b * bar_s * rate)
        out[i:i + len(kick)] += kick
        for beat in (2, 4):
            i = round((b * bar_s + (beat - 1) * quarter) * rate)
            out[i:i + len(snare)] += snare
    return corpus._stereo(corpus._norm(out)), rate


def reported_tempo(signal, rate) -> float:
    """The median tempo across the run, or 0.0 if it never settled."""
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: rate, clock=lambda: now[0])
    an._ensure_plan(rate)
    seen = []
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / rate
        an._analyse_once()
        if an._frame.tempo_bpm > 0:
            seen.append(an._frame.tempo_bpm)
    return float(np.median(seen)) if seen else 0.0


@pytest.mark.parametrize("bpm", [100.0, 128.0, 150.0, 174.0])
def test_the_tempo_is_reported_in_the_octave_it_was_played_in(bpm):
    got = reported_tempo(*backbeat(bpm))
    assert got == pytest.approx(bpm, rel=0.06), (
        f"{bpm} BPM came back as {got:.1f}"
    )


def test_a_hundred_and_fifty_is_not_halved():
    """The exact regression this range exists for.

    The fold's floor used to be 0.4 s, which is 150 BPM exactly. A 150 BPM
    track measures 0.39999..., fell through the floor, doubled, and was
    reported as 75 -- and 150 is one of the most common tempos there is.
    """
    got = reported_tempo(*backbeat(150.0))
    assert got > 120.0, f"reported {got:.1f}, which is the half-speed answer"


def test_the_range_covers_the_tempos_music_is_actually_at():
    assert 60.0 / TEMPO_MIN_S >= 180.0, "too slow a ceiling for fast music"
    assert 60.0 / TEMPO_MAX_S <= 60.0, "too fast a floor for slow music"
    # 150 must sit strictly inside, not on the boundary where it was before.
    assert TEMPO_MIN_S < 0.4 < TEMPO_MAX_S


def test_silence_and_noise_report_no_tempo():
    for scenario in ("silence", "noise"):
        signal, rate, _truth = corpus.SCENARIOS[scenario]()
        assert reported_tempo(signal, rate) == 0.0, scenario


def test_a_tempo_that_is_never_agreed_on_is_not_invented():
    """A detector that fires irregularly must produce 0.0, not a number.
    Saying nothing is much better than handing a mode a confident wrong one."""
    signal, rate, _truth = corpus.SCENARIOS["note_stream"]()
    assert reported_tempo(signal, rate) == 0.0
