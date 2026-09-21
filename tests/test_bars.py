"""Which beat of the bar, from the drum pattern."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.audio.bars import BEATS, MIN_ONSETS, BarTracker  # noqa: E402

PERIOD = 0.5            # 120 BPM
KICK = {"kick": 1.0, "snare": 0.0, "hat": 0.0}
SNARE = {"kick": 0.0, "snare": 1.0, "hat": 0.0}
NOTHING = {"kick": 0.0, "snare": 0.0, "hat": 0.0}

#: Kick on one, snare on two and four: the ordinary backbeat, and determinate
#: because the kick is not repeated on three.
BACKBEAT = [KICK, SNARE, NOTHING, SNARE]


def play(pattern, bars=6, start=0.0, period=PERIOD):
    """Run ``pattern`` — one entry per beat — for ``bars`` bars."""
    bt = BarTracker()
    t = start
    for _ in range(bars):
        for hit in pattern:
            bt.feed(t, period, hit)
            t += period
    return bt


def test_a_backbeat_finds_the_downbeat():
    bt = play(BACKBEAT)
    assert bt.confidence > 0.0
    # The run ends on beat four, which is index three.
    assert bt.beat == BEATS - 1


def test_the_downbeat_is_found_wherever_the_pattern_starts():
    """The tracker must not assume the first onset it hears is beat one."""
    started_on_two = BACKBEAT[1:] + BACKBEAT[:1]
    bt = play(started_on_two)
    assert bt.confidence > 0.0
    # Starting on two means the run ends on beat one of the following bar.
    assert bt.beat == 0


def test_a_symmetric_pattern_is_ambiguous_and_says_so():
    """Kick on one *and* three with a snare either side is genuinely
    ambiguous: beat one and beat three are indistinguishable from the drums
    alone, and a listener uses harmony to tell them apart. Claiming one would
    be inventing information, so the honest answer is none."""
    bt = play([KICK, SNARE, KICK, SNARE])
    assert bt.beat is None
    assert bt.confidence == 0.0


def test_four_on_the_floor_admits_it_does_not_know():
    """Every beat identical: there is no downbeat in the drums at all."""
    bt = play([KICK, KICK, KICK, KICK])
    assert bt.beat is None
    assert bt.confidence == 0.0
    assert bt.phase(10.0) == 0.0


def test_nothing_is_claimed_before_there_is_evidence():
    bt = BarTracker()
    t = 0.0
    for i in range(MIN_ONSETS - 2):
        bt.feed(t, PERIOD, BACKBEAT[i % BEATS])
        t += PERIOD
    assert bt.beat is None and bt.confidence == 0.0


def test_an_unknown_tempo_is_not_a_grid():
    bt = play(BACKBEAT)
    assert bt.confidence > 0.0
    bt.feed(100.0, 0.0, KICK)
    assert bt.beat is None and bt.confidence == 0.0


def test_silence_makes_it_forget():
    bt = play(BACKBEAT)
    assert bt.confidence > 0.0
    bt.reset()
    assert bt.beat is None and bt.confidence == 0.0


def test_drums_that_say_nothing_never_produce_a_downbeat():
    bt = play([NOTHING] * BEATS)
    assert bt.beat is None and bt.confidence == 0.0


def test_the_phase_runs_once_round_the_bar():
    bt = play(BACKBEAT, bars=8)
    assert bt.confidence > 0.0
    bar = PERIOD * BEATS
    base = bt.phase(20.0)
    assert bt.phase(20.0 + bar) == pytest.approx(base, abs=1e-6)
    assert bt.phase(20.0 + bar / 4) == pytest.approx((base + 0.25) % 1.0, abs=1e-6)


def test_the_downbeat_lands_where_the_kick_does():
    """Phase 0 has to coincide with beat one, not merely be self-consistent."""
    bt = play(BACKBEAT, bars=8)
    assert bt.confidence > 0.0
    for n in range(3):
        assert bt.phase(n * PERIOD * BEATS) == pytest.approx(0.0, abs=1e-6)


def test_a_tempo_change_throws_the_old_evidence_away():
    bt = play(BACKBEAT)
    assert bt.confidence > 0.0
    bt.feed(50.0, PERIOD * 1.5, KICK)     # clearly a different tempo
    assert bt.beat is None, "evidence counted on the old grid still voted"


def test_hits_off_the_grid_are_not_counted():
    """A fill or a flam sits between beats. Filing it against the nearest one
    would blur every residue towards equal and cost the downbeat."""
    bt = play(BACKBEAT, bars=8)
    before = (bt.beat, round(bt.confidence, 6))
    bt.feed(bt._anchor + PERIOD * 4.5, PERIOD, KICK)   # squarely between beats
    assert (bt.beat, round(bt.confidence, 6)) == before


# ── through the real analyser ────────────────────────────────────────────────
#
# Everything above hands the tracker perfect drum labels on a perfect grid.
# This synthesises a backbeat, runs it through the actual analyser, and feeds
# the tracker whatever that produces -- detection latency, missed first hit,
# imperfect likelihoods and all. Without it, the tracker could be measuring
# nothing but the test's own assumptions.
#
# The backbeat lives here rather than in ``onset_eval`` on purpose: adding a
# scenario there changes the pooled totals the onset scorer's floors are set
# against, and this has nothing to say about onset accuracy.

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import onset_eval as corpus  # noqa: E402

from spektr.analysis import HOP, Analyser  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402

E2E_BPM = 120.0
E2E_BARS = 8


def synth_backbeat(bpm=E2E_BPM, bars=E2E_BARS):
    """Kick on one, snare on two and four. Returns the audio and the bar."""
    rate = corpus.SR
    bar_s = BEATS * 60.0 / bpm
    out = np.zeros(round(bar_s * bars * rate))
    rng = np.random.default_rng(11)
    kick = corpus._struck(55.0, 0.9, 0.4, 0.003, 0.08)
    snare = corpus._noise_burst(0.7, 0.15, 0.04, rng)
    quarter = bar_s / BEATS
    for b in range(bars):
        i = round(b * bar_s * rate)
        out[i:i + len(kick)] += kick
        for beat in (2, 4):
            i = round((b * bar_s + (beat - 1) * quarter) * rate)
            out[i:i + len(snare)] += snare
    return corpus._stereo(corpus._norm(out)), rate, bar_s


def track_real_audio():
    signal, rate, bar_s = synth_backbeat()
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: rate, clock=lambda: now[0])
    an._ensure_plan(rate)
    bt = BarTracker()
    last = 0
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / rate
        an._analyse_once()
        frame = an._frame
        if frame.onset_seq != last:
            last = frame.onset_seq
            period = 60.0 / frame.tempo_bpm if frame.tempo_bpm > 0 else 0.0
            bt.feed(now[0], period, dict(frame.drums))
    return bt, bar_s


#: How far the reported downbeat may sit from the real one, as a share of the
#: bar. Onsets are detected a hop or two after the hit that caused them, so
#: the grid the tracker builds is systematically a little late; at 120 BPM
#: this allowance is about a sixteenth note.
BAR_SLACK = 0.06


def test_it_finds_the_downbeat_in_actual_audio():
    bt, bar_s = track_real_audio()
    assert bt.confidence > 0.0, "no downbeat found in a plain backbeat"

    for bar in range(4, E2E_BARS):
        phase = bt.phase(bar * bar_s)
        # 0.98 and 0.0 are both "on the downbeat"; the phase wraps.
        assert min(phase, 1.0 - phase) < BAR_SLACK, (
            f"bar {bar} downbeat reported at phase {phase:.3f}"
        )


@pytest.mark.parametrize("beat, expected", [(1, 0.25), (2, 0.5), (3, 0.75)])
def test_the_other_beats_land_where_they_should(beat, expected):
    bt, bar_s = track_real_audio()
    assert bt.confidence > 0.0
    phase = bt.phase(5 * bar_s + beat * bar_s / BEATS)
    assert abs(phase - expected) < BAR_SLACK, (
        f"beat {beat + 1} reported at phase {phase:.3f}, expected {expected}"
    )


# ── the gate shuts between hits, and that is not silence ─────────────────────

def frames_over(signal, rate):
    """Every frame the analyser publishes, stamped with its time."""
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: rate, clock=lambda: now[0])
    an._ensure_plan(rate)
    out = []
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / rate
        an._analyse_once()
        out.append((now[0], an._frame))
    return out


def at(series, t):
    return [f for stamp, f in series if stamp <= t][-1]


def test_the_bar_survives_the_gaps_between_hits():
    """Sparse drums leave half a second of near-nothing between hits, and the
    noise gate shuts in it. Forgetting there would clear the grid several
    times a bar and no downbeat would ever be found."""
    signal, rate, bar_s = synth_backbeat()
    series = frames_over(signal, rate)

    # Sample right through one bar, including whatever the gate did.
    phases = [at(series, 6 * bar_s + k * bar_s / BEATS) for k in range(BEATS)]
    assert all(f.bar_confidence > 0.0 for f in phases), (
        "the downbeat was lost inside a bar: "
        f"{[round(f.bar_confidence, 2) for f in phases]}"
    )
    assert any(f.silent for f in phases), (
        "this test is pointless unless the gate actually shuts mid-bar"
    )


def test_a_track_that_stops_is_eventually_forgotten():
    from spektr.analysis import FORGET_AFTER_S

    signal, rate, bar_s = synth_backbeat()
    # Three seconds of nothing after the last hit.
    tail = np.zeros((int(3.0 * rate), 2), dtype=signal.dtype)
    series = frames_over(np.concatenate([signal, tail]), rate)
    end = E2E_BARS * bar_s

    held = at(series, end + FORGET_AFTER_S * 0.5)
    assert held.bar_confidence > 0.0, "forgotten while the silence was brief"

    gone = at(series, end + FORGET_AFTER_S + 1.0)
    assert gone.bar_confidence == 0.0
    assert gone.beat_in_bar is None
    assert gone.bar_phase == 0.0
    assert not any(gone.drums.values()), "the last hit outlived the music"
    assert not gone.chroma.any(), "the last chord outlived the music"
