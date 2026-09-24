"""The beats a mode is handed are the ones a listener would call the beat.

The onset detector is built to find every hit -- the hi-hats, the ghost notes,
the quiet pluck under the chorus -- because tempo, the drum streams and the
bar tracker all need every one of them. A visualiser does not. Given every hit
at close to full strength, a mode spends more than half its beats on the
background, and the kick and snare that carry the song look no different from
a hat twenty-five times quieter.

So each hit also gets an accent: how far the signal's energy jumped, against
the biggest jumps of the last few seconds. Only accented hits reach a mode.
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
from spektr.audio.accent import ACCENT_MIN, AccentMeter  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402

STEP = HOP / corpus.SR


def _level(meter, t0, t1, level):
    t = t0
    while t < t1:
        meter.feed(t, level)
        t += STEP
    return t


def test_a_hit_that_dwarfs_everything_is_a_full_accent():
    m = AccentMeter()
    t = _level(m, 0.0, 0.5, 0.01)
    _level(m, t, t + 0.03, 0.8)
    assert m.judge(t) == pytest.approx(1.0)


def test_a_quiet_hit_after_a_loud_one_is_not_an_accent():
    m = AccentMeter()
    t = _level(m, 0.0, 0.3, 0.01)
    t_loud = t
    t = _level(m, t, t + 0.1, 0.8)
    assert m.judge(t_loud) == pytest.approx(1.0)
    t = _level(m, t, t + 0.2, 0.01)
    t_soft = t
    _level(m, t, t + 0.03, 0.04)
    assert m.judge(t_soft) < 0.1


def test_the_answer_does_not_depend_on_loudness():
    """The same passage at any volume accents the same hits."""
    out = []
    for gain in (0.01, 1.0, 50.0):
        m = AccentMeter()
        t = _level(m, 0.0, 0.3, 0.01 * gain)
        a = t
        t = _level(m, t, t + 0.1, 0.8 * gain)
        t = _level(m, t, t + 0.2, 0.01 * gain)
        b = t
        _level(m, t, t + 0.03, 0.3 * gain)
        out.append((m.judge(a), m.judge(b)))
    for pair in out[1:]:
        assert pair == pytest.approx(out[0])


def test_after_a_loud_passage_the_quieter_one_gets_its_beats_back():
    """A breakdown after a drop still has a beat. It is judged against itself
    once the drop has had a few seconds to be forgotten."""
    m = AccentMeter()
    t = _level(m, 0.0, 0.3, 0.01)
    m.judge(t)
    t = _level(m, t, t + 0.1, 0.9)
    t = _level(m, t, t + 0.2, 0.01)
    for _ in range(12):
        hit = t
        t = _level(m, t, t + 0.05, 0.15)
        a = m.judge(hit)
        t = _level(m, t, t + 0.5, 0.01)
    assert a >= ACCENT_MIN


def test_a_hit_with_no_energy_history_is_not_invented():
    assert AccentMeter().judge(1.0) == 0.0


def test_reset_forgets_the_last_track():
    m = AccentMeter()
    t = _level(m, 0.0, 0.3, 0.01)
    t = _level(m, t, t + 0.1, 0.9)
    m.reset()
    t = _level(m, t, t + 0.3, 0.01)
    hit = t
    _level(m, t, t + 0.05, 0.1)
    assert m.judge(hit) == pytest.approx(1.0)


# ── through the analyser ─────────────────────────────────────────────────────

def foreground_and_background(bg_db: float, bpm=120.0, bars=8, seed=4):
    """A kick and snare backbeat, with hats and soft plucks underneath it.

    Returns the audio and a list of ``(time, kind)`` where kind is ``"main"``
    for the kick and snare and ``"background"`` for everything else.
    """
    rng = np.random.default_rng(seed)
    sr = corpus.SR
    beat = 60.0 / bpm
    out = np.zeros(round(beat * 4 * bars * sr) + sr)
    events = []
    g = 10 ** (bg_db / 20)
    kick = corpus._struck(55.0, 0.9, 0.4, 0.003, 0.08)
    for b in range(bars * 4):
        t = b * beat
        i = round(t * sr)
        if b % 2 == 0:
            out[i:i + len(kick)] += kick
        else:
            s = corpus._noise_burst(0.7, 0.15, 0.04, rng)
            s += corpus._struck(190.0, 0.4, 0.15, 0.002, 0.05)
            out[i:i + len(s)] += s
        events.append((t, "main"))
        th = t + beat / 2
        h = corpus._noise_burst(0.7 * g, 0.05, 0.012, rng)
        j = round(th * sr)
        out[j:j + len(h)] += h
        events.append((th, "background"))
        if b % 3 == 1:
            tp = t + beat * 0.75
            p = corpus._struck(660.0 * (1 + (b % 4) / 8), 0.7 * g, 0.3, 0.004, 0.12)
            j = round(tp * sr)
            out[j:j + len(p)] += p
            events.append((tp, "background"))
    return corpus._stereo(corpus._norm(out)), sr, events


def beats_by_kind(signal, sr, events):
    """Which events the raw counter and the accented counter each fired on."""
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: sr, clock=lambda: now[0])
    an._ensure_plan(sr)
    times = np.array([t for t, _ in events])
    kinds = [k for _, k in events]
    raw = {"main": 0, "background": 0}
    accented = {"main": 0, "background": 0}
    seq = acc = 0
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / sr
        an._analyse_once()
        f = an._frame
        if f.onset_seq == seq and f.accent_seq == acc:
            continue
        i = int(np.argmin(np.abs(times - an._onset.last_t)))
        kind = kinds[i] if abs(times[i] - an._onset.last_t) < 0.06 else None
        if kind is not None:
            raw[kind] += f.onset_seq - seq
            accented[kind] += f.accent_seq - acc
        seq, acc = f.onset_seq, f.accent_seq
    return raw, accented


@pytest.mark.parametrize("bg_db", [-28.0, -22.0])
def test_the_background_stops_being_the_beat(bg_db):
    signal, sr, events = foreground_and_background(bg_db)
    main = sum(1 for _, k in events if k == "main")
    raw, accented = beats_by_kind(signal, sr, events)
    # The detector is untouched: it still hears the background.
    assert raw["background"] > main * 0.8
    # What reaches a mode is the kick and the snare, all of them...
    assert accented["main"] >= main - 2
    # ...and next to none of the rest.
    assert accented["background"] <= 2, accented


def test_a_loud_background_is_still_heard():
    """Hats pushed up to the level of the drums are part of the beat, and a
    rule that dropped them would be deciding on a genre rather than a level."""
    signal, sr, events = foreground_and_background(+2.0)
    raw, accented = beats_by_kind(signal, sr, events)
    assert accented["background"] >= raw["background"] * 0.8


def test_an_accent_is_published_on_the_same_hop_as_its_onset():
    """Judging a hit must not cost it latency: a beat that lands on screen a
    few frames after it is heard looks like a beat being missed."""
    signal, sr, _ = foreground_and_background(-28.0)
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: sr, clock=lambda: now[0])
    an._ensure_plan(sr)
    seq = acc = 0
    late = []
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / sr
        an._analyse_once()
        f = an._frame
        if f.accent_seq != acc and f.onset_seq == seq:
            late.append(now[0])
        seq, acc = f.onset_seq, f.accent_seq
    assert not late


def test_the_accent_counter_survives_silence():
    """Like the raw counter, it must never go backwards across a gap, or the
    widget would read the recovery as a burst of beats."""
    signal, sr, _ = foreground_and_background(-28.0, bars=2)
    gap = np.zeros((sr, 2), dtype=signal.dtype)
    signal = np.concatenate([signal, gap, signal])
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: sr, clock=lambda: now[0])
    an._ensure_plan(sr)
    last = 0
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / sr
        an._analyse_once()
        assert an._frame.accent_seq >= last
        last = an._frame.accent_seq
    assert last > 0
