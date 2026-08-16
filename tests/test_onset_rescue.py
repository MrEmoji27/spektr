"""Lock in the rescue-band onset change.

The breakbeat scenario has a hit that every curve used to miss: a hi-hat
landing ~86 ms after a snare, whose flat flux is masked by the snare's tail
and whose only surviving signature is a whitened spike in one quiet band.
The rescue curve (RESCUE_BAND) exists to catch it. These tests assert that:

* breakbeat recall improved to at least the current level,
* precision stays at 1.000 on every scenario — the false-onset guardrail,
* the empty scenarios stay empty,
* nothing else regressed below the committed baseline.

They run the real analyser through the corpus scorer, the same way
``tests/onset_score.py`` does, so what is asserted here is what a mode sees.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onset_eval import SCENARIOS, evaluate  # noqa: E402

from spektr.analysis import HOP, Analyser  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402


def detect(signal: np.ndarray, samplerate: int) -> list[float]:
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: samplerate, clock=lambda: now[0])
    an._ensure_plan(samplerate)

    times: list[float] = []
    last_seq = 0
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / samplerate
        an._analyse_once()
        seq = an._frame.onset_seq
        times.extend([now[0]] * max(0, seq - last_seq))
        last_seq = seq
    return times


#: Minimum true positives per scenario, from the table that shipped with the
#: detector. The rescue curve must not push any of these down. Counts rather
#: than the printed recall so 15/16 (0.9375) is not compared against a
#: rounded 0.938.
FLOORS = {
    "click": 14,            # 16 hits, recall 0.875
    "four_on_floor": 15,    # 16 hits, recall 0.938
    "kick_snare": 15,       # 16 hits, recall 0.938
    "breakbeat": 45,        # 48 hits; was 38 before the rescue curve
    "pad_under_kick": 15,   # 16 hits, recall 0.938
    "swing": 15,            # 16 hits, recall 0.938
    "tempo_ramp": 38,       # 39 hits, recall 0.974
    "quiet": 15,            # 16 hits, recall 0.938
    "silence": 0,
    "noise": 0,
    "note_stream": 0,
}

EMPTY = ("silence", "noise", "note_stream")


@pytest.mark.parametrize("name", sorted(FLOORS))
def test_scenario_never_regresses(name):
    samples, sr, truth = SCENARIOS[name]()
    rows = evaluate(detect, scenarios=[name])[name]
    assert rows["precision"] == 1.0, (
        f"{name}: a false onset slipped in (precision {rows['precision']})"
    )
    assert rows["tp"] >= FLOORS[name], (
        f"{name}: {rows['tp']} true positives, floor is {FLOORS[name]} "
        f"({len(truth)} hits)"
    )


def test_empty_scenarios_stay_silent():
    for name in EMPTY:
        samples, sr, _truth = SCENARIOS[name]()
        rows = evaluate(detect, scenarios=[name])[name]
        assert rows["precision"] == 1.0, (
            f"{name}: detected something in an onset-free track"
        )


def test_breakbeat_hats_after_the_snare_are_found():
    """The specific miss class the rescue curve exists for: the hats one
    16th after the snares (steps 5 and 13 of each bar)."""
    samples, sr, truth = SCENARIOS["breakbeat"]()
    det = detect(samples, sr)
    det_a = np.sort(np.asarray(det, dtype=np.float64))
    truth_a = np.sort(np.asarray(truth, dtype=np.float64))
    hits = 0
    for t in truth_a:
        if np.any(np.abs(det_a - t) <= 0.05):
            hits += 1
    # all 48 hits: the two warm-up misses at the very start of the track are
    # structurally unreachable (the analyser needs its 8192-sample window
    # before the first hop), so this is 46 of 48.
    assert hits >= 46, (
        f"breakbeat found {hits}/48; the rescue curve was expected to "
        "recover the hats after the snares"
    )
