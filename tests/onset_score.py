"""Score the real onset detector against the corpus in ``onset_eval``.

``onset_eval`` holds the corpus and the scoring, deliberately knowing nothing
about spektr's detector -- it ships a reference detector of its own so the
corpus can be checked without one. This is the missing half: it drives the
actual :class:`~spektr.analysis.Analyser` over each scenario and hands the
onsets it publishes to that scorer.

It is kept separate from ``onset_eval`` on purpose. Whoever tunes the
detector should be able to be checked by a driver they did not write, and a
corpus that imports the thing it grades tends to drift towards grading it
kindly.

Run it from the repo root::

    python tests/onset_score.py

Onsets are read the way the UI reads them -- by differencing ``onset_seq`` on
the published :class:`~spektr.analysis.Frame` -- rather than by reaching into
the detector, so what is measured here is what a mode actually sees.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from onset_eval import SCENARIOS, evaluate  # noqa: E402

from spektr.analysis import HOP, Analyser  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402


def detect(signal: np.ndarray, samplerate: int) -> list[float]:
    """Return the onset times the analyser publishes for ``signal``.

    The analyser's clock is injected rather than left on the wall clock. Its
    gate, its threshold history and its refractory are all in seconds, and
    twenty seconds of audio analysed in a fraction of that would put every
    one of them on the wrong timescale -- so the clock advances with the
    audio, one hop at a time.
    """
    ring = RingBuffer(1 << 16)
    now = [0.0]
    analyser = Analyser(ring, lambda: samplerate, clock=lambda: now[0])
    analyser._ensure_plan(samplerate)

    times: list[float] = []
    last_seq = 0
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        # timestamp the hop at its end: the newest sample the analyser can see
        now[0] = (start + HOP) / samplerate
        analyser._analyse_once()
        seq = analyser._frame.onset_seq
        # the counter can move by more than one if a hop resolved a burst
        times.extend([now[0]] * max(0, seq - last_seq))
        last_seq = seq
    return times


def main() -> int:
    rows = evaluate(detect)

    print(f"{'scenario':<16}{'P':>8}{'R':>8}{'F':>8}")
    print("-" * 40)
    for name in list(SCENARIOS) + ["total"]:
        row = rows.get(name)
        if row is None:
            continue
        if name == "total":
            print("-" * 40)
        print(f"{name:<16}{row['precision']:>8.3f}{row['recall']:>8.3f}"
              f"{row['f']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
