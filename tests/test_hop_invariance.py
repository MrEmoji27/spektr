"""The analysis hop rate must not depend on the capture block size.

This is the invariant behind commit 3af24f8, and it is the one the project had
no way to see for a long time: `tests/onset_score.py` drives the analyser one
HOP at a time, while the real Windows loopback backend hands the ring 512
frames at once. When the analyser ran once per push instead of once per hop, it
analysed at half its designed rate and the corpus went from F 0.948 to F 0.135
— with every gate still green, because no gate pushed anything but a hop.

So this file pushes blocks the way a backend does, and asserts the analyser
produces exactly the same onsets. Not similar: the same, to the sample.

A note on how to get this wrong, because it cost a probe here: an onset has to
be timestamped at *its own hop*, which is what the analyser passes to
`_analyse_once`, not at the end of the block that happened to contain it.
Stamping at the block end makes a correct detector look progressively worse as
the block grows — pure measurement error, and convincing enough to be believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onset_eval import SCENARIOS  # noqa: E402

from spektr.analysis import HOP, Analyser  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402


def detect(signal: np.ndarray, samplerate: int, block: int) -> list[float]:
    """Onset times, pushing ``block`` frames at a time through the real drain."""
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: samplerate, clock=lambda: now[0])
    an._ensure_plan(samplerate)

    times: list[float] = []
    seen = [0]
    inner = an._analyse_once

    def record(end=None, now_arg=None):
        inner(end=end, now=now_arg)
        seq = an._frame.onset_seq
        if seq > seen[0]:
            # the analyser's own per-hop clock, not the caller's block clock
            stamp = now_arg if now_arg is not None else now[0]
            times.extend([stamp] * (seq - seen[0]))
            seen[0] = seq

    an._analyse_once = lambda end=None, now=None: record(end=end, now_arg=now)

    for start in range(0, signal.shape[0] - block + 1, block):
        ring.push(signal[start : start + block])
        now[0] = (start + block) / samplerate
        an._analyse_hops()
    return times


# One percussive scenario and one with a sustained bed under it: between them
# they exercise the threshold history, the refractory and the quiet gate. The
# whole corpus at four block sizes is a minute of CPU and buys nothing extra.
SCENARIO_NAMES = ("kick_snare", "breakbeat")

#: 512 is what the WASAPI loopback backend actually pushes. 1024 is the next
#: doubling. 480 is deliberately not a multiple of HOP: the leftover frames
#: have to be carried to the next push rather than dropped or re-read.
BLOCKS = (512, 1024, 480)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
@pytest.mark.parametrize("block", BLOCKS)
def test_block_size_does_not_change_the_onsets(name, block):
    signal, samplerate, _truth = SCENARIOS[name]()
    baseline = detect(signal, samplerate, HOP)
    got = detect(signal, samplerate, block)

    assert len(got) == len(baseline), (
        f"{name}: pushing {block} frames found {len(got)} onsets, "
        f"{HOP}-frame hops found {len(baseline)}"
    )
    assert got == pytest.approx(baseline, abs=1e-9), (
        f"{name}: pushing {block} frames moved the onset times"
    )


def test_a_hop_at_a_time_still_finds_the_beat():
    """Guard the guard: if the baseline itself were empty this would pass."""
    signal, samplerate, truth = SCENARIOS["kick_snare"]()
    found = detect(signal, samplerate, HOP)
    assert len(found) >= len(truth) * 0.8, (
        f"the baseline detector found {len(found)} onsets against {len(truth)} real ones"
    )


def test_a_backlog_is_skipped_rather_than_ground_through():
    """A stalled reader must not spend the next second catching up.

    ``MAX_HOPS_PER_WAKE`` exists because a suspend or a device hiccup leaves
    more audio in the ring than the analyser can ever work through in real
    time. The recovery is to jump to the present, not to analyse a backlog
    nobody is waiting for any more.
    """
    from spektr.analysis import MAX_HOPS_PER_WAKE

    samplerate = 48000
    ring = RingBuffer(1 << 18)
    an = Analyser(ring, lambda: samplerate, clock=lambda: 0.0)
    an._ensure_plan(samplerate)

    calls = [0]
    inner = an._analyse_once
    an._analyse_once = lambda end=None, now=None: (
        calls.__setitem__(0, calls[0] + 1),
        inner(end=end, now=now),
    )[1]

    backlog = HOP * (MAX_HOPS_PER_WAKE + 40)
    ring.push(np.zeros(backlog, dtype=np.float32))
    an._analyse_hops()

    assert calls[0] <= MAX_HOPS_PER_WAKE
    assert ring.written - an._at < HOP, "the analyser did not catch up to the present"
