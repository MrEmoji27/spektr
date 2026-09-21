"""The picture has to move, and it has to move on the beat.

The floors here are set below what the modes measure today, not at it, so
ordinary retuning has somewhere to go. What they catch is the shape of
failure nothing else in the suite can see: beats that never reach a mode, and
a picture that goes on animating after it has stopped answering the music.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onset_eval import SCENARIOS  # noqa: E402
from reactivity import measure  # noqa: E402

#: One from each family that answers beats differently: bars driven by the
#: spring, a particle mode driven by onsets, and a field.
MODES = ("Bars", "Pulse", "Bubbles")

#: A steady four-on-the-floor and a dense breakbeat. The first says a mode
#: answers an obvious beat; the second says it can still tell them apart when
#: they arrive three times as often.
SIGNALS = ("four_on_floor", "breakbeat")

#: Measured today at 0.008-0.042 on these pairs. A mode under this is not
#: drawing a moving picture at all. Only ``None``, which draws nothing on
#: purpose, sits at zero.
MIN_CHURN = 0.004

#: What each mode's beat response measures today on ``four_on_floor``, as
#: recorded rather than as endorsed.
#:
#: ``Bars`` answers the beat plainly: it changes nearly two and a half times
#: as much of the screen around one as between. ``Pulse`` does too, now that
#: its shockwaves are thrown from the blob's rim and cross it in a quarter of
#: a second instead of hanging in flight for longer than a bar — before that
#: its idle animation moved more of the screen than its waves did, and it
#: measured 0.81 here. ``Fireworks`` answered a beat with one more climbing
#: shell in a sky that already held ten, which measured 1.02; a beat now fires
#: a salvo of mines that burst where they are lit, and it measures 1.93.
#: ``Shooting Star`` laid a beat's meteor train behind the edge of the frame,
#: so the beat landed as one fragment and the rest arrived over the next third
#: of a second; the train now enters head-first and whole, and it measures 1.32.
#: ``Bubbles`` reads no rhythm field at all — its spawn rate follows the low
#: band's *level* (``particles.py:492``), and the ``ctx.pulse`` reads in that
#: file are ``_radial``'s and ``_sonar``'s, neither of which ``bubbles`` calls
#: — so 1.0 is the honest answer for a mode whose blurb promises bubbles from
#: the low end and claims nothing about beats. Saying "no rhythm field" here
#: means the mode *and* everything it calls: ``Radial`` is a one-liner over
#: ``_radial``, which does read ``ctx.pulse``, and a scan of the decorated
#: function alone reports it as reading nothing.
#:
#: These are here so a change that damps a mode's response shows up as a
#: number that fell, not as something someone notices months later.
BASELINE = {"Bars": 2.45, "Pulse": 1.89, "Bubbles": 1.01, "Fireworks": 1.93,
            "Shooting Star": 1.32}

#: How far a baseline may fall before it counts as a regression. Generous,
#: because the corpus is synthetic and a retune is allowed to cost a little.
SLACK = 0.20


@pytest.fixture
def clock(monkeypatch):
    """The signal's clock, in place of the wall's, for the whole widget.

    The widget reads ``time.monotonic`` for ``dt``, for ``ctx.t`` and for the
    morph. Left on the wall it would see a ``dt`` of nothing, because this
    renders ten seconds of audio in a fraction of that, and physics
    integrated in seconds would produce a picture that never moves.
    """
    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("mode", MODES)
def test_every_beat_reaches_the_mode(mode, signal, clock):
    """The onset counter is differenced once, and nothing else eats it.

    Two painters share that counter now. If a second caller ever lands on the
    path that consumes it, the beats it takes never reach the screen, and no
    fingerprint or F-score in the suite would move.
    """
    samples, rate, _truth = SCENARIOS[signal]()
    got = measure(mode, samples, rate, clock)
    assert got["published"] > 0, "the corpus published no onsets at all"
    assert got["lost"] == 0, (
        f"{mode} on {signal}: {got['lost']} of {got['published']} beats "
        f"never reached the mode"
    )


@pytest.mark.parametrize("mode", MODES)
def test_the_picture_keeps_moving(mode, clock):
    samples, rate, _truth = SCENARIOS["four_on_floor"]()
    got = measure(mode, samples, rate, clock)
    assert got["churn"] >= MIN_CHURN, f"{mode} barely moves at all: {got}"


@pytest.mark.parametrize("mode", MODES)
def test_beat_response_has_not_been_damped(mode, clock):
    """Whatever a mode answered the beat with, it still answers with."""
    samples, rate, _truth = SCENARIOS["four_on_floor"]()
    got = measure(mode, samples, rate, clock)
    floor = BASELINE[mode] * (1.0 - SLACK)
    assert got["beat_ratio"] >= floor, (
        f"{mode} answers the beat less than it used to: it changes "
        f"{got['beat_churn']:.5f} of the screen around a beat against "
        f"{got['idle_churn']:.5f} between them ({got['beat_ratio']}x, "
        f"was {BASELINE[mode]}x)"
    )
