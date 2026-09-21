"""Which beat of the bar we are on, from what the drums did.

The onset detector says *when* the beats are and roughly how fast. It has no
idea which of them is beat one, and that is the difference between a mode that
pulses and a mode that phrases: a change that lands on the downbeat reads as
musical, and the same change a beat early reads as a glitch.

Nothing in a spectrum marks a downbeat. What marks it is the pattern: in
almost everything spektr will ever be pointed at, the kick is on one and the
snare answers on two and four. So this takes the drum likelihoods
:mod:`spektr.audio.drums` already produces, files each onset against the beat
grid the tempo tracker already has, and asks which rotation of that grid best
matches the pattern.

Two deliberate limits, both the same kind of honesty the detector keeps:

* **It assumes four beats to the bar.** Three-four exists and this will be
  wrong about it. Reporting a wrong bar confidently is worse than reporting
  none, so :attr:`confidence` is what a caller checks, not the phase.
* **It refuses to guess.** A four-on-the-floor with no snare has no downbeat
  anyone can hear, and the right answer there is "I don't know" rather than
  whichever of the four was fractionally ahead. That is what
  :data:`MIN_MARGIN` is for.
"""
from __future__ import annotations

import numpy as np

#: Beats to the bar. See the module docstring for why this is not a setting.
BEATS = 4

#: Where each drum is expected to fall, indexed by beat within the bar.
#:
#: The kick carries the downbeat and, more weakly, the third beat — weakly
#: because a kick on three is common but far from universal, and scoring it as
#: hard as the downbeat makes a two-beat rotation look as good as the right
#: one. The snare's two-and-four is the stronger signal of the pair and is
#: what actually breaks that tie.
KICK_ON = (1.0, 0.0, 0.45, 0.0)
SNARE_ON = (0.0, 1.0, 0.0, 1.0)

#: The same templates with their average taken out, which is what the scoring
#: actually correlates against.
#:
#: This matters more than it looks. A snare on two and four is unchanged by
#: rotating the bar a half turn, so it scores identically for "beat one is
#: here" and "beat one is two beats away" — and as a raw sum that identical
#: part is *added to both*, swamping the kick, which is the only term that can
#: tell them apart. On a plain backbeat the winner led by 12% of the total and
#: was thrown away as indecisive, while the kick on its own led by 55%.
#:
#: Taking the mean out leaves a correlation: it answers "is the mass
#: distributed like this template" and ignores how much of it there is. Even
#: spread now scores exactly zero rather than scoring well everywhere, which
#: is why a four-on-the-floor falls out as unknown instead of as a tie that
#: happens to be under the margin.
_KICK_C = np.asarray(KICK_ON) - np.mean(KICK_ON)
_SNARE_C = np.asarray(SNARE_ON) - np.mean(SNARE_ON)

#: How much evidence before an answer is offered at all, in onsets.
MIN_ONSETS = 8

#: How far the best rotation must beat the runner-up, as a share of the best
#: score. Below this the pattern does not actually say which beat is one.
MIN_MARGIN = 0.15

#: How much a bar of evidence fades per bar. Old material should stop voting
#: once the music has moved on, but not so fast that a fill erases the metre.
DECAY = 0.85


class BarTracker:
    """Files onsets against the beat grid and reports where the bar starts.

    Fed one onset at a time, with the drum likelihoods for it and the current
    beat period. Cheap: a handful of floats per onset and no allocation, so it
    can sit in the analyser's path without costing it anything.
    """

    __slots__ = (
        "_anchor", "_down", "_kick", "_period", "_seen", "_snare",
        "beat", "confidence",
    )

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        #: Accumulated kick and snare mass per residue of the beat grid.
        self._kick = np.zeros(BEATS)
        self._snare = np.zeros(BEATS)
        #: The onset every later one is counted from, and the period in force
        #: when it was taken. Both are dropped when the tempo moves.
        self._anchor: float | None = None
        self._period = 0.0
        self._seen = 0
        #: Which residue of the grid the downbeat sits on, or ``None`` while
        #: the pattern has not said. This is the answer; :attr:`beat` and
        #: :meth:`phase` are both read off it.
        self._down: int | None = None
        #: Which beat of the bar the last onset was, or ``None`` when unknown.
        self.beat: int | None = None
        #: 0..1. Zero means the pattern does not say, which is common and not
        #: a failure — plenty of music has no downbeat to find.
        self.confidence = 0.0

    def reset(self) -> None:
        """Forget everything. For a new track, or coming out of silence."""
        self._reset()

    def feed(self, t: float, period: float, drums: dict) -> None:
        """File one onset, then re-read the metre.

        ``t`` is when the onset landed, ``period`` the seconds between beats
        the tempo tracker currently believes, and ``drums`` the likelihoods
        from :func:`spektr.audio.drums.classify`.
        """
        if period <= 0.0:
            # No tempo means no grid to file against; anything counted now
            # would be filed against a guess.
            self._down = None
            self.beat = None
            self.confidence = 0.0
            return

        if self._anchor is None or abs(period - self._period) > self._period * 0.12:
            # The grid this evidence was counted on no longer exists. Keeping
            # it would file new onsets against old spacing, which is how a
            # tracker ends up confidently a beat out for the rest of a song.
            self._kick[:] = 0.0
            self._snare[:] = 0.0
            self._seen = 0
            self._anchor = t
            self._period = period

        steps = (t - self._anchor) / period
        residue = int(round(steps)) % BEATS
        # A hit that does not sit near the grid is a flam, a fill or a wrong
        # tempo; counting it would blur every residue towards equal.
        if abs(steps - round(steps)) > 0.25:
            return

        self._kick[residue] += float(drums.get("kick", 0.0))
        self._snare[residue] += float(drums.get("snare", 0.0))
        self._seen += 1

        if self._seen % (BEATS * 2) == 0:
            self._kick *= DECAY
            self._snare *= DECAY

        self._score(residue)

    def _score(self, residue: int) -> None:
        """Pick the rotation of the grid that best fits the drum pattern."""
        if self._seen < MIN_ONSETS:
            self._down = None
            self.beat = None
            self.confidence = 0.0
            return

        scores = np.array([
            float(
                np.dot(self._kick, np.roll(_KICK_C, k))
                + np.dot(self._snare, np.roll(_SNARE_C, k))
            )
            for k in range(BEATS)
        ])

        best = int(np.argmax(scores))
        top = scores[best]
        # Zero or below means nothing correlates: evenly spread mass, or a
        # pattern that fits no rotation better than the average one.
        if top <= 1e-9:
            self._down = None
            self.beat = None
            self.confidence = 0.0
            return

        rest = np.delete(scores, best)
        margin = (top - float(rest.max())) / top
        if margin < MIN_MARGIN:
            # Two rotations fit about as well, so the pattern is not saying
            # which beat is one. A steady four-on-the-floor lives here.
            self._down = None
            self.beat = None
            self.confidence = 0.0
            return

        # ``best`` is the rotation that puts the downbeat on residue ``best``,
        # so the beat this onset landed on is its distance from there.
        self._down = best
        self.beat = (residue - best) % BEATS
        self.confidence = float(min(1.0, margin / 0.5))

    def phase(self, now: float) -> float:
        """0..1 across the bar, 0.0 on the downbeat.

        Returns 0.0 whenever the downbeat is unknown, the same way
        ``beat_phase`` reports 0.0 on an unknown tempo. A caller that needs to
        tell "on the downbeat" from "no idea" checks :attr:`confidence`.
        """
        if self._down is None or self._anchor is None or self._period <= 0.0:
            return 0.0
        # Downbeats land on the anchor plus whole bars, offset by whichever
        # residue the pattern picked out.
        steps = (now - self._anchor) / self._period
        return float(((steps - self._down) % BEATS) / BEATS)
