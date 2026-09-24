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

#: Where a change of chord falls. Overwhelmingly the downbeat, and after that
#: the half bar.
#:
#: This is the cue that settles what the drums cannot. A kick on one and
#: three with a snare either side is symmetric: rotating the bar by two beats
#: maps the pattern exactly onto itself, so no amount of drum evidence can say
#: which kick is the downbeat. A listener resolves it by harmony -- the chord
#: changes on the one -- and so does this. Dynamics would have been the other
#: candidate and are not available: the analyser's auto-sensitivity flattens
#: accents so thoroughly that a kick played at 0.65 measured a *larger* rise
#: than the same kick at 1.0 in the same bar.
HARMONY_ON = (1.0, 0.0, 0.30, 0.0)

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
_HARMONY_C = np.asarray(HARMONY_ON) - np.mean(HARMONY_ON)


def _rolls(template: np.ndarray) -> np.ndarray:
    """``template`` at every rotation, one per row: row ``k`` is
    ``np.roll(template, k)``. Built once rather than on every hit."""
    return np.stack([np.roll(template, k) for k in range(len(template))])


_KICK_R = _rolls(_KICK_C)
_SNARE_R = _rolls(_SNARE_C)
_HARMONY_R = _rolls(_HARMONY_C)

#: How much the harmony cue counts against the drums. Set so it decides a
#: symmetric pattern, where the drum terms cancel exactly, without being able
#: to overrule a plain backbeat where the drums are unambiguous.
HARMONY_WEIGHT = 2.0

#: How much evidence before an answer is offered at all, in onsets.
MIN_ONSETS = 8

#: How far the best rotation must beat the runner-up, as a share of the best
#: score. Below this the pattern does not actually say which beat is one.
MIN_MARGIN = 0.15

#: How many onsets in a row have to agree on a new tempo before the grid is
#: rebuilt around it.
#:
#: The tempo estimate wobbles on real material -- measured across a 120 to
#: 150 BPM change it reported 150, then 75, then nothing, then 75 again
#: within a few seconds. Rebuilding on the first disagreement meant the
#: accumulated metre was thrown away faster than eight onsets could refill
#: it, and the downbeat was never found again for the rest of the track.
#: Waiting for agreement costs a beat or two after a real change and survives
#: the wobble.
REGRID_AFTER = 4

#: A tempo estimate that is out by an octave is not recoverable here.
#:
#: Gap-based tempo estimation is prone to octave errors and a backbeat
#: provokes them: kick on one with snares on two and four gives gaps of 0.4,
#: 0.8, 0.4 seconds, and on a 150 BPM track the analyser settled on 75. That
#: makes the bar cover eight real beats, no rotation fits, and the downbeat
#: stays unknown for as long as the estimate is wrong.
#:
#: Refiling the recent onsets at half and double the period was tried and
#: removed. It cannot be done safely: the fit scores of two grids are not
#: comparable, because a coarser grid concentrates the same evidence into
#: fewer residues and scores higher for it. Used as a tiebreak it overruled
#: grids that were working; used as a fallback it manufactured a confident
#: downbeat for a four-on-the-floor, which has none. Reporting nothing while
#: the tempo is wrong is the correct failure, and fixing it belongs in the
#: tempo estimate rather than here.

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
        "_anchor", "_down", "_harmony", "_kick", "_period", "_prev_residue",
        "_regrid", "_regrid_for", "_seen", "_snare", "beat", "confidence",
    )

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        #: Accumulated kick and snare mass per residue of the beat grid.
        self._kick = np.zeros(BEATS)
        self._snare = np.zeros(BEATS)
        #: How much the harmony moved at each residue of the grid.
        self._harmony = np.zeros(BEATS)
        #: Which residue the previous onset sat on. Harmony is credited
        #: backwards to it; see :meth:`feed`.
        self._prev_residue: int | None = None
        #: A tempo that disagrees with the grid in force, and how many onsets
        #: in a row have agreed with it. See :data:`REGRID_AFTER`.
        self._regrid: float | None = None
        self._regrid_for = 0
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

    def feed(self, t: float, period: float, drums: dict,
             harmony: float = 0.0) -> None:
        """File one onset, then re-read the metre.

        ``t`` is when the onset landed, ``period`` the seconds between beats
        the tempo tracker currently believes, ``drums`` the likelihoods from
        :func:`spektr.audio.drums.classify`, and ``harmony`` how far the chord
        moved since the previous onset -- 0.0 when there is nothing tonal, or
        when the caller has no chroma to offer.
        """
        if period <= 0.0:
            # No tempo means no grid to file against; anything counted now
            # would be filed against a guess.
            self._down = None
            self.beat = None
            self.confidence = 0.0
            return

        moved = (
            self._anchor is None
            or abs(period - self._period) > self._period * 0.12
        )
        if moved and self._anchor is not None:
            # Wait for the new tempo to be confirmed before believing it. A
            # single wobbling estimate must not cost the metre.
            agrees = (
                self._regrid is not None
                and abs(period - self._regrid) <= self._regrid * 0.12
            )
            self._regrid_for = self._regrid_for + 1 if agrees else 1
            self._regrid = period
            if self._regrid_for < REGRID_AFTER:
                return
        else:
            self._regrid, self._regrid_for = None, 0

        if moved:
            # The grid this evidence was counted on no longer exists. Keeping
            # it would file new onsets against old spacing, which is how a
            # tracker ends up confidently a beat out for the rest of a song.
            self._kick[:] = 0.0
            self._snare[:] = 0.0
            self._harmony[:] = 0.0
            self._prev_residue = None
            self._down = None
            self._seen = 0
            self._anchor = t
            self._period = period
            self._regrid, self._regrid_for = None, 0
            # The onsets themselves are kept: they are still evidence, just
            # evidence that has to be refiled against the new spacing.

        steps = (t - self._anchor) / period
        residue = int(round(steps)) % BEATS
        # A hit that does not sit near the grid is a flam, a fill or a wrong
        # tempo; counting it would blur every residue towards equal.
        if abs(steps - round(steps)) > 0.25:
            return

        self._kick[residue] += float(drums.get("kick", 0.0))
        self._snare[residue] += float(drums.get("snare", 0.0))
        # Harmony is credited to the *previous* onset, not this one. The
        # measurement is "how far the chord moved between the last hit and
        # this one", and a chord that changes on the downbeat has not visibly
        # moved at the instant the downbeat is struck -- it has moved by the
        # time the next hit lands. Crediting it forwards put the estimate a
        # beat late and the reported downbeat half a bar out.
        if self._prev_residue is not None:
            self._harmony[self._prev_residue] += max(0.0, float(harmony))
        self._prev_residue = residue
        self._seen += 1

        if self._seen % (BEATS * 2) == 0:
            self._kick *= DECAY
            self._snare *= DECAY
            self._harmony *= DECAY

        self._score(residue)

    @staticmethod
    def _rotations(kick, snare, harmony) -> np.ndarray:
        """How well each rotation of a grid fits the evidence on it."""
        return (_KICK_R @ np.asarray(kick) + _SNARE_R @ np.asarray(snare)
                + HARMONY_WEIGHT * (_HARMONY_R @ np.asarray(harmony)))


    def _score(self, residue: int) -> None:
        """Pick the grid, and the rotation of it, that the drums best fit."""
        if self._seen < MIN_ONSETS:
            self._down = None
            self.beat = None
            self.confidence = 0.0
            return

        scores = self._rotations(self._kick, self._snare, self._harmony)

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
