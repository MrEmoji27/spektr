"""What key the music is in, or an honest refusal.

:mod:`spektr.audio.chroma` says which pitch classes are sounding right now.
A key is not a property of a moment: C major and A minor contain the same
twelve weights, and the difference between them is which note the music keeps
returning to over a phrase. So this accumulates chroma over seconds and scores
the result against the twenty-four major and minor profiles.

The profiles are Krumhansl and Kessler's probe-tone ratings, which is the
standard that every later method is measured against. They are correlations,
not a chord dictionary: what they encode is how strongly each scale degree
belongs to a key, so a passage that never plays the seventh still lands on the
right key from the weight of everything else.

What it deliberately does not do is guess. Three states, not two:

* a key, with a confidence
* ``None`` with confidence 0.0 — not enough has been heard yet
* ``None`` with a confidence above zero — enough has been heard and it is
  genuinely ambiguous, which is the state a chromatic or atonal passage sits
  in, and the state a relative major and minor pair sit in until the music
  commits

A visualiser that recolours on a key change must not do it on a coin flip, so
:attr:`uncertain` is a state a caller can draw differently rather than a
failure it has to hide.
"""
from __future__ import annotations

import numpy as np

from .chroma import NOTES

#: Krumhansl-Kessler probe-tone profiles, rooted on C. The major profile's
#: peaks are the tonic, the fifth and the third, in that order, which is what
#: makes the correlation able to tell a key from its relative minor at all.
MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

#: How long the running average of the chroma leans back, in seconds. A key is
#: established over a phrase; anything much shorter tracks the current chord
#: instead, which is a different question and a much less stable answer.
MEMORY_S = 8.0

#: How much evidence before any answer is offered, in seconds of tonal audio.
#: Silence and percussion do not count towards it, so this is eight seconds of
#: music rather than eight seconds of wall clock.
MIN_EVIDENCE_S = 4.0

#: How well the best key has to actually fit before it is named at all.
#:
#: The obvious gate — "the winner must beat the runner-up by some margin" —
#: does not work here, and measuring said so plainly: a chromatic passage
#: produced a *larger* margin (0.15 of the spread) than a clean I-IV-V-I in C
#: (0.06), because noise picks a winner as decisively as music does. What
#: separates them is not how far ahead the winner is but how well it fits:
#: real keys correlated at 0.58 to 0.87, while a chromatic run managed 0.38
#: and a single repeated note 0.46.
#:
#: So the gate is on the fit, and the margin only shades the confidence. A
#: relative major and minor sitting a hair apart both fit well; which of the
#: two it is comes out as a low confidence rather than as a refusal.
MIN_FIT = 0.55

#: The fit at which confidence reaches its ceiling, measured above MIN_FIT.
FIT_RANGE = 0.30

#: How much better a challenger has to score than the key currently being
#: reported before the answer changes, as a share of the incumbent's score.
#:
#: Without this the reported key flickers, and a relative major and minor are
#: where it flickers worst: they share every note, so their scores sit a hair
#: apart and noise alone decides which is ahead from one frame to the next. A
#: mode that recolours on the key would strobe. Real modulations clear this
#: easily -- a new key is not a hair better, it is a different shape.
SWITCH_MARGIN = 0.06

#: ...and how long the challenger has to stay ahead, in seconds. A passing
#: chord borrowed from another key wins for a moment; a modulation does not
#: give the old key back.
SWITCH_HOLD_S = 2.5

#: Hysteresis only applies once a full :data:`MEMORY_S` of tonal audio has
#: been heard. Before that the average is still filling and its early winner
#: is not an incumbent worth defending — holding one cost the right answer on
#: a cadence in C, which settles correctly a few seconds in but had already
#: locked onto its relative minor.


#: How much of a note's energy its harmonics put on other pitch classes.
#:
#: A played note is not one frequency. Its third harmonic is an octave and a
#: fifth above the fundamental, which folds onto the pitch class seven
#: semitones up; its fifth harmonic folds onto the major third, four semitones
#: up. In a sawtooth — near enough to most instruments for this — the nth
#: harmonic has amplitude ``1/n``, so those two land at a third and a fifth of
#: the fundamental's weight. These are those two numbers, not fitted
#: constants.
#:
#: Left in, this is not a small error: it is a systematic one, and it always
#: points the same way. Every note quietly votes for its own fifth, so the
#: accumulated chroma of a piece in C looks like a piece in G, and the
#: estimator returned the dominant for every key it was given — C major read
#: as G major, D major as A major, A minor as E major. Subtracting the
#: harmonics each class would have produced puts all four back on the tonic.
FIFTH_BLEED = 0.33
THIRD_BLEED = 0.20


def deharmonise(chroma) -> np.ndarray:
    """Take out the energy a note's own harmonics put on other classes.

    Only for naming a key. :func:`spektr.audio.chroma.fold` deliberately does
    not do this: a mode colouring by harmony wants what is *sounding*, and a
    note's harmonics are part of that sound. Deciding what key a piece is in
    is the opposite question, and there the fifths are an artefact.
    """
    c = np.asarray(chroma, dtype=np.float64)
    if c.size != 12:
        return np.zeros(12)
    out = c - FIFTH_BLEED * np.roll(c, 7) - THIRD_BLEED * np.roll(c, 4)
    # Clipped rather than allowed negative: a class the correction pushes
    # below zero carried nothing of its own, and a negative weight would then
    # count as evidence *against* every key containing it.
    return np.maximum(out, 0.0)


def _centred(profile) -> np.ndarray:
    a = np.asarray(profile, dtype=np.float64)
    return a - a.mean()


_MAJOR_C = _centred(MAJOR)
_MINOR_C = _centred(MINOR)


def score(chroma) -> np.ndarray:
    """Correlate ``chroma`` against all 24 keys.

    Returns a ``(2, 12)`` array: row 0 the major keys, row 1 the minor, each
    column a tonic from C. Correlation rather than dot product, so a loud
    passage and a quiet one covering the same notes score the same.
    """
    c = np.asarray(chroma, dtype=np.float64)
    if c.size != 12:
        return np.zeros((2, 12))
    c = c - c.mean()
    norm = float(np.sqrt((c * c).sum()))
    if norm <= 1e-9:
        return np.zeros((2, 12))
    c = c / norm

    out = np.zeros((2, 12))
    for tonic in range(12):
        # Rotate the profile to the tonic rather than the chroma, so the sign
        # of the rotation cannot quietly go the wrong way.
        out[0, tonic] = float(np.dot(c, np.roll(_MAJOR_C, tonic)))
        out[1, tonic] = float(np.dot(c, np.roll(_MINOR_C, tonic)))
    denom_major = float(np.sqrt((_MAJOR_C ** 2).sum()))
    denom_minor = float(np.sqrt((_MINOR_C ** 2).sum()))
    out[0] /= denom_major
    out[1] /= denom_minor
    return out


def name(tonic: int, minor: bool) -> str:
    """``"C major"``, ``"A minor"``, and so on."""
    return f"{NOTES[tonic % 12]} {'minor' if minor else 'major'}"


def _index_of(key_name: str) -> int:
    """Where a key sits in a flattened :func:`score` array."""
    tonic, _, quality = key_name.partition(" ")
    return (12 if quality == "minor" else 0) + NOTES.index(tonic)


class KeyEstimator:
    """Accumulates chroma and reports a key when the evidence supports one.

    Fed the chroma of each analysed frame along with how much time that frame
    covers. Frames with nothing tonal in them are ignored rather than counted
    as evidence of a flat profile, so a drum break does not wash out the key
    of the track it sits in.
    """

    __slots__ = (
        "_avg", "_challenger", "_challenger_for", "_heard",
        "confidence", "key", "uncertain",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._avg: np.ndarray | None = None
        #: Seconds of *tonal* audio folded in so far.
        self._heard = 0.0
        #: The key trying to take over, and how long it has been ahead.
        self._challenger: str | None = None
        self._challenger_for = 0.0
        #: ``"C major"`` or ``None``. See the module docstring for the three
        #: states this and :attr:`confidence` make between them.
        self.key: str | None = None
        #: 0..1. With ``key is None`` a non-zero value means "heard enough,
        #: still ambiguous".
        self.confidence = 0.0
        #: True when there is evidence but no clear winner.
        self.uncertain = False

    def feed(self, chroma, dt: float) -> None:
        c = np.asarray(chroma, dtype=np.float64)
        if c.size != 12 or not c.any() or dt <= 0.0:
            return

        # Exponential average in seconds, so the answer does not depend on
        # what rate the analyser happens to run at.
        alpha = 1.0 - float(np.exp(-dt / MEMORY_S))
        if self._avg is None:
            self._avg = c.copy()
        else:
            self._avg += (c - self._avg) * alpha
        self._heard = min(self._heard + dt, MEMORY_S * 4)
        self._decide(dt)

    def _decide(self, dt: float = 0.0) -> None:
        if self._avg is None or self._heard < MIN_EVIDENCE_S:
            self.key = None
            self.confidence = 0.0
            self.uncertain = False
            return

        scores = score(deharmonise(self._avg))
        flat = scores.ravel()
        order = np.argsort(flat)[::-1]
        best, second = float(flat[order[0]]), float(flat[order[1]])
        spread = float(flat.std())

        if best < MIN_FIT or spread <= 1e-9:
            # Nothing fits a key well enough to name one. Atonal, chromatic,
            # percussion-only, or simply too little of the scale played.
            self.key = None
            self.confidence = 0.0
            self.uncertain = True
            return

        row, tonic = divmod(int(order[0]), 12)
        winner = name(tonic, minor=bool(row))

        # Hysteresis. The winner only takes over when it is clearly better
        # than what is already being reported and has stayed that way.
        settled = self._heard >= MEMORY_S
        if settled and self.key is not None and winner != self.key:
            held = float(flat[_index_of(self.key)])
            ahead = best > held * (1.0 + SWITCH_MARGIN)
            if winner == self._challenger and ahead:
                self._challenger_for += dt
            else:
                self._challenger = winner if ahead else None
                self._challenger_for = 0.0
            if not ahead or self._challenger_for < SWITCH_HOLD_S:
                # Keep reporting the incumbent, and say how well *it* fits.
                best = held
                winner = self.key
            else:
                self._challenger = None
                self._challenger_for = 0.0
        else:
            self._challenger = None
            self._challenger_for = 0.0

        self.key = winner

        # Two things make an answer trustworthy, and both belong in the
        # number: how well the winner fits, and how far clear of the next one
        # it is. A key sitting a hair above its relative minor is still the
        # best reading of the notes -- it just is not a confident one.
        fit = float(np.clip((best - MIN_FIT) / FIT_RANGE, 0.0, 1.0))
        lead = float(np.clip((best - second) / spread / 0.6, 0.0, 1.0))
        self.confidence = fit * (0.5 + 0.5 * lead)
        self.uncertain = False
