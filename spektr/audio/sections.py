"""Is the track building, dropping, breaking down, or just running?

Everything else in :mod:`spektr.audio` answers a question about a moment: what
note is sounding, what drum hit, which beat of the bar. This answers one about
a *stretch* — the shape a track makes over eight or sixteen bars — because
that is the scale at which a visualiser's big gestures want to land. A mode
that blooms on the drop and holds back through the build is following the
music; the same mode firing on a loud beat is only following the level.

Three things give the shape away, and all three come free from the bands the
analyser already computes:

* **level** — how loud, taken from the frame's ``rms`` rather than from the
  bands. The bands have been through the analyser's auto-sensitivity, which
  exists to keep a quiet source visible and therefore removes most of the
  level dynamics a section change is made of. Measured on a synthetic build
  and drop, the bass came back at barely 1.0x its own average through the
  bands and 3x through the rms.
* **low** — the *share* of the energy sitting under
  :data:`spektr.audio.analysis.BASS_CUT_HZ`, not how much of it there is.
  A share survives the auto-sensitivity, which scales every band together.
  Filtering the bass out is how nearly every build is made, and putting it
  back is the drop.
* **brightness** — where the energy sits, as a share of the way up the band
  range. A build sweeps upward; a breakdown does not.

Each is compared against its own slower average, so what is reported is
*change* rather than level, and a quiet track and a loud one look the same.

Honest about what it is
-----------------------
This is a heuristic over three numbers, not a structural analysis. It will
call a sudden loud chorus a drop, because from the spectrum that is what a
drop is. :attr:`confidence` says how strongly the evidence points, and
``"steady"`` is the answer most of the time, which is correct: most of a
track is not a transition.

**The thresholds below are provisional.** They were set against one
synthetic build-and-drop, and a synthetic track cannot settle a perceptual
question: whether a passage "is a build" is a judgement about music, and the
only way to tune these is against real tracks with someone listening. The
build and the drop are read reliably on the synthetic material and the
breakdown boundary is the least trustworthy of the three. Nothing in the app
reads this yet; it is wired into no ``Ctx`` field on purpose, so that
tightening these numbers later cannot change anything already on screen.
"""
from __future__ import annotations

import numpy as np

#: The states reported. ``steady`` is not a failure to decide — it is the
#: answer for most of most tracks.
STATES = ("steady", "build", "drop", "breakdown")

#: How far back the fast and slow averages lean, in seconds. The fast one is
#: about a bar at an ordinary tempo; the slow one is the phrase it sits in, so
#: the difference between them is what "changing" means here.
FAST_S = 1.2
SLOW_S = 8.0

#: A build sweeps upward. This is how much brighter than its own slow average
#: the sound has to be, in units of the band range, before that counts.
BUILD_BRIGHT = 0.035

#: ...and how far the bass has to have thinned against its own average. Nearly
#: every build is made by filtering the low end out; this is what separates a
#: build from the track simply getting louder.
BUILD_LOW = 0.80

#: A drop is the low end coming back, hard. The ratio the bass has to jump by
#: against its slow average.
DROP_LOW = 1.45

#: ...and how much of the recent past must have been a build for a jump to be
#: read as a drop rather than as an ordinary loud bar. In seconds.
DROP_AFTER_BUILD_S = 2.0

#: A breakdown is the floor falling out: this much of the slow level, or less.
BREAKDOWN_LEVEL = 0.55

#: How long a state must hold before it is reported, in seconds.
#:
#: A bar with a snare in it is brighter and thinner in the bass than the bar
#: before it, which is the same shape a build has — over one bar, a backbeat
#: *is* a tiny build. At 0.35 s that flickered three times inside a steady
#: section. A real build lasts bars, so requiring the condition to persist
#: costs nothing and removes the whole class of false positive.
HOLD_S = 1.6

#: How long a drop stays "the drop" once called. It is a moment, not a
#: stretch, and a mode reacting to it wants an edge rather than a plateau.
DROP_S = 1.5


def _ema(old, new, dt: float, tau: float):
    """Exponential average in seconds, so the analyser's rate cannot change it."""
    if old is None:
        return new
    return old + (new - old) * (1.0 - float(np.exp(-dt / tau)))


def _envelope(old, new, dt: float, tau: float):
    """Rises instantly, falls slowly — a loudness meter, not an average.

    Level has to be read this way and the shares do not. Between two drum hits
    the signal really is near silence, so a mean of the rms over a second
    tracks how *sparse* the bar is rather than how loud it is, and on a plain
    kick-and-snare pattern that dipped far enough to report a breakdown twice
    a bar. Holding the peak and letting it fall measures the loudness a
    listener hears instead.
    """
    if old is None or new >= old:
        return new
    return old * float(np.exp(-dt / tau))


def features(bands, bass_bands: int, rms: float | None = None) -> tuple[float, float, float]:
    """``(level, low_share, brightness)`` for one frame.

    ``low_share`` and ``brightness`` are both shares of the frame's own
    energy, so the analyser's auto-sensitivity cannot move them: it multiplies
    every band by the same number. ``level`` is the one quantity that has to
    come from outside the bands for the same reason, and ``rms`` is what the
    caller should pass.
    """
    b = np.asarray(bands, dtype=np.float64)
    if b.size == 0:
        return 0.0, 0.0, 0.0
    total = float(b.sum())
    level = float(b.mean()) if rms is None else float(rms)
    if total <= 1e-9:
        return level, 0.0, 0.0
    cut = max(1, min(int(bass_bands), b.size))
    low = float(b[:cut].sum()) / total
    centre = float(np.dot(b, np.arange(b.size)) / total)
    return level, low, centre / max(1, b.size - 1)


class SectionTracker:
    """Reports which of :data:`STATES` the track is in."""

    __slots__ = (
        "_bright_fast", "_bright_slow", "_build_for", "_drop_for", "_held",
        "_level_fast", "_level_slow", "_low_fast", "_low_slow", "_pending",
        "confidence", "state",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._level_fast = self._level_slow = None
        self._low_fast = self._low_slow = None
        self._bright_fast = self._bright_slow = None
        #: Seconds the build condition has held, which is what makes a later
        #: jump in the bass a drop rather than a loud bar.
        self._build_for = 0.0
        #: Seconds since a drop was called, while it is still the drop.
        self._drop_for = 0.0
        self._pending: str = "steady"
        self._held = 0.0
        #: One of :data:`STATES`.
        self.state = "steady"
        #: 0..1 in how strongly the evidence points at the current state.
        #: Always 0.0 while ``steady``.
        self.confidence = 0.0

    def feed(self, bands, dt: float, bass_bands: int = 6,
             rms: float | None = None) -> None:
        if dt <= 0.0:
            return
        b = np.asarray(bands, dtype=np.float64)
        if b.size == 0 or float(b.sum()) <= 1e-9:
            # A gated frame carries no spectrum, so its shares are zeros that
            # mean "nothing measured" rather than "no bass". Averaging them in
            # would drag every share towards zero through ordinary gaps.
            return
        level, low, bright = features(b, bass_bands, rms)

        self._level_fast = _envelope(self._level_fast, level, dt, FAST_S)
        self._level_slow = _envelope(self._level_slow, level, dt, SLOW_S)
        self._low_fast = _ema(self._low_fast, low, dt, FAST_S)
        self._low_slow = _ema(self._low_slow, low, dt, SLOW_S)
        self._bright_fast = _ema(self._bright_fast, bright, dt, FAST_S)
        self._bright_slow = _ema(self._bright_slow, bright, dt, SLOW_S)

        want, strength = self._read(dt)
        if want == self._pending:
            self._held += dt
        else:
            self._pending, self._held = want, 0.0

        # A drop is announced the moment it is seen. Waiting out the hold
        # would put it late, and late is the one thing a drop cannot be.
        if want == "drop" or self._held >= HOLD_S:
            self.state = want
            self.confidence = 0.0 if want == "steady" else strength

    def _read(self, dt: float) -> tuple[str, float]:
        if self._level_slow is None or self._level_slow <= 1e-9:
            return "steady", 0.0

        level_ratio = self._level_fast / max(self._level_slow, 1e-9)
        low_ratio = self._low_fast / max(self._low_slow, 1e-9)
        bright_lift = self._bright_fast - self._bright_slow

        # ── drop ──
        # Called first: a drop is loud, bright and bass-heavy all at once, and
        # would otherwise satisfy nothing or be mistaken for a build's tail.
        if self._drop_for > 0.0:
            self._drop_for = max(0.0, self._drop_for - dt)
            if self._drop_for > 0.0:
                return "drop", 1.0
        if low_ratio >= DROP_LOW and self._build_for >= DROP_AFTER_BUILD_S:
            self._drop_for = DROP_S
            self._build_for = 0.0
            return "drop", float(np.clip((low_ratio - DROP_LOW) / 0.8 + 0.5, 0.0, 1.0))

        # ── build ──
        building = bright_lift >= BUILD_BRIGHT and low_ratio <= BUILD_LOW
        if building:
            self._build_for += dt
            near = bright_lift / BUILD_BRIGHT
            return "build", float(np.clip((near - 1.0) / 1.5 + 0.4, 0.0, 1.0))
        # The memory of a build decays rather than clearing, so a bar of the
        # bass poking through mid-build does not disqualify the drop.
        self._build_for = max(0.0, self._build_for - dt * 0.5)

        # ── breakdown ──
        if level_ratio <= BREAKDOWN_LEVEL:
            return "breakdown", float(
                np.clip((BREAKDOWN_LEVEL - level_ratio) / 0.35 + 0.3, 0.0, 1.0)
            )

        return "steady", 0.0
