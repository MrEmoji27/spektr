"""Frame-rate independent easing.

The original had a subtle bug worth spelling out. Attack and release were
correctly dt-corrected::

    attack = 1.0 - math.exp(-dt / self._attack_tau)      # correct
    self._vel[i] += (target - cur) * k
    self._vel[i] *= self._damping                        # not correct

``_damping`` was a raw per-frame multiply. So when the adaptive pacer moved
between 18 and 45 fps, the damping ratio changed with it and the bars visibly
changed weight mid-song. Peak hold had the same problem — ``_PEAK_HOLD = 8``
frames is 178 ms at 45 fps and 444 ms at 18 fps.

This is a proper damped spring parameterised in real units (seconds), with
sub-stepping so a long frame can never destabilise it. The feel is constant
whatever the frame rate does.
"""

from __future__ import annotations

import numpy as np

#: Integration step. Longer frames are split into several of these.
_MAX_STEP = 1.0 / 90.0

#: Motion personalities, as keyword arguments for :class:`Spring`.
#:
#: Both are expressed in seconds and damping ratios, so neither depends on
#: the frame rate — deliberately *not* cava's gravity/integral filters, which
#: are framerate-dependent by construction (``framerate_mod = 66/framerate``).
#: The point of ``glide`` is to reproduce cava's *feel* — the lazy,
#: centre-weighted look — without importing the bug that motivated this
#: module in the first place.
PROFILES = {
    # The tuning everything else was calibrated against. Kept as written
    # parameters rather than a bare constructor call so a profile is one
    # table entry, not a branch somewhere in the widget.
    "snappy": dict(attack=0.09, release=0.30, attack_zeta=0.85, release_zeta=1.0),
    # Slower to rise (a kick swells rather than snaps), overdamped on the
    # fall so bars sink without overshoot, and paired with the pre-blend and
    # neighbour spread below — between them the transients get rounded off
    # and the sustained middle of the spectrum accumulates height, which is
    # exactly why cava reads as busier in the centre.
    "glide": dict(attack=0.24, release=0.75, attack_zeta=0.95, release_zeta=1.05),
}

#: Temporal pre-blend applied to band targets in the ``glide`` profile, in
#: seconds. This is cava's noise-reduction smoothing done dt-correctly: the
#: raw spectrum is exponential-blended toward the spring's target before the
#: spring sees it, so a one-frame transient arrives already softened.
GLIDE_BLEND_TAU = 0.06

#: Spatial neighbour decay for the ``glide`` profile — cava's monstercat
#: filter. Each bar's level leaks to its neighbours multiplied by this per
#: cell of distance, taken as a maximum, so a hot band fattens the ones
#: beside it instead of standing alone.
GLIDE_SPREAD_DECAY = 0.62


class Spring:
    """A vector of critically-ish damped springs.

    ``response`` is the time to substantially reach a new target. ``zeta`` is
    the damping ratio: 1.0 is critical (no overshoot), slightly below adds a
    little life on transients, above makes it sluggish.
    """

    __slots__ = ("_wa", "_wr", "_za", "_zr", "v", "x")

    def __init__(
        self,
        n: int,
        attack: float = 0.09,
        release: float = 0.30,
        attack_zeta: float = 0.85,
        release_zeta: float = 1.0,
    ):
        self.x = np.zeros(n, dtype=np.float64)
        self.v = np.zeros(n, dtype=np.float64)
        self._wa = 5.0 / max(1e-3, attack)
        self._wr = 5.0 / max(1e-3, release)
        self._za = attack_zeta
        self._zr = release_zeta

    def step(self, targets: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(targets, dtype=np.float64)
        if target.shape != self.x.shape:
            self.x = np.resize(self.x, target.shape)
            self.v = np.resize(self.v, target.shape)

        remaining = max(0.0, min(dt, 0.25))
        while remaining > 1e-9:
            h = remaining if remaining < _MAX_STEP else _MAX_STEP
            remaining -= h

            rising = target > self.x
            w = np.where(rising, self._wa, self._wr)
            z = np.where(rising, self._za, self._zr)

            accel = (w * w) * (target - self.x) - (2.0 * z * w) * self.v
            self.v += accel * h
            self.x += self.v * h

        np.clip(self.x, 0.0, 1.0, out=self.x)
        # kill velocity at the rails so a clamped band doesn't stay wound up
        self.v[(self.x <= 0.0) & (self.v < 0.0)] = 0.0
        self.v[(self.x >= 1.0) & (self.v > 0.0)] = 0.0
        return self.x

    def retune(
        self,
        attack: float,
        release: float,
        attack_zeta: float = 0.85,
        release_zeta: float = 1.0,
    ) -> None:
        """Swap the spring's constants live — how a motion profile is applied.

        Position and velocity are kept: retuning mid-song eases from wherever
        the bars are now rather than teleporting them, which is the difference
        between changing character and resetting the picture.
        """
        self._wa = 5.0 / max(1e-3, attack)
        self._wr = 5.0 / max(1e-3, release)
        self._za = attack_zeta
        self._zr = release_zeta


class Peaks:
    """Peak markers with hold and fall measured in seconds."""

    __slots__ = ("_fall", "_hold", "_t", "_until", "value")

    def __init__(self, n: int, hold: float = 0.35, fall: float = 0.55):
        self.value = np.zeros(n, dtype=np.float64)
        self._until = np.zeros(n, dtype=np.float64)
        self._hold = hold
        self._fall = fall
        self._t = 0.0

    def step(self, values: np.ndarray, dt: float) -> np.ndarray:
        self._t += dt
        if self.value.shape != values.shape:
            self.value = np.array(values, dtype=np.float64)
            self._until = np.zeros(values.shape, dtype=np.float64)

        hit = values >= self.value
        self.value[hit] = values[hit]
        self._until[hit] = self._t + self._hold

        falling = ~hit & (self._t > self._until)
        self.value[falling] = np.maximum(
            values[falling], self.value[falling] - self._fall * dt
        )
        return self.value


def spread(values: np.ndarray, decay: float = GLIDE_SPREAD_DECAY) -> np.ndarray:
    """Cava's monstercat filter: a hot bar fattens its neighbours.

    Each level leaks outward multiplied by ``decay`` per cell of distance,
    taken as a maximum with what was already there, so spreading can only
    ever raise a bar and a flat field is unchanged. Two passes — one per
    direction — because the leak has to carry through intermediate bars to
    reach the far side of a group.

    A plain Python loop over at most N_BANDS elements, deliberately: the two
    passes are sequential by nature (each bar reads the already-updated one
    beside it), and that is exactly the dependency a vectorised maximum would
    silently get wrong at the boundary.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    n = out.shape[0]
    for i in range(1, n):
        leaked = out[i - 1] * decay
        if leaked > out[i]:
            out[i] = leaked
    for i in range(n - 2, -1, -1):
        leaked = out[i + 1] * decay
        if leaked > out[i]:
            out[i] = leaked
    return out


class Trace:
    """A dt-correct exponential blend, for the scope trace.

    The old code used a fixed ``a = 0.45`` per frame, which meant the waveform
    smoothed differently at different frame rates just like the bars did.
    """

    __slots__ = ("_tau", "value")

    def __init__(self, tau: float = 0.03):
        self.value: np.ndarray | None = None
        self._tau = tau

    def step(self, fresh: np.ndarray, dt: float) -> np.ndarray:
        fresh = np.asarray(fresh, dtype=np.float64)
        if self.value is None or self.value.shape != fresh.shape:
            self.value = fresh.copy()
            return self.value
        a = 1.0 - np.exp(-max(dt, 1e-6) / self._tau)
        self.value += (fresh - self.value) * a
        return self.value
