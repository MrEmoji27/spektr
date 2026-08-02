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


class Spring:
    """A vector of critically-ish damped springs.

    ``response`` is the time to substantially reach a new target. ``zeta`` is
    the damping ratio: 1.0 is critical (no overshoot), slightly below adds a
    little life on transients, above makes it sluggish.
    """

    __slots__ = ("x", "v", "_wa", "_wr", "_za", "_zr")

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


class Peaks:
    """Peak markers with hold and fall measured in seconds."""

    __slots__ = ("value", "_until", "_hold", "_fall", "_t")

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


class Trace:
    """A dt-correct exponential blend, for the scope trace.

    The old code used a fixed ``a = 0.45`` per frame, which meant the waveform
    smoothed differently at different frame rates just like the bars did.
    """

    __slots__ = ("value", "_tau")

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
