"""Pay for a mode's first frame before it is on screen.

Loading a mode lazily is cheap: its module imports in a few milliseconds. What
is not cheap is the first frame. A mode builds its geometry, look-up tables
and grids into scratch the first time it draws at a size, and for the heaviest
that is 20 to 35 ms at 200x50, twice the whole frame budget, on exactly the
frame a morph starts. So the widget names the modes it expects to draw next
(see ``AudioVisualizer._mode_window``) and this draws each of them once, on a
background thread, into the same scratch the render path will use.

**What a warm-up draws.** A silent-level frame with ``dt`` of zero: the bands
are all zero and no time passes, so the mode builds what it builds at this
size and advances no animation. Its first real frame is the frame it would
have drawn cold, only without the setup in it.

**Never two threads in one mode.** A mode's scratch is not built for
concurrent use, so the render path calls :meth:`ModeWarmer.claim` before it
draws: a warm-up still queued for that mode is dropped, and one already
running is waited for. The wait is at most what the mode would have cost cold,
which is the cost this exists to remove, so it is never worse than before.

**Built-ins only.** A plugin's code has made no promise to be safe off the
main thread, and a plugin that is slow to start is the plugin's to fix.

A warm-up that raises is swallowed. The render path owns failure: it
quarantines, it tells the user, and it would do both again on the first real
frame. Warming a broken mode must not be what takes the app down.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .. import modes as mode_registry
from ..analysis import N_BANDS, WAVE_POINTS
from ..modes import Ctx

#: How long the thread waits for more work before it exits. It is started
#: again by the next request, so an idle app is not holding a thread open.
IDLE_S = 30.0

#: How long after it is first asked the thread starts, and the pause between
#: one warm-up and the next. The first request comes as the app mounts, and a
#: warm-up then competes with the first paint for the interpreter -- the one
#: frame nobody should wait on for the sake of one that may never be shown.
SETTLE_S = 0.75
GAP_S = 0.05


def warm_ctx(w: int, h: int, palette, state: dict, level: float = 0.0) -> Ctx:
    """A frame with nothing in it: bands at ``level``, no time passing."""
    bands = np.full(N_BANDS, level, dtype=np.float64)
    wave = np.zeros(WAVE_POINTS, dtype=np.float64)
    return Ctx(
        w=w, h=h, bands=bands, peaks=bands.copy(), bands_l=bands.copy(),
        bands_r=bands.copy(), wave=wave, stereo=np.zeros((WAVE_POINTS, 2)),
        frame=0, t=0.0, dt=0.0, energy=float(level), silent=False,
        palette=palette, state=state,
    )


class ModeWarmer:
    """One background thread drawing first frames for modes not yet shown."""

    def __init__(self, settle: float = SETTLE_S, gap: float = GAP_S) -> None:
        self.settle = settle
        self.gap = gap
        self._cv = threading.Condition()
        self._queue: list[tuple] = []
        self._running: str | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False

    # ── the render path's side ───────────────────────────────────────────────

    def request(self, name: str, state: dict, w: int, h: int, palette) -> None:
        """Draw ``name`` once into ``state`` at ``w`` x ``h``, soon."""
        with self._cv:
            if self._stopped or self._running == name:
                return
            if any(job[0] == name for job in self._queue):
                return
            self._queue.append((name, state, w, h, palette))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="spektr-warm", daemon=True
                )
                self._thread.start()
            self._cv.notify_all()

    def claim(self, name: str) -> None:
        """Make ``name`` safe to draw: drop its queued warm-up, or wait out
        the one that is running."""
        with self._cv:
            self._queue = [job for job in self._queue if job[0] != name]
            while self._running == name:
                self._cv.wait()

    def busy(self, name: str) -> bool:
        with self._cv:
            return self._running == name or any(j[0] == name for j in self._queue)

    def wait(self, name: str, timeout: float) -> bool:
        """Block until ``name`` has no warm-up pending. True if it finished."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._running == name or any(j[0] == name for j in self._queue):
                left = end - time.monotonic()
                if left <= 0.0:
                    return False
                self._cv.wait(left)
        return True

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._queue.clear()
            self._cv.notify_all()

    # ── the thread ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        with self._cv:
            self._cv.wait_for(lambda: self._stopped, timeout=self.settle)
        while True:
            with self._cv:
                if not self._queue and not self._stopped:
                    self._cv.wait(IDLE_S)
                if self._stopped or not self._queue:
                    self._thread = None
                    return
                name, state, w, h, palette = self._queue.pop(0)
                self._running = name
            try:
                self._warm(name, state, w, h, palette)
            except Exception:
                pass
            finally:
                with self._cv:
                    self._running = None
                    self._cv.notify_all()
                    # Leave the render path a breath between two warm-ups.
                    self._cv.wait_for(lambda: self._stopped, timeout=self.gap)

    @staticmethod
    def _warm(name: str, state: dict, w: int, h: int, palette) -> None:
        m = mode_registry.ensure_loaded(name)
        if m is None or m.is_plugin:
            return
        m.fn(warm_ctx(w, h, palette, state))
