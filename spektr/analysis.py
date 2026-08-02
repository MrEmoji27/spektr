"""Spectrum analysis, on its own clock.

This is the fix that matters most. The old code called ``_analyse()`` from
inside the frame timer, so a 2048-sample block at 48 kHz (23.4 blocks/sec) was
being sampled by a 30-45 fps render loop. A third of frames re-read a buffer
they had already seen, and when the adaptive pacer sped up, blocks were dropped
outright. The result was beat-rate aliasing — bands appearing to stall and then
jump — which no amount of easing can hide, because the target sequence itself
is stepped.

Here the analyser owns a thread and runs on a **hop** of 512 samples with a
2048-sample window, i.e. 75% overlap and ~94 analyses/sec at 48 kHz. That is
comfortably faster than any sane frame rate, so the renderer always has a fresh
target and the spring has something continuous to chase.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .capture import RingBuffer

N_BANDS = 32          # internal resolution; modes downsample for chunky looks
FFT_SIZE = 2048
HOP = 512             # 75% overlap
WAVE_POINTS = 512     # downsampled scope trace

#: How far above the noise gate a signal must be before it is drawn at full
#: strength. Kept deliberately narrow: the gate already rejects digital silence
#: outright, so this only has to cover the sliver just above it. An earlier
#: value of 6x was far too wide — it crushed quiet-but-real sources, which is
#: exactly the case auto-gain exists to rescue.
_KNEE = 2.0


@dataclass
class Frame:
    """One analysis result. Treated as immutable once published."""

    seq: int = 0
    bands: np.ndarray = field(default_factory=lambda: np.zeros(N_BANDS))
    bands_l: np.ndarray = field(default_factory=lambda: np.zeros(N_BANDS))
    bands_r: np.ndarray = field(default_factory=lambda: np.zeros(N_BANDS))
    wave: np.ndarray = field(default_factory=lambda: np.zeros(WAVE_POINTS))
    stereo: np.ndarray = field(default_factory=lambda: np.zeros((WAVE_POINTS, 2)))
    rms: float = 0.0
    silent: bool = True
    #: 0..1 — how far above the noise gate this frame sits. Anything near zero
    #: is hugging the floor, which is worth surfacing when someone is trying to
    #: work out why the display is twitching at nothing.
    confidence: float = 0.0

    @property
    def energy(self) -> float:
        return float(self.bands.mean())


class Analyser:
    """Overlapped FFT running independently of the render loop."""

    def __init__(self, ring: RingBuffer, samplerate_getter):
        self._ring = ring
        self._get_sr = samplerate_getter
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame = Frame()
        self._seq = 0

        # cached per (samplerate, size)
        self._win: np.ndarray | None = None
        self._win_key = None
        self._edges_key = None
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None

        # ── loudness handling, constants carried over from the tuned original ──
        self._env = 1e-4          # decaying loudness envelope, for auto gain
        self._target_rms = 0.06   # where auto gain aims
        self._max_gain = 1200.0   # enough for a quiet mix, not enough to chase hiss
        self._gate_abs = 8e-5     # fixed absolute gate; see note below
        self._gate_hold_s = 0.30
        self._hold_until = 0.0
        self.sensitivity = 1.0    # manual trim, [ and ]

    # ── lifecycle ──
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    @property
    def frame(self) -> Frame:
        with self._lock:
            return self._frame

    # ── tuning ──
    def nudge_sensitivity(self, factor: float) -> float:
        self.sensitivity = max(0.15, min(8.0, self.sensitivity * factor))
        return self.sensitivity

    def nudge_gate(self, factor: float) -> float:
        self._gate_abs = max(1e-6, min(2e-3, self._gate_abs * factor))
        return self._gate_abs

    @property
    def gate(self) -> float:
        return self._gate_abs

    # ── band layout ──
    def _ensure_tables(self, sr: int, n: int) -> None:
        if self._win_key != n:
            self._win = np.hanning(n).astype(np.float64)
            self._win_key = n

        key = (sr, n)
        if self._edges_key == key:
            return
        bin_hz = sr / n
        edges = np.logspace(math.log10(20.0), math.log10(min(20000.0, sr / 2)), N_BANDS + 1)
        half = n // 2 + 1
        lo = np.clip((edges[:-1] / bin_hz).astype(np.int32), 1, half - 1)
        hi = np.clip((edges[1:] / bin_hz).astype(np.int32), 1, half - 1)
        hi = np.maximum(hi, lo + 1)   # every band gets at least one bin
        self._lo, self._hi = lo, hi
        self._edges_key = key

    def _bands_from(self, spectrum: np.ndarray) -> np.ndarray:
        """Mean magnitude per band, mapped to a 0..1 dB scale."""
        cum = np.concatenate(([0.0], np.cumsum(spectrum)))
        total = cum[self._hi] - cum[self._lo]
        width = (self._hi - self._lo).astype(np.float64)
        mean = total / np.maximum(width, 1.0)
        db = (20.0 * np.log10(mean + 1e-10) + 10.0) / 50.0
        return np.clip(db, 0.0, 1.0)

    # ── main loop ──
    def _run(self) -> None:
        last_written = 0
        while self._running:
            written = self._ring.written
            if written - last_written < HOP:
                time.sleep(0.002)
                continue
            # if we fell behind, jump to now rather than grinding through backlog
            last_written = written

            try:
                self._analyse_once()
            except Exception:
                time.sleep(0.05)

    def _analyse_once(self) -> None:
        buf = self._ring.latest(FFT_SIZE)
        if buf is None:
            return

        sr = int(self._get_sr() or 48000)
        n = buf.shape[0]
        self._ensure_tables(sr, n)

        left = buf[:, 0].astype(np.float64)
        right = buf[:, 1].astype(np.float64)
        mono = (left + right) * 0.5

        rms = float(np.sqrt(np.mean(mono * mono))) + 1e-12
        now = time.monotonic()

        # Noise gate: a *fixed* absolute threshold, deliberately. An adaptive
        # floor collapses onto the music when playback is already running as
        # spektr starts, which shuts the gate on real audio. Real inputs leave a
        # wide gap between idle hiss and music, so a constant threshold
        # separates them cleanly and can never self-close.
        if rms > self._gate_abs:
            self._hold_until = now + self._gate_hold_s
        if now > self._hold_until:
            self._publish(Frame(seq=self._seq + 1, rms=rms, silent=True))
            return

        # Soft knee above the gate.
        #
        # A hard gate plus aggressive auto-gain is a bad pair: a signal one
        # part in a thousand above the threshold still gets multiplied by up to
        # _max_gain and drawn at full height. Idle hiss from a loopback tap sits
        # exactly there, and the display ends up dancing to the noise floor.
        # Fading in over the first couple of doublings above the gate means real
        # audio (which is orders of magnitude louder) is untouched, while
        # anything hugging the floor stays visibly small.
        knee = float(np.clip((rms / self._gate_abs - 1.0) / (_KNEE - 1.0), 0.0, 1.0))

        # Auto gain: monitor sources (Stereo Mix) can sit 60 dB below a loopback
        # tap. Track a decaying envelope and normalise against it so any source
        # lands in the same visual range.
        if rms > self._env:
            self._env = rms                       # jump up instantly
        else:
            self._env += (rms - self._env) * 0.02  # ease down slowly
        gain = self._target_rms / max(self._env, 1e-9)
        gain = max(1.0, min(self._max_gain, gain)) * self.sensitivity

        left = left * gain
        right = right * gain
        mono = mono * gain

        win = self._win
        spec_m = np.abs(np.fft.rfft(mono * win))
        spec_l = np.abs(np.fft.rfft(left * win))
        spec_r = np.abs(np.fft.rfft(right * win))

        step = max(1, n // WAVE_POINTS)
        wave = np.clip(mono[::step][:WAVE_POINTS], -1.5, 1.5) * knee
        stereo = np.stack(
            (
                np.clip(left[::step][:WAVE_POINTS], -1.5, 1.5),
                np.clip(right[::step][:WAVE_POINTS], -1.5, 1.5),
            ),
            axis=1,
        )

        self._publish(
            Frame(
                seq=self._seq + 1,
                bands=self._bands_from(spec_m) * knee,
                bands_l=self._bands_from(spec_l) * knee,
                bands_r=self._bands_from(spec_r) * knee,
                wave=wave,
                stereo=stereo,
                rms=rms,
                silent=False,
                confidence=knee,
            )
        )

    def _publish(self, frame: Frame) -> None:
        self._seq = frame.seq
        with self._lock:
            self._frame = frame


def resample_bands(bands: np.ndarray, n_out: int) -> np.ndarray:
    """Downsample the internal 32 bands to however many a mode wants to draw.

    Uses area-averaging rather than nearest-neighbour so that widening the
    terminal doesn't make bars pop between values.
    """
    n_in = len(bands)
    if n_out == n_in:
        return bands
    if n_out > n_in:
        x = np.linspace(0.0, n_in - 1, n_out)
        return np.interp(x, np.arange(n_in), bands)
    edges = np.linspace(0, n_in, n_out + 1)
    cum = np.concatenate(([0.0], np.cumsum(bands)))
    idx = edges.astype(np.int32)
    total = cum[idx[1:]] - cum[idx[:-1]]
    width = np.maximum(idx[1:] - idx[:-1], 1)
    return total / width
