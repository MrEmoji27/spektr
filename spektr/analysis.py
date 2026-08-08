"""Spectrum analysis, on its own clock, with cava's band distribution.

Two independent problems are solved here, and it is worth keeping them apart.

**Timing.** The old code called ``_analyse()`` from inside the frame timer, so a
2048-sample block at 48 kHz (23.4 blocks/sec) was being sampled by a 30-45 fps
render loop. A third of frames re-read a buffer they had already seen, and when
the adaptive pacer sped up, blocks were dropped outright. The result was
beat-rate aliasing — bands appearing to stall and then jump — which no amount of
easing can hide, because the target sequence itself is stepped. So the analyser
owns a thread and runs on a fixed hop, faster than any sane frame rate.

**Band distribution.** The first version spread 32 log-spaced bands over
20 Hz-20 kHz and read them from a single 2048-point FFT. At 48 kHz that is
23.4 Hz per bin, which means the bands below ~90 Hz were all reading the *same
two bins*: five of the thirty-two were exact duplicates of an earlier band and
nine were under two bins wide. The bottom quarter of the display was one number
drawn four times, which is why the bass moved as a slab. Meanwhile three bands
sat above 10 kHz, where music has almost nothing, so the right-hand end never
moved.

cava (github.com/karlstav/cava) solved this years ago, and the parts worth
taking are its band plan and its sensitivity loop:

* **two FFT sizes** — a long window for the bass, where frequency resolution
  matters and time resolution does not, and a shorter one above 100 Hz, where
  the reverse is true. At 48 kHz that is 8192 (5.9 Hz per bin) and 4096.
* **50 Hz to 10 kHz**, not 20 Hz to 20 kHz. Both ends of the wider range are
  places where music is silent and hardware is noisy.
* **strictly disjoint bin ranges** — where the exponential distribution clumps
  two bars onto one bin, the plan pushes the later bar up by one rather than
  letting them read the same data.
* **an eq tilt of f^0.85** — spectra fall off with frequency, so without a tilt
  the treble bars barely leave the floor.
* **sensitivity from overshoot, not loudness.** This is the subtle one. The old
  auto-gain normalised on frame RMS, so a kick raised the RMS, which lowered
  the gain, which shrank *the whole display on the beat* — backwards. cava never
  looks at loudness: it scales down 2% per frame while any bar is clipping and
  creeps up 0.1% per frame when none is.

What is deliberately not taken is cava's smoothing. Its gravity falloff and
integral filter are framerate-dependent by construction (``framerate_mod =
66 / framerate``), and spektr already solved that with a spring integrated in
seconds — see :mod:`spektr.motion`.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .capture import RingBuffer

N_BANDS = 32          # internal resolution; modes downsample for chunky looks
#: Analysis stride — ~188 analyses/sec at 48 kHz.
#:
#: This was 512, i.e. ~94/sec, which put a hard ceiling on how much of the
#: audio the picture could actually resolve: above that frame rate consecutive
#: frames read the same spectrum, and the waveform modes (Scope, Gonio, ECG,
#: Cassette) draw a genuinely identical trace because they only step when
#: ``frame.seq`` changes.
#:
#: Halving it costs no frequency resolution — that is set by the *window*
#: (``bass_size``/``mid_size``), not the hop, so the extra work is redundant
#: overlap and nothing else. What it buys beyond frame headroom is onset
#: precision: every hit detector in the codebase runs a ~30 ms fast envelope
#: (``ctx.dt / 0.03``), which contained about three analyses at 94 Hz and now
#: contains six, so transients land sooner and more sharply.
#:
#: Cost is linear in the rate: measured at 0.447 ms per analysis, 4.2% of one
#: core at 94 Hz and ~8.4% at 188 Hz.
HOP = 256
WAVE_POINTS = 512     # downsampled scope trace

#: cava's defaults, and they are the right ones. Below 50 Hz you are looking at
#: room rumble and DC offset; above 10 kHz there is nothing in most material
#: loud enough to move a bar.
LOW_CUT_HZ = 50.0
HIGH_CUT_HZ = 10000.0

#: Bands whose lower edge falls below this are read from the long window.
BASS_CUT_HZ = 100.0

#: cava works in the amplitude range of 16-bit PCM and its eq constant (1/2^28)
#: is scaled for that. spektr's samples are floats in ±1, so they are scaled
#: into cava's units rather than re-deriving all of its constants.
_PCM_SCALE = 32768.0

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


# ── band plan ────────────────────────────────────────────────────────────────

def _fft_sizes(rate: int) -> tuple[int, int]:
    """cava's window sizes for a sample rate: (bass, mid+treble).

    The steps are cava's, so the plan matches bar for bar. 48 kHz and 44.1 kHz
    both land on 8192/4096 — 5.9 Hz and 11.7 Hz per bin respectively, against
    the 23.4 Hz of the single 2048 window this replaces.
    """
    size = 512
    if 8125 < rate <= 16250:
        size *= 2
    elif 16250 < rate <= 32500:
        size *= 4
    elif 32500 < rate <= 75000:
        size *= 8
    elif 75000 < rate <= 150000:
        size *= 16
    elif 150000 < rate <= 300000:
        size *= 32
    elif rate > 300000:
        size *= 64
    return size * 2, size


class BandPlan:
    """Which FFT bins each bar reads, and what to multiply the sum by.

    A near-line-for-line port of the loop in cava's ``cava_init``. It is written
    out longhand rather than vectorised on purpose: the interesting part is the
    corrections — the clump push-up, the hand-off between the two windows — and
    those are sequential by nature. It runs once per sample rate.
    """

    __slots__ = ("bass_bar", "bass_size", "cutoff", "eq", "lower", "mid_size", "rate", "upper")

    def __init__(self, rate: int, bars: int = N_BANDS,
                 low: float = LOW_CUT_HZ, high: float = HIGH_CUT_HZ,
                 bass_cut: float = BASS_CUT_HZ):
        high = min(high, rate / 2 - 1)
        self.rate = rate
        self.bass_size, self.mid_size = _fft_sizes(rate)
        bass_half, mid_half = self.bass_size // 2, self.mid_size // 2

        lower = [0] * (bars + 1)
        upper = [0] * (bars + 1)
        cutoff = [0.0] * (bars + 1)
        rel = [0.0] * (bars + 1)

        # distributes the bars exponentially between the two cut-offs
        frequency_constant = math.log10(low / high) / (1.0 / (bars + 1) - 1.0)
        min_bandwidth = rate / self.bass_size
        bass_bar = 0
        first_bar = True

        for n in range(bars + 1):
            coeff = -frequency_constant + (n + 1) / (bars + 1) * frequency_constant
            cutoff[n] = high * 10.0 ** coeff
            if n > 0 and cutoff[n - 1] >= cutoff[n]:
                cutoff[n] = cutoff[n - 1] + min_bandwidth

            rel[n] = cutoff[n] / (rate / 2)          # remember nyquist

            if cutoff[n] < bass_cut:
                lower[n] = min(int(rel[n] * bass_half), bass_half)
                bass_bar += 1
                if bass_bar > 1:
                    first_bar = False
            else:
                lower[n] = min(int(math.ceil(rel[n] * mid_half)), mid_half)
                if n == bass_bar:
                    # the hand-off: this bar is the first on the short window,
                    # so the previous bar's upper edge still belongs to the long
                    # one and has to be expressed in its bins
                    first_bar = True
                    if n > 0:
                        upper[n - 1] = int(rel[n] * bass_half) - 1
                else:
                    first_bar = False

            if n > 0:
                if not first_bar:
                    upper[n - 1] = lower[n] - 1

                    # Where the exponential distribution clumps — several bars
                    # landing on one bin down in the bass — push each bar up to
                    # the next free bin instead of letting them read identical
                    # data. This is the fix for the duplicate-band problem.
                    if lower[n] <= lower[n - 1]:
                        half = bass_half if n < bass_bar else mid_half
                        if lower[n - 1] + 1 < half + 1:
                            lower[n] = lower[n - 1] + 1
                            upper[n - 1] = lower[n] - 1
                else:
                    if upper[n - 1] < lower[n - 1]:
                        upper[n - 1] = lower[n - 1] + 1

            # the cut-off actually achieved, after all of the above
            half = bass_half if n < bass_bar else mid_half
            rel[n] = lower[n] / half
            cutoff[n] = rel[n] * (rate / 2)

        self.bass_bar = bass_bar
        self.lower = np.array(lower[:bars], dtype=np.int64)
        self.upper = np.array(upper[:bars], dtype=np.int64)
        self.cutoff = np.array(cutoff, dtype=np.float64)

        # cava's hard-coded eq. The 1/2^28 normalises the raw FFT magnitudes;
        # f^0.85 is the treble boost that stops the top bars flatlining; the
        # last two divisions make bars comparable across window sizes and bin
        # counts. The exponent is the one number here you might taste-tune.
        eq = np.full(bars, 1.0 / 2.0 ** 28)
        eq *= self.cutoff[1:bars + 1] ** 0.85
        sizes = np.where(np.arange(bars) < bass_bar, self.bass_size, self.mid_size)
        eq /= np.log2(sizes)
        eq /= (self.upper - self.lower + 1)
        self.eq = eq

    def describe(self) -> str:
        lines = [f"  {'bar':>3}  {'range':>15}  {'window':>6}  {'bins':>11}  {'width':>5}"]
        for n in range(len(self.lower)):
            window = "bass" if n < self.bass_bar else "mid"
            lines.append(
                f"  {n:>3}  {self.cutoff[n]:>6.0f}-{self.cutoff[n + 1]:<6.0f} Hz  {window:>6}  "
                f"{self.lower[n]:>4}-{self.upper[n]:<4}  {self.upper[n] - self.lower[n] + 1:>5}"
            )
        return "\n".join(lines)


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

        self._plan: BandPlan | None = None
        self._plan_key = ()
        self._bass_win: np.ndarray | None = None
        self._mid_win: np.ndarray | None = None
        #: How many bands to resolve. Raising this past the default rebuilds
        #: the plan so the extra bars are real bin ranges; lowering it is left
        #: to the modes, which resample and can do it without a rebuild.
        self._bars = N_BANDS

        # ── cava's autosens ──
        #: Multiplies every bar. Moves down fast while anything is clipping and
        #: up slowly when nothing is, so it settles wherever the material sits.
        self._sens = 1.0
        #: Until the first clip, climb ten times faster — this is what makes a
        #: quiet source reach full height in a second instead of two minutes.
        self._sens_init = True

        # ── loudness handling, for the waveform only ──
        # The band path no longer uses this. The scope modes still need a quiet
        # source scaled up to a visible trace, and that is a time-domain
        # question with a different answer to the spectrum's.
        self._env = 1e-4
        self._target_rms = 0.06
        self._max_gain = 1200.0
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
        return self.set_gate(self._gate_abs * factor)

    def set_gate(self, value: float) -> float:
        self._gate_abs = max(1e-6, min(2e-3, float(value)))
        return self._gate_abs

    @property
    def gate(self) -> float:
        return self._gate_abs

    @property
    def plan(self) -> BandPlan:
        """The band plan for the current sample rate, built on demand."""
        self._ensure_plan(int(self._get_sr() or 48000))
        return self._plan

    def set_bands(self, n: int) -> int:
        """Ask for ``n`` resolved bands. Returns what was actually taken.

        Below the native 32 there is nothing to gain from a rebuild — a mode
        asking for 12 bars resamples 32 down and gets the same picture — so the
        plan stays put and only the drawing changes. Above it, the plan is
        rebuilt so each extra bar is its own range of FFT bins rather than an
        interpolated copy of its neighbour, which is the whole difference
        between more bars and more detail.
        """
        want = max(N_BANDS, min(256, int(n)))
        if want != self._bars:
            self._bars = want
            self._plan = None          # rebuilt on the next analysis
        return self._bars

    @property
    def bands(self) -> int:
        return self._bars

    def _ensure_plan(self, sr: int) -> None:
        key = (sr, self._bars)
        if self._plan is not None and self._plan_key == key:
            return
        self._plan = BandPlan(sr, bars=self._bars)
        self._plan_key = key
        # periodic Hann, matching cava's 0.5*(1-cos(2*pi*i/(N-1)))
        self._bass_win = np.hanning(self._plan.bass_size)
        self._mid_win = np.hanning(self._plan.mid_size)
        self._sens = 1.0
        self._sens_init = True

    # ── main loop ──
    def _run(self) -> None:
        last_written = 0
        while self._running:
            written = self._ring.written
            if written - last_written < HOP:
                # Poll finer than the hop period, which at HOP=256/48 kHz is
                # 5.3 ms. The old 2 ms sleep was a fifth of a 10.7 ms period
                # and is over a third of this one — enough quantisation to
                # show up as jitter in the analysis interval.
                time.sleep(0.001)
                continue
            # if we fell behind, jump to now rather than grinding through backlog
            last_written = written

            try:
                self._analyse_once()
            except Exception:
                time.sleep(0.05)

    def _band_sums(self, spec: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Sum the magnitudes in each bar's bin range, inclusive of both ends.

        A running sum turns 32 slices into two gathers. cava loops; at 94
        analyses a second and three signals, the loop is worth avoiding.
        """
        cum = np.concatenate(([0.0], np.cumsum(spec)))
        return cum[np.minimum(upper + 1, len(spec))] - cum[lower]

    def _analyse_once(self) -> None:
        sr = int(self._get_sr() or 48000)
        self._ensure_plan(sr)
        plan = self._plan

        buf = self._ring.latest(plan.bass_size)
        if buf is None:
            return

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
            # zeros at the *current* band count: a silent frame that is a
            # different length to a live one would resize the springs on every
            # pause
            quiet = np.zeros(self._bars)
            self._publish(Frame(
                seq=self._seq + 1, rms=rms, silent=True,
                bands=quiet, bands_l=quiet, bands_r=quiet,
            ))
            return

        # Soft knee above the gate.
        #
        # A hard gate plus aggressive gain is a bad pair: a signal one part in a
        # thousand above the threshold still gets scaled up and drawn at full
        # height. Idle hiss from a loopback tap sits exactly there, and the
        # display ends up dancing to the noise floor. Fading in over the first
        # couple of doublings above the gate means real audio (which is orders
        # of magnitude louder) is untouched, while anything hugging the floor
        # stays visibly small.
        knee = float(np.clip((rms / self._gate_abs - 1.0) / (_KNEE - 1.0), 0.0, 1.0))

        # ── spectra ──
        # Both windows end at the newest sample: the long one reaches further
        # back for bass resolution, the short one stays responsive up top.
        bass_l = left * _PCM_SCALE
        bass_r = right * _PCM_SCALE
        mid_l = bass_l[-plan.mid_size:]
        mid_r = bass_r[-plan.mid_size:]

        spec_bass_l = np.abs(np.fft.rfft(bass_l * self._bass_win))
        spec_bass_r = np.abs(np.fft.rfft(bass_r * self._bass_win))
        spec_mid_l = np.abs(np.fft.rfft(mid_l * self._mid_win))
        spec_mid_r = np.abs(np.fft.rfft(mid_r * self._mid_win))

        cut = plan.bass_bar
        lower, upper = plan.lower, plan.upper

        # sized from the plan, not the module constant — the band count is
        # settable and the plan is the only thing that knows the current one
        raw_l = np.empty(len(lower))
        raw_r = np.empty(len(lower))
        if cut:
            raw_l[:cut] = self._band_sums(spec_bass_l, lower[:cut], upper[:cut])
            raw_r[:cut] = self._band_sums(spec_bass_r, lower[:cut], upper[:cut])
        raw_l[cut:] = self._band_sums(spec_mid_l, lower[cut:], upper[cut:])
        raw_r[cut:] = self._band_sums(spec_mid_r, lower[cut:], upper[cut:])

        raw_l *= plan.eq
        raw_r *= plan.eq

        # ── cava's autosens ──
        # Judged before the manual trim, so pressing ] actually makes the bars
        # taller instead of being cancelled out over the next second. That is a
        # deliberate departure: cava has no manual trim to fight with.
        scaled_l = raw_l * self._sens
        scaled_r = raw_r * self._sens
        overshoot = bool(scaled_l.max() > 1.0 or scaled_r.max() > 1.0)

        rate = sr / HOP                        # analyses per second
        framerate_mod = 66.0 / rate
        if overshoot:
            self._sens *= 1.0 - 0.02 * framerate_mod
            self._sens_init = False
        else:
            self._sens *= 1.0 + 0.001 * framerate_mod
            if self._sens_init:
                self._sens *= 1.0 + 0.1 * framerate_mod

        trim = self.sensitivity * knee
        bands_l = np.clip(scaled_l * trim, 0.0, 1.0)
        bands_r = np.clip(scaled_r * trim, 0.0, 1.0)
        # Mono is the mean of the two channels' magnitudes rather than the
        # spectrum of L+R: summing in the time domain cancels out-of-phase
        # content, which would make a wide stereo mix look thinner than it is.
        bands = (bands_l + bands_r) * 0.5

        # ── waveform ──
        # Still normalised on a loudness envelope, because a scope trace is
        # about amplitude over time and has no bars to overshoot.
        if rms > self._env:
            self._env = rms                       # jump up instantly
        else:
            self._env += (rms - self._env) * 0.02  # ease down slowly
        gain = self._target_rms / max(self._env, 1e-9)
        gain = max(1.0, min(self._max_gain, gain)) * self.sensitivity

        tail = plan.mid_size
        step = max(1, tail // WAVE_POINTS)
        wave = np.clip(mono[-tail:][::step][:WAVE_POINTS] * gain, -1.5, 1.5) * knee
        stereo = np.stack(
            (
                np.clip(left[-tail:][::step][:WAVE_POINTS] * gain, -1.5, 1.5),
                np.clip(right[-tail:][::step][:WAVE_POINTS] * gain, -1.5, 1.5),
            ),
            axis=1,
        )

        self._publish(
            Frame(
                seq=self._seq + 1,
                bands=bands,
                bands_l=bands_l,
                bands_r=bands_r,
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
