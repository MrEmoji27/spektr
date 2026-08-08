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
#: frames read the same spectrum, and the waveform modes (Scope, Gonio, ECG)
#: draw a genuinely identical trace because they only step when
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

#: Analyses per second at a nominal 48 kHz. Only used to size the ceiling on
#: "unlimited" frame rate — past roughly twice this, consecutive frames read
#: the same spectrum and the extra ones are interpolation, not new audio. The
#: real device rate is whatever the capture reports; this is a planning figure.
ANALYSES_PER_SEC = 48000 / HOP
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

    # ── rhythm ──
    #
    # A *count*, not a flag, and that is the whole design. The analyser runs at
    # 187.5 Hz while the widget reads only the newest Frame per rendered frame;
    # at 60 fps that discards two analyses in every three. A boolean ``onset``
    # would therefore be missed most of the time it was true, and missed
    # differently at every frame rate — the exact class of bug the dt work went
    # in to kill. A monotonic counter cannot be missed: a reader compares it to
    # the last value it saw and learns how many beats happened in between, no
    # matter how long it looked away.
    #: Monotonic count of onsets detected since start. Never resets, including
    #: across silence — a reset would read as a burst of beats to anyone
    #: differencing it.
    onset_seq: int = 0
    #: 0..1 strength of the most recently detected onset.
    onset_strength: float = 0.0
    #: 0..1 raw onset-detection-function value for this hop, before peak
    #: picking. Continuous, so it is safe to read at any rate; useful for
    #: modes that want "how percussive is right now" rather than discrete hits.
    flux: float = 0.0
    #: Estimated tempo. 0.0 means unknown, which is a state modes must handle —
    #: never divide by this without checking.
    tempo_bpm: float = 0.0
    #: 0..1 position within the current beat, 0.0 on the beat. 0.0 whenever
    #: :attr:`tempo_bpm` is 0.0.
    beat_phase: float = 0.0

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


class OnsetDetector:
    """Turns a per-hop spectrum into discrete beats.

    Half-wave rectified spectral flux with an adaptive threshold. Flux, rather
    than level, is the whole reason this exists: three modes previously rolled
    their own beat detection out of ``ctx.energy`` — the mean band level — and
    a mean cannot tell a kick drum from the track simply getting louder. Put a
    sustained pad under a four-on-the-floor and an energy detector either fires
    continuously or not at all, because total energy barely dips between hits.
    Flux asks a different question: *how much of the spectrum just got louder
    than it was*. A steady pad contributes nothing to that however loud it is,
    because it is not changing. Only the attack does.

    Rectified because only increases are onsets. A note ending is a large
    spectral change and not a beat, so the negative half is discarded rather
    than taken as magnitude.

    Log compression before differencing. Raw magnitudes make the flux scale
    with absolute level, so a quiet passage detects nothing and a loud one
    detects everything; comparing log spectra measures *proportional* change,
    which is roughly what a listener hears and roughly level-independent.

    Deliberately a class rather than a function: this is stateful — previous
    spectrum, a moving threshold, a refractory clock — and hiding that in
    module globals would make two analysers in one process share it silently.
    """

    #: Shortest gap between two onsets, seconds. Below roughly this a single
    #: drum hit's attack and its body register as two events. 50 ms also caps
    #: detection at 1200 BPM, far above any real pulse.
    REFRACTORY_S = 0.05
    # Measured across 0.04-0.07 against the evaluation corpus: total F moved
    # by 0.004 and dense-material recall not at all, so this is the standard
    # value rather than a tuned one. The limit on fast material is PEAK_FRAC,
    # not this.

    #: Half-width of the peak-picking neighbourhood, in hops. A peak must be
    #: the largest flux value within this many hops either side. Costs that
    #: many hops of latency (5.3 ms each), which is the price of not reporting
    #: one drum hit as two.
    PEAK_SPAN = 4

    #: Seconds of flux history behind the adaptive threshold. Long enough to
    #: span a bar at slow tempi, short enough to follow a track that changes
    #: density. Too long and a quiet intro sets the bar for a loud chorus.
    HISTORY_S = 0.7

    #: Log compression constant. Standard in the onset literature; large
    #: enough that quiet spectral detail still registers.
    GAMMA = 1000.0

    #: Threshold is the running median plus this many mean-absolute-deviations.
    #: Median and MAD rather than mean and standard deviation because onsets
    #: are exactly the outliers being looked for, and they drag a mean upward
    #: until the detector stops being able to see them.
    K_MAD = 3.0

    #: A peak must also reach this fraction of the recent peak flux.
    #:
    #: Median-plus-deviation alone collapses on sparse percussive material. A
    #: four-on-the-floor kick is a short event in a mostly quiet bar, so the
    #: median flux sits near zero and the deviation with it, and the threshold
    #: goes to roughly nothing -- at which point every faint ripple between
    #: hits clears it. Measured: a spurious detection 278 ms after every kick,
    #: in the same place every bar, giving exactly two detections per beat.
    #: Requiring a candidate to be a real fraction of how hard things have
    #: recently hit is what a bare median cannot express.
    PEAK_FRAC = 0.65

    def __init__(self) -> None:
        self.seq = 0
        self.strength = 0.0
        self.flux = 0.0
        self.tempo_bpm = 0.0
        self.beat_phase = 0.0

        self._level = 0.0
        self._prev: np.ndarray | None = None
        self._hist: np.ndarray | None = None      # circular flux history
        self._hi = 0
        self._filled = 0
        #: last three (time, flux) samples — a peak is picked from the middle
        #: one, so detection lags by a single hop (~5.3 ms) in exchange for
        #: not firing on the leading edge of every slow swell.
        self._win: list[tuple[float, float]] = []
        self._peak = 1e-9
        self._last_t = -1e9
        self._beats: list[float] = []
        self._span = self.PEAK_SPAN * 2 + 1

    def feed(self, spectrum: np.ndarray, now: float) -> None:
        """Consume one hop's magnitude spectrum, taken at time ``now``.

        ``now`` is passed rather than read from a clock here so the caller
        owns the timeline — an offline harness drives this on the signal's
        clock, and onsets are judged to within tens of milliseconds.
        """
        # Normalise before compressing, or the compression is not compression.
        #
        # log1p(GAMMA * x) only behaves logarithmically once GAMMA * x is
        # comfortably above 1; below that it is very nearly linear. So the same
        # code measures *relative* spectral change at ordinary listening levels
        # and *absolute* change on a quiet track — two different detectors
        # chosen by volume, which is the opposite of the level independence
        # this was supposed to provide. Measured on a track at -40 dB:
        # precision 0.06, sixteen false positives and fifteen misses in one
        # scenario, while the identical material at full level scored 0.91.
        #
        # Dividing by a slow running mean of the spectrum's own magnitude puts
        # the input to log1p in the same range whatever the track's level. The
        # follower is deliberately sluggish: it should track "how loud is this
        # recording" and not "how loud is this beat", or it would normalise
        # away the very transients being detected.
        total = float(spectrum.sum()) / max(spectrum.size, 1)
        if self._level <= 0.0:
            self._level = max(total, 1e-12)
        else:
            self._level += (total - self._level) * 0.002
        cur = np.log1p(self.GAMMA * spectrum / max(self._level, 1e-12))

        if self._prev is None or self._prev.shape != cur.shape:
            # No previous spectrum to difference against, or the window
            # changed size under us (a sample-rate change rebuilds the plan).
            # Either way this hop has no meaningful flux.
            self._prev = cur
            self.flux = 0.0
            return

        diff = cur - self._prev
        self._prev = cur
        np.maximum(diff, 0.0, out=diff)
        raw = float(diff.sum()) / diff.size

        # Peak decays so ``strength`` means "hard, for this passage" rather
        # than "hard, compared with the loudest thing since startup". Half a
        # second-ish of memory: 0.996 per hop at 187.5 Hz is ~0.47 a second.
        self._peak = max(raw, self._peak * 0.996)
        self.flux = min(1.0, raw / self._peak) if self._peak > 0 else 0.0

        if self._hist is None:
            n = max(8, int(self.HISTORY_S * 187.5))
            self._hist = np.zeros(n, dtype=np.float64)
            self._hi = 0
            self._filled = 0

        if not self._win:
            # Seed the left half of the peak-picking neighbourhood with
            # silence rather than waiting for it to refill.
            #
            # A candidate cannot be judged until there is history either side
            # of it, so an empty window costs PEAK_SPAN hops before anything
            # can fire. That is invisible in continuous music and fatal for
            # sparse percussion: the noise gate shuts between widely spaced
            # hits, and the next attack arrives *during* the refill and is
            # never tested. Measured on an impulse track at 120 BPM — sixteen
            # gate closures for sixteen beats, and none of the sixteen
            # detected.
            #
            # Seeding zeros is not a trick to dodge the wait; it is what the
            # recent past actually contained. The gate was shut, so the flux
            # was nothing, and a transient arriving now genuinely does tower
            # over the silence behind it.
            step = 1.0 / 187.5
            self._win = [
                (now - (self.PEAK_SPAN - k) * step, 0.0)
                for k in range(self.PEAK_SPAN)
            ]

        self._win.append((now, raw))
        if len(self._win) > self._span:
            self._win.pop(0)

        if self._filled >= 8 and len(self._win) == self._span:
            hist = self._hist[: self._filled]
            med = float(np.median(hist))
            mad = float(np.mean(np.abs(hist - med)))
            thresh = max(med + self.K_MAD * mad, self.PEAK_FRAC * self._peak)

            t_mid, b = self._win[self.PEAK_SPAN]
            # A peak has to dominate a whole neighbourhood, not merely beat its
            # two neighbours. The analysis window is 4096 samples -- 85 ms --
            # so a single drum hit slides through it over many hops and its
            # flux forms a broad hump, not a spike. Testing only the adjacent
            # pair finds several maxima inside one hump and reports one hit as
            # two: measured precision 0.49 against recall 0.97, almost exactly
            # double-counting.
            #
            # Strictly greater on the left and not less on the right, so a
            # plateau resolves to its first sample rather than firing twice.
            left = all(b > v for _, v in self._win[: self.PEAK_SPAN])
            right = all(b >= v for _, v in self._win[self.PEAK_SPAN + 1 :])
            if left and right and b > thresh and t_mid - self._last_t >= self.REFRACTORY_S:
                self._last_t = t_mid
                self.seq += 1
                self.strength = min(1.0, b / self._peak) if self._peak > 0 else 0.0
                self._note_beat(t_mid)

        self._hist[self._hi] = raw
        self._hi = (self._hi + 1) % len(self._hist)
        self._filled = min(self._filled + 1, len(self._hist))

        self._update_phase(now)

    # ── tempo ──
    def _note_beat(self, t: float) -> None:
        self._beats.append(t)
        if len(self._beats) > 48:
            del self._beats[:-48]

    def _update_phase(self, now: float) -> None:
        """Estimate tempo from inter-onset intervals, and phase from the last.

        Deliberately conservative. A histogram of gaps between recent onsets
        is a crude tempo tracker — it has no notion of metre and will happily
        lock to eighth notes — so it reports 0.0 (unknown) unless a clear
        plurality of intervals agree. Saying nothing is much better than
        handing a mode a confident wrong number to sync animation to.
        """
        if len(self._beats) < 6:
            self.tempo_bpm = 0.0
            self.beat_phase = 0.0
            return

        gaps = np.diff(np.asarray(self._beats[-24:]))
        gaps = gaps[(gaps > 0.25) & (gaps < 2.0)]      # 30..240 BPM
        if gaps.size < 4:
            self.tempo_bpm = 0.0
            self.beat_phase = 0.0
            return

        # Cluster around the median gap rather than averaging: a missed beat
        # produces a double-length interval, and a mean quietly splits the
        # difference between right and twice-right.
        med = float(np.median(gaps))
        near = gaps[np.abs(gaps - med) < med * 0.18]
        if near.size < max(3, int(gaps.size * 0.5)):
            self.tempo_bpm = 0.0
            self.beat_phase = 0.0
            return

        period = float(near.mean())
        # Fold into a musically plausible range. A detector that fires on
        # every eighth note is not wrong about the music, but 240 BPM is the
        # wrong number to hand something trying to pulse on the beat.
        while period < 0.4:
            period *= 2.0
        while period > 1.2:
            period *= 0.5

        self.tempo_bpm = 60.0 / period
        self.beat_phase = float(((now - self._last_t) / period) % 1.0)

    def reset_continuity(self) -> None:
        """Called when the input goes silent.

        Drops the history used for differencing without touching :attr:`seq`.
        The first hop after a gap must not be compared against the spectrum
        from before it — that difference is the size of the whole silence and
        would fire a spurious onset on every un-pause. The count itself has to
        survive, or a reader differencing it sees the gap as beats.
        """
        self._prev = None
        self._win.clear()
        self.flux = 0.0
        self.strength = 0.0
        self.beat_phase = 0.0


class Analyser:
    """Overlapped FFT running independently of the render loop."""

    def __init__(self, ring: RingBuffer, samplerate_getter, clock=time.monotonic):
        self._ring = ring
        self._get_sr = samplerate_getter
        self._onset = OnsetDetector()
        # Injectable so an offline harness can drive the analyser on the
        # *signal's* timeline rather than the wall's. Onset detection is
        # judged on timing to within tens of milliseconds, and a corpus that
        # renders twenty seconds of audio in two hundred milliseconds of CPU
        # would otherwise have every event compared against the wrong clock.
        self._clock = clock
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
        now = self._clock()

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
            # The onset counter is carried through silence rather than left at
            # its default. Publishing 0 here would look like the count going
            # backwards, and anyone differencing it across the gap would read
            # the recovery as a burst of beats that never played.
            self._onset.reset_continuity()
            self._publish(Frame(
                seq=self._seq + 1, rms=rms, silent=True,
                bands=quiet, bands_l=quiet, bands_r=quiet,
                onset_seq=self._onset.seq,
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

        # Rhythm comes off the band sums here, and the position in the chain is
        # the point.
        #
        # *After* the eq, because that is a fixed per-band constant: it cannot
        # create change where there was none, and it lifts the treble where
        # transients actually live.
        #
        # *Before* autosens, because that is not constant. It walks up and down
        # continuously to keep the display in range, and a detector watching a
        # signal multiplied by a moving number sees change that is not in the
        # music.
        #
        # And band sums rather than the raw spectrum, which was the first
        # version and was wrong. Differencing 2049 FFT bins meant ~2000 of them
        # sat at the noise floor, where log1p(1000 x) amplifies numerical
        # jitter enormously, and summing that much rectified noise swamped the
        # few bins carrying an actual attack. A dead-steady drone scored 72
        # onsets. Thirty-two band sums average that jitter away, and cost less.
        self._onset.feed(raw_l + raw_r, now)

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
                onset_seq=self._onset.seq,
                onset_strength=self._onset.strength,
                flux=self._onset.flux,
                tempo_bpm=self._onset.tempo_bpm,
                beat_phase=self._onset.beat_phase,
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
