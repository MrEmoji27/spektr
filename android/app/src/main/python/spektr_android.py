"""The Python half of the Android port: one object Kotlin talks to.

Kotlin owns the audio and the view; everything between those two is the
desktop engine running unmodified. This module is the only new Python in the
port, and it exists to keep it that way — it adapts the engine's shape to a
single call per frame rather than forking anything to suit Android.

Three things it has to reconcile:

**The analyser pulls, Kotlin pushes.** ``Analyser`` runs its own thread and
reads from a ``RingBuffer`` whenever it likes. Android's ``AudioRecord`` hands
us buffers on its own schedule instead. Both sides agree on the ring, so the
adaptation is just ``push()`` here and the analyser thread on the other side,
exactly as on desktop where the capture thread does the same job.

**Modes may return two arrays or three.** A mode drawing through braille
returns ``(codes, cidx)``; one drawing through the half-block ``▀`` trick
returns ``(codes, cidx, bidx)`` and needs a background colour per cell too.
The design document predates the third array and describes a two-array
contract; packing only two would silently drop the background of every
half-block mode, which is now several of the best-looking ones. So the wire
format carries a plane count and Kotlin honours it.

**One crossing per frame, not three.** Every JNI crossing costs, and the port's
main measured risk is this boundary. Returning three Python lists would cross
three times and convert each element; instead everything is packed into one
``bytes`` that Kotlin wraps and reads directly.
"""

from __future__ import annotations

import struct
import time

import numpy as np

from spektr.analysis import Analyser, N_BANDS
from spektr.capture import RingBuffer
from spektr.modes import MODES, Ctx
from spektr.motion import Peaks, Spring, Trace
from spektr.palette import BUILTIN, Palette

#: Wire format version. Kotlin refuses a buffer it does not recognise rather
#: than reading a stale layout as though it were current.
WIRE_VERSION = 1

#: How much of the ramp's low end OLED mode fades into true black, as a
#: fraction of the ramp. An OLED pixel showing #000000 is *off* — that is the
#: whole point of the panel — and a theme whose darkest step is #1d2021 lights
#: every cell of a mostly-dark picture to show something indistinguishable
#: from black at arm's length. Fading the bottom third rather than only index
#: 0 is what makes the difference visible: the dim end is where nearly all the
#: area of a dark mode lives.
_OLED_KNEE = 0.35


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def _blacken(hexes: list[str]) -> list[str]:
    """Fade the bottom of a ramp to true black, in linear light.

    Linear rather than on the hex values: scaling sRGB bytes darkens the
    midtones far more than it darkens the bottom, so the ramp loses its shape
    instead of just losing its floor. The hues are untouched — this only takes
    light away, and only where the ramp was nearly black already.
    """
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=np.float64)
    lin = _to_linear(rgb / 255.0)
    n = len(hexes)
    fade = np.clip(np.arange(n) / max(1.0, n * _OLED_KNEE), 0.0, 1.0)
    out = np.clip(np.rint(_to_srgb(lin * fade[:, None]) * 255.0), 0, 255).astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in out]


#: ``codes`` are Unicode codepoints and run past U+FFFF (braille sits at
#: U+2800, the block elements lower), so the grid is int32 rather than a
#: narrower type. ``cidx``/``bidx`` are ramp indices bounded by 64 (asserted in
#: tests/bench.py), so a byte each is enough and keeps the buffer small.
_HEADER = struct.Struct("<4sHHHH")   # magic, version, planes, w, h
_MAGIC = b"SPKT"


class Engine:
    """One per app. Holds the ring, the analyser and the mode's scratch state."""

    def __init__(self, samplerate: int = 48000, bars: int = N_BANDS) -> None:
        # Two seconds of stereo headroom. The analyser only ever asks for the
        # most recent window, so this is slack against scheduler jitter rather
        # than a queue — if Android stalls us we drop old audio, which is the
        # right failure for a visualiser.
        self._ring = RingBuffer(samplerate * 2)
        self._sr = samplerate
        self._analyser = Analyser(self._ring, lambda: self._sr)
        self._analyser.set_bands(bars)
        self._analyser.start()

        self._t0 = time.monotonic()
        self._last_t = self._t0
        self._frame = 0
        self._state: dict = {}
        self._mode_name: str | None = None
        self._modes = {m.name: m for m in MODES}
        self._theme = BUILTIN["gruvbox"]
        self._palette = Palette(self._theme)
        self._swatches: list[list[str]] | None = None
        self._oled = False

        # Debug counters; see stats().
        self._stats_t0 = self._t0
        self._stats_frames = 0
        self._stats_dt = 0.0
        self._stats_energy = 0.0
        self._stats_onsets = 0
        self._stats_band_peak = 0.0
        self._stats_sample_peak = 0.0

        # The motion layer, which is not in the analyser and not in the modes.
        #
        # ``Analyser`` publishes a raw ``Frame``: bands as measured, and no
        # peaks, energy or smoothed trace at all. Everything that makes the
        # picture move like the desktop app lives between the two, in the
        # widget — and a port that skips it does not fail, it just renders a
        # jittery version of the same modes and looks subtly wrong forever.
        # Same objects and same constants as ``AudioVisualizer.__init__``.
        self._bars = bars
        self._spring = Spring(bars)
        self._peaks = Peaks(bars)
        self._stereo_l = Spring(bars)
        self._stereo_r = Spring(bars)
        self._trace = Trace(tau=0.028)
        #: Last analyser sequence the trace was stepped for. The wave is only
        #: advanced on a genuinely new block, exactly as the widget does it.
        self._last_seq = -1
        #: Onset counter as of the previous rendered frame, for ``ctx.onsets``.
        self._last_onset_seq = 0

    # ── what the engine is actually seeing ──
    def stats(self) -> list[float]:
        """Counters since the last call, for the debug build's log line.

        A port has no window onto itself. Everything about how a mode looks —
        whether it launches too often, whether it moves too fast — is decided
        by numbers that live on this side of the boundary, and on a tablet the
        only way to read them is a log line. So they are collected here rather
        than inferred from the picture.

        ``[fps, mean dt ms, mean energy, onsets/s, peak band, peak sample]``.
        Resets on read, so each line describes its own interval.
        """
        now = time.monotonic()
        span = max(1e-6, now - self._stats_t0)
        n = max(1, self._stats_frames)
        out = [
            self._stats_frames / span,
            self._stats_dt / n * 1000.0,
            self._stats_energy / n,
            self._stats_onsets / span,
            self._stats_band_peak,
            self._stats_sample_peak,
        ]
        self._stats_t0 = now
        self._stats_frames = 0
        self._stats_dt = 0.0
        self._stats_energy = 0.0
        self._stats_onsets = 0
        self._stats_band_peak = 0.0
        self._stats_sample_peak = 0.0
        return [float(v) for v in out]

    # ── audio in ──
    def push(self, pcm: bytes, channels: int = 2) -> None:
        """Hand one AudioRecord buffer to the analyser.

        Float32 little-endian, interleaved, which is what
        ``AudioFormat.ENCODING_PCM_FLOAT`` gives us. Mono is widened rather
        than special-cased downstream: the engine's stereo modes expect two
        columns and would otherwise need an Android-only branch.
        """
        buf = np.frombuffer(pcm, dtype="<f4")
        if channels == 1:
            buf = np.repeat(buf[:, None], 2, axis=1)
        else:
            buf = buf.reshape(-1, 2)
        if buf.size:
            self._stats_sample_peak = max(
                self._stats_sample_peak, float(np.abs(buf).max())
            )
        self._ring.push(buf)

    # ── configuration ──
    def set_theme(self, name: str) -> bool:
        spec = BUILTIN.get(name)
        if spec is None:
            return False
        self._theme = spec
        self._palette = Palette(spec)
        return True

    # ── colours, as flat lists ──
    #
    # Kotlin used to reach across and read ``BUILTIN[name]`` and
    # ``Palette.hexes`` itself. It cannot: Chaquopy's ``PyObject.get`` is
    # *attribute* access, so ``BUILTIN.get("gruvbox")`` asked a dict for an
    # attribute named gruvbox, got null, and the ``!!`` after it threw a
    # NullPointerException before the first frame — which is exactly what the
    # tablet showed. Subscripting a dict from Kotlin needs an explicit
    # ``__getitem__`` call, and code that has to know that to be correct is
    # code that belongs on this side of the boundary.
    #
    # Both return plain lists of ``#rrggbb`` strings. Not dicts: reading a dict
    # from Kotlin lands on the same attribute-versus-item trap. A list crosses
    # as a list and is unambiguous.

    def ramp_hexes(self) -> list[str]:
        """The ramp, in index order — what a cell's colour index selects."""
        return list(self._palette.hexes)

    def chrome_hexes(self) -> list[str]:
        """``[background, foreground]`` for the app's own chrome."""
        return [self._theme.bg, self._theme.fg]

    def colours(self) -> list[str]:
        """``[bg, fg, *ramp]`` as Kotlin should draw them, OLED applied.

        The modes never see this. They emit ramp *indices* and Kotlin turns
        those into colours, so the whole OLED treatment is a remap of this one
        list — no mode changes, no engine fork, and nothing for the desktop to
        carry.
        """
        if not self._oled:
            return [self._theme.bg, self._theme.fg, *self._palette.hexes]
        return ["#000000", self._theme.fg, *_blacken(list(self._palette.hexes))]

    def use_theme(self, name: str, oled: bool = False) -> list[str] | None:
        """Switch theme and hand back every colour Kotlin needs, in one call.

        ``[bg, fg, *ramp]``, or ``None`` for a theme that does not exist —
        which is what a config saved by a newer build looks like to an older
        one. Combined rather than ``set_theme`` plus two reads because those
        three can half-fail: a theme that switches and then fails to yield its
        ramp leaves Kotlin drawing the old colours over the new background,
        and nothing in the app would say so.

        ``oled`` rides along for the same reason: the toggle and the theme
        both decide the same list, and applying them in two calls means one
        frame drawn with the new theme and the old floor.
        """
        if not self.set_theme(name):
            return None
        self._oled = bool(oled)
        return self.colours()

    def set_oled(self, on: bool) -> list[str]:
        """Toggle true black and return the colours that follow from it."""
        self._oled = bool(on)
        return self.colours()

    def set_sensitivity(self, value: float) -> float:
        """Manual trim on top of the analyser's autosens — desktop's ``[``/``]``.

        Autosens normalises the bands to the loudest thing it has heard
        recently, which is what stops a quiet track drawing a flat line. It
        cannot know how *busy* you want the picture, and modes that trigger on
        level rather than draw it — Fireworks launches at
        ``0.35 + energy * 7`` per second — turn that preference into a rate.
        Same range as the desktop's, so a value means the same on both.
        """
        v = max(0.15, min(8.0, float(value)))
        self._analyser.sensitivity = v
        return v

    def mode_names(self) -> list[str]:
        """The modes the picker offers — everything except the hidden ones.

        The twelve hidden variants draw through Unicode 16 octants (U+1CD00
        and up), which landed in 2024 and which no font on Android carries
        yet. Offering them would put twelve entries in the picker that render
        as tofu, and tofu on a visualiser reads as a crash rather than as a
        missing glyph. ``render`` still accepts them by name, exactly as
        desktop keeps a hidden mode selectable from a config file.
        """
        return [m.name for m in MODES if not m.hidden]

    def theme_names(self) -> list[str]:
        return sorted(BUILTIN)

    def theme_swatches(self, n: int = 6) -> list[list[str]]:
        """``[name, bg, fg, *n ramp colours]`` per theme, for painting the picker.

        Fifty-four theme names is a list, not a choice — nobody knows what
        "ayu-mirage" looks like, and finding out by selecting each in turn is
        the whole afternoon. The picker draws the colours instead.

        Built once and cached: constructing a ``Palette`` interpolates the
        whole ramp, and doing that for every theme is worth about a tenth of a
        second. That is nothing on first open and everything at 30 fps, so it
        never happens on a frame.
        """
        if self._swatches is None:
            rows = []
            for name in sorted(BUILTIN):
                spec = BUILTIN[name]
                hexes = Palette(spec).hexes
                # Spread across the ramp rather than taking the first n: the
                # low end of most ramps is near-background, so the first six
                # colours of a dozen themes are six near-identical smudges.
                step = max(1, len(hexes) // n)
                picked = list(hexes[::step][:n])
                while len(picked) < n:                       # very short ramps
                    picked.append(hexes[-1])
                rows.append([name, spec.bg, spec.fg, *picked])
            self._swatches = rows
        return self._swatches

    # ── one frame ──
    def render(self, name: str, w: int, h: int) -> bytes:
        """Run one mode at this grid size and return the packed grid.

        ``w``/``h`` come from Kotlin's measured cell metrics every frame. The
        engine is resolution-agnostic on desktop for exactly this reason, so
        nothing here caches a size or objects to it changing mid-session — a
        rotation is just a different pair of numbers.
        """
        mode = self._modes.get(name)
        if mode is None:
            raise KeyError(f"no mode named {name!r}")

        # Mode scratch is keyed to the mode, and switching mode must not hand
        # the next one the last one's arrays. This mirrors the desktop widget,
        # which drops the dict on a switch for the same reason.
        if name != self._mode_name:
            self._state = {}
            self._mode_name = name

        now = time.monotonic()
        dt = min(0.2, max(1e-4, now - self._last_t))
        self._last_t = now
        self._frame += 1

        # ``Analyser.frame`` is a property holding the newest published Frame —
        # there is no ``snapshot()``, and the Frame carries neither ``peaks``
        # nor ``energy`` nor ``bars``. This block is the widget's ``_tick``,
        # which is where those actually come from.
        f = self._analyser.frame
        if len(f.bands) != len(self._spring.x):
            # The band count is settable at runtime, so the springs follow the
            # analyser rather than a constant fixed at construction.
            n = len(f.bands)
            self._bars = n
            self._spring = Spring(n)
            self._peaks = Peaks(n)
            self._stereo_l = Spring(n)
            self._stereo_r = Spring(n)

        self._spring.step(f.bands, dt)
        self._peaks.step(self._spring.x, dt)
        self._stereo_l.step(f.bands_l, dt)
        self._stereo_r.step(f.bands_r, dt)
        if f.seq != self._last_seq:
            self._trace.step(f.wave, dt)
            self._last_seq = f.seq

        # Differenced once here, not in every mode that wants beats. Clamped at
        # zero because a restarted analyser counts from nothing again, and a
        # negative delta is not a burst of beats played backwards.
        onsets = max(0, f.onset_seq - self._last_onset_seq)
        self._last_onset_seq = f.onset_seq

        energy = float(self._spring.x.mean())
        self._stats_frames += 1
        self._stats_dt += dt
        self._stats_energy += energy
        self._stats_onsets += onsets
        self._stats_band_peak = max(self._stats_band_peak, float(self._spring.x.max()))

        ctx = Ctx(
            w=w, h=h,
            bands=self._spring.x, peaks=self._peaks.value,
            bands_l=self._stereo_l.x, bands_r=self._stereo_r.x,
            wave=self._trace.value if self._trace.value is not None else f.wave,
            stereo=f.stereo,
            frame=self._frame,
            t=now - self._t0,
            dt=dt,
            energy=energy, silent=f.silent,
            palette=self._palette, state=self._state,
            bars=self._bars,
            onset_seq=f.onset_seq, onsets=onsets,
            onset_strength=f.onset_strength,
            flux=f.flux, tempo_bpm=f.tempo_bpm, beat_phase=f.beat_phase,
        )

        out = mode.fn(ctx)
        codes, cidx = out[0], out[1]
        bidx = out[2] if len(out) == 3 else None
        return _pack(codes, cidx, bidx)


def _pack(codes: np.ndarray, cidx: np.ndarray, bidx: np.ndarray | None) -> bytes:
    """Header then planes, each C-contiguous, no padding between them.

    ``tobytes`` on an already-contiguous array is a straight memcpy; the
    ``ascontiguousarray`` calls are there because a mode is free to return a
    slice or a transpose and several do.
    """
    h, w = codes.shape
    planes = 3 if bidx is not None else 2
    parts = [
        _HEADER.pack(_MAGIC, WIRE_VERSION, planes, w, h),
        np.ascontiguousarray(codes, dtype="<i4").tobytes(),
        np.ascontiguousarray(cidx, dtype=np.uint8).tobytes(),
    ]
    if bidx is not None:
        parts.append(np.ascontiguousarray(bidx, dtype=np.uint8).tobytes())
    return b"".join(parts)
