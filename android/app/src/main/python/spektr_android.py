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
#: than reading a stale layout as though it were current. v2 adds the float
#: field plane (``planes == 4``) for the port's terrain view; the glyph and
#: index layouts are unchanged from v1.
WIRE_VERSION = 2

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


def _clamp_bars(n: int) -> int:
    """The desktop widget's rule for a band-count setting: 0 fits, else 8..64."""
    n = int(n)
    return 0 if n <= 0 else max(8, min(64, n))


class _RecordingPalette:
    """A palette that remembers the floats modes hand it just before quantising.

    Every mode ends its frame with ``ctx.ramp(field)`` — one call to
    :meth:`Palette.indices` carrying the per-cell floats the whole picture is
    shaded by. Immediately afterwards those floats are crushed to 64 ramp
    indices, which is invisible in flat colour and fatal as terracing once
    the terrain view lights the surface. This proxy sits between the modes
    and the real palette (the engine hands its Ctx a palette either way), so
    the terrain path can ship the *pre-quantisation* floats over the wire
    without touching a single mode or any shared engine file — the port's
    founding rule holds; this lives entirely on the Android side.

    Only the last call's array is kept: nearly every mode shades through
    exactly one ``ramp`` call per frame, and where one does not, the shape
    check at pack time falls back to the quantised indices rather than
    shipping a mis-mapped height map.
    """

    def __init__(self, inner: Palette) -> None:
        self._inner = inner
        self.last_norm: np.ndarray | None = None

    def indices(self, norm) -> np.ndarray:
        arr = np.asarray(norm, dtype=np.float64)
        self.last_norm = arr
        return self._inner.indices(arr)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _field_float(codes, cidx, bidx, rec: _RecordingPalette):
    """The picture those glyphs stood for, as float heights instead of indices.

    Same geometry walk as :func:`_field` — braille gives 4x2 dots per cell,
    half-blocks give two rows, everything else one — but each pixel carries a
    normalised float for the GPU to displace and light, not a 64-step index.
    Background pixels are 0.0.

    The heights come from the recording palette, best source first:

    1. A recording shaped exactly like this geometry's output plane — what
       the half-block modes produce, since they ramp their smooth field at
       full field resolution (Chladni measured at (2h, w)). Zero loss; the
       quantiser never touched these numbers.
    2. A recording shaped like the cell grid — the braille modes' shape,
       whose dots then share their cell's height.
    3. The quantised indices as floats. Still a valid height map, just with
       the 64-step terracing this path exists to avoid; a mode that shades
       through several ``ramp`` calls lands here rather than shipping a
       mis-mapped surface.
    """
    h, w = codes.shape
    hexmax = max(1, len(rec._inner.hexes) - 1)
    idx_float = cidx.astype(np.float32) / hexmax

    rec_norm = rec.last_norm

    def clipped(a):
        return np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)

    braille = (codes >= 0x2800) & (codes <= 0x28FF)
    upper = codes == 0x2580
    full = codes == 0x2588
    lower = codes == 0x2584
    blank = (codes == 0) | (codes == 0x20)

    if braille.any():
        sr, sc = 4, 2
        out_h, out_w = h * sr, w * sc
        if rec_norm is not None and tuple(rec_norm.shape) == (out_h, out_w):
            out = clipped(rec_norm).copy()
        else:
            norm = None
            if rec_norm is not None and tuple(rec_norm.shape) == (h, w):
                norm = clipped(rec_norm)
            out = np.zeros((out_h, out_w), dtype=np.float32)
            if norm is None:
                norm = idx_float
            bits = np.where(braille, codes - 0x2800, 0)
            for bit, (dy, dx) in enumerate(_BRAILLE_BITS):
                on = ((bits >> bit) & 1).astype(bool)
                sub = out[dy::sr, dx::sc]
                np.copyto(sub, norm, where=on)
            # Non-braille marks (peak ticks, box drawing) read as solid cells.
            solid = ~braille & ~blank
            if solid.any():
                for dy in range(sr):
                    for dx in range(sc):
                        sub = out[dy::sr, dx::sc]
                        np.copyto(sub, norm, where=solid)
        return out, out_w, out_h

    if upper.any() or full.any() or lower.any():
        bg = bidx.astype(np.float32) / hexmax \
            if bidx is not None else np.zeros((h, w), dtype=np.float32)
        if rec_norm is not None and tuple(rec_norm.shape) == (h * 2, w):
            # Full-resolution floats straight off the mode's own field.
            base = clipped(rec_norm)
            blank2 = np.repeat(blank, 2, axis=0)
            return np.where(blank2, np.float32(0.0), base), w, h * 2
        norm = None
        if rec_norm is not None and tuple(rec_norm.shape) == (h, w):
            norm = clipped(rec_norm)
        if norm is None:
            norm = idx_float
        top = np.where(full | upper, norm, np.where(blank, 0.0, bg))
        bot = np.where(full | lower, norm, np.where(blank, 0.0, bg))
        other = ~(upper | full | lower | blank)
        top = np.where(other, norm, top)
        bot = np.where(other, norm, bot)
        out = np.empty((h * 2, w), dtype=np.float32)
        out[0::2] = top
        out[1::2] = bot
        return out, w, h * 2

    solid = ~blank
    if rec_norm is not None and tuple(rec_norm.shape) == (h, w):
        return np.where(solid, clipped(rec_norm), np.float32(0.0)), w, h
    return np.where(solid, idx_float, np.float32(0.0)), w, h


class Engine:
    """One per app. Holds the ring, the analyser and the mode's scratch state."""

    def __init__(self, samplerate: int = 48000, bars: int = 16) -> None:
        # Two seconds of stereo headroom. The analyser only ever asks for the
        # most recent window, so this is slack against scheduler jitter rather
        # than a queue — if Android stalls us we drop old audio, which is the
        # right failure for a visualiser.
        self._ring = RingBuffer(samplerate * 2)
        self._sr = samplerate
        self._analyser = Analyser(self._ring, lambda: self._sr)
        #: How many bars to draw: the user's setting, with ``0`` meaning "fit
        #: the grid" — exactly what ``config.bands`` means on desktop. The
        #: analyser is asked for the same number, so a count past its native
        #: resolution resolves into real FFT-bin ranges rather than as
        #: interpolated copies of neighbouring bands.
        self._bars_wanted = _clamp_bars(bars)
        resolved = self._analyser.set_bands(self._bars_wanted or N_BANDS)
        self._analyser.start()

        self._t0 = time.monotonic()
        self._last_t = self._t0
        self._frame = 0
        self._state: dict = {}
        self._mode_name: str | None = None
        self._modes = {m.name: m for m in MODES}
        self._theme = BUILTIN["gruvbox"]
        self._palette = Palette(self._theme)
        self._rec = _RecordingPalette(self._palette)
        self._swatches: list[list[str]] | None = None
        self._oled = False
        self._field_mode = False

        #: The terrain family — modes built for the GLES view. Selecting one
        #: makes render() ship float heights instead of the index plane; no
        #: separate switch exists, because "3D" is a property of the mode,
        #: not a lens over all of them.
        self._terrain_modes = {"Swell", "Terra"}

        # Debug counters; see stats().
        self._stats_t0 = self._t0
        self._stats_frames = 0
        self._stats_dt = 0.0
        self._stats_energy = 0.0
        self._stats_onsets = 0
        self._stats_band_peak = 0.0
        self._stats_sample_peak = 0.0
        self._stats_render = 0.0
        self._stats_render_max = 0.0

        #: When audio last arrived. Capture can stop without anyone telling
        #: the engine — see _feed_silence.
        self._last_push = self._t0

        # The motion layer, which is not in the analyser and not in the modes.
        #
        # ``Analyser`` publishes a raw ``Frame``: bands as measured, and no
        # peaks, energy or smoothed trace at all. Everything that makes the
        # picture move like the desktop app lives between the two, in the
        # widget — and a port that skips it does not fail, it just renders a
        # jittery version of the same modes and looks subtly wrong forever.
        # Same objects and same constants as ``AudioVisualizer.__init__``,
        # sized to the count the analyser actually resolved — the wanted
        # setting and the resolved count only agree at or below N_BANDS.
        self._spring = Spring(resolved)
        self._peaks = Peaks(resolved)
        self._stereo_l = Spring(resolved)
        self._stereo_r = Spring(resolved)
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

        ``[fps, mean dt ms, mean energy, onsets/s, peak band, peak sample,
        mean render ms, worst render ms]``. Resets on read, so each line
        describes its own interval rather than the whole session — a burst
        that only shows in the average of one second disappears into the
        average of five hundred.
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
            self._stats_render / n * 1000.0,
            self._stats_render_max * 1000.0,
        ]
        self._stats_t0 = now
        self._stats_frames = 0
        self._stats_dt = 0.0
        self._stats_energy = 0.0
        self._stats_onsets = 0
        self._stats_band_peak = 0.0
        self._stats_sample_peak = 0.0
        self._stats_render = 0.0
        self._stats_render_max = 0.0
        return [float(v) for v in out]

    #: How long the engine waits before deciding the audio has stopped. Long
    #: enough to ride out a scheduler hiccup between AudioRecord buffers —
    #: those arrive about every 43 ms — and short enough that a stopped
    #: capture stops the picture rather than freezing it.
    _SILENCE_AFTER = 0.25

    def _feed_silence(self, now: float) -> None:
        """Write silence when nothing is arriving, so the picture can settle.

        The analyser pulls: it reads the most recent window out of the ring
        whenever it likes. If capture stops, nothing overwrites that window,
        so it keeps reading the same audio and publishing the same bands —
        and every mode goes on drawing whatever level was playing at the
        moment the audio stopped, indefinitely. Switching mode does not help,
        because the new mode is handed the same frozen numbers.

        That is what happens on Android whenever the projection ends: the
        service tears down the AudioRecord and simply stops pushing. Nothing
        in the pull model can distinguish "no new audio" from "the same audio
        again", so the engine has to say so itself.

        Silence rather than a reset, because silence is the truth — the
        speakers are not playing — and it lets the springs and peak-holds
        decay the way they do at the end of any quiet passage, instead of
        snapping to zero.
        """
        gap = now - self._last_push
        if gap < self._SILENCE_AFTER:
            return
        # One frame's worth, so the ring is refreshed at the rate it is being
        # read rather than all at once.
        n = max(1, int(self._sr * min(gap, 0.05)))
        self._last_push = now
        self._ring.push(np.zeros((n, 2), dtype=np.float32))

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
        self._last_push = time.monotonic()
        self._ring.push(buf)

    # ── configuration ──
    def set_theme(self, name: str) -> bool:
        spec = BUILTIN.get(name)
        if spec is None:
            return False
        self._theme = spec
        self._palette = Palette(spec)
        self._rec = _RecordingPalette(self._palette)
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

    def set_bands(self, n: int) -> int:
        """Change how many bars are drawn, live — desktop's band-count setting.

        One control, two mechanisms, same as the widget: up to the analyser's
        native resolution the modes simply draw fewer bars out of the same
        analysis; past it, the analyser resolves more bands for real. ``0``
        fits the grid. Returns what was taken, so Kotlin can show the clamped
        value rather than the one it asked for.
        """
        self._bars_wanted = _clamp_bars(n)
        self._analyser.set_bands(self._bars_wanted or N_BANDS)
        return self._bars_wanted

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
        began = time.monotonic()
        self._feed_silence(began)
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
            # analyser rather than a constant fixed at construction — and the
            # mode's scratch goes with them: cached geometry can be sized to
            # the old count. The desktop widget drops the same dict on this
            # event (``_resize_bands``).
            n = len(f.bands)
            self._spring = Spring(n)
            self._peaks = Peaks(n)
            self._stereo_l = Spring(n)
            self._stereo_r = Spring(n)
            self._state = {}

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
            palette=self._rec, state=self._state,
            bars=self._bars_wanted,
            onset_seq=f.onset_seq, onsets=onsets,
            onset_strength=f.onset_strength,
            flux=f.flux, tempo_bpm=f.tempo_bpm, beat_phase=f.beat_phase,
        )

        out = mode.fn(ctx)
        codes, cidx = out[0], out[1]
        bidx = out[2] if len(out) == 3 else None
        if self._field_mode and name in self._terrain_modes:
            # The terrain family: float heights for the GLES displacement,
            # pre-quantisation — these modes hand ctx.ramp their whole field
            # in one call, so the recording is exact. planes == 4 says
            # "w*h float32 little-endian".
            plane, fw, fh = _field_float(codes, cidx, bidx, self._rec)
            out_bytes = b"".join((
                _HEADER.pack(_MAGIC, WIRE_VERSION, 4, fw, fh),
                np.ascontiguousarray(plane, dtype="<f4").tobytes(),
            ))
        elif self._field_mode:
                plane, fw, fh = _field(codes, cidx, bidx)
                out_bytes = b"".join((
                    _HEADER.pack(_MAGIC, WIRE_VERSION, 1, fw, fh),
                    np.ascontiguousarray(plane, dtype=np.uint8).tobytes(),
                ))
        else:
            out_bytes = _pack(codes, cidx, bidx)
        spent = time.monotonic() - began
        self._stats_render += spent
        self._stats_render_max = max(self._stats_render_max, spent)
        return out_bytes

    def set_field_mode(self, on: bool) -> None:
        """Draw as a picture rather than as glyphs.

        The glyph grid is the terminal's constraint, not the mode's: Chladni
        computes a smooth nodal field and then throws most of it away choosing
        a half-block to stand for each cell. Android has a canvas and no such
        constraint, so this returns the field itself and lets Kotlin scale it —
        which is the difference between a 118x34 mosaic and a picture.
        """
        self._field_mode = bool(on)


#: Ramp indices are 0..63, so 255 is free to mean "nothing here — paint the
#: background". A separate plane would double the buffer to carry one bit.
FIELD_EMPTY = 255

#: Braille dot bits, in Unicode's order, as (sub-row, sub-col) in a 4x2 cell.
#: Dots 1-3 run down the left column, 4-6 down the right, and 7-8 are the low
#: pair added when braille went to eight dots — which is why the last two are
#: not where counting straight down would put them.
_BRAILLE_BITS = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1))


def _field(codes: np.ndarray, cidx: np.ndarray, bidx: np.ndarray | None) -> tuple[np.ndarray, int, int]:
    """Unpack a glyph grid into the picture those glyphs would have drawn.

    A cell is not a pixel. Braille carries eight, the half-block trick carries
    two, and the mode computed its field at that finer resolution before
    choosing a glyph to approximate it with. Drawing the glyph throws the
    difference away; this puts it back, so the bitmap renderer can draw what
    the mode actually computed instead of a 118x34 mosaic of it.

    Sub-cell shape follows the geometry rather than being fixed: braille
    modes give 4x2 a cell, block modes 2x1, and a mode drawing text gives 1x1
    because there is nothing finer in it to recover.
    """
    h, w = codes.shape
    fg = cidx.astype(np.uint8)
    bg = bidx.astype(np.uint8) if bidx is not None else None

    braille = (codes >= 0x2800) & (codes <= 0x28FF)
    if braille.any():
        sr, sc = 4, 2
        out = np.full((h * sr, w * sc), FIELD_EMPTY, dtype=np.uint8)
        bits = np.where(braille, codes - 0x2800, 0)
        for bit, (dr, dc) in enumerate(_BRAILLE_BITS):
            on = (bits >> bit) & 1
            sub = out[dr::sr, dc::sc]
            np.copyto(sub, fg, where=on.astype(bool))
        # Cells that are not braille still have to say something. Anything
        # solid fills its whole cell; a blank leaves the background showing.
        solid = ~braille & (codes != 0) & (codes != 0x20)
        if solid.any():
            for dr in range(sr):
                for dc in range(sc):
                    sub = out[dr::sr, dc::sc]
                    np.copyto(sub, fg, where=solid)
        return out, w * sc, h * sr

    # The half-block plane: `▀` is the top half in fg over the bottom half in
    # bg, which is two rows of picture per row of cells and the reason these
    # modes look twice as tall as they read.
    upper = codes == 0x2580
    full = codes == 0x2588
    lower = codes == 0x2584
    if upper.any() or full.any() or lower.any():
        top = np.where(full | upper, fg, bg if bg is not None else FIELD_EMPTY)
        bot = np.where(full | lower, fg, bg if bg is not None else FIELD_EMPTY)
        blank = (codes == 0) | (codes == 0x20)
        if bg is None:
            top = np.where(blank, FIELD_EMPTY, top)
            bot = np.where(blank, FIELD_EMPTY, bot)
        other = ~(upper | full | lower | blank)
        top = np.where(other, fg, top)
        bot = np.where(other, fg, bot)
        out = np.empty((h * 2, w), dtype=np.uint8)
        out[0::2] = top
        out[1::2] = bot
        return out, w, h * 2

    # Nothing sub-cell to recover — text, box drawing, geometric shapes.
    solid = (codes != 0) & (codes != 0x20)
    base = bg if bg is not None else np.full_like(fg, FIELD_EMPTY)
    return np.where(solid, fg, base).astype(np.uint8), w, h


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
