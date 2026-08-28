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

import math
import struct
import time

import numpy as np

from spektr.analysis import N_BANDS, Analyser
from spektr.capture import RingBuffer
from spektr.modes import MODES, Ctx
from spektr.motion import (
    GLIDE_BLEND_TAU,
    PROFILES,
    Peaks,
    Spring,
    Trace,
    spread,
)
from spektr.palette import BUILTIN, Palette

#: Wire format version. Kotlin refuses a buffer it does not recognise rather
#: than reading a stale layout as though it were current. v3 replaces v2's
#: float *height* plane with the scene parameter block (``planes == 5``): the
#: port no longer ships a picture for its 3D view, it ships the music. The
#: glyph and index layouts are unchanged from v1.
WIRE_VERSION = 3

#: The scene family — raymarched solids the GLES view draws.
#:
#: Deliberately *not* spektr modes. A mode is a function from a Ctx to a grid
#: of glyphs, and there is no grid of glyphs that is a raymarched metaball;
#: pretending otherwise would mean writing a second, flat, worse version of
#: each of these to satisfy a contract nothing needs them to keep. So the
#: scene lives in the fragment shader and Python's whole job is to describe
#: what the music is doing.
#:
#: That also makes them nearly free. The height-mapped view this replaces cost
#: 33 ms a frame on the tablet — a mode's worth of numpy, every frame, to
#: build a picture the GPU then had to be told about. A scene frame is 160
#: bytes and no numpy at all.
SCENES = ("Metaball", "Wormhole", "Monolith", "Lattice")
SCENE_INDEX = {name: i for i, name in enumerate(SCENES)}

#: Scene parameter block: a fixed header of scalars, then the bands. Kotlin
#: reads the same layout by index, so these two halves are one definition in
#: two languages and have to move together.
#:
#:  0 scene   1 t        2 energy   3 bass     4 mid      5 treble
#:  6 pulse   7 hardest  8 beat     9 tempo   10 flux    11 peak band
#: 12 tilt   13 hits    14 silent  15 —      16.. bands
SCENE_HEAD = 16
SCENE_BANDS = 24
SCENE_FLOATS = SCENE_HEAD + SCENE_BANDS

#: How fast a hit fades out of a scene, in seconds. Long enough that a shape
#: is still moving when the next beat lands at 120 bpm, short enough that it
#: is not simply always on.
_PULSE_TAU = 0.28

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
        self._swatches: list[list[str]] | None = None
        self._oled = False
        self._field_mode = False

        #: Decays every frame, kicked to 1.0 by an onset. The scenes want a
        #: hit as a shape over time rather than as the instant it happened,
        #: and a fragment shader has no memory between frames to build one
        #: from — so the envelope is integrated here and shipped as a number.
        self._pulse = 0.0
        self._hardest = 0.0
        self._hits = 0

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
        # The motion profile starts on snappy, the default, and is switched
        # live by ``set_motion`` — same table as the desktop widget, so a
        # profile means the same feel on both platforms.
        self._motion = "snappy"
        params = PROFILES[self._motion]
        self._spring = Spring(resolved, **params)
        self._peaks = Peaks(resolved)
        self._stereo_l = Spring(resolved, **params)
        self._stereo_r = Spring(resolved, **params)
        self._trace = Trace(tau=0.028)
        # Glide's temporal pre-blends, one per spring exactly as the widget
        # keeps them. Idle until glide is selected; re-seeded from live audio
        # on the first glide frame.
        self._band_blend = Trace(tau=GLIDE_BLEND_TAU)
        self._stereo_l_blend = Trace(tau=GLIDE_BLEND_TAU)
        self._stereo_r_blend = Trace(tau=GLIDE_BLEND_TAU)
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

    def set_motion(self, name: str) -> str:
        """Switch the motion profile, live — desktop settings' motion row.

        Same table and same conditioning as ``AudioVisualizer.set_motion``,
        so ``glide`` feels identical on both platforms. Springs are retuned
        in place rather than replaced, so position and velocity survive the
        switch; the pre-blends are dropped because history accumulated under
        one profile is history the other never saw. Returns what was taken,
        so Kotlin can show the settled value rather than the ask.
        """
        if name not in PROFILES:
            return self._motion
        if name != self._motion:
            self._motion = name
            for spring in (self._spring, self._stereo_l, self._stereo_r):
                spring.retune(**PROFILES[name])
            for blend in (self._band_blend, self._stereo_l_blend, self._stereo_r_blend):
                blend.value = None
        return name

    def mode_names(self) -> list[str]:
        """The modes the picker offers — everything except the hidden ones.

        The twelve hidden variants draw through Unicode 16 octants (U+1CD00
        and up), which landed in 2024 and which no font on Android carries
        yet. Offering them would put twelve entries in the picker that render
        as tofu, and tofu on a visualiser reads as a crash rather than as a
        missing glyph. ``render`` still accepts them by name, exactly as
        desktop keeps a hidden mode selectable from a config file.
        """
        return [*SCENES, *(m.name for m in MODES if not m.hidden)]

    def scene_names(self) -> list[str]:
        """The subset of [mode_names] the GLES view draws, for Kotlin's switch."""
        return list(SCENES)

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
        scene = SCENE_INDEX.get(name)
        mode = None if scene is not None else self._modes.get(name)
        if mode is None and scene is None:
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
            # event (``_resize_bands``). Rebuilt under the current motion
            # profile, so a resize never silently resets the personality.
            n = len(f.bands)
            params = PROFILES[self._motion]
            self._spring = Spring(n, **params)
            self._peaks = Peaks(n)
            self._stereo_l = Spring(n, **params)
            self._stereo_r = Spring(n, **params)
            self._state = {}

        # Motion profile conditioning, exactly as the widget's _tick does it:
        # glide pre-blends temporally then spreads to neighbours; snappy
        # feeds the raw spectrum straight through.
        if self._motion == "glide":
            bands_t = spread(self._band_blend.step(f.bands, dt))
            bands_l_t = spread(self._stereo_l_blend.step(f.bands_l, dt))
            bands_r_t = spread(self._stereo_r_blend.step(f.bands_r, dt))
        else:
            bands_t, bands_l_t, bands_r_t = f.bands, f.bands_l, f.bands_r

        self._spring.step(bands_t, dt)
        self._peaks.step(self._spring.x, dt)
        self._stereo_l.step(bands_l_t, dt)
        self._stereo_r.step(bands_r_t, dt)
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

        # Onset envelope, integrated here because the shader cannot keep one.
        self._pulse *= math.exp(-dt / _PULSE_TAU)
        if onsets:
            hard = min(1.0, max(0.0, float(f.onset_strength)))
            self._pulse = 1.0
            self._hardest = hard
            self._hits += onsets

        if scene is not None:
            out_bytes = self._pack_scene(scene, f, energy, now - self._t0)
            spent = time.monotonic() - began
            self._stats_render += spent
            self._stats_render_max = max(self._stats_render_max, spent)
            return out_bytes

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
            bars=self._bars_wanted,
            onset_seq=f.onset_seq, onsets=onsets,
            onset_strength=f.onset_strength,
            flux=f.flux, tempo_bpm=f.tempo_bpm, beat_phase=f.beat_phase,
        )

        out = mode.fn(ctx)
        codes, cidx = out[0], out[1]
        bidx = out[2] if len(out) == 3 else None
        if self._field_mode:
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

    def _pack_scene(self, scene: int, f, energy: float, t: float) -> bytes:
        """One scene frame: ``planes == 5``, a flat block of float32.

        No geometry crosses here and no picture does either. A raymarched
        scene is defined entirely by its own shader; what it cannot know is
        what the music is doing, and that is all of forty numbers. Kotlin
        hands them straight to a uniform array.

        The bands are resampled to a fixed [SCENE_BANDS] rather than shipped at
        whatever count the analyser resolved. A shader indexing a uniform array
        wants a constant bound, and the scenes read bands as *positions along a
        spectrum* — a metaball's third orbit, a lattice's third rank — which is
        a fraction of the way up, not an FFT bin. Resampling keeps that meaning
        fixed while the band-count setting moves underneath it.
        """
        bands = self._spring.x
        if len(bands) != SCENE_BANDS:
            bands = np.interp(
                np.linspace(0.0, 1.0, SCENE_BANDS),
                np.linspace(0.0, 1.0, len(bands)),
                bands,
            )
        left = float(self._stereo_l.x.mean())
        right = float(self._stereo_r.x.mean())
        head = [
            float(scene),
            # Wrapped, not raw. A float32 loses its fractional resolution as it
            # grows, and a scene left running overnight would drift from
            # smooth motion into visible stepping. An hour is longer than any
            # period in these shaders, so the wrap is invisible.
            float(t % 3600.0),
            float(np.clip(energy, 0.0, 1.0)),
            float(np.clip(bands[:SCENE_BANDS // 6].mean() * 1.4, 0.0, 1.0)),
            float(np.clip(bands[SCENE_BANDS // 6:SCENE_BANDS // 2].mean() * 1.6, 0.0, 1.0)),
            float(np.clip(bands[SCENE_BANDS // 2:].mean() * 2.0, 0.0, 1.0)),
            float(self._pulse),
            float(self._hardest),
            float(f.beat_phase),
            float(np.clip(f.tempo_bpm / 200.0, 0.0, 1.5)),
            float(np.clip(f.flux, 0.0, 1.0)),
            float(np.clip(bands.max(), 0.0, 1.0)),
            float(np.clip((right - left) * 4.0, -1.0, 1.0)),
            float(self._hits % 1024),
            1.0 if f.silent else 0.0,
            0.0,
        ]
        assert len(head) == SCENE_HEAD
        block = np.asarray(head + list(np.clip(bands, 0.0, 1.0)), dtype="<f4")
        return b"".join((
            _HEADER.pack(_MAGIC, WIRE_VERSION, 5, SCENE_FLOATS, 1),
            block.tobytes(),
        ))

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
