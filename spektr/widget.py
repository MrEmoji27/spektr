"""The visualiser widget.

Renders through ``render_line`` returning ``Strip`` objects rather than
``render`` returning a Rich ``Text``. The old path meant Textual ran the whole
widget through a Rich console render every frame — the cache never helped,
because ``refresh()`` on every tick invalidated it — costing 2-4 ms at
fullscreen no matter which mode was active. Building Strips directly skips that
pass entirely.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
from textual.reactive import reactive
from textual.strip import Strip
from textual.widget import Widget

from . import display as display_probe
from . import modes as mode_registry
from .analysis import ANALYSES_PER_SEC, N_BANDS, Analyser
from .capture import Capture
from .config import FPS_MAX, FPS_UNLIMITED, Settings
from .modes import Ctx
from .motion import Peaks, Spring, Trace
from .palette import AUTO, RAMP_STEPS, Palette, all_themes, theme_from_textual
from .plugins import BadModeOutput, Quarantine, validate
from .render import SPACE, make_strips

#: A plugin allowed to eat the whole frame budget would stutter the entire UI,
#: so anything slower than this gets its previous frame reused on alternate
#: ticks. cliamp caps Lua plugins at 10 ms for the same reason.
SLOW_MODE_MS = 11.0

#: How long an animated theme's colour loop takes to turn once, in seconds.
#: Expressed as a duration rather than a rate tied to RAMP_STEPS, so raising
#: the ramp's resolution for a smoother gradient doesn't also speed up the
#: animation — the two used to share one constant and silently coupled.
RAINBOW_SECONDS_PER_CYCLE = 10.7


class AudioVisualizer(Widget):
    # background is set from the active theme in _paint_background(); the
    # value here is only what shows for the instant before the first theme
    # is applied
    DEFAULT_CSS = """
    AudioVisualizer {
        height: 1fr;
        background: #000000;
    }
    """

    mode_name: reactive[str] = reactive("Bars")

    def __init__(
        self,
        device=None,
        settings: Settings | None = None,
        allow_mic: bool = False,
        config_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.settings = settings or Settings()
        #: where user themes are read from; None means the platform default
        #: from palette.config_dir(). Handed straight to all_themes() so an
        #: injected config root redirects the theme-list read exactly like it
        #: already redirects the writes.
        self._config_dir = config_dir

        self.capture = Capture(device=device, allow_mic=allow_mic)
        self.analyser = Analyser(self.capture.ring, lambda: self.capture.samplerate)
        self.analyser.sensitivity = self.settings.sensitivity
        if self.settings.bands:
            self.analyser.set_bands(self.settings.bands)

        self.palette = Palette()
        self._themes = all_themes(self._config_dir)
        self._theme_name = self.settings.theme

        self._spring = Spring(N_BANDS)
        self._stereo_l = Spring(N_BANDS)
        self._stereo_r = Spring(N_BANDS)
        self._peaks = Peaks(N_BANDS)
        self._trace = Trace(tau=0.028)

        self._mode_state: dict[str, dict] = {}
        self._strips: list[Strip] | None = None
        self._frame = 0
        #: Onset counter as of the previous rendered frame, for ``ctx.onsets``.
        self._last_onset_seq = 0
        self._t0 = time.monotonic()
        self._last = self._t0
        self._timer = None
        # The rate the user asked for, and the rate we are currently running.
        # Adaptive pacing moves the second one; the first is what gets saved,
        # so a slow machine never quietly rewrites the preference — and it is
        # also the ceiling adaptive pacing is allowed to climb back up to.
        # "Unlimited" is resolved to a concrete number here, once, and never
        # again. Everything downstream — the adaptive step-down, the recovery
        # ceiling, the ``1000 / fps`` budget — needs a finite target, and
        # keeping the resolution at this one point means none of that had to
        # learn about the sentinel.
        self._unlimited = display_probe.unlimited_fps(int(2 * ANALYSES_PER_SEC))
        self._target_fps = self._resolve_fps(self.settings.fps)
        self._fps = self._target_fps
        self._build_ms: float | None = None
        self._last_seq = -1

        self._preview: str | None = None  # theme name being previewed
        self._preview_mode: str | None = None

        self.quarantine = Quarantine()
        #: called with (mode_name, message) when a mode is disabled
        self.on_mode_disabled = None
        self._mode_ms: dict[str, float] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.apply_theme(self._theme_name, remember=False)
        self.mode_name = (
            self.settings.mode if mode_registry.get(self.settings.mode) else "Bars"
        )
        try:
            self.watch(self.app, "theme", self._on_app_theme, init=False)
        except Exception:  # noqa: BLE001 — not mounted yet, on_mount will redo it
            pass

        self.capture.start()
        self.analyser.start()
        # Pass the raw setting, not the resolved rate: ``requested=True``
        # writes it back to ``settings``, and writing the resolved number there
        # would silently turn "unlimited" into "144 fps" on first run and pin
        # the preference to whatever monitor happened to be attached.
        self._retime(self.settings.fps, requested=True)

    def on_unmount(self) -> None:
        self.analyser.stop()
        self.capture.stop()

    def _on_app_theme(self, *_args) -> None:
        if self._theme_name == AUTO:
            self.apply_theme(AUTO, remember=False)

    # ── status ───────────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self.capture.status

    @property
    def on_mic(self) -> bool:
        return self.capture.on_mic

    @property
    def perf(self) -> str:
        ms = self._build_ms or 0.0
        return f"{self.mode_name} · {ms:.1f} ms/frame · {self._fps} fps"

    @property
    def level(self) -> str:
        """Input level against the gate — the readout for "why is it moving?"."""
        f = self.analyser.frame
        gate = self.analyser.gate
        ratio = f.rms / gate if gate else 0.0
        if f.silent:
            return f"gated · input {f.rms:.2e} is below the gate {gate:.1e}"
        return (
            f"input {f.rms:.2e} · gate {gate:.1e} · {ratio:.1f}x over · "
            f"strength {f.confidence * 100:.0f}%"
        )

    # ── modes ────────────────────────────────────────────────────────────────

    @property
    def mode_names(self) -> list[str]:
        """Every mode the picker and the cycle keys offer.

        Quarantined ones are left out so cycling cannot land you back on
        something broken, and hidden ones because they have been superseded —
        both are still registered and both still render if something asks for
        them by name, which is what ``--mode`` and a saved config do.
        """
        return [
            m.name
            for m in mode_registry.listed()
            if not self.quarantine.is_disabled(m.name)
        ]

    def set_mode(self, name: str, *, remember: bool = True) -> None:
        """Switch to a mode by name — including a hidden one, deliberately.

        Hiding is about what the interface *offers*, not about what it will
        run: a config file or ``--mode`` naming a superseded mode has to keep
        working, or hiding one would silently change what someone's setup does.
        """
        if mode_registry.get(name) is None or self.quarantine.is_disabled(name):
            return
        self.mode_name = name
        self._strips = None
        if remember:
            self._preview_mode = None
            self.settings.mode = name
        self.refresh()

    def preview_mode(self, name: str) -> None:
        """Show a mode without committing to it — for the picker."""
        if self._preview_mode is None:
            self._preview_mode = self.mode_name
        self.set_mode(name, remember=False)

    def commit_mode(self) -> None:
        self._preview_mode = None
        self.settings.mode = self.mode_name

    def cancel_mode_preview(self) -> None:
        if self._preview_mode is not None:
            self.set_mode(self._preview_mode, remember=False)
            self._preview_mode = None

    def cycle_mode(self, step: int = 1) -> str:
        names = self.mode_names
        if not names:
            return self.mode_name
        try:
            i = (names.index(self.mode_name) + step) % len(names)
        except ValueError:
            i = 0  # current mode was just quarantined out of the list
        self.set_mode(names[i])
        return self.mode_name

    def _quarantine_mode(self, name: str, detail: str) -> None:
        """Disable a mode that keeps failing and move somewhere safe."""
        first_line = (
            detail.strip().splitlines()[-1] if detail.strip() else "unknown error"
        )
        m = mode_registry.get(name)
        who = f"plugin {m.plugin}" if m and m.plugin else "mode"
        if self.on_mode_disabled is not None:
            try:
                self.on_mode_disabled(name, f"{who} {name} disabled — {first_line}")
            except Exception:
                pass
        if self.mode_name == name:
            fallback = next((n for n in self.mode_names if n != "None"), "None")
            self.set_mode(fallback, remember=False)

    # ── themes ───────────────────────────────────────────────────────────────

    @property
    def theme_names(self) -> list[str]:
        return [AUTO, *self._themes.keys()]

    @property
    def theme_name(self) -> str:
        return self._theme_name

    def apply_theme(self, name: str, *, remember: bool = True) -> str:
        if name == AUTO:
            derived = theme_from_textual(self.app)
            if derived is not None:
                self.palette.set(derived)
            self._theme_name = AUTO
        else:
            theme = self._themes.get(name)
            if theme is None:
                self._themes = all_themes(self._config_dir)
                theme = self._themes.get(name)
            if theme is None:
                return self.palette.note
            self.palette.set(theme)
            self._theme_name = name

        if remember:
            self._preview = None
            self.settings.theme = self._theme_name
        self._paint_background()
        self._strips = None
        self.refresh()
        return self.palette.note

    def _paint_background(self) -> None:
        """Fill the terminal with the theme's own background colour.

        Left to Textual's ``$background`` the widget shows through to whatever
        the terminal is set to, so a dark theme over a light terminal is a
        light rectangle with coloured bars on it. A theme names its background
        for a reason — vantablack means black, gruvbox means #282828 — and the
        cells the modes leave blank should be that colour, not a guess.
        """
        colour = self.palette.theme.bg or "#000000"
        self.styles.background = colour
        try:
            self.screen.styles.background = colour
            self.app.screen.styles.background = colour
        except Exception:
            pass

    def preview_theme_object(self, theme) -> None:
        """Show a ``Theme`` that isn't in the registry yet — the editor's draft.

        ``preview_theme`` takes a *name* and looks it up, which a theme being
        invented does not have. Same restore path though: the name of whatever
        was showing is stashed on ``_preview``, so cancelling out of the editor
        goes through ``cancel_theme_preview`` unchanged.
        """
        if self._preview is None:
            self._preview = self._theme_name
        self.palette.set(theme)
        self._paint_background()
        self._strips = None
        self.refresh()

    def preview_theme(self, name: str) -> str:
        if self._preview is None:
            self._preview = self._theme_name
        return self.apply_theme(name, remember=False)

    def commit_theme(self) -> None:
        self._preview = None
        self.settings.theme = self._theme_name

    def cancel_theme_preview(self) -> None:
        if self._preview is not None:
            self.apply_theme(self._preview, remember=False)
            self._preview = None

    def cycle_theme(self, step: int = 1) -> str:
        names = self.theme_names
        try:
            i = names.index(self._theme_name)
        except ValueError:
            i = 0
        return self.apply_theme(names[(i + step) % len(names)])

    def reload_themes(self) -> int:
        """Re-read user theme files without restarting."""
        self._themes = all_themes(self._config_dir)
        self.apply_theme(self._theme_name, remember=False)
        return len(self._themes)

    # ── audio tuning ─────────────────────────────────────────────────────────

    def nudge_sensitivity(self, factor: float) -> float:
        v = self.analyser.nudge_sensitivity(factor)
        self.settings.sensitivity = v
        return v

    def nudge_gate(self, factor: float) -> float:
        v = self.analyser.nudge_gate(factor)
        self.settings.gate = v
        return v

    # Absolute setters, for the settings panel. The nudge pair above is what
    # the [ ] and g G keys use; a panel showing a value needs to be able to
    # put it somewhere specific rather than only step it.
    def set_sensitivity(self, value: float) -> float:
        self.analyser.sensitivity = max(0.15, min(8.0, float(value)))
        self.settings.sensitivity = self.analyser.sensitivity
        return self.analyser.sensitivity

    def set_gate(self, value: float) -> float:
        v = self.analyser.set_gate(value)
        self.settings.gate = v
        return v

    def restart_capture(self) -> None:
        self.capture.next_source()

    def reset_capture(self) -> None:
        self.capture.reset_source()

    # ── frame loop ───────────────────────────────────────────────────────────

    def _resize_bands(self, n: int) -> None:
        """Rebuild the smoothing state for a new band count.

        Silent frames carry the same length as live ones, so this only runs
        when the setting actually changes — not every time the music pauses.
        """
        self._spring = Spring(n)
        self._stereo_l = Spring(n)
        self._stereo_r = Spring(n)
        self._peaks = Peaks(n)
        self._mode_state.clear()  # cached geometry is sized for the old count
        self._strips = None

    def set_bands(self, n: int) -> int:
        """Change how many bars are drawn, live.

        One control, two mechanisms. Up to the analyser's native resolution the
        modes simply draw fewer bars out of the same analysis; past it, the
        analyser resolves more bands for real. ``0`` fits the terminal width.
        """
        n = int(n)
        self.settings.bands = 0 if n <= 0 else max(8, min(64, n))
        self.analyser.set_bands(self.settings.bands or N_BANDS)
        self._mode_state.clear()
        self._strips = None
        self.refresh()
        return self.settings.bands

    def unlimited_info(self) -> tuple[int, int | None]:
        """``(resolved fps, detected Hz or None)`` for the settings row.

        The row shows both because they are different failures. A probe that
        returns a plausible-but-wrong number and a probe that fails and falls
        back to 60 produce the same *rate*, and a user on a 165 Hz panel who
        sees "display rate unknown" can report the one fact needed to fix it.
        """
        return self._unlimited

    def _resolve_fps(self, fps: int) -> int:
        """Turn a requested rate — possibly the unlimited sentinel — into Hz."""
        if int(fps) == FPS_UNLIMITED:
            return self._unlimited[0]
        return max(15, min(FPS_MAX, int(fps)))

    def _retime(self, fps: int, *, requested: bool = False) -> None:
        """Re-pace the render timer.

        ``requested=True`` means this came from the user (a flag or a keybind)
        and should become the new saved preference. An adaptive re-pace does
        not touch ``settings`` — persisting it both eroded the user's setting
        across sessions and made the recovery branch in ``_tick`` unreachable,
        because the ceiling it compares against had just been lowered to match.

        A requested rate is saved *as requested*, so the unlimited sentinel
        stays a sentinel in the config file; only the resolved number reaches
        the timer and ``_target_fps``.
        """
        resolved = self._resolve_fps(fps)
        if requested:
            self._target_fps = resolved
            self.settings.fps = int(fps)
        if self._timer is not None and resolved == self._fps:
            return
        fps = self._fps = resolved
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(1.0 / fps, self._tick)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.2, max(1e-4, now - self._last))
        self._last = now
        self._frame += 1

        frame = self.analyser.frame
        # The band count is settable at runtime, so the springs have to follow
        # the analyser rather than a module constant. Cheap to check, and the
        # alternative is a shape mismatch the moment someone changes it.
        if len(frame.bands) != len(self._spring.x):
            self._resize_bands(len(frame.bands))

        self._spring.step(frame.bands, dt)
        self._peaks.step(self._spring.x, dt)
        self._stereo_l.step(frame.bands_l, dt)
        self._stereo_r.step(frame.bands_r, dt)
        if frame.seq != self._last_seq:
            self._trace.step(frame.wave, dt)
            self._last_seq = frame.seq

        self._dt = dt
        self._frame_data = frame

        # Pacing is safe to adapt now that the physics is expressed in seconds —
        # changing fps no longer changes how the animation feels, only how
        # finely it is sampled. That was not true before.
        if self._frame % 45 == 0 and self._build_ms is not None:
            budget = 1000.0 / self._fps * 0.5
            if self._build_ms > budget and self._fps > 30:
                # 6 rather than 10: with a coarse step the pacer can only ever
                # sit on multiples of ten, so a machine that comfortably holds
                # 54 gets dropped to 50 and one that wants 48 lands on 40.
                self._retime(self._fps - 6)
            elif self._build_ms < budget * 0.35 and self._fps < self._target_fps:
                self._retime(min(self._fps + 6, self._target_fps))

        # A mode that can't hold the budget gets its previous frame reused on
        # alternate ticks rather than dragging the whole UI down with it. The
        # motion physics is unaffected — it's integrated in seconds, so this
        # halves the sampling rate and changes nothing else.
        cost = self._mode_ms.get(self.mode_name, 0.0)
        if cost > SLOW_MODE_MS and self._frame % 2:
            self.refresh()
            return

        self._strips = None
        self.refresh()

    # ── painting ─────────────────────────────────────────────────────────────

    def render_line(self, y: int) -> Strip:
        strips = self._strips
        if strips is None:
            strips = self._strips = self._build()
        if 0 <= y < len(strips):
            return strips[y]
        return Strip.blank(self.size.width)

    def _animate_ramp(self, codes, cidx, bidx, w: int):
        """Spread one full spectrum across the width.

        Only animated themes (``theme.animated``) call this. The steady flow
        over time comes from rotating the palette's colour loop (see the
        ``set_phase`` call in ``_build``); here we only add the per-column
        offset that lays the rainbow across the bands instead of leaving it a
        single hue. Applied to both the foreground and background index so
        fg/bg pairs stay coherent.

        Rounded rather than floored: floor sends every column in a
        ``RAMP_STEPS/w`` span to the *same* bucket, so at a wide terminal
        (say 200 columns against a 64-step ramp) each colour visibly held for
        three-odd columns before jumping to the next — a staircase, not a
        gradient. Rounding centres each bucket's span on the column nearest
        its true position instead of always taking the low end, which is the
        difference between a smooth sweep and a visibly pixelated one at the
        ramp resolutions this actually runs at.

        Only shifted where ``codes`` is not blank. make_strips run-length
        encodes on colour-index *changes*, and a bar mode's empty space above
        the bars is normally one constant index per row — courtesy of a
        vertical gradient that doesn't vary with column — so it collapses to
        one Segment no matter how wide the terminal is. Shifting blank cells
        by column too broke that for no visible gain: a space has no glyph, so
        its colour is never seen, but make_strips still had to build a
        Segment and look up a Style for every one of those invisible slivers.
        Profiled on Bars at 400x100: a quiet signal (6% of cells lit) still
        cost 4.3 ms in make_strips before this — the same as loud (89% lit) —
        because *every* row was fully fragmented regardless of how much of it
        was actually visible. Masking dropped the quiet case to 1.1 ms; loud
        is now the expensive case (6.8 ms) instead of every frame paying
        loud's price. That constant tax was also large enough on its own to
        occasionally trip the adaptive frame-rate guard below, which reads as
        a stutter that then "catches up" once the average recovers — exactly
        the symptom reported, and tracking real visible cost instead of a
        flat per-frame cost is what removes it, not just makes it smaller.
        """
        cols = np.arange(w, dtype=np.float64)
        shift = np.rint(cols * RAMP_STEPS / max(w, 1)).astype(np.int32)
        lit = codes != SPACE
        shifted = (cidx.astype(np.int32) + shift[None, :]) % RAMP_STEPS
        cidx = np.where(lit, shifted, cidx)
        if bidx is not None:
            shifted_b = (bidx.astype(np.int32) + shift[None, :]) % RAMP_STEPS
            bidx = np.where(lit, shifted_b, bidx)
        return cidx, bidx

    def _build(self) -> list[Strip]:
        w, h = self.size.width, self.size.height
        if w < 2 or h < 1:
            return []

        m = mode_registry.get(self.mode_name)
        if m is None:
            return []

        frame = getattr(self, "_frame_data", None)
        if frame is None:
            from .analysis import Frame

            frame = Frame()

        # Difference the onset counter once, here, rather than in every mode
        # that wants beats. Clamped at zero because a restarted analyser hands
        # back a counter that begins again from nothing, and a negative delta
        # is not a burst of beats played backwards.
        onsets = max(0, frame.onset_seq - self._last_onset_seq)
        self._last_onset_seq = frame.onset_seq

        ctx = Ctx(
            w=w,
            h=h,
            bands=self._spring.x,
            peaks=self._peaks.value,
            bands_l=self._stereo_l.x,
            bands_r=self._stereo_r.x,
            wave=self._trace.value if self._trace.value is not None else frame.wave,
            stereo=frame.stereo,
            frame=self._frame,
            t=time.monotonic() - self._t0,
            dt=getattr(self, "_dt", 1.0 / self._fps),
            energy=float(self._spring.x.mean()),
            silent=frame.silent,
            palette=self.palette,
            state=self._mode_state.setdefault(m.name, {}),
            bars=self.settings.bands,
            onset_seq=frame.onset_seq,
            onsets=onsets,
            onset_strength=frame.onset_strength,
            flux=frame.flux,
            tempo_bpm=frame.tempo_bpm,
            beat_phase=frame.beat_phase,
        )

        from .modes import empty

        t0 = time.perf_counter()
        try:
            out = m.fn(ctx)
            if m.is_plugin:
                # only plugins pay for validation; built-ins are covered by the
                # test suite, and this turns a crash deep inside the strip
                # builder into a message naming the plugin that caused it
                out = validate(out, w, h)
            self.quarantine.record_success(m.name)
        except (Exception, BadModeOutput):
            detail = traceback.format_exc(limit=6)
            if self.quarantine.record_failure(m.name, detail):
                self._quarantine_mode(m.name, detail)
            out = empty(w, h)

        if len(out) == 3:
            codes, cidx, bidx = out
        else:
            codes, cidx = out
            bidx = None

        # animated themes flow their colour ramp. The palette is rotated to the
        # current point on its loop — a fractional phase, so the colours glide a
        # fraction of a step each frame instead of jumping a whole step — and the
        # per-column offset spreads the spectrum across the bands. _build runs
        # each frame, so the rainbow drifts live.
        if getattr(self.palette.theme, "animated", False):
            phase = (time.monotonic() - self._t0) / RAINBOW_SECONDS_PER_CYCLE
            self.palette.set_phase(phase)
            cidx, bidx = self._animate_ramp(codes, cidx, bidx, w)

        strips = make_strips(codes, cidx, self.palette, bidx)
        ms = (time.perf_counter() - t0) * 1000.0
        prev = self._mode_ms.get(m.name)
        self._mode_ms[m.name] = ms if prev is None else prev * 0.7 + ms * 0.3
        self._build_ms = (
            ms if self._build_ms is None else self._build_ms * 0.85 + ms * 0.15
        )
        return strips
