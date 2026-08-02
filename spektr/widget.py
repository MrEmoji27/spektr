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

from . import modes as mode_registry
from .analysis import N_BANDS, Analyser
from .capture import Capture
from .config import Settings
from .modes import Ctx
from .motion import Peaks, Spring, Trace
from .palette import AUTO, Palette, all_themes, theme_from_textual
from .plugins import BadModeOutput, Quarantine, validate
from .render import make_strips

#: A plugin allowed to eat the whole frame budget would stutter the entire UI,
#: so anything slower than this gets its previous frame reused on alternate
#: ticks. cliamp caps Lua plugins at 10 ms for the same reason.
SLOW_MODE_MS = 11.0


class AudioVisualizer(Widget):
    DEFAULT_CSS = """
    AudioVisualizer {
        height: 1fr;
        background: $background;
    }
    """

    mode_name: reactive[str] = reactive("Bars")

    def __init__(self, device=None, settings: Settings | None = None,
                 allow_mic: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings or Settings()

        self.capture = Capture(device=device, allow_mic=allow_mic)
        self.analyser = Analyser(self.capture.ring, lambda: self.capture.samplerate)
        self.analyser.sensitivity = self.settings.sensitivity

        self.palette = Palette()
        self._themes = all_themes()
        self._theme_name = self.settings.theme

        self._spring = Spring(N_BANDS)
        self._stereo_l = Spring(N_BANDS)
        self._stereo_r = Spring(N_BANDS)
        self._peaks = Peaks(N_BANDS)
        self._trace = Trace(tau=0.028)

        self._mode_state: dict[str, dict] = {}
        self._strips: list[Strip] | None = None
        self._frame = 0
        self._t0 = time.monotonic()
        self._last = self._t0
        self._timer = None
        # The rate the user asked for, and the rate we are currently running.
        # Adaptive pacing moves the second one; the first is what gets saved,
        # so a slow machine never quietly rewrites the preference — and it is
        # also the ceiling adaptive pacing is allowed to climb back up to.
        self._target_fps = int(self.settings.fps)
        self._fps = self._target_fps
        self._build_ms: float | None = None
        self._last_seq = -1

        self._preview: str | None = None   # theme name being previewed
        self._preview_mode: str | None = None

        self.quarantine = Quarantine()
        #: called with (mode_name, message) when a mode is disabled
        self.on_mode_disabled = None
        self._mode_ms: dict[str, float] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self.apply_theme(self._theme_name, remember=False)
        self.mode_name = self.settings.mode if mode_registry.get(self.settings.mode) else "Bars"
        try:
            self.watch(self.app, "theme", self._on_app_theme, init=False)
        except Exception:
            pass

        self.capture.start()
        self.analyser.start()
        self._retime(self._target_fps, requested=True)

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
        """Every mode that can currently be selected — quarantined ones are
        hidden, so cycling can't land you back on something broken."""
        return [n for n in mode_registry.names() if not self.quarantine.is_disabled(n)]

    def set_mode(self, name: str, *, remember: bool = True) -> None:
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
            i = 0            # current mode was just quarantined out of the list
        self.set_mode(names[i])
        return self.mode_name

    def _quarantine_mode(self, name: str, detail: str) -> None:
        """Disable a mode that keeps failing and move somewhere safe."""
        first_line = detail.strip().splitlines()[-1] if detail.strip() else "unknown error"
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
                self._themes = all_themes()
                theme = self._themes.get(name)
            if theme is None:
                return self.palette.note
            self.palette.set(theme)
            self._theme_name = name

        if remember:
            self._preview = None
            self.settings.theme = self._theme_name
        self._strips = None
        self.refresh()
        return self.palette.note

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
        self._themes = all_themes()
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

    def restart_capture(self) -> None:
        self.capture.next_source()

    # ── frame loop ───────────────────────────────────────────────────────────

    def _retime(self, fps: int, *, requested: bool = False) -> None:
        """Re-pace the render timer.

        ``requested=True`` means this came from the user (a flag or a keybind)
        and should become the new saved preference. An adaptive re-pace does
        not touch ``settings`` — persisting it both eroded the user's setting
        across sessions and made the recovery branch in ``_tick`` unreachable,
        because the ceiling it compares against had just been lowered to match.
        """
        fps = max(15, min(120, int(fps)))
        if requested:
            self._target_fps = fps
            self.settings.fps = fps
        if self._timer is not None and fps == self._fps:
            return
        self._fps = fps
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(1.0 / fps, self._tick)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.2, max(1e-4, now - self._last))
        self._last = now
        self._frame += 1

        frame = self.analyser.frame
        targets = np.zeros(N_BANDS) if frame.silent else frame.bands

        self._spring.step(targets, dt)
        self._peaks.step(self._spring.x, dt)
        self._stereo_l.step(np.zeros(N_BANDS) if frame.silent else frame.bands_l, dt)
        self._stereo_r.step(np.zeros(N_BANDS) if frame.silent else frame.bands_r, dt)
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
                self._retime(self._fps - 10)
            elif self._build_ms < budget * 0.35 and self._fps < self._target_fps:
                self._retime(self._fps + 10)

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

        strips = make_strips(codes, cidx, self.palette, bidx)
        ms = (time.perf_counter() - t0) * 1000.0
        prev = self._mode_ms.get(m.name)
        self._mode_ms[m.name] = ms if prev is None else prev * 0.7 + ms * 0.3
        self._build_ms = ms if self._build_ms is None else self._build_ms * 0.85 + ms * 0.15
        return strips
