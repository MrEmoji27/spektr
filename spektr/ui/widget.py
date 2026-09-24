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

from .. import display as display_probe
from .. import dissolve
from .. import modes as mode_registry
from ..analysis import ANALYSES_PER_SEC, N_BANDS, Analyser
from ..capture import Capture
from ..config import (
    ECO_BANDS,
    ECO_FPS,
    FPS_MAX,
    FPS_UNLIMITED,
    MOTION_CHOICES,
    MOTION_DEFAULT,
    Settings,
)
from ..modes import Ctx
from ..motion import (
    GLIDE_BLEND_TAU,
    GLIDE_PUNCH_S,
    PROFILES,
    Peaks,
    Spring,
    Trace,
    spread,
)
from ..palette import AUTO, RAMP_STEPS, Palette, all_themes, theme_from_textual
from ..plugins import BadModeOutput, Quarantine, validate
from ..render import SPACE, direct, make_strips
from .warm import ModeWarmer

#: A plugin allowed to eat the whole frame budget would stutter the entire UI,
#: so anything slower than this gets its previous frame reused on alternate
#: ticks. cliamp caps Lua plugins at 10 ms for the same reason.
SLOW_MODE_MS = 11.0

#: How long an animated theme's colour loop takes to turn once, in seconds.
#: Expressed as a duration rather than a rate tied to RAMP_STEPS, so raising
#: the ramp's resolution for a smoother gradient doesn't also speed up the
#: animation — the two used to share one constant and silently coupled.
RAINBOW_SECONDS_PER_CYCLE = 10.7

#: The tempo a morph's sweep is tuned for. A faster track crosses the screen
#: sooner and a slower one takes its time, within the clamp in
#: :meth:`AudioVisualizer._morph_style`: past a point the sweep is either so
#: quick it is not one, or so slow that the change arrives in two halves.
TEMPO_REFERENCE = 120.0


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
        #: Whether the picker, the cycle keys and shuffle offer the subcell
        #: variants. Read from the setting at construction and flipped live by
        #: the settings panel; the modes themselves are registered and
        #: selectable either way, so this only changes what is *offered*.
        self.show_fine = bool(getattr(self.settings, "fine_modes", False))
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

        self.settings.transparent_background = bool(
            getattr(self.settings, "transparent_background", False)
        )
        self.palette = Palette(transparent=self.settings.transparent_background)
        self._themes = all_themes(self._config_dir)
        self._theme_name = self.settings.theme

        #: Which motion personality the bars move with — ``snappy`` (the
        #: default tuning) or ``glide`` (the slower, cava-like feel). The
        #: setting is validated here rather than trusted, because a config
        #: written by a newer or older build may carry a profile this one
        #: does not know; falling back keeps the widget constructible.
        self._motion = (
            self.settings.motion
            if self.settings.motion in MOTION_CHOICES
            else MOTION_DEFAULT
        )
        # Write the settled name back, so an unclamped Settings handed straight
        # to the constructor still saves a value the next build will accept —
        # the same write-back analyser.sensitivity gets two lines above.
        self.settings.motion = self._motion
        self._spring, self._stereo_l, self._stereo_r = self._new_springs(N_BANDS)
        self._peaks = Peaks(N_BANDS)
        self._trace = Trace(tau=0.028)
        # Temporal pre-blends for the glide profile — cava's noise-reduction
        # analogue, one per spring so stereo stays independent. Idle (value
        # None) while snappy is selected, and re-seeded from live audio on
        # the first glide frame, so switching profiles never replays history.
        self._band_blend = Trace(tau=GLIDE_BLEND_TAU)
        self._stereo_l_blend = Trace(tau=GLIDE_BLEND_TAU)
        self._stereo_r_blend = Trace(tau=GLIDE_BLEND_TAU)
        #: The beat counter as glide last saw it, and when the punch-through
        #: it started runs out. See :data:`GLIDE_PUNCH_S`.
        self._glide_accent_seq = 0
        self._punch_until = -1.0

        self._mode_state: dict[str, dict] = {}
        #: The mode shown before this one, most recent first, and the mode
        #: shuffle has already chosen for next. Both are what the window holds
        #: ahead of the menu order; see :meth:`_mode_window`.
        self._recent: list[str] = []
        self._upcoming: str | None = None
        #: Draws the window's first frames off the render path.
        self._warmer = ModeWarmer()
        self._strips: list[Strip] | None = None
        #: The picture's own cells, for the frames that are written straight to
        #: the terminal rather than composed by Textual. See :meth:`_paint`.
        self._screen = direct.Screen()
        self._direct = False
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
        self._dissolve_from: str | None = None
        self._dissolve_started = 0.0
        #: Whether this morph runs for the short span. True for the handful of
        #: switches a person made by hand, where waiting out a shuffle-length
        #: morph would feel like the key did nothing.
        self._dissolve_quick = False
        #: What beats landing during this morph have added to its travel. See
        #: ``dissolve.PUSH``; reset when a morph starts, carried while it runs.
        self._dissolve_push = 0.0
        #: The last picture drawn, and which mode drew it — the morph needs
        #: both, since a frame is only a stand-in for the outgoing mode if
        #: that mode is the one it came from. See :meth:`_outgoing`.
        #: The band count to put back when eco is switched off again.
        self._eco_bands_were: int | None = None
        self._last_frame: tuple | None = None
        self._last_frame_mode: str | None = None
        self._frozen_old: tuple | None = None

        self.quarantine = Quarantine()
        #: called with (mode_name, message) when a mode is disabled
        self.on_mode_disabled = None
        self._mode_ms: dict[str, float] = {}
        self._refresh_mode_window()

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
        self._warmer.stop()
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
        """Every mode the picker, the cycle keys and shuffle offer.

        Quarantined ones are left out so cycling cannot land you back on
        something broken, and hidden ones — the subcell variants — because
        they are opt-in. Both are still registered and both still render if
        something asks for them by name, which is what ``--mode`` and a saved
        config do.

        The loadout narrows what is left. It is deliberately the *last* filter
        and a purely subtractive one: a name it lists that is quarantined,
        hidden or no longer registered does not come back, because the loadout
        says which of the offered modes you want, not which modes exist. An
        empty loadout means no restriction.

        This is the one place the three surfaces agree on what "available"
        means — the picker reads it, ``cycle_mode`` reads it, and shuffle
        reads it — so the loadout could not narrow one and miss another.
        """
        pool = mode_registry.MODES if self.show_fine else mode_registry.listed()
        names = [m.name for m in pool if not self.quarantine.is_disabled(m.name)]
        chosen = set(self.settings.loadout or ())
        if not chosen:
            return names
        kept = [n for n in names if n in chosen]
        # A loadout none of whose modes survived is a loadout that would leave
        # the interface with nothing to offer — no cycling, an empty picker,
        # a shuffle that cannot move. That is never what the user meant, and
        # it is reachable without them doing anything wrong: quarantine can
        # empty a small loadout on its own. Fall back to the full list.
        return kept or names

    def _mode_window(self, extra: str | None = None) -> list[str]:
        """The five modes that keep their working memory, likeliest first.

        The mode on screen; the one it is morphing out of; the one shuffle has
        already picked for next; the one shown before this; then the menu
        order from here, which is what the cycle keys reach. It used to be the
        menu order alone, which is the one thing shuffle and the picker never
        follow -- and the mode just left was the first to be evicted, so going
        straight back to it paid for its whole setup again.
        """
        names = self.mode_names
        if self.mode_name not in names:
            names = [mode.name for mode in mode_registry.MODES]
        if not names:
            return []
        try:
            start = names.index(self.mode_name)
        except ValueError:
            start = 0
        count = min(5, len(names))
        if extra is None:
            extra = getattr(self, "_dissolve_from", None)
        likely = [self.mode_name, extra, self._upcoming, *self._recent[:1]]
        likely += [names[(start + offset) % len(names)] for offset in range(1, count)]
        window: list[str] = []
        for name in likely:
            if name is None or name in window or mode_registry.get(name) is None:
                continue
            window.append(name)
            if len(window) == count:
                break
        return window

    def _refresh_mode_window(self, extra: str | None = None) -> None:
        previous = self._mode_state
        window = self._mode_window(extra)
        self._mode_state = {name: previous.get(name, {}) for name in window}
        for name in window:
            mode_registry.ensure_loaded(name)
        size = self.size
        self._warm_window(size.width, size.height)

    def _warm_window(self, w: int, h: int) -> None:
        """Have the warmer draw a first frame for every mode the window holds
        that has not drawn one at this size. Not the mode on screen, or the
        one it is morphing out of: those are being drawn right now."""
        if w < 2 or h < 1:
            return
        busy = (self.mode_name, self._dissolve_from)
        for name, state in self._mode_state.items():
            if name in busy:
                continue
            if any(isinstance(k, tuple) and k[1:] == (w, h) for k in state):
                continue
            self._warmer.request(name, state, w, h, self.palette)

    def expect(self, name: str | None) -> None:
        """Name the mode that is coming next, so its memory is kept and its
        first frame drawn before it is shown. Shuffle calls this as soon as it
        has picked."""
        if name == self.mode_name:
            name = None
        self._upcoming = name
        self._refresh_mode_window(self._dissolve_from)

    def redraw(self) -> None:
        """Throw away the cached frame and build the next one from scratch.

        For a change that alters how the *current* picture is drawn rather
        than what is drawn — switching subcell geometry, say. Without it the
        cached strips survive until something else invalidates them and the
        setting looks like it did nothing.
        """
        self._invalidate()
        self.refresh()

    def set_mode(
        self,
        name: str,
        *,
        remember: bool = True,
        dissolve: bool = False,
        quick: bool = False,
        from_mode: str | None = None,
    ) -> None:
        """Switch to a mode by name — including a hidden one, deliberately.

        Hiding is about what the interface *offers*, not about what it will
        run: a config file or ``--mode`` naming a hidden mode has to keep
        working, or hiding one would silently change what someone's setup does.

        ``dissolve`` morphs out of what is on screen rather than cutting, and
        is what every switch a person makes passes — shuffle, the cycle keys,
        the picker. ``quick`` shortens the morph to ``FAMILY_SECONDS`` however
        far apart the two modes are. ``from_mode`` names the picture to morph
        out of when that is not the mode currently running: the picker
        previews live, so at the moment its choice is committed the mode on
        screen is the one being committed, and the change worth animating is
        the one from where the picker was opened.
        """
        if mode_registry.get(name) is None or self.quarantine.is_disabled(name):
            return
        previous = self.mode_name
        source = from_mode or previous
        if name != previous:
            self._recent = [previous, *(n for n in self._recent if n not in (previous, name))][:3]
        if name == self._upcoming:
            self._upcoming = None
        if dissolve and name != source:
            self._start_morph(source, quick=quick)
        else:
            self._dissolve_from = None
        self.mode_name = name
        self._refresh_mode_window(self._dissolve_from)
        self._invalidate()
        if remember:
            self._preview_mode = None
            self.settings.mode = name
        self.refresh()

    def _morph_span(self) -> float:
        """How long the morph that is running lasts.

        One of the two lengths ``dissolve`` offers, never anything in between:
        a shuffle between two modes of different families is the long one, and
        everything else — the same family, or a switch someone made by hand —
        is the short one. A keypress that starts a morph the length of a
        shuffle would feel like a key that did nothing.
        """
        if self._dissolve_quick or self._same_family(
            self._dissolve_from or "", self.mode_name
        ):
            return dissolve.FAMILY_SECONDS
        return dissolve.SECONDS

    def _morph_style(self, frame, family: bool) -> dissolve.Style:
        """How this frame of the morph should move.

        The sweep is shortened inside one family — the two pictures are nearly
        the same shape and that morph is the short one — and shortened again
        by a fast tempo, so the change crosses the screen in time with the
        music rather than to a schedule of its own. The bands go in as they
        are and the morph reorders its own sweep with them: the loud parts of
        the picture set off first. And the new picture enters the way its
        family moves, rather than every switch arriving in one order.
        """
        scale = dissolve.SWEEP_FAMILY if family else 1.0
        tempo = float(getattr(frame, "tempo_bpm", 0.0) or 0.0)
        if tempo > 0.0:
            scale *= max(0.7, min(1.4, TEMPO_REFERENCE / tempo))
        # The incoming picture arrives the way its family moves: bars rise,
        # particles burst out, a field ripples. See dissolve.ENTRANCES.
        incoming = mode_registry.get(self.mode_name)
        entrance = dissolve.ENTRANCES.get(incoming.group) if incoming else None
        return dissolve.Style(
            sweep=dissolve.SWEEP * scale,
            wavefront=dissolve.WAVEFRONT * scale,
            levels=self._spring.x,
            entrance=entrance,
        )

    def _start_morph(self, source: str, *, quick: bool) -> None:
        """Begin morphing out of ``source``'s picture into the current mode.

        ``source`` is normally the mode running now, and the frame it drew is
        the picture being carried into the new one. It is not always — the
        picker commits a mode it has already previewed, so what it morphs out
        of is the mode the picker was opened on, which nothing has drawn
        recently. Then there is no frame to hold and the outgoing mode is
        redrawn live for the length of the morph, which is what a morph out of
        a picture that is not on screen costs.
        """
        self._dissolve_from = source
        self._dissolve_started = time.monotonic()
        self._dissolve_quick = quick
        self._dissolve_push = 0.0
        self._frozen_old = (
            self._last_frame if self._last_frame_mode == source else None
        )

    def preview_mode(self, name: str) -> None:
        """Show a mode without committing to it — for the picker.

        Instant, and deliberately: arrowing down the list changes modes faster
        than a morph lasts, so every one of them would start a morph that the
        next keypress would cut off. The morph belongs to the choice, not to
        the browsing — see :meth:`commit_mode`.
        """
        if self._preview_mode is None:
            self._preview_mode = self.mode_name
        self.set_mode(name, remember=False)

    def commit_mode(self, *, dissolve: bool = False) -> None:
        """Keep the previewed mode. ``dissolve`` morphs into it from where the
        picker was opened, which is the change the commit actually makes."""
        source, self._preview_mode = self._preview_mode, None
        if dissolve and source is not None and source != self.mode_name:
            self._start_morph(source, quick=True)
            self._invalidate()
        self.settings.mode = self.mode_name

    def cancel_mode_preview(self) -> None:
        if self._preview_mode is not None:
            back = self._preview_mode
            self._preview_mode = None
            self.set_mode(back, remember=False, dissolve=True, quick=True)

    def cycle_mode(self, step: int = 1) -> str:
        names = self.mode_names
        if not names:
            return self.mode_name
        try:
            i = (names.index(self.mode_name) + step) % len(names)
        except ValueError:
            i = 0  # current mode was just quarantined out of the list
        self.set_mode(names[i], dissolve=True, quick=True)
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
        self._invalidate()
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
        self._invalidate()
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

    def set_motion(self, name: str) -> str:
        """Switch the motion profile, live — the settings panel's motion row.

        Returns what was taken (the clamped name), so a row can show the
        settled value rather than the ask. The springs are retuned in place
        rather than replaced: position and velocity survive, so toggling
        mid-song eases into the new character instead of resetting the bars.
        The glide pre-blends are dropped, not carried — a blend accumulated
        under one profile is history the other profile never saw.
        """
        name = str(name)
        if name not in PROFILES:
            return self._motion
        self.settings.motion = name
        if name != self._motion:
            self._motion = name
            for spring in (self._spring, self._stereo_l, self._stereo_r):
                spring.retune(**PROFILES[name])
            for blend in (self._band_blend, self._stereo_l_blend, self._stereo_r_blend):
                blend.value = None
            # The picture is about to move differently; a cached frame built
            # under the old profile is stale by definition.
            self._strips = None
            self.refresh()
        return name

    def set_transparent_background(self, on: bool) -> bool:
        """Leave empty cells to the terminal, or paint the theme into them, live.

        The settings panel's background row. The palette rebuilds its styles
        on the spot and the cached frame is dropped, so the very next frame
        goes out with the new backgrounds rather than one built under the old
        styles. The widget and screen backgrounds are left as they are either
        way: Textual draws a Line-API widget's lines as they are handed over,
        so those colours never reach a visualizer cell — they only fill space
        no widget covers.
        """
        on = bool(on)
        self.settings.transparent_background = on
        self.palette.set_transparent(on)
        self._invalidate()
        self.refresh()
        return on

    def restart_capture(self) -> None:
        self.capture.next_source()

    def reset_capture(self) -> None:
        self.capture.reset_source()

    # ── frame loop ───────────────────────────────────────────────────────────

    def _new_springs(self, n: int) -> tuple[Spring, Spring, Spring]:
        """Three springs tuned to the current motion profile.

        One factory rather than three call sites, so a profile change can
        never leave the main spring and the stereo pair disagreeing about
        what character they are in.
        """
        params = PROFILES[self._motion]
        return Spring(n, **params), Spring(n, **params), Spring(n, **params)

    def _resize_bands(self, n: int) -> None:
        """Rebuild the smoothing state for a new band count.

        Silent frames carry the same length as live ones, so this only runs
        when the setting actually changes — not every time the music pauses.
        """
        self._spring, self._stereo_l, self._stereo_r = self._new_springs(n)
        self._peaks = Peaks(n)
        # The blends re-seed themselves from live audio on their next step
        # (Trace treats a shape change as a fresh start), which is the right
        # behaviour here too: a resized blend has no honest past.
        self._mode_state.clear()  # cached geometry is sized for the old count
        self._refresh_mode_window(self._dissolve_from)
        self._invalidate()

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
        self._refresh_mode_window(self._dissolve_from)
        self._invalidate()
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

    def affordable(self, name: str) -> bool:
        """Whether ``name`` has drawn inside the frame budget at this size.

        A mode nobody has drawn yet counts as affordable: refusing to try it
        would mean eco silently shrinking the roster to whatever happened to
        run first.
        """
        cost = self._mode_ms.get(name)
        return cost is None or cost <= 1000.0 / max(1, self._target_fps)

    @staticmethod
    def _same_family(a: str, b: str) -> bool:
        """Whether two modes belong to the same group in the picker."""
        one, two = mode_registry.get(a), mode_registry.get(b)
        return one is not None and two is not None and one.group == two.group

    def eco_active(self) -> bool:
        """Whether eco is in force."""
        return self.settings.eco == "on"

    def _apply_eco(self) -> None:
        """Put the frame rate and band count where eco wants them.

        Retiming needs a running app; the settings panel is also built and
        checked without one, so a setter called there changes the setting and
        leaves the clock alone rather than raising.
        """
        try:
            self._apply_eco_now()
        except RuntimeError:
            pass

    def _apply_eco_now(self) -> None:
        # Bands first, then the clock: retiming is the part that needs a
        # running app, so doing it last means the band count still follows
        # when the panel is built without one.
        if self.eco_active():
            if self.settings.bands > ECO_BANDS:
                self._eco_bands_were = self.settings.bands
                self.set_bands(ECO_BANDS)
            self._retime(min(self._fps or ECO_FPS, ECO_FPS))
        else:
            if self._eco_bands_were is not None:
                self.set_bands(self._eco_bands_were)
                self._eco_bands_were = None
            self._retime(self._resolve_fps(self.settings.fps))

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

        # Motion profile: glide conditions the targets before the springs see
        # them — temporal pre-blend first (cava's noise-reduction analogue),
        # then the neighbour spread (its monstercat analogue) — so transients
        # arrive softened and energy leans outward from hot bands. snappy,
        # the default, feeds the raw spectrum straight through; the branch is
        # per frame but costs one attribute compare.
        #
        # Except on a beat. For GLIDE_PUNCH_S after one, the raw spectrum
        # goes past the pre-blend and the springs rise at snappy's speed, so
        # the kick and snare land while the hats and everything between hits
        # still glide. The blends keep following the audio throughout, so
        # nothing jumps when the punch ends.
        punch = {}
        if self._motion == "glide":
            if frame.accent_seq != self._glide_accent_seq:
                self._glide_accent_seq = frame.accent_seq
                self._punch_until = now + GLIDE_PUNCH_S
            blended = (
                self._band_blend.step(frame.bands, dt),
                self._stereo_l_blend.step(frame.bands_l, dt),
                self._stereo_r_blend.step(frame.bands_r, dt),
            )
            if now < self._punch_until:
                blended = tuple(
                    np.maximum(b, raw) for b, raw in
                    zip(blended, (frame.bands, frame.bands_l, frame.bands_r))
                )
                fast = PROFILES["snappy"]
                punch = {"attack": fast["attack"], "attack_zeta": fast["attack_zeta"]}
            bands_t, bands_l_t, bands_r_t = (spread(b) for b in blended)
        else:
            bands_t, bands_l_t, bands_r_t = frame.bands, frame.bands_l, frame.bands_r

        self._spring.step(bands_t, dt, **punch)
        self._peaks.step(self._spring.x, dt)
        self._stereo_l.step(bands_l_t, dt, **punch)
        self._stereo_r.step(bands_r_t, dt, **punch)
        if frame.seq != self._last_seq:
            self._trace.step(frame.wave, dt)
            self._last_seq = frame.seq

        self._dt = dt
        self._frame_data = frame

        # Twice a second, see whether a mode the window holds needs its first
        # frame drawn -- after a resize every one of them does. Cheap when
        # there is nothing to do: five dictionary scans.
        if self._frame % 30 == 0:
            size = self.size
            self._warm_window(size.width, size.height)

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
            if self._painting_directly():
                return  # last frame is still on the screen, untouched
            self.refresh()
            return

        self._paint()

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

    def _render_mode(self, name: str, frame, w: int, h: int, onsets: int) -> tuple:
        m = mode_registry.get(name)
        if m is None:
            from ..modes import empty

            return empty(w, h)
        state = self._mode_state.get(name)
        if state is None:
            self._refresh_mode_window(self._dissolve_from)
            state = self._mode_state.get(name, {})
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
            state=state,
            bars=self.settings.bands,
            onset_seq=frame.accent_seq,
            onsets=onsets,
            onset_strength=frame.accent_strength,
            flux=frame.flux,
            tempo_bpm=frame.tempo_bpm,
            beat_phase=frame.beat_phase,
            drums=frame.drums,
            chroma=frame.chroma,
            bar_phase=frame.bar_phase,
            beat_in_bar=frame.beat_in_bar,
            bar_confidence=frame.bar_confidence,
            key=frame.key,
            key_confidence=frame.key_confidence,
            key_uncertain=frame.key_uncertain,
        )

        # A warm-up of this mode may still be drawing into the same scratch.
        self._warmer.claim(name)
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
            from ..modes import empty

            detail = traceback.format_exc(limit=6)
            if self.quarantine.record_failure(m.name, detail):
                self._quarantine_mode(m.name, detail)
            out = empty(w, h)

        ms = (time.perf_counter() - t0) * 1000.0
        prev = self._mode_ms.get(m.name)
        self._mode_ms[m.name] = ms if prev is None else prev * 0.7 + ms * 0.3
        return out

    def _outgoing(self, name: str, frame, w: int, h: int, onsets: int) -> tuple:
        """The outgoing mode's picture: live if it fits, frozen if it does not.

        Both modes have to be drawn for the length of a dissolve, so a pair of
        expensive ones costs the sum of the two. Measured on the worst pair at
        400x100 that is 39 ms a frame — the app drawing at 25 fps for the whole
        0.6 s. When the pair does not fit the frame budget the outgoing mode's
        last frame is held instead: it stops animating while it fades, which
        over half a second reads as a still layer dissolving away rather than
        as a stutter across the whole screen.
        """
        budget = 1000.0 / max(1, self._fps)
        pair = self._mode_ms.get(name, 0.0) + self._mode_ms.get(self.mode_name, 0.0)
        frozen = self._frozen_old
        # Within a family both pictures are the same kind of thing and both
        # are reacting to the same music, so the outgoing one keeps drawing:
        # its bars go on bouncing while they become the new shape. Freezing it
        # there would stop motion the eye is already following. Family modes
        # are the cheap ones, so this is affordable; if the pair really cannot
        # keep up the frozen frame is still the fallback below.
        if self._same_family(name, self.mode_name) and pair <= budget * 2:
            return self._render_mode(name, frame, w, h, onsets)
        if pair <= budget or frozen is None:
            return self._render_mode(name, frame, w, h, onsets)
        if frozen[0].shape == (h, w):
            return frozen
        # The terminal was resized mid-dissolve, so the held frame is the wrong
        # shape to blend against; drawing it live is the only correct answer.
        return self._render_mode(name, frame, w, h, onsets)

    def _picture(self) -> tuple:
        """This frame's arrays: the mode, the morph, and the animated ramp.

        The picture, with nothing done about how it reaches the terminal —
        which is what the two painters below differ on, and the only thing
        they differ on.
        """
        w, h = self.size.width, self.size.height
        frame = getattr(self, "_frame_data", None)
        if frame is None:
            from ..analysis import Frame

            frame = Frame()

        # Difference the onset counter once, here, rather than in every mode
        # that wants beats. Clamped at zero because a restarted analyser hands
        # back a counter that begins again from nothing, and a negative delta
        # is not a burst of beats played backwards.
        #
        # The accented counter, not the raw one: the detector hears every hat
        # and ghost note, and a mode handed all of them spends most of its
        # beats on the background. See spektr.audio.accent.
        onsets = max(0, frame.accent_seq - self._last_onset_seq)
        self._last_onset_seq = frame.accent_seq

        out = self._render_mode(self.mode_name, frame, w, h, onsets)
        if self._dissolve_from is not None:
            family = self._same_family(self._dissolve_from, self.mode_name)
            progress = (time.monotonic() - self._dissolve_started) / self._morph_span()
            if progress >= 1.0:
                self._dissolve_from = None
                self._frozen_old = None
                self._refresh_mode_window()
            else:
                # A beat landing mid-morph shoves the shapes on: the change
                # lands with the music rather than to its own schedule. Only
                # the travel — the handover keeps its own clock, or a beat
                # would swap the colours before the shapes had met.
                if onsets:
                    self._dissolve_push = min(
                        dissolve.PUSH_MAX,
                        self._dissolve_push + dissolve.PUSH * min(onsets, 2),
                    )
                out = dissolve.blend(
                    self._outgoing(self._dissolve_from, frame, w, h, onsets),
                    out, dissolve.ease(progress),
                    gather=0.0 if family else dissolve.GATHER,
                    push=self._dissolve_push,
                    style=self._morph_style(frame, family),
                )
        self._last_frame = out
        self._last_frame_mode = self.mode_name

        if len(out) == 3:
            codes, cidx, bidx = out
        else:
            codes, cidx = out
            bidx = None

        # animated themes flow their colour ramp. The palette is rotated to the
        # current point on its loop — a fractional phase, so the colours glide a
        # fraction of a step each frame instead of jumping a whole step — and the
        # per-column offset spreads the spectrum across the bands. This runs
        # each frame, so the rainbow drifts live.
        # Which backgrounds are the ramp's floor has to be decided on the
        # mode's own indices: the animation below shifts them by column, after
        # which index 0 no longer means "nothing here".
        clear = bidx == 0 if bidx is not None and self.palette.transparent else None
        if getattr(self.palette.theme, "animated", False):
            phase = (time.monotonic() - self._t0) / RAINBOW_SECONDS_PER_CYCLE
            self.palette.set_phase(phase)
            cidx, bidx = self._animate_ramp(codes, cidx, bidx, w)
            # The colours moved under keys that did not: whatever is on screen
            # was drawn in the ramp's last position, so it all goes out again.
            self._screen.forget()
        return codes, cidx, bidx, clear

    def _build(self) -> list[Strip]:
        """The frame as strips, for Textual to compose and encode."""
        w, h = self.size.width, self.size.height
        if w < 2 or h < 1:
            return []
        t0 = time.perf_counter()
        codes, cidx, bidx, clear = self._picture()
        strips = make_strips(codes, cidx, self.palette, bidx, clear)
        self._note_build(t0)
        return strips

    def _note_build(self, started: float) -> None:
        """Fold this frame's build into the running average the pacer reads."""
        ms = (time.perf_counter() - started) * 1000.0
        self._build_ms = (
            ms if self._build_ms is None else self._build_ms * 0.85 + ms * 0.15
        )

    def _direct_region(self):
        """The part of this widget the chrome is not sitting on.

        The visualiser fills the whole screen and the header and footer are
        painted *over* it, so its own region includes their rows. Writing
        straight to the terminal across all of it wipes them out, which is
        what happened: the chrome vanished a frame after it was drawn. Whatever
        else the screen holds is measured here and kept out of.
        """
        region = self.content_region
        top = bottom = 0
        for other in self.screen.children:
            if other is self or not other.display:
                continue
            r = other.region
            if not r.height or not r.overlaps(region):
                continue
            if r.y <= region.y:                       # docked above the picture
                top = max(top, r.y + r.height - region.y)
            elif r.y + r.height >= region.y + region.height:   # and below it
                bottom = max(bottom, region.y + region.height - r.y)
        height = max(0, region.height - top - bottom)
        return region.__class__(region.x, region.y + top, region.width, height)

    def _painting_directly(self) -> bool:
        """Whether the picture can go straight to the terminal this frame.

        Only while the picture is the only thing on screen: a picker or a
        settings panel is docked over it and a notification floats above it, so
        anything Textual has put there is Textual's to keep. Transparent
        backgrounds are left out too — there a cell that names no colour is
        meant to show the terminal's own through, and that is a rule Textual's
        renderer owns.
        """
        app = self.app
        driver = getattr(app, "_driver", None)
        return (
            self._screen is not None
            and driver is not None
            and not getattr(driver, "is_inline", False)
            and getattr(app, "_overlay", None) is None
            and not getattr(app, "_notifications", None)
            and not self.palette.transparent
        )

    def _paint(self) -> None:
        """Draw the frame, directly or through Textual.

        The two paths are exclusive per frame: Textual composes the whole
        screen whenever it paints, so letting it paint a frame and then writing
        over it from here would be paying twice for one picture.
        """
        direct = self._painting_directly()
        if direct:
            region = self._direct_region()
            self._screen.place(region.width, region.height, region.x, region.y)
        if direct != self._direct:
            # Handing over either way, whatever is on screen is not what the
            # next frame is about to draw.
            self._direct = direct
            self._invalidate()
            self.refresh()
        if not direct:
            self._invalidate()
            self.refresh()
            return
        started = time.perf_counter()
        codes, cidx, bidx, clear = self._picture()
        frame = self._screen.frame(codes, cidx, bidx, self.palette, clear)
        self._note_build(started)
        if not frame:
            return
        app = self.app
        begin = getattr(app, "_begin_update", None)
        if begin is not None:
            begin()
        try:
            app._driver.write(frame)
        finally:
            end = getattr(app, "_end_update", None)
            if end is not None:
                end()
        app._driver.flush()

    def note_textual_paint(self) -> None:
        """Textual has painted over the whole screen; forget ours.

        Called by the app after its own display pass: whatever it drew is on
        top of the visualiser's cells now, so the next direct frame has to draw
        all of them again rather than the ones that changed.
        """
        self._screen.forget()

    def _invalidate(self) -> None:
        """Both painters' caches: the strips Textual holds, and our cells.

        Anything that makes the cached frame stale — a new picture, a different
        theme, another size — has to make it stale for both, or the one that
        was not invalidated goes on drawing under the old one.
        """
        self._strips = None
        self._screen.forget()
