"""spektr — a terminal spectrum analyser for whatever your speakers are doing."""

from __future__ import annotations

import random
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from .. import __version__, config, nowplaying
from .. import loadouts as loadouts_module
from .. import modes as mode_registry
from .. import palette as palette_mod
from .pickers import (
    ColourPicker,
    HelpPanel,
    HexPrompt,
    LoadoutPicker,
    NamePrompt,
    Picker,
    key_label,
)
from .settings import Setting, SettingsPanel
from .widget import AudioVisualizer

#: How often shuffle picks a new mode, in seconds — long enough to actually
#: look at what changed, short enough to read as "shuffling" rather than
#: "occasionally different".
SHUFFLE_MODE_SECONDS = 15.0

#: Theme changes once every this-many mode changes, not every tick. The mode
#: is usually the more interesting axis, and flipping both at once — new
#: shapes *and* new colours in the same instant — reads as the picture
#: breaking rather than as a deliberate change.
SHUFFLE_THEME_EVERY = 3

#: How often to ask the OS what's playing. A media session doesn't change
#: fast enough to need polling every frame, and each poll is a real IPC round
#: trip (WinRT or D-Bus) rather than a free local read.
NOWPLAYING_POLL_SECONDS = 5.0


class Spektr(App):
    # The screen colour is taken over by the active spektr theme once the
    # visualiser mounts — see AudioVisualizer._paint_background.
    # Overlay panels are styled here in App CSS rather than on the widget
    # classes: in this Textual version a widget's own CSS attribute is not
    # applied to widgets that are mounted dynamically (after the app starts),
    # so the panel would render unframed and full-screen. The visualiser sits
    # on the base layer; the docked panel rides on the overlay layer above it,
    # leaving the bands visible and repainting behind it.
    CSS = """
    Screen { layers: base overlay; background: #000000; }
    AudioVisualizer { layer: base; height: 1fr; }

    Picker, SettingsPanel, NamePrompt, HelpPanel {
        layer: overlay;
        dock: right;
        width: auto;
        height: 100%;
        background: transparent;
    }
    Picker > #panel, SettingsPanel > #panel, NamePrompt > #panel, HelpPanel > #panel {
        height: 100%;
        background: $surface;
        border-left: tall $accent;
        padding: 0 1;
    }
    /* Picker lists fifty mode names; SettingsPanel holds eight rows with short
       notes. Both used to share one 32-wide rule, which was fine for the
       settings notes and far too narrow for what the mode picker showed then —
       a blurb per mode, some past sixty characters, every one of them wrapping
       into a jumbled paragraph. The blurbs are no longer shown here (see
       action_pick_mode), so the width is now set by the longest mode NAME plus
       the current-item marker, with room for the theme picker's swatches. Kept
       separate from SettingsPanel because the two still don't want the same
       number. */
    Picker > #panel {
        width: 40;
    }
    SettingsPanel > #panel, NamePrompt > #panel {
        width: 42;
    }
    /* The help panel is two columns — a key and what it does — where the
       others are one, so it needs the room. It was in none of these rules
       when it was added, which is not a panel that looks slightly wrong: with
       no `layer: overlay` and no width it mounted into the screen's own
       vertical layout at full size, pushed the Header and Footer off, and
       drew nothing, because its background is transparent and only #panel
       carries a surface. Pressing `h` looked exactly like a key that hides
       the chrome. */
    HelpPanel > #panel {
        width: 58;
    }
    Picker #title, SettingsPanel #title, NamePrompt #title, HelpPanel #title {
        color: $accent;
        text-style: bold;
        padding: 1 0 0 0;
    }
    Picker Input, SettingsPanel Input, NamePrompt Input {
        border: none;
        background: $surface;
        padding: 0;
        margin: 0 0 1 0;
    }
    Picker OptionList, SettingsPanel OptionList, HelpPanel OptionList {
        background: $surface;
        border: none;
        height: 1fr;
        scrollbar-size-vertical: 1;
    }
    Picker #hint, SettingsPanel #hint, NamePrompt #hint, HelpPanel #hint {
        color: $text-muted;
    }
    """

    TITLE = "spektr"
    SUB_TITLE = "system audio, drawn in the terminal"

    # Textual's default ctrl+p command palette is switched off rather than
    # fixed. Its "Theme" command opens Textual's own theme list — nord,
    # gruvbox, dracula, monokai, solarized, catppuccin-mocha/latte,
    # tokyo-night, several of them namesakes of spektr's own audio themes —
    # unranked and unrelated to the `t` picker's ramp. Picking "nord" there
    # recolours the header and footer chrome, not a single band, which reads
    # as the picker being broken rather than as a second, unrelated feature.
    # spektr already has its own filterable palette (`v` / `t`); a second one
    # a keystroke away, searching a different list under an identical name,
    # is a bug waiting to be filed rather than a feature worth keeping.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("m,space", "cycle_mode", "Mode"),
        Binding("M", "cycle_mode(-1)", "Prev mode", show=False),
        Binding("v", "pick_mode", "Modes"),
        Binding("l", "loadout", "Loadout"),
        Binding("t", "pick_theme", "Themes"),
        Binding("c", "settings", "Settings"),
        Binding("T", "cycle_theme", "Next theme", show=False),
        # Off the footer, not off the key. Adding `l` pushed the row past
        # 80 columns and what fell off the end was `h Help` and `q Quit` —
        # the two a first-time user cannot afford to lose. This is the
        # cheapest thing to drop: a display toggle nobody needs in their
        # first minute, still listed in `h` like every hidden binding.
        Binding("f", "toggle_chrome", "Full screen", show=False),
        Binding("d", "next_source", "Next source", show=False),
        Binding("D", "default_source", "Default output", show=False),
        Binding("r", "reload", "Reload themes + plugins", show=False),
        # s used to be "show source status" — moved into the settings panel
        # (see action_settings' source row) since it's now free, and s for
        # Shuffle is a plainer mnemonic than the a it had before.
        Binding("s", "toggle_shuffle", "Shuffle"),
        Binding("left_square_bracket", "gain(-1)", "Sens -", show=False),
        Binding("right_square_bracket", "gain(1)", "Sens +", show=False),
        Binding("g", "gate(-1)", "Gate -", show=False),
        Binding("G", "gate(1)", "Gate +", show=False),
        Binding("h,question_mark", "help", "Help"),
        Binding("p", "show_perf", "Frame time and FPS", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        device=None,
        settings: config.Settings | None = None,
        allow_mic: bool = False,
        config_dir=None,
    ):
        super().__init__()
        self._device = device
        self._allow_mic = allow_mic
        #: where settings, loadouts, themes, plugins and ascii reels live;
        #: None means the platform default from palette.config_dir()
        self._config_dir = config_dir
        # Clamped whatever the source. `config.load` already does it, but an
        # injected Settings — a test, an embedder —
        # goes straight into the widget, where an fps of "soon" raises out of
        # `int()` before anything is on screen. Nothing in a settings object is
        # worth failing to start over.
        self.settings = (settings or config.load(config_dir)).clamp()
        #: the picker/settings overlay currently mounted, if any
        self._overlay = None
        #: the shuffle timer, or None when shuffle is off
        self._shuffle_timer = None
        self._shuffle_count = 0
        self._loadouts = loadouts_module.load(config_dir)
        #: the capture/device status text — what the header falls back to
        #: when nothing is playing that the OS will report on
        self._capture_status = self.SUB_TITLE
        #: "Artist — Title" from the OS media session, or None
        self._now_playing: str | None = None
        #: Last non-empty identity observed by track-timed shuffle. The first
        #: observation establishes a baseline rather than causing a switch.
        self._shuffle_track: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield AudioVisualizer(
            device=self._device,
            settings=self.settings,
            allow_mic=self._allow_mic,
            config_dir=self._config_dir,
            id="viz",
        )
        yield Footer()

    @property
    def viz(self) -> AudioVisualizer:
        return self.query_one(AudioVisualizer)

    def on_mount(self) -> None:
        self._ask_for_synchronized_output()
        if not self.settings.chrome:
            self._set_chrome(False)
        self.viz.on_mode_disabled = self._mode_disabled
        # give the capture thread a moment, then say what it found
        self.set_timer(1.5, self.action_show_status)
        if self.settings.shuffle:
            self._start_shuffle()
        self.set_interval(NOWPLAYING_POLL_SECONDS, self._poll_now_playing)

    def _ask_for_synchronized_output(self) -> None:
        """Ask the terminal whether it can present a whole frame at once.

        Textual brackets each repaint in synchronized-output markers
        (``CSI ?2026h`` / ``?2026l``) only once the terminal has answered the
        ``CSI ?2026$p`` query saying it supports them — and only its Linux and
        web drivers send that query. On Windows it was never asked, so every
        frame went out unbracketed and Windows Terminal, which does support
        it, painted each one as it arrived. A visualizer rewrites most of the
        screen every frame, and the terminal showed the top of one frame over
        the bottom of the last: matched against the frames spektr actually
        rendered, 95 of 720 captured 60 fps frames at 188x52 were two renders
        stitched together, which on Tunnel read as breaks in the fastest ring.
        With the query sent, 13 were.

        The answer is handled by Textual's own parser, so a terminal that does
        not support the mode simply never replies and nothing changes. Sent
        only on Windows, where Textual does not already do it.
        """
        if sys.platform != "win32":
            return
        driver = getattr(self, "_driver", None)
        if driver is None or getattr(driver, "is_inline", False) or getattr(driver, "is_headless", False):
            return
        try:
            driver.write("\x1b[?2026$p")
            driver.flush()
        except Exception:
            pass

    def _mode_disabled(self, name: str, message: str) -> None:
        """A mode failed repeatedly and was quarantined."""
        self.notify(
            f"{message}\nrun: spektr plugins doctor",
            severity="error",
            timeout=8,
        )

    def on_unmount(self) -> None:
        config.save(self.settings, config_dir=self._config_dir)

    def _save_settings(self) -> None:
        """Persist the current settings immediately.

        Called after a mode or theme is committed (and after the settings
        panel closes), so a preference survives even an abrupt exit that never
        reaches :meth:`on_unmount`. The unmount save and the atexit handler
        remain as backstops; this is the one that makes the commit itself
        durable rather than pending.
        """
        config.save(self.settings, config_dir=self._config_dir)

    # ── overlay panels ────────────────────────────────────────────────────────
    # The pickers/settings are docked overlay widgets on the *same* screen as
    # the visualiser (see layers in the CSS), so the bands stay visible and
    # repainting behind them. Mounting is owned by the App; the panel reports
    # its result through on_done and then removes itself.

    def _open_overlay(self, widget, on_done) -> None:
        """Mount a Picker/SettingsPanel and route its completion to on_done."""
        self._close_overlay()

        def finished(*args) -> None:
            if self._overlay is widget:
                self._overlay = None
            on_done(*args)
            widget.remove()

        widget._on_done = finished
        self._overlay = widget
        self.mount(widget)

    def _close_overlay(self) -> None:
        w = self._overlay
        if w is not None:
            self._overlay = None
            w._on_done = None
            w.remove()

    def check_action(self, action, parameters) -> bool:
        # While an overlay panel is open it owns the keyboard: the global keys
        # (cycle, pick, quit, gain, …) must not fire underneath it.
        return self._overlay is None

    # ── modes ────────────────────────────────────────────────────────────────

    def action_cycle_mode(self, step: int = 1) -> None:
        self.notify(f"mode — {self.viz.cycle_mode(step)}", timeout=1)
        self._save_settings()

    def action_pick_mode(self) -> None:
        viz = self.viz
        # Names only. The blurbs are still on every mode and still printed by
        # the ``--list`` CLI; they are just not shown here. At fifty modes the
        # picker was a wall of prose — two lines each, the dim line longer than
        # the name — and scanning it for a mode you already knew the name of
        # meant reading past a description you did not want. The plugin marker
        # stays, because that is provenance rather than description: it is the
        # only place the UI says a mode did not ship with spektr.
        labels = {
            m.name: (f"·{m.plugin}" if m.is_plugin else "")
            for m in mode_registry.listed()
        }
        # The subcell variants are registered as "(o)" and shown as "(q)"
        # whenever the cell setting says quadrant — the suffix reports the
        # geometry being drawn, and that is a setting, not a property of the
        # mode. Selection still carries the registered name.
        shown = {m.name: mode_registry.label(m.name) for m in mode_registry.listed()}

        def done(choice: str | None) -> None:
            if choice is None:
                viz.cancel_mode_preview()
            else:
                viz.set_mode(choice)
                viz.commit_mode()
                self._save_settings()

        self._open_overlay(
            Picker(
                "visualizers",
                viz.mode_names,
                current=viz.mode_name,
                on_preview=viz.preview_mode,
                labels=labels,
                display=shown,
            ),
            done,
        )

    # ── themes ───────────────────────────────────────────────────────────────

    def action_pick_theme(self) -> None:
        viz = self.viz

        def done(choice: str | None) -> None:
            if choice is None:
                viz.cancel_theme_preview()
            else:
                viz.apply_theme(choice)
                viz.commit_theme()
                self._save_settings()
                self.notify(f"theme — {viz.palette.note}", timeout=2)

        self._open_overlay(
            Picker(
                "themes",
                viz.theme_names,
                current=viz.theme_name,
                on_preview=viz.preview_theme,
            ),
            done,
        )

    def action_cycle_theme(self) -> None:
        self.notify(f"theme — {self.viz.cycle_theme()}", timeout=2)
        self._save_settings()

    def action_new_theme(self) -> None:
        """Live theme editor, starting from whatever is on screen.

        Reached from the settings panel's "theme editor" row, not from a key
        of its own. Everything that configures spektr lives behind ``c``; a
        second entry point would be one more thing to know about for no gain.

        Colour *selection* rides the same live-preview rule as the nudges.
        The "swatches" row opens a named-colour picker for whichever slot the
        colour row is on, and the "hex" row opens a text field that takes
        ``#rrggbb``, ``#rgb`` or a colour name — both push straight into the
        draft and repaint the visualiser behind the panel, so you judge a
        pick the same way you judge a nudge: by watching bars move in it,
        not by looking at six swatches. Selection replaces the walk; the
        three nudge rows stay for the fine adjustment after it.

        Four slots, not six. A theme has six, but low/mid/high/accent is what
        someone who has not read the ramp documentation can pick meaningfully;
        bg and fg are derived from those until the moment you edit one, which
        is also what flips the panel into showing them.

        The visibility rule from the audit suite runs live on the draft and
        surfaces as a warning line rather than as a refusal — the check exists
        to stop people accidentally making an invisible theme, not to stop them
        deliberately making a subtle one.
        """
        viz = self.viz
        draft = palette_mod.ThemeDraft(viz.palette.theme, name=viz.theme_name)
        state = {"slot": 0}

        def apply() -> None:
            viz.preview_theme_object(draft.to_theme("preview"))

        def slot_name() -> str:
            return draft.slots[min(state["slot"], len(draft.slots) - 1)]

        def step_slot(delta: int) -> None:
            state["slot"] = (state["slot"] + delta) % len(draft.slots)
            apply()

        def nudge(which: str, amount: float):
            def go(delta: int) -> None:
                draft.nudge(slot_name(), which, delta * amount)
                apply()

            return go

        def show_slot(_v=None) -> str:
            name = slot_name()
            derived = name in palette_mod.ADVANCED_SLOTS and not draft.advanced
            return f"{name}  {draft.hex_of(name)}" + ("  (derived)" if derived else "")

        def component(which: str):
            def show(_v=None) -> str:
                value = draft.component(slot_name(), which)
                filled = int(round(value * 12))
                return f"{'█' * filled}{'·' * (12 - filled)}  {value:.2f}"

            return show

        def show_warning(_v=None) -> str:
            problems = draft.problems()
            return "looks fine" if not problems else f"[!] {problems[0]}"

        def show_slots(_v=None) -> str:
            return "6 — bg and fg editable" if draft.advanced else "4 — bg and fg derived"

        def step_slots(_delta: int) -> None:
            draft.set_advanced(not draft.advanced)
            state["slot"] = min(state["slot"], len(draft.slots) - 1)
            apply()

        def swatches() -> list[tuple[str, str]]:
            """(hex, label) pairs for the picker: the base theme's own colours
            first, then the named set, deduped by hex keeping the first label."""
            seen: dict[str, str] = {}
            for slot in draft.slots:
                seen.setdefault(draft.hex_of(slot), f"{draft.name} {slot}")
            for name, colour in palette_mod.NAMED_COLOURS.items():
                seen.setdefault(colour, name)
            return list(seen.items())

        def open_picker() -> None:
            def preview(colour: str) -> None:
                draft.set_slot(slot_name(), colour)
                apply()

            def picked(colour: str | None) -> None:
                if colour is not None:
                    draft.set_slot(slot_name(), colour)
                    apply()
                self.call_after_refresh(show_panel)

            self._open_overlay(
                ColourPicker(
                    f"colour — {slot_name()}",
                    [h for h, _ in swatches()],
                    dict(swatches()),
                    current=draft.hex_of(slot_name()),
                    on_preview=preview,
                ),
                picked,
            )

        def open_hex() -> None:
            def got(colour: str | None) -> None:
                if colour is not None:
                    draft.set_slot(slot_name(), colour)
                    apply()
                self.call_after_refresh(show_panel)

            self._open_overlay(
                HexPrompt(
                    f"hex — {slot_name()}",
                    current=draft.hex_of(slot_name()),
                ),
                got,
            )

        def done() -> None:
            def named(name: str | None) -> None:
                if name is None:
                    viz.cancel_theme_preview()
                    return
                final = palette_mod.available_theme_name(name, config_dir=self._config_dir)
                try:
                    palette_mod.save_user_theme(
                        draft.to_theme(final), config_dir=self._config_dir
                    )
                except OSError as exc:
                    viz.cancel_theme_preview()
                    self.notify(f"could not save theme: {exc}", severity="error", timeout=4)
                    return
                viz.reload_themes()
                viz.apply_theme(final)
                viz.commit_theme()
                self.notify(f"saved theme — {final}", timeout=3)

            self._open_overlay(NamePrompt("name this theme", draft.name), named)

        def rows() -> list[Setting]:
            return [
                Setting(
                    "slot", "colour", [], show_slot, None,
                    "which colour the rows below change",
                    step=step_slot,
                    live=lambda: None,
                ),
                Setting("hue", "hue", [], component("h"), None, "",
                        step=nudge("h", 0.02), live=lambda: None),
                Setting("sat", "saturation", [], component("s"), None, "",
                        step=nudge("s", 0.04), live=lambda: None),
                Setting("lum", "lightness", [], component("l"), None, "",
                        step=nudge("l", 0.03), live=lambda: None),
                Setting(
                    "swatches", "swatches", [], str, None,
                    "named colours for the selected slot — the nudge rows "
                    "above still fine-tune",
                    step=lambda delta: self.call_after_refresh(open_picker) if delta > 0 else None,
                    live=lambda: f"→ pick for {slot_name()}",
                ),
                Setting(
                    "hex", "hex", [], str, None,
                    "#rrggbb, #rgb, or a colour name",
                    step=lambda delta: self.call_after_refresh(open_hex) if delta > 0 else None,
                    live=lambda: "→ type a colour",
                ),
                Setting(
                    "slots", "slots", [], show_slots, None,
                    "four is enough for a working theme; six if you want the "
                    "background and text picked by hand",
                    step=step_slots,
                    live=lambda: None,
                ),
                Setting(
                    "visible", "check", [], show_warning, None,
                    "the same rule the test suite applies to built-in themes",
                    step=lambda _d: None,
                    live=lambda: None,
                ),
            ]

        def show_panel() -> None:
            self._open_overlay(SettingsPanel(rows(), {}), done)

        apply()
        show_panel()

    # ── shuffle ──────────────────────────────────────────────────────────────
    # A screensaver toggle: pick a random mode every SHUFFLE_MODE_SECONDS, and
    # a random theme every SHUFFLE_THEME_EVERY-th tick. Random rather than the
    # sequential cycle m/T already do — a fixed rotation through the same list
    # in the same order is exactly what makes a screensaver feel like a slideshow
    # instead of a surprise.

    def _set_shuffle_scope(self, scope: str) -> None:
        """Change what shuffle cycles. Does not switch it on — `s` does that.

        Editing a preference should not start something the user did not ask to
        start, so the timer is left alone. The row says so when shuffle is off,
        rather than silently doing nothing.
        """
        if scope in config.SHUFFLE_SCOPES:
            self.settings.shuffle_scope = scope
            self._save_settings()

    def _set_shuffle_timing(self, timing: str) -> None:
        if timing not in config.SHUFFLE_TIMINGS:
            return
        self.settings.shuffle_timing = timing
        self._shuffle_track = self._now_playing
        self._stop_shuffle()
        if self.settings.shuffle:
            self._start_shuffle()
        self._save_settings()

    def action_toggle_shuffle(self) -> None:
        self.settings.shuffle = not self.settings.shuffle
        self._save_settings()
        if self.settings.shuffle:
            self._start_shuffle()
            self.notify(f"shuffle on — {self.settings.shuffle_scope}", timeout=2)
        else:
            self._stop_shuffle()
            self.notify("shuffle off", timeout=2)

    def _start_shuffle(self) -> None:
        if self.settings.shuffle_timing == "track":
            self._shuffle_track = self._now_playing
        elif self._shuffle_timer is None:
            self._shuffle_count = 0
            self._shuffle_timer = self.set_interval(SHUFFLE_MODE_SECONDS, self._shuffle_tick)

    def _stop_shuffle(self) -> None:
        if self._shuffle_timer is not None:
            self._shuffle_timer.stop()
            self._shuffle_timer = None

    def _shuffle_tick(self) -> None:
        # a picker or the settings panel open means the user is mid-decision;
        # yanking the mode out from under them would be exactly the wrong kind
        # of surprise
        if self._overlay is not None:
            return

        viz = self.viz
        scope = self.settings.shuffle_scope
        self._shuffle_count += 1

        if scope in ("modes", "both"):
            others = [n for n in viz.mode_names if n != viz.mode_name]
            if viz.eco_active():
                # In eco, shuffle stops offering the modes this machine cannot
                # draw at rate. Judged on what they measured here rather than a
                # fixed list, so it follows the terminal size too: the same
                # mode is cheap in a small window and expensive fullscreen.
                affordable = [n for n in others if viz.affordable(n)]
                others = affordable or others
            if others:
                viz.set_mode(random.choice(others), dissolve=True)
                viz.commit_mode()

        # Every tick when themes are the only thing moving, every
        # SHUFFLE_THEME_EVERY-th when they ride along with the modes. Changing
        # both on the same tick reads as the picture breaking rather than as a
        # deliberate change, which is why they are staggered — but that reason
        # disappears when the mode is holding still, and a theme that changed
        # only every third tick would look like shuffle had stopped working.
        theme_due = scope == "themes" or (
            scope == "both" and self._shuffle_count % SHUFFLE_THEME_EVERY == 0
        )
        if theme_due:
            theme_others = [n for n in viz.theme_names if n != viz.theme_name]
            if theme_others:
                viz.apply_theme(random.choice(theme_others))
                viz.commit_theme()

        self.notify(f"shuffle — {viz.mode_name} · {viz.palette.name}", timeout=2)

    # ── loadouts ─────────────────────────────────────────────────────────────

    def action_loadout(self, chosen=None) -> None:
        """The one panel for which modes the interface offers.

        Built over the *unfiltered* roster rather than ``viz.mode_names``:
        that property is already narrowed by the active loadout, so feeding it
        back in would show only what is currently ticked and there would be no
        way to add a mode back. Quarantined modes stay out — a mode that
        crashes is not something to hand someone a tickbox for — and the
        subcell variants follow ``fine_modes`` as they do everywhere else.

        ``chosen`` overrides what opens ticked, which is how the panel comes
        back with your picks intact after naming a loadout.
        """
        viz = self.viz
        pool = mode_registry.MODES if viz.show_fine else mode_registry.listed()
        offer = [m.name for m in pool if not viz.quarantine.is_disabled(m.name)]
        labels = {m.name: (f"·{m.plugin}" if m.is_plugin else "") for m in pool}
        shown = {m.name: mode_registry.label(m.name) for m in pool}

        def forget(name: str) -> None:
            self._loadouts.pop(name, None)
            loadouts_module.save(self._loadouts, config_dir=self._config_dir)

        def done(result) -> None:
            # None is cancel. Otherwise ("apply"|"save", names), where an
            # empty list is a real answer meaning "no restriction" — which is
            # why cancel cannot be represented as one.
            if result is None:
                return
            what, names = result
            if what == "save":
                self._name_loadout(names)
                return
            self.settings.loadout = names
            # config.save directly rather than through a helper: this panel is
            # the only thing in the file that persists outside the settings
            # panel's own path, and reaching for a method defined elsewhere in
            # the class made the loadout depend on it existing.
            config.save(self.settings, config_dir=self._config_dir)
            # The running mode is left alone even when it falls outside the
            # new loadout: yanking the picture away is a worse surprise than
            # a mode that stays until you next move. `cycle_mode` already
            # handles the current name being absent, so the next `m` steps
            # into the loadout on its own.
            n = len(viz.mode_names)
            self.notify(
                f"loadout off — all {n} modes" if not names
                else f"loadout — {n} mode{'' if n == 1 else 's'}",
                timeout=2)

        self._open_overlay(
            LoadoutPicker(
                "loadout",
                offer,
                chosen=self.settings.loadout if chosen is None else chosen,
                saved=self._loadouts,
                on_delete=forget,
                current=viz.mode_name,
                labels=labels,
                display=shown,
            ),
            done,
        )

    def _name_loadout(self, names: list[str]) -> None:
        """Ask for a name, then hand the panel back with the picks intact.

        A round trip rather than a prompt inside the panel, because an overlay
        replaces the one before it — but the user should not be able to tell:
        they press ``s``, type a name, and are back where they were with
        everything still ticked, whether they named it or escaped.
        """
        def named(name: str | None) -> None:
            if name:
                # Everything ticked stores as empty, and a named loadout that
                # restricts nothing is a trap rather than a feature — there
                # would be no way to tell it from the ones that do.
                if not names:
                    self.notify("nothing to save — a loadout needs a narrower "
                                "pick than every mode", timeout=3)
                else:
                    self._loadouts[name] = names
                    loadouts_module.save(self._loadouts, config_dir=self._config_dir)
                    self.notify(f"saved — {name}", timeout=2)
            self.action_loadout(chosen=names)

        self._open_overlay(NamePrompt("save loadout as…", placeholder="name"), named)

    def action_reload(self) -> None:
        """Re-read themes and plugins from disk without restarting.

        The main reason this exists is authoring: editing a plugin and pressing
        `r` is a much shorter loop than quitting and relaunching. Note that an
        edit invalidates the plugin's approval, so a changed file will show up
        as untrusted until you re-approve it — which is the point.
        """
        from ..plugins import reload_all

        viz = self.viz
        themes = viz.reload_themes()
        viz.quarantine.clear()

        loaded = reload_all(config_dir=self._config_dir)
        ok = sum(1 for p in loaded if p.loaded)
        stale = [p.name for p in loaded if not p.trusted]
        broken = [p.name for p in loaded if p.error]

        parts = [f"{themes} themes", f"{ok} plugins"]
        if stale:
            parts.append(f"{len(stale)} need re-approval: {', '.join(stale)}")
        if broken:
            parts.append(f"{len(broken)} failed: {', '.join(broken)}")

        # a mode that vanished with its plugin can't stay selected
        if mode_registry.get(viz.mode_name) is None:
            viz.set_mode("Bars")

        self.notify(
            " · ".join(parts),
            severity="warning" if (stale or broken) else "information",
            timeout=4,
        )

    # ── audio ────────────────────────────────────────────────────────────────

    def action_gain(self, direction: int) -> None:
        v = self.viz.nudge_sensitivity(1.35 if direction > 0 else 1 / 1.35)
        self.notify(f"sensitivity ×{v:.2f}", timeout=1)

    def action_gate(self, direction: int) -> None:
        self.viz.nudge_gate(1.4 if direction > 0 else 1 / 1.4)
        self.notify(
            "gate raised — quieter inputs cut off sooner"
            if direction > 0
            else "gate lowered — picks up quieter audio",
            timeout=1,
        )

    def action_next_source(self) -> None:
        self.viz.restart_capture()
        self.notify("next audio source…", timeout=2)
        self.set_timer(3.5, self.action_show_status)

    def action_default_source(self) -> None:
        """Back to whatever the OS calls the default output.

        `d` is sticky for the session, so without this, cycling one past the
        device you wanted means restarting spektr.
        """
        self.viz.reset_capture()
        self.notify("back to the default output…", timeout=2)
        self.set_timer(3.5, self.action_show_status)

    def action_show_status(self) -> None:
        viz = self.viz
        status = viz.status
        self._capture_status = status
        self._update_subtitle()

        if viz.on_mic:
            severity = "error"
        elif status.startswith("listening"):
            severity = "information"
        else:
            severity = "warning"

        self.notify(f"{status}\n{viz.level}", severity=severity, timeout=7)

    # ── now playing ──────────────────────────────────────────────────────────
    # spektr taps raw audio — it has no idea what's making the sound, only
    # that something is. Track metadata comes from the OS media session
    # instead (see nowplaying.py), polled on a timer rather than pushed,
    # because neither SMTC nor MPRIS offers spektr a change notification
    # worth wiring up for a header line.

    async def _poll_now_playing(self) -> None:
        track = await nowplaying.current()
        current = str(track) if track else None
        previous = self._shuffle_track
        self._now_playing = current
        if current is not None:
            self._shuffle_track = current
            if (
                previous is not None
                and current != previous
                and self.settings.shuffle
                and self.settings.shuffle_timing == "track"
            ):
                self._shuffle_tick()
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        # now playing wins when there is one — it's the more interesting fact
        # once music is actually going: capture status is diagnostic, this is
        # content
        self.sub_title = self._now_playing or self._capture_status

    def action_help(self) -> None:
        """Every key, and where the files are — generated, never written out.

        The keys come from ``BINDINGS`` and the counts from the registry, so
        this cannot describe a version of the app that no longer exists. That
        is not a hypothetical worry: the README drifted twice in a day before
        anything checked it, and a help screen is read at exactly the moment
        someone is least able to tell that it is wrong.

        The hidden bindings are included. They are hidden from the footer
        because it has room for eight, not because they are secret — and the
        ones you cannot see are the ones a help screen is for.
        """
        self._open_overlay(HelpPanel(self._help_sections()), lambda *a: None)

    def _help_sections(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """The help panel's contents, split out so a test can parse them.

        These are Textual markup strings by the time the panel is done with
        them, and markup that does not parse takes the app down from inside
        the compositor rather than merely looking wrong — see
        ``pickers.markup_safe``. A test can only catch that if it can get at
        the rows without mounting an app, which is what this is for.
        """
        viz = self.viz
        listed = len(mode_registry.listed())
        opt_in = len([m for m in mode_registry.MODES if m.hidden])
        cells = mode_registry.CELL_SUFFIX.get(self.settings.cells, "?")

        sections = [
            ("keys", [
                (key_label(b.key), b.description)
                for b in self.BINDINGS
                if b.description
            ]),
            # Everything below the generated key list exists because a key
            # and its one-line description do not tell you what the thing
            # *is*. "Loadout" is not an explanation of a loadout.
            ("in a picker — v, t", [
                ("↑ ↓", "move — it previews as you go"),
                ("type", "filter the list"),
                ("enter", "keep it"),
                ("esc", "cancel — what you had comes back"),
            ]),
            # The loadout panel does not follow the rules above: it picks a
            # set rather than one thing, so the list holds the cursor instead
            # of a filter box and typing does not filter.
            ("in the loadout — l", [
                ("↑ ↓", "move"),
                ("space", "on a mode: pick it in or out"),
                ("", "on a ★: load that set"),
                ("s", "name what is picked, and keep it"),
                ("d", "delete the ★ under the cursor"),
                ("a  n", "pick all, or none"),
                ("/", "filter — enter comes back to the list"),
                ("enter", "apply and close"),
                ("esc", "cancel — nothing changes"),
            ]),
            ("in the settings panel — c", [
                ("↑ ↓", "move between rows"),
                ("← →", "change it — applies live"),
                ("esc", "close"),
            ]),
            # Short lines on purpose. A long one wraps with no hanging
            # indent and stops reading as a list — and someone opening the
            # help is already lost, so this is the worst place to be wordy.
            ("what these mean", [
                ("loadout", "which modes v, m and shuffle offer you."),
                ("", "pick a few and the rest stop turning up."),
                ("", "pick all, and nothing is held back."),
                ("★", "a loadout you named. space loads it."),
                ("shuffle", "swaps mode and/or theme on a timer"),
                ("", "or when the song changes."),
                ("", "s starts it; c chooses what and when."),
                ("subcells", "(o) draws 2x4 dots per character, (q) 2x2."),
                ("", "the extra pairs are opt-in in c, and want"),
                ("", "a font with Unicode 16 octants."),
                ("·name", "that mode came from a plugin, not spektr."),
                ("dropped", "a mode that keeps crashing is switched"),
                ("", "off until restart, rather than taking"),
                ("", "the app down with it."),
            ]),
            ("now", [
                ("mode", mode_registry.label(viz.mode_name)),
                ("theme", viz.palette.name),
                ("source", viz.status),
                ("modes", f"{listed} offered, {opt_in} more with subcell modes on"),
                # What the cycle keys and shuffle will actually move through,
                # which is the question "why is `m` only giving me four modes"
                # — and the loadout is the answer nobody would think to look
                # for unless the help says it is there.
                ("loadout", (f"{len(viz.mode_names)} of them"
                             if self.settings.loadout else "off — all of them")),
                ("saved", (", ".join(self._loadouts) if self._loadouts
                           else "none yet — pick some modes in l and press s")),
                ("subcells", f"{self.settings.cells} — shown as {cells}"),
            ]),
            ("files", [
                ("config", str(self._config_dir or palette_mod.config_dir())),
                ("", "config.json and loadouts.json are in there,"),
                ("", "and so are themes/, plugins/ and ascii/."),
            ]),
            ("more", [
                ("", f"spektr {__version__} — spektr --help for the command line"),
                ("", "spektr --glyph-test checks this terminal for octants"),
            ]),
        ]
        return sections

    def action_settings(self) -> None:
        """The live settings panel.

        Sensitivity and gate are here as well as on their keys, because a
        keybinding you have to remember is not a discoverable setting — and
        seeing the current value is half of knowing which way to nudge it.
        """
        rows, values = self._settings_rows(self.viz, self.settings)
        self._open_overlay(SettingsPanel(rows, values), self._save_settings)

    def _set_eco(self, value: str) -> str:
        """Take the eco choice and apply it at once."""
        self.settings.eco = value
        self.viz.settings.eco = value
        self.viz._apply_eco()
        self._save_settings()
        return value

    def _settings_rows(self, viz, s) -> "tuple[list[Setting], dict]":
        """Build the panel's rows and the values they open on.

        Split out of :meth:`action_settings` so the shape of the panel can be
        checked without a running app, because two invariants decide whether
        it opens at all and neither is visible from any one row:

        * a row needs either a ``live`` callback or an entry in ``values`` —
          :meth:`SettingsPanel._row_text` subscripts ``values`` directly, so a
          row with neither raises ``KeyError`` the moment the panel opens;
        * a row needs either a ``step`` callback or a non-empty ``choices`` —
          :meth:`Setting.index_of` falls back to ``min(choices)`` for a numeric
          value, and ``min`` of an empty list raises.

        Both hold today by construction, which is exactly the kind of thing
        that stops holding when someone adds a row. tests/test_settings.py
        asserts them over the whole list, in every mode that changes it.
        """

        def show_bands(v):
            return (
                "fit the terminal"
                if v == 0
                else f"{v}" + ("  (resolved)" if v > 32 else "")
            )

        def show_shuffle(v):
            # Says when the setting is inert. Changing a scope while shuffle is
            # off does nothing visible, and a row that looks like it did
            # something is worse than one that admits it did not.
            return f"{v}" if s.shuffle else f"{v}  (off — press s)"

        def show_shuffle_timing(v):
            label = "every track change" if v == "track" else "every 15 seconds"
            return label if s.shuffle else f"{label}  (off — press s)"

        def show_eco(v):
            if v != "auto":
                return v
            return "auto — on" if viz.eco_active() else "auto — not needed"

        def show_fps(v):
            if v != config.FPS_UNLIMITED:
                return f"{v} fps"
            # Show what was actually detected, not just the resolved rate. A
            # probe that silently guessed wrong and a probe that failed and
            # fell back both produce a number; only one of them is worth
            # reporting, and the user cannot tell which from the number alone.
            got, detected = viz.unlimited_info()
            if detected is None:
                return f"unlimited (experimental) — display rate unknown, using {got}"
            return f"unlimited (experimental) — detected {detected} Hz"

        rows = [
            Setting(
                "fps",
                "frame rate",
                config.FPS_CHOICES,
                show_fps,
                lambda v: viz._retime(v, requested=True),
                "motion is timed in seconds, so this is smoothness only; "
                "unlimited caps to the detected display rate and is only worth "
                "it with resources to spare",
            ),
            Setting(
                "bands",
                "bands",
                config.BAND_CHOICES,
                show_bands,
                viz.set_bands,
                "more bars, more detail",
            ),
            Setting(
                "motion",
                "motion",
                config.MOTION_CHOICES,
                lambda v: v,
                viz.set_motion,
                "reactive or smooth movement",
            ),
            Setting(
                "sensitivity",
                "sensitivity",
                (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                lambda v: f"x{v:g}",
                viz.set_sensitivity,
                "scales the input",
            ),
            Setting(
                "gate",
                "noise gate",
                (1e-5, 3e-5, 8e-5, 2e-4, 5e-4, 1e-3),
                lambda v: f"{v:.0e}",
                viz.set_gate,
                "quieter than this is silence",
            ),
            Setting(
                "shuffle_scope",
                "shuffle",
                config.SHUFFLE_SCOPES,
                show_shuffle,
                self._set_shuffle_scope,
                f"cycles modes/themes every {int(SHUFFLE_MODE_SECONDS)}s",
            ),
            Setting(
                "shuffle_timing",
                "shuffle when",
                config.SHUFFLE_TIMINGS,
                show_shuffle_timing,
                self._set_shuffle_timing,
                (
                    "song changes use the OS now-playing service"
                    if nowplaying.available()
                    else "song changes unavailable here — no OS now-playing service"
                ),
            ),
            Setting(
                "transparent_background",
                "background",
                (False, True),
                lambda v: "terminal (see-through)" if v else "theme (solid)",
                viz.set_transparent_background,
                "see-through shows terminal opacity; light themes want a light terminal",
            ),
            Setting(
                "chrome",
                "header + footer",
                (True, False),
                lambda v: "shown" if v else "hidden",
                self._set_chrome,
                "same as f",
            ),
            Setting(
                "fine_modes",
                "subcell modes",
                (False, True),
                lambda v: "shown" if v else "hidden",
                self._set_fine_modes,
                "higher-resolution variants",
            ),
            Setting(
                "eco",
                "eco mode",
                config.ECO_CHOICES,
                show_eco,
                self._set_eco,
                "easier on an older machine: 30 fps, fewer bars, and shuffle "
                "keeps away from the heaviest visuals",
            ),
            Setting(
                "cells",
                "subcell shape",
                ("octant", "quadrant"),
                lambda v: v,
                self._set_cells,
                "rounder vs blockier edges",
            ),
            # An action row, not a value. It lives here because the editor is
            # otherwise only reachable from a keybinding nobody has been told
            # about — and "a keybinding you have to remember is not a
            # discoverable setting" is the reason this whole panel exists.
            # Deferred to after the refresh because opening the editor removes
            # this panel, and doing that from inside its own key handler is
            # asking for trouble.
            Setting(
                "theme_editor",
                "theme editor",
                [],
                live=lambda: "→ build a new theme",
                step=lambda delta: (
                    self.call_after_refresh(self.action_new_theme) if delta > 0 else None
                ),
                note="tweak four colours live",
            ),
            # No fixed choices to step through and nothing to read out of
            # values — the audio source is whatever the capture thread
            # currently holds, which changes on its own schedule as the new
            # device settles. step/live are exactly the escape hatch Setting
            # documents itself: right reuses next_source's cycle, left
            # reuses default_source's reset, and the row polls the live
            # status (see SettingsPanel.on_mount) rather than reading a
            # snapshot taken when the key was pressed.
            Setting(
                "source",
                "source",
                [],
                live=lambda: viz.status,
                step=lambda delta: viz.restart_capture() if delta > 0 else viz.reset_capture(),
                note="→ next device · ← default",
            ),
        ]
        values = {
            # The *requested* rate, not viz._target_fps, which is the resolved
            # one: with unlimited selected that is a concrete 144, which
            # matches the literal 144 stop in FPS_CHOICES, so the row opened
            # reading "144 fps" and stepping away from it silently converted
            # the preference from "unlimited" into a fixed rate. Adaptive
            # pacing never writes settings.fps, so this stays what was asked
            # for even while the pacer is throttling.
            "fps": s.fps,
            "bands": s.bands,
            "motion": s.motion,
            "fine_modes": s.fine_modes,
            "eco": s.eco,
            "cells": s.cells,
            "sensitivity": s.sensitivity,
            "gate": s.gate,
            "chrome": s.chrome,
            "transparent_background": s.transparent_background,
            "shuffle_scope": s.shuffle_scope,
            "shuffle_timing": s.shuffle_timing,
        }

        return rows, values

    def action_show_perf(self) -> None:
        self.notify(f"{self.viz.perf}\n{self.viz.level}", timeout=4)

    # ── chrome ───────────────────────────────────────────────────────────────

    def _set_chrome(self, visible: bool) -> None:
        for w in (self.query_one(Header), self.query_one(Footer)):
            w.display = visible
        self.settings.chrome = visible
        self._save_settings()

    def action_toggle_chrome(self) -> None:
        self._set_chrome(not self.settings.chrome)

    def _set_fine_modes(self, on: bool) -> None:
        """Put the subcell variants on the menu, or take them off again.

        Nothing about the modes changes — they are registered either way, and
        selectable by name either way. This is only whether the picker, the
        cycle keys and shuffle offer them.

        Turning it *off* while one is showing leaves it showing: the mode is
        still perfectly good, and yanking the picture out from under someone
        because a menu setting changed would be the wrong reading of what this
        does. It simply stops being offered next time.
        """
        self.settings.fine_modes = bool(on)
        self.viz.show_fine = bool(on)
        self._save_settings()

    def _set_cells(self, shape: str) -> None:
        """Switch every subcell mode between octant and quadrant geometry."""
        from ..render import set_cell_mode

        set_cell_mode(shape)
        self.settings.cells = shape
        self.viz.redraw()
        self._save_settings()

