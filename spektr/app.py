"""spektr — a terminal spectrum analyser for whatever your speakers are doing."""

from __future__ import annotations

import random
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from . import __version__, asciiart, config, nowplaying
from . import modes as mode_registry
from . import presets as presets_module
from .pickers import NamePrompt, Picker, Setting, SettingsPanel
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

    Picker, SettingsPanel, NamePrompt {
        layer: overlay;
        dock: right;
        width: auto;
        height: 100%;
        background: transparent;
    }
    Picker > #panel, SettingsPanel > #panel, NamePrompt > #panel {
        height: 100%;
        background: $surface;
        border-left: tall $accent;
        padding: 0 1;
    }
    /* Picker holds up to 29 mode names, each with a blurb that can run past
       sixty characters; SettingsPanel holds five rows with short notes. Both
       used to share one 32-wide rule, which was fine for the settings notes
       and far too narrow for the mode blurbs — every one of them wrapped,
       and wrapped text with no hanging indent reads as a jumbled paragraph
       rather than a list. Widened, and given each its own value now that
       they don't actually want the same one. */
    Picker > #panel {
        width: 40;
    }
    SettingsPanel > #panel, NamePrompt > #panel {
        width: 42;
    }
    Picker #title, SettingsPanel #title, NamePrompt #title {
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
    Picker OptionList, SettingsPanel OptionList {
        background: $surface;
        border: none;
        height: 1fr;
        scrollbar-size-vertical: 1;
    }
    Picker #hint, SettingsPanel #hint, NamePrompt #hint {
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
        Binding("t", "pick_theme", "Themes"),
        Binding("c", "settings", "Settings"),
        Binding("T", "cycle_theme", "Next theme", show=False),
        Binding("f", "toggle_chrome", "Full screen"),
        Binding("d", "next_source", "Next source", show=False),
        Binding("D", "default_source", "Default output", show=False),
        Binding("r", "reload", "Reload themes + plugins", show=False),
        # s used to be "show source status" — moved into the settings panel
        # (see action_settings' source row) since it's now free, and s for
        # Shuffle is a plainer mnemonic than the a it had before.
        Binding("s", "toggle_shuffle", "Shuffle"),
        Binding("l", "load_preset", "Load preset", show=False),
        Binding("L", "save_preset", "Save preset", show=False),
        Binding("left_square_bracket", "gain(-1)", "Sens -", show=False),
        Binding("right_square_bracket", "gain(1)", "Sens +", show=False),
        Binding("g", "gate(-1)", "Gate -", show=False),
        Binding("G", "gate(1)", "Gate +", show=False),
        Binding("p", "show_perf", "Perf", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        device=None,
        settings: config.Settings | None = None,
        allow_mic: bool = False,
    ):
        super().__init__()
        self._device = device
        self._allow_mic = allow_mic
        self.settings = settings or config.load()
        asciiart.restore(self.settings.ascii_reel, self.settings.ascii_fx)
        #: the picker/settings overlay currently mounted, if any
        self._overlay = None
        #: the shuffle timer, or None when shuffle is off
        self._shuffle_timer = None
        self._shuffle_count = 0
        self._presets = presets_module.load()
        #: the capture/device status text — what the header falls back to
        #: when nothing is playing that the OS will report on
        self._capture_status = self.SUB_TITLE
        #: "Artist — Title" from the OS media session, or None
        self._now_playing: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield AudioVisualizer(
            device=self._device,
            settings=self.settings,
            allow_mic=self._allow_mic,
            id="viz",
        )
        yield Footer()

    @property
    def viz(self) -> AudioVisualizer:
        return self.query_one(AudioVisualizer)

    def on_mount(self) -> None:
        if not self.settings.chrome:
            self._set_chrome(False)
        self.viz.on_mode_disabled = self._mode_disabled
        # give the capture thread a moment, then say what it found
        self.set_timer(1.5, self.action_show_status)
        if self.settings.shuffle:
            self._start_shuffle()
        self.set_interval(NOWPLAYING_POLL_SECONDS, self._poll_now_playing)

    def _mode_disabled(self, name: str, message: str) -> None:
        """A mode failed repeatedly and was quarantined."""
        self.notify(
            f"{message}\nrun: spektr plugins doctor",
            severity="error",
            timeout=8,
        )

    def on_unmount(self) -> None:
        config.save(self.settings)

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

    def action_pick_mode(self) -> None:
        viz = self.viz
        labels = {
            m.name: (f"{m.blurb}  ·{m.plugin}" if m.is_plugin else m.blurb)
            for m in mode_registry.MODES
        }

        def done(choice: str | None) -> None:
            if choice is None:
                viz.cancel_mode_preview()
            else:
                viz.set_mode(choice)
                viz.commit_mode()

        self._open_overlay(
            Picker(
                "visualizers",
                viz.mode_names,
                current=viz.mode_name,
                on_preview=viz.preview_mode,
                labels=labels,
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

    # ── shuffle ──────────────────────────────────────────────────────────────
    # A screensaver toggle: pick a random mode every SHUFFLE_MODE_SECONDS, and
    # a random theme every SHUFFLE_THEME_EVERY-th tick. Random rather than the
    # sequential cycle m/T already do — a fixed rotation through the same list
    # in the same order is exactly what makes a screensaver feel like a slideshow
    # instead of a surprise.

    def action_toggle_shuffle(self) -> None:
        self.settings.shuffle = not self.settings.shuffle
        if self.settings.shuffle:
            self._start_shuffle()
        else:
            self._stop_shuffle()
        self.notify(f"shuffle {'on' if self.settings.shuffle else 'off'}", timeout=2)

    def _start_shuffle(self) -> None:
        if self._shuffle_timer is None:
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
        others = [n for n in viz.mode_names if n != viz.mode_name]
        if others:
            viz.set_mode(random.choice(others))
            viz.commit_mode()

        self._shuffle_count += 1
        if self._shuffle_count % SHUFFLE_THEME_EVERY == 0:
            theme_others = [n for n in viz.theme_names if n != viz.theme_name]
            if theme_others:
                viz.apply_theme(random.choice(theme_others))
                viz.commit_theme()

        self.notify(f"shuffle — {viz.mode_name} · {viz.palette.name}", timeout=2)

    # ── presets ──────────────────────────────────────────────────────────────
    # A named snapshot of mode + theme + the four settings-panel numbers.
    # Loading previews mode and theme exactly like the `t`/`v` pickers do —
    # arrow through, see it, escape puts back what you had. fps/bands/
    # sensitivity/gate apply immediately as you arrow instead, same as the `c`
    # panel already does for those four; there's no existing "undo" concept
    # for them anywhere in the app, so a preset preview doesn't invent one.

    def action_save_preset(self) -> None:
        def done(name: str | None) -> None:
            if not name:
                return
            viz = self.viz
            s = self.settings
            self._presets[name] = {
                "mode": viz.mode_name,
                "theme": viz.theme_name,
                "fps": viz._target_fps,
                "bands": s.bands,
                "sensitivity": s.sensitivity,
                "gate": s.gate,
            }
            presets_module.save(self._presets)
            self.notify(f"preset saved — {name}", timeout=2)

        self._open_overlay(NamePrompt("save preset as…", placeholder="name"), done)

    def action_load_preset(self) -> None:
        if not self._presets:
            self.notify("no presets saved yet — press L to save the current look", timeout=3)
            return

        viz = self.viz

        def preview(name: str) -> None:
            p = self._presets.get(name)
            if p is None:
                return
            viz.preview_mode(p["mode"])
            viz.preview_theme(p["theme"])
            viz._retime(p["fps"], requested=True)
            viz.set_bands(p["bands"])
            viz.set_sensitivity(p["sensitivity"])
            viz.set_gate(p["gate"])

        def done(name: str | None) -> None:
            if name is None:
                viz.cancel_mode_preview()
                viz.cancel_theme_preview()
            else:
                viz.commit_mode()
                viz.commit_theme()
                self.notify(f"preset — {name}", timeout=2)

        self._open_overlay(
            Picker("presets", list(self._presets), current=None, on_preview=preview),
            done,
        )

    def action_reload(self) -> None:
        """Re-read themes and plugins from disk without restarting.

        The main reason this exists is authoring: editing a plugin and pressing
        `r` is a much shorter loop than quitting and relaunching. Note that an
        edit invalidates the plugin's approval, so a changed file will show up
        as untrusted until you re-approve it — which is the point.
        """
        from .plugins import reload_all

        viz = self.viz
        themes = viz.reload_themes()
        viz.quarantine.clear()
        asciiart.reload()

        loaded = reload_all()
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
        self._now_playing = str(track) if track else None
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        # now playing wins when there is one — it's the more interesting fact
        # once music is actually going: capture status is diagnostic, this is
        # content
        self.sub_title = self._now_playing or self._capture_status

    def action_settings(self) -> None:
        """The live settings panel.

        Sensitivity and gate are here as well as on their keys, because a
        keybinding you have to remember is not a discoverable setting — and
        seeing the current value is half of knowing which way to nudge it.
        """
        viz = self.viz
        s = self.settings

        def show_bands(v):
            return (
                "fit the terminal"
                if v == 0
                else f"{v}" + ("  (resolved)" if v > 32 else "")
            )

        rows = [
            Setting(
                "fps",
                "frame rate",
                config.FPS_CHOICES,
                lambda v: f"{v} fps",
                lambda v: viz._retime(v, requested=True),
                "the motion is timed in seconds, so this changes smoothness only",
            ),
            Setting(
                "bands",
                "bands",
                config.BAND_CHOICES,
                show_bands,
                viz.set_bands,
                "above 32 the analyser resolves more, below it modes draw fewer",
            ),
            Setting(
                "sensitivity",
                "sensitivity",
                (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                lambda v: f"x{v:g}",
                viz.set_sensitivity,
                "trim on top of the automatic gain",
            ),
            Setting(
                "gate",
                "noise gate",
                (1e-5, 3e-5, 8e-5, 2e-4, 5e-4, 1e-3),
                lambda v: f"{v:.0e}",
                viz.set_gate,
                "below this, input counts as silence",
            ),
            Setting(
                "chrome",
                "header + footer",
                (True, False),
                lambda v: "shown" if v else "hidden",
                self._set_chrome,
                "same as f",
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
                note="→ next candidate · ← back to the system default",
            ),
        ]
        values = {
            "fps": viz._target_fps,
            "bands": s.bands,
            "sensitivity": s.sensitivity,
            "gate": s.gate,
            "chrome": s.chrome,
        }

        # Flipbook-only rows: two more settings with no fixed choice list
        # (which reel, which effect) that would just be dead weight on the
        # other 40 modes, so they're appended only while it's the active one
        # — action_settings rebuilds this list fresh every time the panel
        # opens, so there's no stale-panel-shape problem to guard against.
        if viz.mode_name == "Flipbook":
            rows.append(
                Setting(
                    "ascii_reel",
                    "ascii reel",
                    [],
                    live=self._ascii_reel_label,
                    step=self._step_ascii_reel,
                    note=f"drop .txt frames into {asciiart.ascii_dir()}",
                ),
            )
            rows.append(
                Setting(
                    "ascii_fx",
                    "ascii fx",
                    ("warp", "dissolve", "lit"),
                    lambda v: v,
                    self._set_ascii_fx,
                    "warp breathes it, dissolve scatters it in quiet, lit just lights it",
                ),
            )
            values["ascii_fx"] = s.ascii_fx

        self._open_overlay(SettingsPanel(rows, values), lambda *a: None)

    def _ascii_reel_label(self) -> str:
        r = asciiart.current()
        return f"{r.name} ({r.n_frames} frame{'s' if r.n_frames != 1 else ''})" if r else "none found"

    def _step_ascii_reel(self, delta: int) -> None:
        r = asciiart.step_reel(delta)
        self.settings.ascii_reel = r.name if r else ""

    def _set_ascii_fx(self, fx: str) -> None:
        asciiart.restore(self.settings.ascii_reel, fx)
        self.settings.ascii_fx = fx

    def action_show_perf(self) -> None:
        self.notify(f"{self.viz.perf}\n{self.viz.level}", timeout=4)

    # ── chrome ───────────────────────────────────────────────────────────────

    def _set_chrome(self, visible: bool) -> None:
        for w in (self.query_one(Header), self.query_one(Footer)):
            w.display = visible
        self.settings.chrome = visible

    def action_toggle_chrome(self) -> None:
        self._set_chrome(not self.settings.chrome)


# ── cli ──────────────────────────────────────────────────────────────────────

_USAGE = """spektr — terminal spectrum analyser for system audio

usage: spektr [options]
       spektr plugins <command>

  --diagnose         probe every source and report what it delivers
                     (says which device the OS calls the default, and which
                      endpoint spektr resolved it to — start here if flat)
  --monitor          run the app's own capture path headlessly and show
                     frames/level/gate/bars once a second — use when
                     --diagnose looks fine but the display does not move
  --devices          list every audio device and exit
  --device <n>       force a capture device by index
  --mode <name>      start in a given visualiser
  --theme <name>     start with a given theme
  --fps <n>          frame rate cap (default 60)
  --mic              allow the microphone as an automatic source
  --no-plugins       skip loading plugins this run
  --list-modes       print visualiser names and exit
  --list-themes      print theme names and exit
  --version          print version and exit
  -h, --help         this text

plugin commands:

  spektr plugins list            what's installed, and whether it's trusted
  spektr plugins trust <name>    review and approve a plugin's contents
  spektr plugins untrust <name>  revoke approval
  spektr plugins remove <name>   delete it from disk
  spektr plugins doctor          why isn't mine loading?
  spektr plugins path            print the plugins folder
"""

_TRUST_WARNING = """
  This is Python. It runs with your privileges — it can read your files
  and reach the network. spektr cannot sandbox it. Read it first.
"""


def _plugins_cli(argv: list[str]) -> int:
    from . import plugins as P

    cmd = argv[0] if argv else "list"
    arg = argv[1] if len(argv) > 1 else None
    folder = P.plugins_dir()

    if cmd == "path":
        print(folder)
        return 0

    if cmd == "list":
        # load_all() does the discovery itself and returns the *loaded* records
        found = P.load_all()
        if not found:
            print(f"no plugins in {folder}")
            print("drop a .py file there — see docs/plugins.md")
            return 0
        width = max(len(p.name) for p in found)
        for p in found:
            modes = ", ".join(p.modes) if p.modes else "—"
            print(f"  {p.name:<{width}}  {p.status:<9}  {modes}")
            if p.error:
                print(f"  {'':<{width}}  {p.error.splitlines()[-1]}")
        return 0

    if cmd == "doctor":
        found = P.load_all()
        if not found:
            print(f"no plugins in {folder}")
            return 0
        for p in found:
            print(f"\n{p.name}  [{p.status}]")
            print(f"  path    {p.path}")
            print(f"  sha256  {p.digest[:16]}…  ({p.lines} lines)")
            if not p.trusted:
                print("  not approved — run: spektr plugins trust " + p.name)
            if p.modes:
                print(f"  modes   {', '.join(p.modes)}")
            if p.error:
                print("  error:")
                for line in p.error.splitlines():
                    print(f"    {line}")
        return 0

    if cmd in ("trust", "untrust", "remove"):
        if not arg:
            print(f"usage: spektr plugins {cmd} <name>")
            return 2
        if cmd == "trust":
            p = next((x for x in P.discover() if x.name == arg), None)
            if p is None:
                print(f"no plugin named {arg!r} in {folder}")
                return 1
            print(f"\n  plugin  {p.name}")
            print(f"  source  {p.path}")
            print(f"  sha256  {p.digest}")
            print(f"  size    {p.lines} lines")
            print(_TRUST_WARNING)
            if "--yes" not in argv:
                try:
                    reply = input("  Trust this plugin? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    reply = ""
                if reply not in ("y", "yes"):
                    print("  not trusted")
                    return 1
            ok, msg = P.trust(arg)
        else:
            ok, msg = (P.untrust if cmd == "untrust" else P.remove)(arg)
        print(f"  {msg}" if ok else msg)
        return 0 if ok else 1

    print(f"unknown plugin command: {cmd}")
    print(_USAGE.split("plugin commands:")[1])
    return 2


def _arg(argv, flag, cast=str):
    if flag not in argv:
        return None
    try:
        return cast(argv[argv.index(flag) + 1])
    except (IndexError, ValueError):
        print(f"{flag} needs a value")
        raise SystemExit(2)


def main() -> None:
    argv = sys.argv[1:]

    # A console inherits whatever code page the terminal happens to be in.
    # spektr prints braille and the → between theme colours, so force UTF-8
    # before anything writes — otherwise --list-themes and --diagnose raise
    # UnicodeEncodeError on a default Windows console. The frozen exe already
    # does this in packaging/entry.py; this covers the pip-installed script.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Before anything touches the audio libraries: their destructors can raise
    # during interpreter shutdown, and the traceback lands on screen after the
    # UI is gone, which reads as a crash on exit.
    from .capture import install_shutdown_filter

    install_shutdown_filter()

    if argv and argv[0] == "plugins":
        raise SystemExit(_plugins_cli(argv[1:]))

    if "-h" in argv or "--help" in argv:
        print(_USAGE)
        return
    if "--version" in argv:
        print(f"spektr {__version__}")
        return
    if "--diagnose" in argv:
        from .capture import diagnose

        print(diagnose())
        return
    if "--monitor" in argv:
        from .capture import monitor

        monitor()
        return
    if "--devices" in argv:
        from .capture import describe_devices

        print(describe_devices())
        return

    if "--no-plugins" not in argv:
        from .plugins import load_all

        for p in load_all():
            if p.error:
                print(f"plugin {p.name}: {p.error.splitlines()[-1]}", file=sys.stderr)

    if "--list-modes" in argv:
        for m in mode_registry.MODES:
            tag = f"  [{m.plugin}]" if m.is_plugin else ""
            print(f"  {m.name:<10} {m.blurb}{tag}")
        return
    if "--list-themes" in argv:
        from .palette import AUTO, all_themes

        print(f"  {AUTO:<18} follow the terminal theme")
        for name, th in all_themes().items():
            print(f"  {name:<18} {th.low} → {th.high}")
        return

    settings = config.load()
    mode = _arg(argv, "--mode")
    if mode:
        if mode_registry.get(mode) is None:
            print(f"unknown mode: {mode}   (see: spektr --list-modes)")
            return
        settings.mode = mode
    theme = _arg(argv, "--theme")
    if theme:
        settings.theme = theme
    fps = _arg(argv, "--fps", int)
    if fps:
        settings.fps = fps
    settings.clamp()

    Spektr(
        device=_arg(argv, "--device", int),
        settings=settings,
        allow_mic="--mic" in argv,
    ).run()


if __name__ == "__main__":
    main()
