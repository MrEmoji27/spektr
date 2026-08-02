"""spektr — a terminal spectrum analyser for whatever your speakers are doing."""

from __future__ import annotations

import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from . import config
from . import modes as mode_registry
from .pickers import Picker, Setting, SettingsPanel
from .widget import AudioVisualizer

__version__ = "0.2.0"


class Spektr(App):
    # The screen colour is taken over by the active spektr theme once the
    # visualiser mounts — see AudioVisualizer._paint_background.
    CSS = """
    Screen { background: #000000; }
    AudioVisualizer { height: 1fr; }
    """

    TITLE = "spektr"
    SUB_TITLE = "system audio, drawn in the terminal"

    BINDINGS = [
        Binding("m,space", "cycle_mode", "Mode"),
        Binding("M", "cycle_mode(-1)", "Prev mode", show=False),
        Binding("v", "pick_mode", "Modes"),
        Binding("t", "pick_theme", "Themes"),
        Binding("c", "settings", "Settings"),
        Binding("T", "cycle_theme", "Next theme", show=False),
        Binding("f", "toggle_chrome", "Full screen"),
        Binding("s", "show_status", "Source"),
        Binding("d", "next_source", "Next source", show=False),
        Binding("D", "default_source", "Default output", show=False),
        Binding("r", "reload", "Reload themes + plugins", show=False),
        Binding("left_square_bracket", "gain(-1)", "Sens -", show=False),
        Binding("right_square_bracket", "gain(1)", "Sens +", show=False),
        Binding("g", "gate(-1)", "Gate -", show=False),
        Binding("G", "gate(1)", "Gate +", show=False),
        Binding("p", "show_perf", "Perf", show=False),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, device=None, settings: config.Settings | None = None,
                 allow_mic: bool = False):
        super().__init__()
        self._device = device
        self._allow_mic = allow_mic
        self.settings = settings or config.load()

    def compose(self) -> ComposeResult:
        yield Header()
        yield AudioVisualizer(
            device=self._device, settings=self.settings,
            allow_mic=self._allow_mic, id="viz",
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

    def _mode_disabled(self, name: str, message: str) -> None:
        """A mode failed repeatedly and was quarantined."""
        self.notify(
            f"{message}\nrun: spektr plugins doctor",
            severity="error",
            timeout=8,
        )

    def on_unmount(self) -> None:
        config.save(self.settings)

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

        self.push_screen(
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

        self.push_screen(
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
        self.sub_title = status

        if viz.on_mic:
            severity = "error"
        elif status.startswith("listening"):
            severity = "information"
        else:
            severity = "warning"

        self.notify(f"{status}\n{viz.level}", severity=severity, timeout=7)

    def action_settings(self) -> None:
        """The live settings panel.

        Sensitivity and gate are here as well as on their keys, because a
        keybinding you have to remember is not a discoverable setting — and
        seeing the current value is half of knowing which way to nudge it.
        """
        viz = self.viz
        s = self.settings

        def show_bands(v):
            return "fit the terminal" if v == 0 else f"{v}" + ("  (resolved)" if v > 32 else "")

        rows = [
            Setting(
                "fps", "frame rate", config.FPS_CHOICES,
                lambda v: f"{v} fps",
                lambda v: viz._retime(v, requested=True),
                "the motion is timed in seconds, so this changes smoothness only",
            ),
            Setting(
                "bands", "bands", config.BAND_CHOICES,
                show_bands,
                viz.set_bands,
                "above 32 the analyser resolves more, below it modes draw fewer",
            ),
            Setting(
                "sensitivity", "sensitivity",
                (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                lambda v: f"x{v:g}",
                viz.set_sensitivity,
                "trim on top of the automatic gain",
            ),
            Setting(
                "gate", "noise gate",
                (1e-5, 3e-5, 8e-5, 2e-4, 5e-4, 1e-3),
                lambda v: f"{v:.0e}",
                viz.set_gate,
                "below this, input counts as silence",
            ),
            Setting(
                "chrome", "header + footer", (True, False),
                lambda v: "shown" if v else "hidden",
                self._set_chrome,
                "same as f",
            ),
        ]
        values = {
            "fps": viz._target_fps, "bands": s.bands, "sensitivity": s.sensitivity,
            "gate": s.gate, "chrome": s.chrome,
        }
        self.push_screen(SettingsPanel(rows, values))

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
