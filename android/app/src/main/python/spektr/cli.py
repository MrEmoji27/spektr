"""Command-line entry point and compatibility access to the UI shell."""

from __future__ import annotations

import atexit
import sys

from . import __version__, config
from . import modes as mode_registry


def __getattr__(name):
    from .ui import app as shell

    try:
        return getattr(shell, name)
    except AttributeError:
        raise AttributeError(f"module 'spektr.app' has no attribute {name!r}") from None

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
  --fps <n>          frame rate cap, 15-240 (default 60)
  --fps unlimited    run at the detected display refresh rate (experimental)
  --mic              allow the microphone as an automatic source
  --no-plugins       skip loading plugins this run
  --list-modes       print visualiser names and exit
  --list-themes      print theme names and exit
  --glyph-test       can this terminal draw the (o) subcell modes? and exit
  --cells quadrant   draw the subcell modes as (q) — block elements only
                     instead of Unicode 16 octants — half the resolution,
                     works in every font. Saved, so set it once.
  --background terminal
                     leave the visualizer's empty cells to the terminal, so
                     its opacity or acrylic shows through; --background theme
                     paints the theme's colour again (the default). Saved.
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


def _glyph_test() -> None:
    """Show whether this terminal can draw the modes that need octants.

    There is no way to ask a terminal "do you have a glyph for U+1CD1E". A
    missing one renders as a replacement box, not as an error, and the modes
    that use them are perfectly happy — so the only reliable detector is a
    person looking at a row of characters. Hence this: print the ones that
    matter beside a reference row everything can draw, and say plainly what to
    conclude.

    The three rows are three different questions. Block Elements have been in
    every terminal font for decades and are the control. The octant block is
    Unicode 16 (2024) and is what the ``Fine`` and ``Ultra`` modes are built
    on. The last row is the handful of patterns Unicode did *not* put in the
    octant block, which fonts ship least reliably — spektr already avoids
    those in what it draws, and they are here because seeing which of them
    work says how far ahead of the standard a font actually is.
    """
    import os

    from .render import OCTANT_BASE, OCTANT_LUT

    def row(codes):
        return " ".join(chr(int(c)) for c in codes)

    print()
    print("  terminal:")
    for var in ("TERM_PROGRAM", "TERM", "COLORTERM", "WT_SESSION", "KITTY_WINDOW_ID"):
        val = os.environ.get(var)
        if val:
            print(f"    {var:<16} {val if len(val) < 40 else val[:37] + '...'}")
    print()

    print("  1. block elements — decades old, every font has them:")
    print("     " + row((0x2580, 0x2584, 0x258C, 0x2590, 0x2596, 0x2597, 0x2598,
                         0x2599, 0x259A, 0x259B, 0x259C, 0x259D, 0x259E, 0x259F, 0x2588)))
    print()
    print("  2. quadrants alone — the safe fallback renderer, 2x2 per cell:")
    print("     " + row((0x2596, 0x2597, 0x2598, 0x259D, 0x2580, 0x2584, 0x258C, 0x2590)))
    print()

    print("  3. the octant block, U+1CD00..U+1CDE5 — what the Fine modes draw.")
    print("     Say which ROWS are broken; a font can ship part of this block.")
    for i in range(0, 230, 16):
        end = min(i + 16, 230)
        print(f"     {OCTANT_BASE + i:05X}  " + row(range(OCTANT_BASE + i, OCTANT_BASE + end)))
    print()

    print("  4. outside the octant block — spektr never draws these; they only")
    print("     say how complete the font is:")
    print("     " + row((0x1CEA8, 0x1CEAB, 0x1CEA3, 0x1CEA0, 0x1FBE6, 0x1FBE7, 0x1FB82, 0x1FB85)))
    print()
    print("  Row 1 broken -> nothing here will work; the font is very old.")
    print("  Rows 1-2 fine, row 3 broken or patchy -> run spektr --cells quadrant.")
    print("     The Fine and Ultra modes then draw with row 2's characters: 2x2")
    print("     subcells instead of 2x4, still twice what the plain modes get,")
    print("     and nothing outside Block Elements. The setting is saved.")
    print(f"  ({len(OCTANT_LUT)} patterns total, 230 in the block plus block elements)")
    print()


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

    if "--check-modes" in argv:
        # Loads every mode's code, which is the one thing --version and
        # --list-modes do not: they read the catalogue. A frozen build that is
        # missing a mode module passes those two and dies on its first frame,
        # which is exactly how 0.5.5 first shipped. The release builds run
        # this, so that cannot happen quietly again.
        mode_registry.load_all()
        broken = [m.name for m in mode_registry.MODES if m.fn is None]
        if broken:
            print(f"modes that failed to load: {broken}", file=sys.stderr)
            raise SystemExit(1)
        print(f"{len(mode_registry.MODES)} modes loaded")
        return

    if "--list-modes" in argv:
        # Hidden modes are listed here and nowhere else in the UI. They are
        # still selectable by name, so a listing that omitted them would make
        # them unfindable rather than merely unoffered.
        for m in mode_registry.MODES:
            tag = f"  [{m.plugin}]" if m.is_plugin else ""
            mark = "  (opt-in — settings, or --mode)" if m.hidden else ""
            print(f"  {m.name:<10} {m.blurb}{tag}{mark}")
        return
    if "--glyph-test" in argv:
        _glyph_test()
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
    cells = _arg(argv, "--cells")
    if cells:
        if cells not in ("octant", "quadrant"):
            print(f"unknown cell geometry: {cells}   (octant or quadrant)")
            return
        settings.cells = cells
    background = _arg(argv, "--background")
    if background:
        if background not in ("theme", "terminal"):
            print(f"unknown background: {background}   (theme or terminal)")
            return
        settings.transparent_background = background == "terminal"
    # Set before any mode draws: the subcell packers read it, so a mode never
    # has to know which geometry it is being rendered into.
    from .render import set_cell_mode

    set_cell_mode(settings.cells)
    # ``unlimited`` spelled out, and 0 as its numeric form. ``if fps:`` would
    # have quietly dropped the sentinel on the floor, since 0 is falsy.
    fps_raw = _arg(argv, "--fps")
    if fps_raw is not None:
        if fps_raw.strip().lower() in ("unlimited", "max"):
            settings.fps = config.FPS_UNLIMITED
        else:
            try:
                settings.fps = int(fps_raw)
            except ValueError:
                print("--fps needs a number, or 'unlimited'")
                raise SystemExit(2) from None
    settings.clamp()

    from .ui.app import Spektr

    app = Spektr(
        device=_arg(argv, "--device", int),
        settings=settings,
        allow_mic="--mic" in argv,
    )
    # Flush the session's settings at interpreter exit as well as on a clean
    # unmount. Closing the terminal or Ctrl+C can skip Textual's teardown on
    # some platforms, and a preference that only commits on a well-behaved quit
    # resets on every exit path people actually use. atexit fires on normal
    # shutdown and on KeyboardInterrupt; it is registered here, not in the
    # class, so a test constructing a Spektr never writes the real config.
    atexit.register(config.save, app.settings, app._config_dir)
    app.run()


if __name__ == "__main__":
    main()
