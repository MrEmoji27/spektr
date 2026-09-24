"""Command-line entry point and compatibility access to the UI shell."""

from __future__ import annotations

import atexit
import difflib
import sys

from . import __version__, config


def __getattr__(name):
    # Python itself asks modules for dunder names -- ``from .cli import main``
    # checks for ``__path__`` -- and forwarding those imported the whole UI
    # toolkit for ``spektr --version``.
    if name.startswith("__"):
        raise AttributeError(name)
    from .ui import app as shell

    try:
        return getattr(shell, name)
    except AttributeError:
        raise AttributeError(f"module 'spektr.app' has no attribute {name!r}") from None


_USAGE = """spektr: a terminal spectrum analyser for system audio

usage: spektr [options]
       spektr plugins <command>

start up
  --mode <name>        start in a given mode (see --list-modes)
  --theme <name>       start with a given theme (see --list-themes)
  --motion <m>         snappy or glide: reactive or smooth bars
  --morph <m>          clean or classic: how one mode changes into the next
  --bands <n>          how many bars: 8 to 64, or 0 to fit the terminal
  --fps <n>            frame rate cap, 15 to 240, or unlimited for the
                       display's own refresh rate (experimental)
  --eco <on|off>       30 fps, fewer bars, shuffle skips the heavy modes
  --shuffle <what>     modes, themes, both, or off
  --cells quadrant     draw the (o) modes with block elements only: half the
                       detail, works in every font (see --glyph-test)
  --background terminal
                       leave empty cells to the terminal, so its opacity or
                       acrylic shows through; --background theme undoes it

  Everything above is saved, so set it once.

audio
  --devices            list every audio device and exit
  --device <n>         use this device, by its number in --devices
  --mic                allow the microphone as an automatic source
  --diagnose           probe every source and report what it delivers: which
                       device the system calls the default, and which one
                       spektr chose. Start here if the picture is flat
  --monitor            run the capture headlessly and print level, gate and
                       bars once a second, for when --diagnose looks fine but
                       nothing moves

look it up
  --list-modes         every mode, including the opt-in ones
  --list-themes        every theme
  --glyph-test         can this terminal draw the (o) modes?
  --no-plugins         skip loading plugins this run
  --version            print the version
  -h, --help           this text

plugin commands:

  spektr plugins list            what is installed, and whether it is trusted
  spektr plugins trust <name>    review and approve a plugin's contents
  spektr plugins untrust <name>  revoke approval
  spektr plugins remove <name>   delete it from disk
  spektr plugins doctor          why is mine not loading?
  spektr plugins path            print the plugins folder
"""

#: Every option, and whether it takes a value. Anything else on the command
#: line is a mistake worth saying out loud: a mistyped flag used to be ignored,
#: and spektr started as if nothing had been asked.
_OPTIONS = {
    "--mode": True, "--theme": True, "--motion": True, "--morph": True,
    "--bands": True, "--fps": True, "--eco": True, "--shuffle": True,
    "--cells": True, "--background": True, "--device": True,
    "--devices": False, "--mic": False, "--diagnose": False, "--monitor": False,
    "--list-modes": False, "--list-themes": False, "--glyph-test": False,
    "--no-plugins": False, "--version": False, "-h": False, "--help": False,
    # for the release builds: loads every mode's code (see below)
    "--check-modes": False,
}


def _fail(message: str) -> None:
    print(f"spektr: {message}", file=sys.stderr)
    raise SystemExit(2)


def _suggest(word: str, choices) -> str:
    """``   (did you mean X?)`` for the closest of ``choices``, or ``""``."""
    lowered = {str(c).lower(): str(c) for c in choices}
    near = difflib.get_close_matches(word.lower(), list(lowered), n=1, cutoff=0.6)
    return f"   (did you mean {lowered[near[0]]}?)" if near else ""


def _check(argv: list[str]) -> None:
    """Refuse anything on the command line that spektr does not understand."""
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _OPTIONS:
            if _OPTIONS[token]:
                if i + 1 >= len(argv) or argv[i + 1] in _OPTIONS:
                    _fail(f"{token} needs a value   (see: spektr --help)")
                i += 1
        elif token.startswith("-"):
            _fail(f"unknown option {token}"
                  + (_suggest(token, _OPTIONS) or "   (see: spektr --help)"))
        else:
            _fail(f"unexpected {token!r}   (see: spektr --help)")
        i += 1


def _choice(argv, flag: str, choices) -> str | None:
    """The value given for ``flag``, checked against ``choices``, any case."""
    raw = _arg(argv, flag)
    if raw is None:
        return None
    for choice in choices:
        if raw.lower() == str(choice).lower():
            return str(choice)
    names = [str(c) for c in choices]
    listed = ", ".join(names[:-1]) + f" or {names[-1]}" if len(names) > 1 else names[0]
    _fail(f"{flag} takes {listed}, not {raw!r}{_suggest(raw, choices)}")


def _mode_named(name: str, registry) -> str:
    """The registered name for ``name``, whatever its case, or a clear no."""
    if registry.get(name) is not None:
        return name
    names = registry.names()
    for known in names:
        if known.lower() == name.lower():
            return known
    _fail(f"no mode called {name!r}"
          + (_suggest(name, names) or "   (see: spektr --list-modes)"))


def _theme_named(name: str) -> str:
    """The theme called ``name``, whatever its case, or a clear no."""
    from .palette import AUTO, all_themes

    names = [AUTO, *all_themes()]
    for known in names:
        if known.lower() == name.lower():
            return known
    _fail(f"no theme called {name!r}"
          + (_suggest(name, names) or "   (see: spektr --list-themes)"))


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

    if argv and argv[0] == "plugins":
        raise SystemExit(_plugins_cli(argv[1:]))
    _check(argv)

    # The two that need nothing, answered before anything heavy is imported.
    if "-h" in argv or "--help" in argv:
        print(_USAGE)
        return
    if "--version" in argv:
        print(f"spektr {__version__}")
        return

    # Before anything touches the audio libraries: their destructors can raise
    # during interpreter shutdown, and the traceback lands on screen after the
    # UI is gone, which reads as a crash on exit.
    from .capture import install_shutdown_filter

    install_shutdown_filter()

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

    from . import modes as mode_registry

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
        width = max(len(m.name) for m in mode_registry.MODES)
        for m in mode_registry.MODES:
            tag = f"  [{m.plugin}]" if m.is_plugin else ""
            mark = "  (opt-in: settings, or --mode)" if m.hidden else ""
            print(f"  {m.name:<{width}}  {m.blurb}{tag}{mark}")
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
        settings.mode = _mode_named(mode, mode_registry)
    theme = _arg(argv, "--theme")
    if theme:
        settings.theme = _theme_named(theme)
    cells = _choice(argv, "--cells", ("octant", "quadrant"))
    if cells:
        settings.cells = cells
    background = _choice(argv, "--background", ("theme", "terminal"))
    if background:
        settings.transparent_background = background == "terminal"
    for flag, name, choices in (
        ("--motion", "motion", config.MOTION_CHOICES),
        ("--morph", "morph", config.MORPH_CHOICES),
        ("--eco", "eco", config.ECO_CHOICES),
    ):
        value = _choice(argv, flag, choices)
        if value:
            setattr(settings, name, value)
    shuffle = _choice(argv, "--shuffle", (*config.SHUFFLE_SCOPES, "off"))
    if shuffle:
        settings.shuffle = shuffle != "off"
        if shuffle != "off":
            settings.shuffle_scope = shuffle
    bands = _arg(argv, "--bands")
    if bands is not None:
        if not bands.isdigit() or not (int(bands) == 0 or 8 <= int(bands) <= 64):
            _fail(f"--bands takes 8 to 64, or 0 to fit the terminal, not {bands!r}")
        settings.bands = int(bands)
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
                _fail(f"--fps takes a number from 15 to 240, or unlimited, not {fps_raw!r}")
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
