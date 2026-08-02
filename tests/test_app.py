"""Headless smoke test — boots the app, drives the UI, checks nothing explodes.

Runs without an audio device, so it works in CI. Capture will report that it
found nothing; that's expected and is itself worth asserting, because a missing
device used to kill the capture thread silently.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cp1252 consoles cannot encode the … and — in this file's output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_T0 = time.monotonic()


def mark(message: str) -> None:
    print(f"  [{time.monotonic() - _T0:5.1f}s] {message}", flush=True)

from spektr import config                      # noqa: E402
from spektr.app import Spektr                   # noqa: E402
from spektr.modes import MODES                  # noqa: E402
from spektr.palette import all_themes           # noqa: E402


async def main() -> int:
    problems: list[str] = []

    # Two concessions to the test harness, neither of which touches what's
    # under test. Pilot waits for the app to go idle between keystrokes, and an
    # app repainting at 60 fps essentially never does — so the render loop is
    # slowed right down here. Toasts hold the app busy for their full timeout
    # for the same reason, so they're stubbed out.
    settings = config.Settings(fps=15)

    app = Spektr(settings=settings)
    app.notify = lambda *a, **k: None  # type: ignore[method-assign]
    async with app.run_test(size=(120, 32)) as pilot:
        viz = app.viz

        # every mode must survive being selected and painted
        for m in MODES:
            viz.set_mode(m.name)
            try:
                strips = viz._build()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"mode {m.name}: {exc}")
                continue
            if len(strips) != viz.size.height:
                problems.append(f"mode {m.name}: {len(strips)} strips, want {viz.size.height}")
        await pilot.pause()
        mark(f"{len(MODES)} modes painted")

        # every theme must build a ramp and repaint
        themes = list(all_themes())
        for name in themes:
            viz.apply_theme(name)
            if len(viz.palette.styles) != 64:
                problems.append(f"theme {name}: ramp is {len(viz.palette.styles)} steps")
            viz._build()
        viz.apply_theme("auto")
        await pilot.pause()
        mark(f"{len(themes)} themes applied (+ auto)")

        # keyboard surface
        for key in ("m", "M", "T", "f", "f", "s", "p", "left_square_bracket",
                    "right_square_bracket", "g", "G", "r"):
            await pilot.press(key)
        await pilot.pause()
        mark("keybindings OK")

        # pickers: open, filter, arrow, escape (restores), then open and commit
        viz.apply_theme("gruvbox")
        before = viz.theme_name
        await pilot.press("t")
        await pilot.pause()
        await pilot.press("n", "o", "r")
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("escape")
        await pilot.pause()
        if viz.theme_name != before:
            problems.append(f"escape did not restore theme: {before} -> {viz.theme_name}")
        mark(f"theme picker OK (restored {viz.theme_name})")

        # the picker opens highlighted on the current item, so start somewhere
        # with room to move down
        viz.set_mode("Bars")
        mode_before = viz.mode_name
        await pilot.press("v")
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        if viz.mode_name == mode_before:
            problems.append("mode picker did not commit a new selection")
        mark(f"mode picker OK (committed {viz.mode_name})")

        # resize must not break cached geometry
        for size in ((40, 10), (200, 50), (80, 24)):
            viz.set_mode("Pulse")
            await pilot.resize_terminal(*size)
            await pilot.pause()
            viz.set_mode("Warp")
            await pilot.pause()
        mark("resize OK")

        status = viz.status
        if "starting" in status:
            problems.append("capture never reported a result")
        mark(f"capture status: {status}")

    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for p in problems:
            print("  ", p)
        return 1
    print("\nall good")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
