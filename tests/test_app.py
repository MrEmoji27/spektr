"""Headless smoke test — boots the app, drives the UI, checks nothing explodes.

Runs without an audio device, so it works in CI. Capture will report that it
found nothing; that's expected and is itself worth asserting, because a missing
device used to kill the capture thread silently.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The app writes its settings back on exit (app.on_unmount), and the test
# drives real UI changes through those settings — including saving a preset,
# which writes presets.json next to config.json. Both config.py and presets.py
# build their path from palette.config_dir(), so redirecting that one function
# scratches both at once rather than needing a matching patch per module added
# under this directory later.
import spektr.palette as _palette  # noqa: E402

_scratch_dir = Path(tempfile.mkdtemp(prefix="spektr-test-"))
_palette.config_dir = lambda: _scratch_dir  # type: ignore[attr-defined]

# cp1252 consoles cannot encode the … and — in this file's output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_T0 = time.monotonic()


def mark(message: str) -> None:
    print(f"  [{time.monotonic() - _T0:5.1f}s] {message}", flush=True)


from spektr import config  # noqa: E402
from spektr.app import Spektr  # noqa: E402
from spektr.modes import MODES  # noqa: E402
from spektr.palette import RAMP_STEPS, all_themes  # noqa: E402


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
                problems.append(
                    f"mode {m.name}: {len(strips)} strips, want {viz.size.height}"
                )
        await pilot.pause()
        mark(f"{len(MODES)} modes painted")

        # every theme must build a ramp and repaint
        themes = list(all_themes())
        for name in themes:
            viz.apply_theme(name)
            if len(viz.palette.styles) != RAMP_STEPS:
                problems.append(
                    f"theme {name}: ramp is {len(viz.palette.styles)} steps"
                )
            viz._build()
        viz.apply_theme("auto")
        await pilot.pause()
        mark(f"{len(themes)} themes applied (+ auto)")

        # keyboard surface. s (shuffle) is pressed twice — on then off — so it
        # doesn't leave a 15s timer running that could fire mid-test and change
        # the mode out from under an assertion later in this file. L (save
        # preset) opens an overlay that escape then cancels, for the same
        # reason: left open, it would swallow every keypress after it via
        # check_action. l (load preset) is safe bare — with no presets saved
        # yet it just notifies and never opens anything. show_status has no
        # key of its own any more (folded into the settings panel's source
        # row) but startup and the source keys still call it internally, so
        # it's called directly here rather than left completely untested.
        app.action_show_status()
        for key in (
            "m",
            "M",
            "T",
            "f",
            "f",
            "d",
            "D",
            "p",
            "left_square_bracket",
            "right_square_bracket",
            "g",
            "G",
            "r",
            "s",
            "s",
            "l",
            "L",
            "escape",
        ):
            await pilot.press(key)
        await pilot.pause()
        if app.settings.shuffle:
            problems.append("shuffle left on after toggling it twice")
        if app._overlay is not None:
            problems.append("an overlay was left open after the keybinding sweep")
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
            problems.append(
                f"escape did not restore theme: {before} -> {viz.theme_name}"
            )
        mark(f"theme picker OK (restored {viz.theme_name})")

        # rainbow theme is animated: its ramp drifts over time + position
        from spektr.palette import BUILTIN

        if not BUILTIN["rainbow"].animated:
            problems.append("rainbow theme should be marked animated")
        mark("rainbow animation flag OK")

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

        # settings panel: open, step a row, change a value, close. This used to
        # crash on open because the Setting constructor's render/note args were
        # swapped, and the key-surface loop below never presses c — so a
        # regression here would otherwise go unnoticed.
        await pilot.press("c")
        await pilot.pause()
        settings_open = app._overlay is not None
        if not settings_open:
            problems.append("settings panel did not open")
        else:
            await pilot.press("down")  # move to the bands row
            await pilot.press("right")  # change the value
            # source is the last row (fps, bands, sensitivity, gate, chrome,
            # source) — it has no choices list, only step/live, which used to
            # be untested entirely since the row didn't exist before this.
            for _ in range(4):
                await pilot.press("down")
            await pilot.press("right")  # next_source via step, not apply
            await pilot.pause()
            await pilot.press("escape")  # close (enter is eaten by OptionList)
            await pilot.pause()
            if app._overlay is not None:
                problems.append("settings panel did not close")
        mark(
            f"settings panel OK ({'opened+stepped+closed' if settings_open else 'did not open'})"
        )

        # presets: save the current look under a name, switch away, load it
        # back, and confirm both the name-prompt and the preset picker itself
        # round-trip correctly.
        viz.set_mode("Flame")
        viz.commit_mode()
        viz.apply_theme("synthwave")
        viz.commit_theme()
        await pilot.pause()

        await pilot.press("L")
        await pilot.pause()
        for ch in "smoke test":
            await pilot.press("space" if ch == " " else ch)
        await pilot.press("enter")
        await pilot.pause()
        if "smoke test" not in app._presets:
            problems.append(f"preset was not saved: {list(app._presets)}")

        viz.set_mode("Bars")
        viz.commit_mode()
        viz.apply_theme("classic")
        viz.commit_theme()
        await pilot.pause()

        await pilot.press("l")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        if viz.mode_name != "Flame" or viz.theme_name != "synthwave":
            problems.append(
                f"preset did not apply: mode={viz.mode_name} theme={viz.theme_name}"
            )
        mark(f"presets OK (saved+loaded {list(app._presets)})")

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
