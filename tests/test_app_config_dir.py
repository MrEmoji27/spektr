"""The app hands its config directory to the visualiser.

The whole injection chain — ``Spektr(config_dir=…)`` down through
``compose``'s ``AudioVisualizer(config_dir=…)`` into the widget's theme list —
is one seam: inject a directory at the app level and the widget's theme list
must come from it, not from the real user directory. The widget-level tests
cover the widget alone; this one starts one level up, where a leak would mean
the chain was broken at the app.

The negative half of the assertion (the real user directory's themes must not
appear) is why the injected theme gets a random name: it can never collide
with a theme a developer happens to have saved, so a leak is always visible.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cp1252 consoles cannot encode this file's output characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spektr import config  # noqa: E402
from spektr.app import Spektr  # noqa: E402
from spektr.palette import BUILTIN, load_user_themes  # noqa: E402


def test_config_dir_reaches_the_widgets_theme_list() -> None:
    async def run() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="spektr-app-config-test-"))
        theme_dir = tmp / "themes"
        theme_dir.mkdir()
        # unique, so it can never collide with a real user theme
        name = f"injected-{uuid.uuid4().hex[:8]}"
        (theme_dir / f"{name}.toml").write_text(
            'low = "#00ff41"\nmid = "#ffb000"\nhigh = "#ff3300"\n',
            encoding="utf-8",
        )

        app = Spektr(settings=config.Settings(fps=15), config_dir=tmp)
        app.notify = lambda *a, **k: None  # type: ignore[method-assign]
        async with app.run_test(size=(80, 24)):
            viz = app.viz

            user_themes = set(viz._themes) - set(BUILTIN)
            assert name in user_themes, f"injected theme {name} not in the widget's list"
            assert user_themes == {name}, f"real user themes leaked in: {user_themes}"
            assert name in viz.theme_names

            real_user = set(load_user_themes())
            assert not (user_themes & real_user), "the real user directory leaked through"

            assert viz.apply_theme(name)
            assert viz.theme_name == name

    asyncio.run(run())