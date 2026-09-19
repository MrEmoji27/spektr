"""Tests never touch the real config folder.

The app saves settings when it closes (``Spektr.on_unmount``), into the
platform config folder unless it was given a ``config_dir``. Tests that start
the app without one used to overwrite the user's own settings on every run.
``conftest.py`` points the config folder at a temporary directory for every
test; this checks that it does.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import palette  # noqa: E402


def _real_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        return Path(base) / "spektr"
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "spektr"


def test_the_config_folder_is_not_the_real_one_during_tests():
    assert palette.config_dir().resolve() != _real_config_dir().resolve()


def test_an_app_closed_in_a_test_does_not_write_the_real_config():
    import asyncio

    from spektr import config
    from spektr.app import Spektr

    real = _real_config_dir() / "config.json"
    before = real.read_bytes() if real.exists() else None

    async def open_and_close() -> None:
        app = Spektr(settings=config.Settings(fps=15, theme="gruvbox"))
        app.notify = lambda *a, **k: None  # type: ignore[method-assign]
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()

    asyncio.run(open_and_close())
    after = real.read_bytes() if real.exists() else None
    assert after == before
    assert (palette.config_dir() / "config.json").exists()
