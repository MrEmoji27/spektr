"""The command line behaves as it did in 0.5.0.

Listing flags are run in a subprocess against an empty config folder and
their output compared with tests/golden/cli/. Flags that start the app are
checked by running main() with the app's run() replaced, so the settings it
would start with can be inspected. Update the pinned output, after a change
that is meant to alter it, with: python tests/test_cli_compat.py --update
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spektr import app as app_module, config, palette  # noqa: E402

PINNED = Path(__file__).resolve().parent / "golden" / "cli"
LISTINGS = {
    "help": ["--help"],
    "list-modes": ["--list-modes", "--no-plugins"],
    "list-themes": ["--list-themes", "--no-plugins"],
}


def _run(args: list[str], home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, APPDATA=str(home), XDG_CONFIG_HOME=str(home),
               PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "spektr.app", *args],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=120,
    )


@pytest.mark.parametrize("name", sorted(LISTINGS))
def test_listing_output_is_unchanged(name, tmp_path):
    result = _run(LISTINGS[name], tmp_path)
    assert result.returncode == 0, result.stderr
    want = (PINNED / f"{name}.txt").read_text(encoding="utf-8")
    assert result.stdout == want, (
        f"`spektr {' '.join(LISTINGS[name])}` output changed. If intended, run "
        "python tests/test_cli_compat.py --update and say so in the commit."
    )


def test_version_prints_the_version(tmp_path):
    from spektr import __version__
    result = _run(["--version"], tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == f"spektr {__version__}"


def test_plugins_subcommand_runs(tmp_path):
    result = _run(["plugins", "list"], tmp_path)
    assert result.returncode == 0, result.stderr


# ── flags that start the app ─────────────────────────────────────────────────


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """Run main() with argv; return the app it would have started, or None."""
    monkeypatch.setattr(palette, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(app_module.atexit, "register", lambda *a, **k: None)
    started = []
    monkeypatch.setattr(app_module.Spektr, "run", lambda self, *a, **k: started.append(self))

    def run(*argv: str):
        monkeypatch.setattr(sys, "argv", ["spektr", "--no-plugins", *argv])
        app_module.main()
        return started[-1] if started else None

    return run


def test_no_flags_starts_with_the_saved_settings(launch):
    app = launch()
    assert app is not None
    assert app.settings.mode == config.Settings().mode


def test_mode_flag(launch):
    assert launch("--mode", "Tunnel").settings.mode == "Tunnel"


def test_unknown_mode_does_not_start(launch, capsys):
    assert launch("--mode", "No Such Mode") is None
    assert "unknown mode" in capsys.readouterr().out


def test_theme_flag(launch):
    assert launch("--theme", "gruvbox").settings.theme == "gruvbox"


def test_cells_flag(launch):
    from spektr import render
    try:
        assert launch("--cells", "quadrant").settings.cells == "quadrant"
    finally:
        render.set_cell_mode("octant")


def test_background_flag(launch):
    assert launch("--background", "terminal").settings.transparent_background is True


@pytest.mark.parametrize("value, expected", [("30", 30), ("unlimited", config.FPS_UNLIMITED)])
def test_fps_flag(launch, value, expected):
    assert launch("--fps", value).settings.fps == expected


def test_fps_flag_rejects_junk(launch):
    with pytest.raises(SystemExit) as exc:
        launch("--fps", "fast")
    assert exc.value.code == 2


def test_device_and_mic_flags(launch):
    app = launch("--device", "3", "--mic")
    assert app._device == 3
    assert app._allow_mic is True


def _update() -> None:
    import tempfile
    PINNED.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as home:
        for name, args in LISTINGS.items():
            result = _run(args, Path(home))
            if result.returncode != 0:
                raise SystemExit(f"{name} failed: {result.stderr}")
            (PINNED / f"{name}.txt").write_text(result.stdout, encoding="utf-8")
            print(f"wrote {name}.txt")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    else:
        print(__doc__)
