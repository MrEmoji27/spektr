"""The command line says what it did not understand, and does what it did.

A mistyped flag used to be ignored and spektr started as if nothing had been
asked; an unknown mode printed a line and exited as a success; half the
settings had no flag at all. And ``spektr --version`` took half a second,
because answering it imported the whole UI toolkit.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from spektr import cli, config

ROOT = Path(__file__).resolve().parent.parent


class _App:
    """Stands in for the UI: records the settings it was started with."""

    started: list = []

    def __init__(self, device=None, settings=None, allow_mic=False):
        self.settings = settings
        self._config_dir = None

    def run(self):
        _App.started.append(self.settings)


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run ``spektr <args>`` in-process with a fresh config and no real UI.

    The saved config is never the user's: ``config.load`` hands back defaults
    and the save registered for interpreter exit is dropped.
    """
    import atexit

    import spektr.ui.app as shell

    monkeypatch.setattr(shell, "Spektr", _App)
    monkeypatch.setattr(config, "load", lambda *a, **k: config.Settings())
    monkeypatch.setattr(atexit, "register", lambda *a, **k: None)
    _App.started = []

    def go(*args):
        monkeypatch.setattr(sys, "argv", ["spektr", "--no-plugins", *args])
        cli.main()
        return _App.started[-1] if _App.started else None

    return go


def _fails(run, capsys, *args) -> str:
    with pytest.raises(SystemExit) as exc:
        run(*args)
    assert exc.value.code == 2
    return capsys.readouterr().err


def test_a_mistyped_flag_is_refused_with_a_suggestion(run, capsys):
    assert "did you mean --mode?" in _fails(run, capsys, "--mdoe", "Bars")


def test_a_stray_word_is_refused(run, capsys):
    assert "unexpected 'Bars'" in _fails(run, capsys, "Bars")


def test_a_flag_without_its_value_is_refused(run, capsys):
    assert "--mode needs a value" in _fails(run, capsys, "--mode")


def test_an_unknown_mode_fails_and_suggests(run, capsys):
    assert "did you mean Bars?" in _fails(run, capsys, "--mode", "Barz")


def test_an_unknown_theme_fails_and_suggests(run, capsys):
    assert "did you mean gruvbox?" in _fails(run, capsys, "--theme", "gruvbx")


def test_names_are_matched_whatever_their_case(run):
    s = run("--mode", "jp bars", "--theme", "GRUVBOX")
    assert s.mode == "JP Bars" and s.theme == "gruvbox"


def test_every_setting_has_a_flag(run):
    s = run("--motion", "glide", "--morph", "classic", "--eco", "on",
            "--bands", "24", "--shuffle", "themes")
    assert (s.motion, s.morph, s.eco, s.bands) == ("glide", "classic", "on", 24)
    assert s.shuffle and s.shuffle_scope == "themes"
    assert run("--shuffle", "off").shuffle is False


@pytest.mark.parametrize("args, says", [
    (("--motion", "glyde"), "snappy or glide"),
    (("--eco", "maybe"), "on or off"),
    (("--bands", "7"), "8 to 64"),
    (("--fps", "fast"), "15 to 240"),
    (("--cells", "hex"), "octant or quadrant"),
])
def test_a_bad_value_names_the_good_ones(run, capsys, args, says):
    assert says in _fails(run, capsys, *args)


def test_the_listing_lines_up(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["spektr", "--no-plugins", "--list-modes"])
    cli.main()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    import spektr.modes as M

    width = max(len(m.name) for m in M.MODES)
    # every blurb starts in the same column
    assert all(ln[2 + width:4 + width] == "  " for ln in lines)


def _in_a_fresh_python(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8")


def test_python_dash_m_runs_it():
    out = _in_a_fresh_python("-m", "spektr", "--version")
    assert out.returncode == 0 and out.stdout.startswith("spektr ")


def test_the_quick_answers_do_not_load_the_ui():
    """``--version`` and ``--help`` took half a second: resolving ``main`` on
    the launcher's module imported the whole UI toolkit, and so did a Python
    probe for ``__path__`` answered by a forwarding ``__getattr__``."""
    code = ("import sys; sys.argv = ['spektr', '--version']; "
            "from spektr.app import main; main(); "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('textual', 'numpy') or m.startswith('spektr.ui')))")
    out = _in_a_fresh_python("-c", code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"
