"""The legacy app path forwards both the shell and command line live."""
from __future__ import annotations


def test_app_path_forwards_shell_and_cli_live(monkeypatch):
    import spektr.app as legacy
    import spektr.cli as cli
    import spektr.ui.app as shell

    app_sentinel = object()
    main_sentinel = object()
    monkeypatch.setattr(shell, "Spektr", app_sentinel)
    monkeypatch.setattr(cli, "main", main_sentinel)
    assert legacy.Spektr is app_sentinel
    assert legacy.main is main_sentinel
    assert isinstance(legacy._USAGE, str)
