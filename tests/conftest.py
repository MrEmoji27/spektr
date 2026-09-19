"""Shared test setup.

Every test runs against a temporary config folder. The app saves its settings
when it closes, into the platform config folder unless it was given a
``config_dir``, so a test that starts the app without one used to overwrite
the user's real settings on every run. Tests that want a specific folder
still pass ``config_dir`` or patch ``palette.config_dir`` themselves; this is
the floor underneath them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr import palette  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path_factory, monkeypatch):
    folder = tmp_path_factory.mktemp("spektr-config")
    monkeypatch.setattr(palette, "config_dir", lambda: folder)
    return folder
