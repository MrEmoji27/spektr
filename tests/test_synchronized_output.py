"""spektr asks the terminal for synchronized output on Windows.

Textual brackets repaints in ``CSI ?2026h`` / ``?2026l`` only after the
terminal answers the ``CSI ?2026$p`` query, and only its Linux and web drivers
send that query. Windows Terminal supports the mode, but was never asked, so a
frame arrived unbracketed and was painted as it came: matched against the
frames spektr rendered, 95 of 720 captured 60 fps frames of Tunnel at 188x52
were two renders stitched together, and 0 of 470 unoccluded frames once the
query was sent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.app as app_mod  # noqa: E402


class _Driver:
    def __init__(self, inline=False, headless=False):
        self.is_inline = inline
        self.is_headless = headless
        self.written: list[str] = []

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass


def _ask(driver, platform, monkeypatch):
    monkeypatch.setattr(app_mod.sys, "platform", platform)
    fake = type("A", (), {"_driver": driver})()
    app_mod.Spektr._ask_for_synchronized_output(fake)
    return driver.written


def test_the_query_is_sent_on_windows(monkeypatch):
    assert _ask(_Driver(), "win32", monkeypatch) == ["\x1b[?2026$p"]


def test_nothing_is_sent_elsewhere_or_without_a_real_screen(monkeypatch):
    assert _ask(_Driver(), "linux", monkeypatch) == []
    assert _ask(_Driver(headless=True), "win32", monkeypatch) == []
    assert _ask(_Driver(inline=True), "win32", monkeypatch) == []
