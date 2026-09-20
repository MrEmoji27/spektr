"""Every built-in mode draws exactly what it drew when the golden file was
recorded. See ``golden.py`` for what is fingerprinted and how to update it."""
from __future__ import annotations

import sys

import golden
import pytest

RECORDED = golden.load_golden() if golden.GOLDEN.exists() else {}
MODES = golden.builtin_modes()
SAME_PLATFORM = RECORDED.get(golden.PLATFORM_KEY) == sys.platform


def _colours_only(mode_name: str) -> bool:
    """True when this mode's glyphs cannot be compared on this machine.

    A handful of modes decide a subcell on a float comparison that lands on
    the threshold, and the last bit of that float is not the same on every
    CPU. On the machine that recorded the file they are pinned whole; anywhere
    else only their colours are.
    """
    return mode_name in golden.PLATFORM_SENSITIVE and not SAME_PLATFORM


def test_the_golden_file_exists():
    assert golden.GOLDEN.exists(), "record it: python tests/golden.py --update"


def test_every_mode_has_golden_output():
    missing = sorted({
        m.name for m in MODES for c in golden.CASES
        if golden.key(m.name, c, _colours_only(m.name)) not in RECORDED
    })
    assert not missing, f"no golden output for {missing}: python tests/golden.py --update"


def test_the_golden_file_names_no_mode_that_is_gone():
    live = {m.name for m in MODES}
    stale = sorted({
        k.split("|")[0] for k in RECORDED
        if k != golden.PLATFORM_KEY and k.split("|")[0] not in live
    })
    assert not stale, f"golden output for modes that no longer exist: {stale}"


@pytest.mark.parametrize("mode", MODES, ids=[m.name for m in MODES])
def test_mode_draws_what_it_drew(mode):
    colours_only = _colours_only(mode.name)
    changed = [
        c.id for c in golden.CASES
        if golden.key(mode.name, c, colours_only) in RECORDED
        and golden.fingerprint(mode, c, colours_only)
        != RECORDED[golden.key(mode.name, c, colours_only)]
    ]
    assert not changed, (
        f"{mode.name} draws differently in {changed}. If that is intended, run "
        "python tests/golden.py --update and say so in the commit message."
    )
