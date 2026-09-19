"""Every built-in mode draws exactly what it drew when the golden file was
recorded. See ``golden.py`` for what is fingerprinted and how to update it."""
from __future__ import annotations

import pytest

import golden

RECORDED = golden.load_golden() if golden.GOLDEN.exists() else {}
MODES = golden.builtin_modes()


def test_the_golden_file_exists():
    assert golden.GOLDEN.exists(), "record it: python tests/golden.py --update"


def test_every_mode_has_golden_output():
    missing = sorted({m.name for m in MODES
                      for c in golden.CASES if golden.key(m.name, c) not in RECORDED})
    assert not missing, f"no golden output for {missing}: python tests/golden.py --update"


def test_the_golden_file_names_no_mode_that_is_gone():
    live = {golden.key(m.name, c) for m in MODES for c in golden.CASES}
    stale = sorted({k.split("|")[0] for k in RECORDED if k not in live})
    assert not stale, f"golden output for modes that no longer exist: {stale}"


@pytest.mark.parametrize("mode", MODES, ids=[m.name for m in MODES])
def test_mode_draws_what_it_drew(mode):
    changed = [
        c.id for c in golden.CASES
        if golden.key(mode.name, c) in RECORDED
        and golden.fingerprint(mode, c) != RECORDED[golden.key(mode.name, c)]
    ]
    assert not changed, (
        f"{mode.name} draws differently in {changed}. If that is intended, run "
        "python tests/golden.py --update and say so in the commit message."
    )
