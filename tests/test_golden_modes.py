"""Every built-in mode draws what it drew when the golden file was recorded.

On the machine that recorded it, cell for cell. On any other machine, by the
shape of the picture: see ``golden.TOLERANCE`` for why an exact comparison
cannot hold across operating systems.
"""
from __future__ import annotations

import sys

import golden
import pytest

RECORDED = golden.load_golden() if golden.GOLDEN.exists() else {}
MODES = golden.builtin_modes()
SAME_PLATFORM = RECORDED.get(golden.PLATFORM_KEY) == sys.platform


def test_the_golden_file_exists():
    assert golden.GOLDEN.exists(), "record it: python tests/golden.py --update"


def test_every_mode_has_golden_output():
    missing = sorted({
        m.name for m in MODES for c in golden.CASES
        if golden.key(m.name, c) not in RECORDED
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
    changed = []
    for case in golden.CASES:
        recorded = RECORDED.get(golden.key(mode.name, case))
        if recorded is None:
            continue
        digest, shape = golden.measure(mode, case)
        if digest == recorded[0]:
            continue
        if SAME_PLATFORM:
            changed.append(case.id)
        else:
            apart = golden.differs(recorded[1], shape)
            if apart > golden.TOLERANCE:
                changed.append(f"{case.id} (off by {apart:.3f})")
    assert not changed, (
        f"{mode.name} draws differently in {changed}. If that is intended, run "
        "python tests/golden.py --update and say so in the commit message."
    )
