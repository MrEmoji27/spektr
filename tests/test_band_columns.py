"""Band gutters must scale to what the bars can afford.

A fixed one-column gutter between every pair of bands only works while the
bands are wide enough to pay for it. Once a band-count control could push the
count past what the screen affords, bands collapsed onto single cells and the
gutters grew into half the picture — thin bright bars ruled apart by dark
lines, worst on narrow phone grids where any count above ~w/4 striped the
screen. ``band_columns`` now steps its gutter plan down (every gap, every
other gap, none, then a nearest-band downsample) while any band would draw
less than two columns.

The properties that have to stay true are pinned here, because "it looks
right on my terminal" is not a property a future change can be checked
against:

* the classic layouts — everything the old algorithm drew with two or more
  columns per band — come out byte-identical to it;
* wherever a gutter exists at all, no bar is narrower than two columns, so
  the picture can never read as stripes again;
* bars are contiguous blocks in spectrum order, and no column is left
  unassigned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.modes import band_columns  # noqa: E402


def _band_columns_before(w: int, n: int):
    """The pre-fix implementation, frozen verbatim as the golden reference."""
    gaps = n - 1
    usable = max(n, w - gaps)
    base, extra = divmod(usable, n)

    col_band = np.zeros(w, dtype=np.int32)
    active = np.zeros(w, dtype=bool)
    x = 0
    for b in range(n):
        width = base + (1 if b < extra else 0)
        end = min(w, x + width)
        col_band[x:end] = b
        active[x:end] = True
        x = end + 1          # leave a gutter
        if x >= w:
            break
    return col_band, active


#: Sizes the bug was actually seen at (phone portrait through desktop wide)
#: plus the defaults every other platform draws.
SWEEP_W = [13, 24, 30, 36, 40, 48, 56, 60, 72, 80, 90, 100, 118, 120,
           128, 160, 200, 300, 400]
SWEEP_N = list(range(8, 65)) + [1, 2, 4]


def test_classic_layouts_are_byte_identical_to_the_old_algorithm():
    """Wherever the old code gave every band two-plus columns, nothing moves."""
    checked = 0
    for w in SWEEP_W:
        for n in SWEEP_N:
            old_cb, old_act = _band_columns_before(w, n)
            # The old layout is worth keeping exactly when it never thinned a
            # band below two columns; that is also precisely when the new one
            # takes the same first branch.
            runs_ok = True
            run = 0
            for x in range(w):
                if old_act[x]:
                    run += 1
                else:
                    if run and run < 2:
                        runs_ok = False
                    run = 0
            if run and run < 2:
                runs_ok = False
            if not runs_ok:
                continue
            got_cb, got_act = band_columns(w, n)
            assert np.array_equal(got_cb, old_cb), f"w={w} n={n}: mapping moved"
            assert np.array_equal(got_act, old_act), f"w={w} n={n}: gutters moved"
            checked += 1
    assert checked > 200, f"golden sweep degenerated ({checked} cases)"


def _active_runs(active: np.ndarray) -> list[int]:
    runs, run = [], 0
    for a in active:
        if a:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def test_no_layout_stripes_the_screen():
    """With gutters present every bar keeps two columns; without them none hide."""
    for w in SWEEP_W:
        for n in SWEEP_N:
            if n > w:
                continue          # the downsample branch has its own test
            _, active = band_columns(w, n)
            lit = int(active.sum())
            assert lit > 0, f"w={w} n={n}: nothing drawn"
            runs = _active_runs(active)
            gutters = w - lit
            if gutters == 0:
                continue          # gapless wall: nothing to separate
            assert min(runs) >= 2, (
                f"w={w} n={n}: a bar drew {min(runs)} column(s) beside "
                f"{gutters} gutter columns — stripes"
            )


def test_bars_are_contiguous_blocks_in_spectrum_order():
    """A band owns one run of columns, and low frequencies sit left."""
    for w in SWEEP_W:
        for n in SWEEP_N:
            col_band, active = band_columns(w, n)
            order = col_band[active]
            assert np.all(np.diff(order) >= 0), f"w={w} n={n}: spectrum folded"
            seen = {}
            prev, start = None, 0
            for x, b in enumerate(order.tolist() + [-1]):
                if b != prev:
                    if prev is not None and prev >= 0:
                        assert prev not in seen, (
                            f"w={w} n={n}: band {prev} split across the row"
                        )
                        seen[prev] = start
                    prev, start = b, x
            assert order.size and 0 <= int(order[0]) and int(order[-1]) < max(n, 1)


def test_more_bands_than_columns_downsamples_gapless():
    """Every column shows the nearest band; nothing folds and nothing hides."""
    for w in (8, 13, 24, 36):
        for n in (w + 1, 64, 256):
            col_band, active = band_columns(w, n)
            assert active.all(), f"w={w} n={n}: dead columns under downsampling"
            assert np.all(np.diff(col_band) >= 0), f"w={w} n={n}: spectrum folded"
            assert col_band.min() >= 0 and col_band.max() < n
