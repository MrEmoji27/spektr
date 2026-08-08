"""Render benchmark and shape check for every mode.

Run with ``python tests/bench.py``. No audio device needed — it feeds the modes
a synthetic spectrum, so it works in CI and over SSH.

The numbers that matter are the totals: every mode should stay well under the
frame budget (16.7 ms at 60 fps) at the largest size you'd plausibly run.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cp1252 consoles cannot encode this file's output characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import make_strips  # noqa: E402

SIZES = [(40, 10), (80, 24), (120, 16), (200, 50), (240, 60), (400, 100)]
BUDGET_MS = 1000.0 / 60.0


def make_ctx(w, h, frame, state, t, palette):
    phase = np.linspace(0.0, 3.0, N_BANDS)
    bands = np.clip(np.abs(np.sin(phase + t * 2.0)) * 0.8, 0.0, 1.0)
    wave = np.sin(np.linspace(0.0, 40.0, 512) + t * 10.0) * 0.7
    stereo = np.stack((wave, np.roll(wave, 7)), axis=1)
    return Ctx(
        w=w, h=h,
        bands=bands, peaks=np.clip(bands * 1.05, 0, 1),
        bands_l=bands * 0.9, bands_r=bands,
        wave=wave, stereo=stereo,
        frame=frame, t=t, dt=1 / 60,
        energy=float(bands.mean()), silent=False,
        palette=palette, state=state,
    )


def check(palette) -> list[str]:
    """Every mode must return correctly shaped, in-range arrays at every size."""
    fails: list[str] = []
    for w, h in SIZES:
        for m in M.MODES:
            state: dict = {}
            try:
                for f in range(4):
                    out = m.fn(make_ctx(w, h, f, state, f / 60, palette))
                    codes, cidx = out[0], out[1]
                    bidx = out[2] if len(out) == 3 else None
                    assert codes.shape == (h, w), f"codes {codes.shape} != {(h, w)}"
                    assert cidx.shape == (h, w), f"cidx {cidx.shape} != {(h, w)}"
                    assert 0 <= cidx.min() and cidx.max() < 64, "ramp index out of range"
                    strips = make_strips(codes, cidx, palette, bidx)
                    assert len(strips) == h
                    assert all(s.cell_length == w for s in strips)
            except Exception as exc:  # noqa: BLE001
                fails.append(f"{w}x{h} {m.name}: {exc}")
                traceback.print_exc(limit=2)
    return fails


def bench(palette, sizes=((120, 16), (200, 50), (240, 60), (400, 100)), n=20, batches=5, warm=25) -> int:
    """Time every mode at every size, reporting best and typical per mode.

    A single wall-clock loop around N frames is what this used to do, and it
    is noisy at 400x100: numpy's allocator churn between modes, GC pauses and
    CPU state made a *different* random set of modes land "over budget" on
    every run — modes that are 6-9 ms when timed alone. Per-frame medians with
    a warm-up helped, but on this machine's power management even those land
    inside multi-second slow phases: byte-identical code measured 13.5 ms and
    18.3 ms in consecutive runs of this file.

    So: several batches, and two numbers out of them. ``best`` is the min batch
    median — the fastest phase that recurs. It is the stable one, so it is what
    to compare across runs when looking for a regression. ``typical`` is the
    median batch, and it is what the budget gate uses, because whether a mode
    drops frames is a question about its ordinary case. Gating on ``best``
    passes anything whose fast phase happens to fit, which is how a mode that
    stutters in practice gets recorded as fine.

    Run this with nothing else going: two overlapping runs of this file
    measured Pulse at a 16.5 ms best against a 23.3 ms typical, a 41% spread
    that was entirely the two processes contending.
    """
    import gc
    import statistics

    over = 0
    for w, h in sizes:
        print(f"\n== {w}x{h} " + "=" * 46)
        print(
            f"{'mode':<10} {'build':>8} {'strips':>8} {'best':>8} {'typical':>9} {'fps':>7}"
        )
        for m in M.MODES:
            state: dict = {}
            counter = [0]

            def step():
                out = m.fn(make_ctx(w, h, counter[0], state, counter[0] / 60, palette))
                counter[0] += 1
                make_strips(out[0], out[1], palette, out[2] if len(out) == 3 else None)

            def batch_medians_ms() -> tuple[float, float]:
                """Median build and median strips time from one batch.

                Both halves are timed inside the *same* frame rather than in
                two separate loops. Timing them apart lets the two loops land
                in different machine phases, and then the reported numbers
                stop being consistent with each other: one run printed a mode
                at 15.15 ms build against a 12.79 ms total, with strips
                clamped to zero to hide the contradiction.
                """
                builds, strips_ = [], []
                for _ in range(n):
                    t0 = time.perf_counter()
                    out = m.fn(make_ctx(w, h, counter[0], state, counter[0] / 60, palette))
                    t1 = time.perf_counter()
                    make_strips(out[0], out[1], palette, out[2] if len(out) == 3 else None)
                    t2 = time.perf_counter()
                    counter[0] += 1
                    builds.append(t1 - t0)
                    strips_.append(t2 - t1)
                return (
                    statistics.median(builds) * 1000.0,
                    statistics.median(strips_) * 1000.0,
                )

            # warm the mode's scratch buffers, caches and the heap before timing
            for _ in range(warm):
                step()

            gc.collect()

            # (build, strips, total) per batch, sorted by total
            runs = sorted(
                ((b, s, b + s) for b, s in (batch_medians_ms() for _ in range(batches))),
                key=lambda bst: bst[2],
            )
            # Two statistics, because they answer different questions and
            # reporting only one of them is how a mode gets declared fine.
            #
            # ``capacity`` is the best batch — the fastest phase that recurs.
            # It is the stable one: a real regression moves it, machine noise
            # does not, so it is what to compare across runs.
            #
            # ``typical`` is the median batch, and it is what the *gate* uses.
            # Whether a mode drops frames for someone is a question about its
            # ordinary case, not its best one, and gating on the best case
            # quietly passes anything whose fast phase happens to fit. Build
            # and strips are reported from the capacity batch so the two
            # columns still describe one window.
            build, strips, capacity = runs[0]
            typical = runs[len(runs) // 2][2]

            flag = "  <-- over budget" if typical > BUDGET_MS else ""
            if typical > BUDGET_MS:
                over += 1
            print(
                f"{m.name:<10} {build:7.2f}ms {strips:7.2f}ms "
                f"{capacity:7.2f}ms {typical:8.2f}ms "
                f"{1000 / max(typical, 1e-6):7.0f}{flag}"
            )
    return over


if __name__ == "__main__":
    palette = Palette(BUILTIN["gruvbox"])

    print(f"{len(M.MODES)} modes, checking shapes at {len(SIZES)} sizes…")
    fails = check(palette)
    if fails:
        print(f"\n{len(fails)} FAILURES:")
        for f in fails:
            print("  ", f)
        raise SystemExit(1)
    print("all modes OK")

    over = bench(palette)
    print(f"\nframe budget at 60 fps: {BUDGET_MS:.1f} ms — {over} mode/size pairs over")
