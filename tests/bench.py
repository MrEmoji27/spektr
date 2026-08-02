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

import spektr.modes as M                      # noqa: E402
from spektr.analysis import N_BANDS           # noqa: E402
from spektr.modes import Ctx                  # noqa: E402
from spektr.palette import BUILTIN, Palette   # noqa: E402
from spektr.render import make_strips         # noqa: E402

SIZES = [(40, 10), (80, 24), (120, 16), (200, 50), (240, 60)]
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


def bench(palette, sizes=((120, 16), (200, 50), (240, 60)), n=60) -> int:
    over = 0
    for w, h in sizes:
        print(f"\n== {w}x{h} " + "=" * 46)
        print(f"{'mode':<10} {'build':>8} {'strips':>8} {'total':>8}   {'fps':>7}")
        for m in M.MODES:
            state: dict = {}
            for f in range(4):
                m.fn(make_ctx(w, h, f, state, f / 60, palette))

            t0 = time.perf_counter()
            for f in range(n):
                m.fn(make_ctx(w, h, f, state, f / 60, palette))
            build = (time.perf_counter() - t0) / n * 1000

            t0 = time.perf_counter()
            for f in range(n):
                out = m.fn(make_ctx(w, h, f, state, f / 60, palette))
                make_strips(out[0], out[1], palette, out[2] if len(out) == 3 else None)
            total = (time.perf_counter() - t0) / n * 1000

            flag = "  <-- over budget" if total > BUDGET_MS else ""
            if total > BUDGET_MS:
                over += 1
            print(
                f"{m.name:<10} {build:7.2f}ms {total - build:7.2f}ms "
                f"{total:7.2f}ms {1000 / max(total, 1e-6):7.0f}{flag}"
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
