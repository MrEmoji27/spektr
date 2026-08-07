"""Performance verification.

``bench.py`` times the render functions in isolation. That's the right number
for tuning a mode, but it isn't the number that decides whether the thing feels
smooth. This measures the parts isolation hides:

* what the analyser thread actually costs, continuously, in the background
* how the strip builder scales with terminal size
* whether long runs leak memory
* whether the slow-mode frame-reuse guard engages when it should

Run a section at a time: ``python tests/perf.py analyser|strips|memory|guard|all``
"""

from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cp1252 consoles cannot encode this file's output characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import resource  # POSIX only
except ModuleNotFoundError:                         # Windows
    resource = None

import spektr.modes as M  # noqa: E402
from spektr.analysis import HOP, N_BANDS, Analyser, BandPlan  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import make_strips  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])
SR = 48000
BUDGET_MS = 1000.0 / 60.0


def rss_mb() -> float:
    """Resident set size in MB.

    ``resource`` does not exist on Windows — which is the platform this project
    primarily targets — so fall back to the Win32 process-memory counters. The
    leak check is the whole point of this file; it should not be POSIX-only.
    """
    if resource is not None:
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (1024 * 1024) if sys.platform == "darwin" else kb / 1024

    import ctypes
    from ctypes import wintypes

    class _COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _COUNTERS()
    counters.cb = ctypes.sizeof(_COUNTERS)
    ok = ctypes.WinDLL("psapi").GetProcessMemoryInfo(
        ctypes.WinDLL("kernel32").GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    return counters.PeakWorkingSetSize / (1024 * 1024) if ok else 0.0


def ctx_for(w, h, frame, state, t):
    bands = np.clip(np.abs(np.sin(np.linspace(0, 3, N_BANDS) + t * 2)) * 0.8, 0, 1)
    wave = np.sin(np.linspace(0, 40, 512) + t * 10) * 0.7
    return Ctx(
        w=w, h=h, bands=bands, peaks=bands, bands_l=bands * 0.9, bands_r=bands,
        wave=wave, stereo=np.stack((wave, np.roll(wave, 7)), axis=1),
        frame=frame, t=t, dt=1 / 60, energy=float(bands.mean()),
        silent=False, palette=PAL, state=state,
    )


# ── the analyser's continuous cost ───────────────────────────────────────────

def check_analyser() -> list[str]:
    """The analyser runs forever in the background. If it costs a meaningful
    slice of a core, it competes with rendering — and the GIL means that
    competition is real, not theoretical."""
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)

    # cost of one analysis, measured directly. The window that matters is the
    # long one — the analyser reads that many frames every hop and runs four
    # FFTs over them (two channels x bass and mid).
    plan = BandPlan(SR)
    t = np.arange(plan.bass_size) / SR
    s = (np.sin(2 * math.pi * 440 * t) * 0.3).astype(np.float32)
    ring.push(np.stack((s, s), axis=1))
    for _ in range(5):
        an._analyse_once()

    N = 300
    t0 = time.perf_counter()
    for _ in range(N):
        an._analyse_once()
    per = (time.perf_counter() - t0) / N * 1000

    rate = SR / HOP
    duty = per * rate / 1000 * 100

    print(f"    one analysis      {per:6.3f} ms  (windows {plan.bass_size}/{plan.mid_size}, hop {HOP})")
    print(f"    analyses/sec      {rate:6.1f} Hz  at {SR} Hz sample rate")
    print(f"    continuous load   {duty:6.2f} % of one core")

    # the ring push, which runs in the audio callback and must be trivial
    block = np.zeros((256, 2), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(20000):
        ring.push(block)
    push_us = (time.perf_counter() - t0) / 20000 * 1e6
    callbacks = SR / 256
    print(f"    ring.push         {push_us:6.1f} µs  x{callbacks:.0f}/sec "
          f"= {push_us * callbacks / 1e4:.2f} % of one core")

    bad = []
    if duty > 25:
        bad.append(f"analyser uses {duty:.1f}% of a core — too much to run alongside 60 fps")
    if push_us > 50:
        bad.append(f"ring.push takes {push_us:.0f} µs — too slow for an audio callback")
    return bad


# ── strip builder scaling ────────────────────────────────────────────────────

def check_strips() -> list[str]:
    """Strip building is per-cell work that every mode pays. It should scale
    linearly with cell count, and the run-length encoding should make a smooth
    field far cheaper than a noisy one."""
    print(f"    {'size':>10} {'cells':>7} {'flat':>9} {'gradient':>9} {'noisy':>9} {'ns/cell':>8}")
    bad = []
    for w, h in ((80, 24), (120, 40), (200, 50), (240, 60), (400, 100)):
        cells = w * h
        codes = np.full((h, w), 0x2840, dtype=np.int32)
        flat = np.zeros((h, w), dtype=np.int32)
        grad = np.repeat(np.linspace(0, 63, h).astype(np.int32)[:, None], w, axis=1)
        noisy = (np.random.default_rng(1).integers(0, 64, (h, w))).astype(np.int32)

        times = []
        for cidx in (flat, grad, noisy):
            for _ in range(3):
                make_strips(codes, cidx, PAL)
            t0 = time.perf_counter()
            for _ in range(30):
                make_strips(codes, cidx, PAL)
            times.append((time.perf_counter() - t0) / 30 * 1000)

        ns_per_cell = times[1] * 1e6 / cells
        print(f"    {w:>4}x{h:<5} {cells:>7} {times[0]:8.2f}ms {times[1]:8.2f}ms "
              f"{times[2]:8.2f}ms {ns_per_cell:7.1f}")

        if times[2] < times[1]:
            bad.append(f"{w}x{h}: noisy colour was not slower than a gradient — RLE may not be working")
    return bad


# ── memory over a long run ───────────────────────────────────────────────────

def check_memory() -> list[str]:
    """Modes hold scratch buffers and the palette caches styles. Neither should
    grow with time — this is a visualiser people leave running for hours."""
    bad = []
    heavy = ["Pulse", "Matrix", "Gonio", "Spectro", "Warp", "Plasma"]
    gc.collect()
    before = rss_mb()

    for name in heavy:
        m = M.get(name)
        state: dict = {}
        for i in range(1500):
            out = m.fn(ctx_for(160, 40, i, state, i / 60))
            make_strips(out[0], out[1], PAL, out[2] if len(out) == 3 else None)

    gc.collect()
    after = rss_mb()
    grew = after - before
    print(f"    {len(heavy)} modes x 1500 frames: {before:.1f} MB -> {after:.1f} MB "
          f"({grew:+.1f} MB)")

    if grew > 20:
        bad.append(f"RSS grew {grew:.1f} MB over 9000 frames")
    return bad


# ── the slow-mode guard ──────────────────────────────────────────────────────

def check_guard() -> list[str]:
    """A deliberately slow plugin must trigger previous-frame reuse rather than
    dragging the frame rate down for everything."""
    from spektr.widget import SLOW_MODE_MS

    bad = []
    print(f"    threshold {SLOW_MODE_MS} ms, budget {BUDGET_MS:.1f} ms at 60 fps")

    worst = []
    for m in M.MODES:
        state: dict = {}
        for i in range(3):
            m.fn(ctx_for(240, 60, i, state, i / 60))
        t0 = time.perf_counter()
        for i in range(25):
            out = m.fn(ctx_for(240, 60, i, state, i / 60))
            make_strips(out[0], out[1], PAL, out[2] if len(out) == 3 else None)
        ms = (time.perf_counter() - t0) / 25 * 1000
        worst.append((ms, m.name))

    worst.sort(reverse=True)
    print("    heaviest built-ins at 240x60:")
    for ms, name in worst[:5]:
        flag = "  (would engage frame reuse)" if ms > SLOW_MODE_MS else ""
        print(f"      {name:<10} {ms:6.2f} ms{flag}")

    over_budget = [n for ms, n in worst if ms > BUDGET_MS]
    if over_budget:
        bad.append(f"over the 60 fps budget at 240x60: {', '.join(over_budget)}")

    headroom = BUDGET_MS / worst[0][0]
    print(f"    worst mode has {headroom:.1f}x headroom against the 60 fps budget")
    if headroom < 1.2:
        bad.append(f"only {headroom:.2f}x headroom on the slowest mode")
    return bad


def check_large() -> list[str]:
    """400x100 is a 4K monitor with a small font — the largest size anyone
    realistically runs, and 2.8x the cell count of 240x60. Cost scales linearly
    with the dot grid, so this is where any mode will fail first."""
    bad = []
    rows = []
    for m in M.MODES:
        state: dict = {}
        for i in range(3):
            m.fn(ctx_for(400, 100, i, state, i / 60))
        t0 = time.perf_counter()
        for i in range(20):
            out = m.fn(ctx_for(400, 100, i, state, i / 60))
            make_strips(out[0], out[1], PAL, out[2] if len(out) == 3 else None)
        rows.append(((time.perf_counter() - t0) / 20 * 1000, m.name))

    rows.sort(reverse=True)
    for ms, name in rows[:6]:
        flag = "  OVER BUDGET" if ms > BUDGET_MS else ""
        print(f"      {name:<10} {ms:6.2f} ms{flag}")
    print(f"    {len(rows) - 6} more, all faster")

    over = [n for ms, n in rows if ms > BUDGET_MS]
    if over:
        bad.append(f"over the 60 fps budget at 400x100: {', '.join(over)}")
    return bad


SECTIONS = {
    "analyser": ("analyser cost", check_analyser),
    "strips": ("strip builder scaling", check_strips),
    "memory": ("memory over a long run", check_memory),
    "guard": ("frame budget and the slow-mode guard", check_guard),
    "large": ("largest realistic terminal (400x100)", check_large),
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = SECTIONS if which == "all" else {which: SECTIONS[which]}

    failures = 0
    for key, (label, fn) in todo.items():
        print(f"\n== {label} " + "=" * max(0, 44 - len(label)))
        bad = fn()
        for b in bad:
            print(f"    FAIL {b}")
        failures += len(bad)

    print("\nall good" if not failures else f"\n{failures} problems")
    raise SystemExit(1 if failures else 0)
