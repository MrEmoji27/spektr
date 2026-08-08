"""Equivalence and speed harness for rewrites of :func:`spektr.render.make_strips`.

``make_strips`` is 80-90% of the frame budget for gradient modes -- Chladni
Extreme at 400x100 spends 10.21 ms here against 2.63 ms in the mode itself --
so it is worth optimising hard. Optimising it *safely* needs something that can
say "the new one draws exactly what the old one drew", for every mode, at every
size, under every theme. That is this file.

Run with ``python tests/strips_equiv.py``. Exit code 0 means equivalent.

WHY THERE IS A COPY OF THE OLD FUNCTION IN HERE
-----------------------------------------------
:func:`reference_make_strips` below is a frozen, verbatim copy of the
implementation as of commit b4e916c. A rewrite replaces the real one, and at
that moment the thing to compare against no longer exists anywhere in the tree.
Keeping the old behaviour pinned here is what makes the comparison possible at
all, and pinning it as *code* rather than as recorded output means the matrix
can grow later without regenerating fixtures.

It is dead weight the day the rewrite lands and stays correct, and it should be
deleted then. Until then it is the only definition of "right".

WHAT EQUIVALENCE MEANS HERE
---------------------------
Not "looks the same". Same number of strips, same ``cell_length``, same number
of segments per strip, and per segment the same text and the same style. A
rewrite that merges two adjacent runs the old one kept is *not* equivalent even
though the terminal output is identical -- it changes what Textual diffs, and
"identical, but with different segment boundaries" is a claim to make
deliberately, not by accident.
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from rich.segment import Segment
from textual.strip import Strip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, RAMP_STEPS, Palette  # noqa: E402
from spektr.render import make_strips  # noqa: E402

# One theme per distinct rle_tol. The tolerance is what the merge loop keys on,
# so a rewrite can easily be correct at tol=0 (where the merge never fires) and
# wrong everywhere else. Testing one theme would have hidden exactly that.
THEMES = ["classic", "gruvbox", "nord", "tokyo-night"]

SIZES = [(80, 24), (200, 50), (400, 100), (13, 3), (1, 1), (2, 40), (240, 1)]


# ── frozen reference: make_strips as of b4e916c ──────────────────────────────

def reference_make_strips(codes, cidx, palette, bidx=None):
    """DO NOT EDIT. Verbatim copy of the implementation being replaced."""
    h, w = codes.shape
    styles = palette.styles
    strips: list[Strip] = []
    tol = palette.rle_tol

    text_all = codes.astype("<u4", copy=False).tobytes().decode("utf-32-le", errors="replace")

    if bidx is None:
        for y in range(h):
            base = y * w
            idx = cidx[y]
            change = np.flatnonzero(idx[1:] != idx[:-1]) + 1
            if change.size == 0:
                strips.append(
                    Strip([Segment(text_all[base : base + w], styles[int(idx[0])])], w)
                )
                continue
            starts = [0, *change.tolist()]
            pick = idx[starts].tolist()
            if np.any(np.abs(np.diff(idx[starts])) <= tol):
                ms = [0]
                mv = [pick[0]]
                v0 = pick[0]
                for k in range(1, len(starts)):
                    v = pick[k]
                    if v > v0 + tol or v < v0 - tol:
                        ms.append(starts[k])
                        mv.append(v)
                        v0 = v
                segs = [
                    Segment(text_all[base + s : base + e], styles[c])
                    for s, e, c in zip(ms, [*ms[1:], w], mv)
                ]
            else:
                segs = [
                    Segment(text_all[base + s : base + e], styles[c])
                    for s, e, c in zip(starts, [*starts[1:], w], pick)
                ]
            strips.append(Strip(segs, w))
        return strips

    cache = palette.pair_styles
    pair_style = palette.pair_style
    for y in range(h):
        base = y * w
        fi = cidx[y]
        bi = bidx[y]
        change = np.flatnonzero((fi[1:] != fi[:-1]) | (bi[1:] != bi[:-1])) + 1
        starts = [0, *change.tolist()]
        fvals = fi[starts].tolist()
        bvals = bi[starts].tolist()
        if np.any(
            (np.abs(np.diff(fi[starts])) <= tol)
            & (np.abs(np.diff(bi[starts])) <= tol)
        ):
            ms = [0]
            mv = [fvals[0] * RAMP_STEPS + bvals[0]]
            f0, b0 = fvals[0], bvals[0]
            for k in range(1, len(starts)):
                f, b = fvals[k], bvals[k]
                if (
                    f > f0 + tol
                    or f < f0 - tol
                    or b > b0 + tol
                    or b < b0 - tol
                ):
                    ms.append(starts[k])
                    mv.append(f * RAMP_STEPS + b)
                    f0, b0 = f, b
            starts = ms
            keys = mv
        else:
            keys = (fi[starts].astype(np.int32) * RAMP_STEPS + bi[starts]).tolist()
        segs = []
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else w
            st = cache.get(keys[k])
            if st is None:
                st = pair_style(keys[k])
            segs.append(Segment(text_all[base + s : base + e], st))
        strips.append(Strip(segs, w))
    return strips


# ── comparison ───────────────────────────────────────────────────────────────

def differences(want: list[Strip], got: list[Strip], where: str) -> list[str]:
    """Every way the two can disagree, reported with enough detail to debug."""
    out: list[str] = []
    if len(want) != len(got):
        return [f"{where}: {len(got)} strips, expected {len(want)}"]
    for y, (a, b) in enumerate(zip(want, got)):
        if a.cell_length != b.cell_length:
            out.append(f"{where} row {y}: cell_length {b.cell_length} != {a.cell_length}")
        sa, sb = a._segments, b._segments
        if len(sa) != len(sb):
            out.append(
                f"{where} row {y}: {len(sb)} segments, expected {len(sa)}"
                f"  (texts {''.join(s.text for s in sb)[:24]!r})"
            )
            continue
        for i, (x, y2) in enumerate(zip(sa, sb)):
            if x.text != y2.text:
                out.append(f"{where} row {y} seg {i}: text {y2.text!r} != {x.text!r}")
                break
            if x.style != y2.style:
                out.append(f"{where} row {y} seg {i}: style {y2.style!r} != {x.style!r}")
                break
        if len(out) > 40:
            out.append(f"{where}: ... further differences suppressed")
            return out
    return out


def ctx_for(w, h, frame, state, t, palette):
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


# ── 1. every mode, every size, every tolerance ───────────────────────────────

def check_modes() -> list[str]:
    bad: list[str] = []
    for theme in THEMES:
        pal = Palette(BUILTIN[theme])
        for w, h in SIZES:
            for m in M.MODES:
                state: dict = {}
                for f in range(4):
                    out = m.fn(ctx_for(w, h, f, state, f / 60, pal))
                    codes, cidx = out[0], out[1]
                    bidx = out[2] if len(out) == 3 else None
                    want = reference_make_strips(codes, cidx, pal, bidx)
                    got = make_strips(codes, cidx, pal, bidx)
                    bad += differences(
                        want, got, f"{theme} {w}x{h} {m.name} f{f}"
                    )
                    if len(bad) > 60:
                        bad.append("... stopping, too many failures")
                        return bad
    return bad


# ── 2. adversarial patterns the modes may never produce ──────────────────────

def check_synthetic() -> list[str]:
    """Colour patterns chosen to sit exactly on the merge loop's edges.

    Real modes are a sample of the input space, not a cover of it. A rewrite
    can pass every mode and still be wrong on, say, a row whose indices step by
    exactly ``tol`` -- the boundary between merging and splitting, and the
    single easiest thing to get off by one.
    """
    bad: list[str] = []
    rng = np.random.default_rng(20260809)
    for theme in THEMES:
        pal = Palette(BUILTIN[theme])
        tol = pal.rle_tol
        for w, h in ((64, 8), (200, 4), (7, 5)):
            cases = {
                "uniform": np.zeros((h, w), np.int32),
                "alternating": (np.arange(w, dtype=np.int32) % 2)[None, :].repeat(h, 0),
                "ramp": (np.arange(w, dtype=np.int32) % RAMP_STEPS)[None, :].repeat(h, 0),
                # steps of exactly tol, tol+1 and tol-1: the merge boundary
                "step_tol": (np.arange(w, dtype=np.int32) * max(tol, 1) % RAMP_STEPS)[None, :].repeat(h, 0),
                "step_tol_plus": (np.arange(w, dtype=np.int32) * (tol + 1) % RAMP_STEPS)[None, :].repeat(h, 0),
                "random": rng.integers(0, RAMP_STEPS, (h, w), dtype=np.int32),
                "sparse": np.where(
                    rng.random((h, w)) < 0.05,
                    rng.integers(0, RAMP_STEPS, (h, w)),
                    0,
                ).astype(np.int32),
                # a slow drift: many tiny steps, which is where the merge earns
                # its keep and where an off-by-one accumulates visibly
                "drift": np.clip(
                    (np.arange(w) // max(1, w // RAMP_STEPS)), 0, RAMP_STEPS - 1
                ).astype(np.int32)[None, :].repeat(h, 0),
            }
            codes = np.full((h, w), ord("#"), np.int32)
            for name, cidx in cases.items():
                for bname, bidx in (
                    ("fg", None),
                    ("fg+bg", np.roll(cidx, 3, axis=1).copy()),
                    ("fg+bg-same", cidx.copy()),
                ):
                    where = f"{theme} {w}x{h} synth:{name}/{bname}"
                    want = reference_make_strips(codes, cidx, pal, bidx)
                    got = make_strips(codes, cidx, pal, bidx)
                    bad += differences(want, got, where)
                    if len(bad) > 60:
                        bad.append("... stopping, too many failures")
                        return bad
    return bad


# ── 3. the bidx path on every mode, not just the four that use it ────────────

def check_forced_bidx() -> list[str]:
    """Only 4 of 45 modes return a background index, so the two-colour path is
    barely exercised by :func:`check_modes`. Synthesising a background for the
    other 41 covers it against real codepoint and colour data."""
    bad: list[str] = []
    pal = Palette(BUILTIN["nord"])          # rle_tol 2, so the merge is live
    for w, h in ((120, 30), (37, 9)):
        for m in M.MODES:
            state: dict = {}
            out = m.fn(ctx_for(w, h, 3, state, 0.05, pal))
            codes, cidx = out[0], out[1]
            bidx = ((cidx + 7) % RAMP_STEPS).astype(cidx.dtype)
            want = reference_make_strips(codes, cidx, pal, bidx)
            got = make_strips(codes, cidx, pal, bidx)
            bad += differences(want, got, f"forced-bg {w}x{h} {m.name}")
            if len(bad) > 60:
                return bad
    return bad


# ── 4. speed ─────────────────────────────────────────────────────────────────

def speed(sizes=((200, 50), (400, 100)), probe=("Bars", "Plasma", "Maelstrom",
                                                "Chladni Extreme", "Murmuration")):
    """Interleaved A/B, because this machine drifts.

    Timing all of the reference then all of the new one compares two different
    machines: identical code here has measured 13.5 ms and 18.3 ms in
    consecutive runs. Alternating them puts any slow phase in both arms.
    """
    pal = Palette(BUILTIN["gruvbox"])
    print("\n" + "=" * 72)
    print("SPEED -- reference vs current, interleaved")
    print("=" * 72)
    total_a = total_b = 0.0
    for w, h in sizes:
        print(f"\n  {w}x{h}")
        print(f"    {'mode':<18}{'reference':>11}{'current':>10}{'speedup':>9}{'segs':>8}")
        for name in probe:
            m = next((x for x in M.MODES if x.name == name), None)
            if m is None:
                continue
            state: dict = {}
            frames = []
            for f in range(30):
                out = m.fn(ctx_for(w, h, f, state, f / 60, pal))
                frames.append((out[0], out[1], out[2] if len(out) == 3 else None))
            gc.collect()

            da, db = [], []
            for codes, cidx, bidx in frames:
                t0 = time.perf_counter_ns()
                reference_make_strips(codes, cidx, pal, bidx)
                t1 = time.perf_counter_ns()
                got = make_strips(codes, cidx, pal, bidx)
                t2 = time.perf_counter_ns()
                da.append(t1 - t0)
                db.append(t2 - t1)
            a = statistics.median(da) / 1e6
            b = statistics.median(db) / 1e6
            segs = sum(len(s._segments) for s in got)
            total_a += a
            total_b += b
            print(f"    {name:<18}{a:10.2f}ms{b:9.2f}ms{a / max(b, 1e-9):8.2f}x{segs:8d}")
    print(f"\n  overall  {total_a:.2f}ms -> {total_b:.2f}ms  "
          f"({total_a / max(total_b, 1e-9):.2f}x)")


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"{len(M.MODES)} modes x {len(SIZES)} sizes x {len(THEMES)} themes "
          f"(rle_tol {[Palette(BUILTIN[t]).rle_tol for t in THEMES]})")

    fails: list[str] = []
    for label, fn in (
        ("modes", check_modes),
        ("synthetic patterns", check_synthetic),
        ("forced background", check_forced_bidx),
    ):
        print(f"\nchecking {label} ...", end=" ", flush=True)
        bad = fn()
        print("OK" if not bad else f"{len(bad)} DIFFERENCES")
        fails += bad

    if fails:
        print("\nNOT EQUIVALENT:")
        for f in fails[:60]:
            print("  ", f)
        raise SystemExit(1)

    print("\nequivalent to the frozen reference everywhere tested")
    speed()
