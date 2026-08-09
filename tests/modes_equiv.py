"""Equivalence harness for the Flame/Pulse/Arcs speedup.

Pins the current (pre-optimisation) implementations of the three modes as
verbatim reference copies, runs both the reference and the live (optimised)
versions over an identical Ctx sequence at several sizes, and asserts the
produced (codes, cidx) arrays are bit-identical.

Run with: python tests/modes_equiv.py   (exit 0 == equivalent)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import cell_max, noise, pack_braille  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])
SIZES = [(40, 10), (80, 24), (200, 50), (400, 100), (13, 3), (120, 16)]

# ── frozen reference copies (verbatim from the pre-optimisation tree) ────────

def _polar_ref(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols

    def build():
        cx, cy = dc / 2.0, dr / 2.0
        x_scale = cy / max(cx, 1.0)
        xs = (np.arange(dc, dtype=np.float32) - cx) * x_scale
        ys = np.arange(dr, dtype=np.float32) - cy
        dx = xs[None, :]
        dy = ys[:, None]
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        ang = np.arctan2(dy + np.zeros_like(dx), dx + np.zeros_like(dy))
        ang = np.where(ang < 0, ang + 2 * math.pi, ang).astype(np.float32)
        turn = (ang / np.float32(2 * math.pi)).astype(np.float32)
        return dist, turn, max(1.0, cy - 1.0)

    return ctx.scratch("polar", build)


def _angular_bands_ref(ctx: Ctx, turn: np.ndarray, n: int, spin: float) -> np.ndarray:
    steps = 512
    bands = ctx.display_bands(n).astype(np.float32)
    pos = np.linspace(0.0, n, steps, endpoint=False, dtype=np.float32)
    bi = pos.astype(np.int32) % n
    frac = pos - np.floor(pos)
    tm = (1.0 - np.cos(frac * np.float32(math.pi))) * np.float32(0.5)
    lut = bands[bi] * (1.0 - tm) + bands[(bi + 1) % n] * tm

    offset = np.float32(float(spin) % 1.0)
    idx = ((turn + offset) * np.float32(steps)).astype(np.int32) & (steps - 1)
    return lut[idx]


def flame_ref(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    n = ctx.n_display
    band_of = np.minimum((np.arange(dc) * n) // dc, n - 1)
    lvl = ctx.display_bands(n)[band_of][None, :].astype(np.float32)

    y = ((dr - 1 - np.arange(dr)) / max(1, dr - 1)).astype(np.float32)[:, None]
    alive = y <= lvl

    seg = max(1.0, dc / n)
    centre = (band_of * seg + seg / 2.0).astype(np.float32)[None, :]

    a = np.float32(ctx.t * 9.0) + y * np.float32(6.0)
    b = (band_of * np.float32(2.1))[None, :]
    wobble = (np.sin(a) * np.cos(b) + np.cos(a) * np.sin(b)) * np.float32(1.5)

    tip = np.float32(1.0) - y / np.maximum(lvl, np.float32(0.01))
    fw = (np.float32(0.3) + np.float32(0.7) * tip) * np.float32(seg / 2.0)

    edge = np.abs(np.arange(dc, dtype=np.float32)[None, :] - centre + np.float32(0.5) - wobble)
    edge /= np.maximum(fw, np.float32(1e-6))

    dots = alive & (edge < 1.0) & ((edge < 0.7) | (noise((dr, dc), ctx.frame + 31) < 0.6))

    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(np.where(dots, tip, np.float32(0.0))))
    return codes, cidx


_PULSE_WAVES_REF = 2


def pulse_ref(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    dist, turn, max_r = _polar_ref(ctx)
    n = min(16, ctx.n_display)

    avg = ctx.energy
    nrg = _angular_bands_ref(ctx, turn, n, ctx.t * (0.10 + avg * 0.30))

    st = ctx.scratch(
        "pulse",
        lambda: {
            "born": np.full(_PULSE_WAVES_REF, -99.0),
            "amp": np.zeros(_PULSE_WAVES_REF),
            "acc": 0.0,
        },
    )
    st["acc"] += (0.8 + ctx.energy * 5.0) * ctx.dt
    due = st["acc"] >= 1.0 or st["born"].max() < 0.0
    if due:
        st["acc"] = max(0.0, st["acc"] - 1.0)
    if (ctx.onsets or due) and (ctx.t - st["born"].max()) > 0.12:
        slot = int(np.argmin(st["born"]))
        st["born"][slot] = ctx.t
        strength = ctx.onset_strength if ctx.onsets else min(1.0, ctx.energy * 1.4)
        st["amp"][slot] = float(np.clip(0.35 + strength * 0.9, 0.0, 1.0))

    r = max_r * (0.1 + 0.9 * nrg * nrg)
    nz = ctx.scratch(
        "pulse_grain", lambda: np.random.default_rng(419).random((dr, dc)).astype(np.float32)
    )

    core = dist < 1.0
    inside = (dist <= r) & (r >= 1.0)
    prox = np.where(inside, dist / np.maximum(r, 1e-6), 0.0)
    lit = inside & ((prox > 0.45) | (nz < 0.3 + prox * 0.7))

    hv = nrg * np.clip(1.0 - (dist - r) * 0.25, 0.0, 1.0) * 0.4
    lit |= (~inside) & (nz < hv)

    for k in range(_PULSE_WAVES_REF):
        age = ctx.t - st["born"][k]
        if not (0.0 <= age < 0.9):
            continue
        phase = age / 0.9
        strength = float(st["amp"][k]) * (1.0 - phase)
        if strength <= 0.06:
            continue
        band = 1.0 + strength * 3.0
        edge = np.abs(dist - max_r * phase)
        near = edge < band
        lit |= near & (nz < (1.0 - edge / band) * strength)

    lit |= core
    codes = pack_braille(lit)
    heat = np.where(lit, prox, 0.0)
    heat[core] = 0.2
    cidx = ctx.ramp(cell_max(heat))
    return codes, cidx


def arcs_ref(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return np.full((ctx.h, ctx.w), 32, dtype=np.int32), np.zeros((ctx.h, ctx.w), dtype=np.int32)

    dist, turn, max_r = _polar_ref(ctx)
    n = int(min(12, max(4, ctx.n_display // 2)))
    lv = ctx.display_bands(n).astype(np.float32)

    st = ctx.scratch("arcs_peak", lambda: np.zeros(n, dtype=np.float32))
    if st.shape[0] != n:
        st = np.zeros(n, dtype=np.float32)
        ctx.state[("arcs_peak", ctx.w, ctx.h)] = st
    np.maximum(st - np.float32(1.5 * ctx.dt), lv, out=st)

    steps = 512
    u = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float32)
    lut = np.zeros(steps, dtype=np.float32)
    for j in range(n):
        level = float(st[j])
        breath = 0.012 * math.sin(ctx.t * 1.7 + j * 0.8)
        r = 0.12 + 0.84 * level + breath
        width = 0.010 + 0.028 * level
        near = np.abs(u - r)
        np.maximum(
            lut,
            np.where(near < width, (1.0 - near / width) * (0.30 + 0.70 * level), 0.0),
            out=lut,
        )

    idx = np.clip((dist * np.float32(steps / max_r)).astype(np.int32), 0, steps - 1)
    heat = lut[idx]

    nrg = _angular_bands_ref(ctx, turn, min(12, n), ctx.t * 0.05)
    heat *= 0.55 + 0.45 * nrg

    lit = heat > 0.06
    codes = pack_braille(lit)
    cidx = ctx.ramp(cell_max(np.where(lit, heat, 0.0)))
    return codes, cidx


# ── harness ──────────────────────────────────────────────────────────────────

def ctx_for(w, h, frame, state, t, dt=1 / 60, bands=None, onsets=0, onset_strength=0.0):
    if bands is None:
        bands = np.clip(np.abs(np.sin(np.linspace(0, 3, N_BANDS) + t * 2)) * 0.8, 0.0, 1.0)
    wave = np.sin(np.linspace(0, 40, 512) + t * 10) * 0.7
    stereo = np.stack((wave, np.roll(wave, 7)), axis=1)
    return Ctx(
        w=w, h=h, bands=bands, peaks=np.clip(bands * 1.05, 0, 1),
        bands_l=bands * 0.9, bands_r=bands, wave=wave, stereo=stereo,
        frame=frame, t=t, dt=dt, energy=float(bands.mean()),
        silent=False, palette=PAL, state=state, onsets=onsets,
        onset_strength=onset_strength,
    )


def check_mode(name: str, ref, frames=40) -> list[str]:
    """Run ref and the live mode over the same Ctx sequence; compare output."""
    bad: list[str] = []
    for w, h in SIZES:
        # force resizes mid-sequence so scratch caches are rebuilt in both arms
        state_a: dict = {}
        state_b: dict = {}
        t = 0.0
        for f in range(frames):
            if f % 13 == 12:
                w, h = (80, 24) if (w, h) == (400, 100) else ((400, 100) if (w, h) == (80, 24) else (w, h))
            # sprinkle onsets so the shockwave path is exercised in both arms
            onsets = 1 if f % 17 == 0 else 0
            strength = 0.7 if onsets else 0.0
            ctx = ctx_for(w, h, f, None, t, onsets=onsets, onset_strength=strength)
            ctx_a = Ctx(w=ctx.w, h=ctx.h, bands=ctx.bands, peaks=ctx.peaks,
                        bands_l=ctx.bands_l, bands_r=ctx.bands_r, wave=ctx.wave,
                        stereo=ctx.stereo, frame=ctx.frame, t=ctx.t, dt=ctx.dt,
                        energy=ctx.energy, silent=ctx.silent, palette=PAL,
                        state=state_a, onsets=ctx.onsets,
                        onset_strength=ctx.onset_strength)
            ctx_b = Ctx(w=ctx.w, h=ctx.h, bands=ctx.bands, peaks=ctx.peaks,
                        bands_l=ctx.bands_l, bands_r=ctx.bands_r, wave=ctx.wave,
                        stereo=ctx.stereo, frame=ctx.frame, t=ctx.t, dt=ctx.dt,
                        energy=ctx.energy, silent=ctx.silent, palette=PAL,
                        state=state_b, onsets=ctx.onsets,
                        onset_strength=ctx.onset_strength)
            want = ref(ctx_a)
            got = M.get(name).fn(ctx_b)
            for arr, label in ((want[0], "codes"), (want[1], "cidx")):
                if not np.array_equal(arr, got[0] if label == "codes" else got[1]):
                    bad.append(
                        f"{name} {w}x{h} f{f} t={t:.3f}: {label} differs "
                        f"({np.count_nonzero(arr != got[0] if label == 'codes' else arr != got[1])} cells)"
                    )
                    break
            t += 1 / 60
            if len(bad) > 10:
                return bad
    return bad


if __name__ == "__main__":
    bad: list[str] = []
    for name, ref in (("Flame", flame_ref), ("Pulse", pulse_ref), ("Arcs", arcs_ref)):
        print(f"checking {name} ...", end=" ", flush=True)
        b = check_mode(name, ref)
        print("OK" if not b else f"{len(b)} DIFFERENCES")
        bad += b

    if bad:
        print("\nNOT EQUIVALENT:")
        for b in bad[:20]:
            print("  ", b)
        raise SystemExit(1)
    print("\nall three modes equivalent to their frozen references")
