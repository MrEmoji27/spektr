"""``Maelstrom`` — a real 2D incompressible fluid sim, stirred by the music.

Every field mode elsewhere in spektr (``Plasma``, ``Chladni``) is analytic: a
closed-form function of position and time, evaluated fresh every frame with
no memory of the frame before it. This is the opposite kind of thing — a
solver carrying real physical state (a velocity field and a dye field)
forward across frames, using Jos Stam's stable-fluids method: semi-Lagrangian
advection (unconditionally stable at any ``dt``, which matters at a variable
frame rate), vorticity confinement (puts back the small swirls advection's
numerical diffusion would otherwise smooth away), and Jacobi-iterated
pressure projection (the step that actually enforces incompressibility —
without it the flow has no internal pressure and never curls back on itself,
and it just reads as smoke drifting through nothing).

The whole thing is simulated on a small **fixed** internal grid, independent
of terminal size, then upsampled onto the real dot grid for display. A
20-ish-iteration Jacobi solve is trivial at 54x96 and would not be at
dot-grid resolution on a large terminal (up to 800x400). This is the same
lesson ``RAMP_STEPS``'s RLE tension and ``render.frac`` vs ``np.mod`` already
taught this codebase, generalised: simulate at the resolution the maths
needs, render at the resolution the eye needs, and don't confuse the two.
"""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_max, pack_braille
from . import Ctx, empty, mode

_SIM_H, _SIM_W = 54, 96


def _neighbours(x: np.ndarray):
    """Edge-clamped up/down/left/right neighbour grids — a solid-walled box,
    not a periodic one, so nothing teleports across the screen."""
    p = np.pad(x, 1, mode="edge")
    return p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]


def _sample_bilinear(field: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    h, w = field.shape
    y = np.clip(y, 0.0, h - 1.001)
    x = np.clip(x, 0.0, w - 1.001)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1, x1 = y0 + 1, x0 + 1
    fy, fx = y - y0, x - x0
    return (
        field[y0, x0] * (1 - fy) * (1 - fx)
        + field[y0, x1] * (1 - fy) * fx
        + field[y1, x0] * fy * (1 - fx)
        + field[y1, x1] * fy * fx
    )


def _advect(field: np.ndarray, vy: np.ndarray, vx: np.ndarray, dt: float) -> np.ndarray:
    """Semi-Lagrangian: trace each cell backward through the flow one ``dt``
    and sample there, rather than push values forward (which is what makes
    an explicit advection scheme blow up at anything but a tiny time step)."""
    h, w = field.shape
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    return _sample_bilinear(field, rows - vy * dt, cols - vx * dt)


def _jacobi_diffuse(field: np.ndarray, diff: float, dt: float, iters: int) -> np.ndarray:
    alpha = diff * dt
    beta = 1.0 + 4.0 * alpha
    x = field.copy()
    for _ in range(iters):
        up, down, left, right = _neighbours(x)
        x = (field + alpha * (up + down + left + right)) / beta
    return x


def _curl(vy: np.ndarray, vx: np.ndarray) -> np.ndarray:
    _, _, vy_l, vy_r = _neighbours(vy)
    vx_u, vx_d, _, _ = _neighbours(vx)
    return (vy_r - vy_l) * 0.5 - (vx_d - vx_u) * 0.5


def _apply_vorticity(vy: np.ndarray, vx: np.ndarray, dt: float, strength: float):
    """Pushes each cell along the gradient of |curl|, scaled by the curl
    itself — the standard stable-fluids trick for putting small swirls back
    in that the numerical diffusion of a coarse grid would otherwise erase.
    """
    w = _curl(vy, vx)
    up, down, left, right = _neighbours(np.abs(w))
    gx = (right - left) * 0.5
    gy = (down - up) * 0.5
    norm = np.sqrt(gx * gx + gy * gy) + 1e-5
    nx, ny = gx / norm, gy / norm
    return vy + strength * (-nx * w) * dt, vx + strength * (ny * w) * dt


def _project(vy: np.ndarray, vx: np.ndarray, iters: int):
    """Solve a Poisson equation for pressure (Jacobi relaxation) and subtract
    its gradient from velocity. This is what enforces incompressibility —
    the field has no sources or sinks except where forcing just put them —
    and it's the step that makes flow curl back on itself instead of just
    drifting outward forever."""
    _, _, vx_l, vx_r = _neighbours(vx)
    vy_u, vy_d, _, _ = _neighbours(vy)
    div = (vx_r - vx_l) * 0.5 + (vy_d - vy_u) * 0.5

    pressure = np.zeros_like(vy)
    for _ in range(iters):
        up, down, left, right = _neighbours(pressure)
        pressure = (up + down + left + right - div) * 0.25

    up, down, left, right = _neighbours(pressure)
    grad_y = (down - up) * 0.5
    grad_x = (right - left) * 0.5
    return vy - grad_y, vx - grad_x


def _state() -> dict:
    return {
        "vy": np.zeros((_SIM_H, _SIM_W), dtype=np.float64),
        "vx": np.zeros((_SIM_H, _SIM_W), dtype=np.float64),
        "dye": np.zeros((_SIM_H, _SIM_W), dtype=np.float64),
        "rng": np.random.default_rng(131),
    }


@mode("Maelstrom", group="fields", blurb="a real fluid sim, stirred by the music")
def maelstrom(ctx: Ctx):
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 16 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("maelstrom", _state)
    vy, vx, dye = st["vy"], st["vx"], st["dye"]
    rng = st["rng"]
    dt = min(max(ctx.dt, 0.0), 1.0 / 20.0)   # capped so a stall can't blow the sim up

    bass = ctx.range(0.0, 0.15)
    treble = ctx.range(0.6, 1.0)

    # A real onset drops a radial burst into the sim. The old fast/slow
    # envelope pair over the bass band fired on any rise in level — a swell
    # stirred the pot exactly as hard as a drum hit — where the analyser's
    # spectral-flux detector fires on transients wherever they sit in the
    # band plan, so a snare bursts as well as a kick.
    hit = bool(ctx.onsets)

    # ── 1. force / emit — every input is a forcing term, nothing is drawn
    # directly. Bass is a hose straight up the middle; the spread spectrum both
    # nudges horizontal velocity along the bottom edge and injects dye there,
    # so the emitter *is* the spectrum and the flow carries its print upward;
    # a hit drops a radial impulse burst.
    #
    # Emission is per *second*, not per frame. These were plain per-frame
    # additions, which is a rate in disguise: the same music injects 2.4x the
    # force and dye at 144 fps that it does at 60, so the sim looks like a
    # different fluid on a different display. Scaling by ``dt * 60`` keeps the
    # tuned appearance at 60 fps exactly as it was and makes every other rate
    # match it.
    emit = dt * 60.0
    r = 3
    hx = _SIM_W // 2
    y0, y1 = _SIM_H - 2 - r, _SIM_H - 1
    vy[y0:y1, hx - r : hx + r] -= (2.0 + bass * 14.0) * emit
    dye[y0:y1, hx - r : hx + r] = np.clip(
        dye[y0:y1, hx - r : hx + r] + bass * 0.9 * emit, 0.0, 1.0
    )

    spread_bands = ctx.display_bands(_SIM_W)
    vx[_SIM_H - 2, :] += (spread_bands - spread_bands.mean()) * 6.0 * emit
    # A dye source shaped like the spectrum. Without it the only smoke in the
    # sim came from the centre hose and the hit bursts, so what the individual
    # bands did was visible in the *motion* and never in the material being
    # moved — the fluid was reacting to a spectrum the picture never showed.
    dye[_SIM_H - 2, :] = np.clip(dye[_SIM_H - 2, :] + spread_bands * 0.55 * emit, 0.0, 1.0)

    if hit:
        cy = rng.uniform(_SIM_H * 0.3, _SIM_H * 0.75)
        cx = rng.uniform(_SIM_W * 0.2, _SIM_W * 0.8)
        yy, xx = np.ogrid[:_SIM_H, :_SIM_W]
        burst = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 18.0)
        ang = rng.uniform(0.0, 2 * math.pi)
        vy = vy + math.sin(ang) * burst * 40.0
        vx = vx + math.cos(ang) * burst * 40.0
        dye = np.clip(dye + burst * 0.8, 0.0, 1.0)

    # ── 2. advect — both velocity (self-advection) and dye, through the
    # pre-advection field
    old_vy, old_vx = vy, vx
    vy = _advect(vy, old_vy, old_vx, dt)
    vx = _advect(vx, old_vy, old_vx, dt)
    dye = _advect(dye, old_vy, old_vx, dt)

    # ── 3. vorticity confinement — treble turns up the swirl ──
    vy, vx = _apply_vorticity(vy, vx, dt, strength=0.6 + treble * 2.6)

    # ── 4. diffuse — light viscosity, a little more for the dye so it
    # spreads like smoke rather than staying a hairline
    vy = _jacobi_diffuse(vy, diff=0.02, dt=dt, iters=3)
    vx = _jacobi_diffuse(vx, diff=0.02, dt=dt, iters=3)
    dye = _jacobi_diffuse(dye, diff=0.05, dt=dt, iters=2)

    # ── 5. project — enforce incompressibility ──
    vy, vx = _project(vy, vx, iters=16)

    vy *= 0.995
    vx *= 0.995
    dye *= 0.997
    np.clip(dye, 0.0, 1.0, out=dye)

    st["vy"], st["vx"], st["dye"] = vy, vx, dye

    up = ctx.scratch(
        "maelstrom_upsample",
        lambda: (
            np.clip((np.arange(dr) * _SIM_H) // max(dr, 1), 0, _SIM_H - 1),
            np.clip((np.arange(dc) * _SIM_W) // max(dc, 1), 0, _SIM_W - 1),
        ),
    )
    ys, xs = up
    field = dye[ys[:, None], xs[None, :]]

    dots = field > 0.06
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx
