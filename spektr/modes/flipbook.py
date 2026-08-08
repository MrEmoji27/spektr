"""``Flipbook`` — plays back a real ASCII animation, not a synthesised one.

Every other mode in spektr draws something computed fresh each frame from the
spectrum. This one draws frames that came from a file: a folder of numbered
``.txt`` frames the user drops into ``<config>/ascii/`` (see
``spektr/asciiart.py`` for discovery and decoding), advanced at a rate the
music sets. Rendered at text-cell resolution — braille sub-cells would
destroy arbitrary glyphs, so this is the one mode in the file group that
can't use the dot grid.

Three switchable effects, chosen in Settings:

* ``lit``    — geometry static, the spectrum lights it in horizontal zones
* ``warp``   — a coherent displacement field samples nearby art, so it breathes
* ``dissolve`` — cells detach in quiet passages and snap home on a hit
"""

from __future__ import annotations

import math

import numpy as np

from .. import asciiart
from ..render import SPACE, noise
from . import Ctx, empty, mode


def _fit(ah: int, aw: int, sh: int, sw: int):
    """Screen cell -> art source cell, plus which screen cells show anything.

    Centred and clipped when the art fits inside the screen; otherwise a
    nearest-neighbour map that stretches the art to fill it. Either way this
    returns full ``(sh, sw)`` index arrays so callers never branch on which
    case they're in.
    """
    if ah <= sh and aw <= sw:
        y0, x0 = (sh - ah) // 2, (sw - aw) // 2
        sy = np.arange(sh)[:, None] - y0
        sx = np.arange(sw)[None, :] - x0
        mask = (sy >= 0) & (sy < ah) & (sx >= 0) & (sx < aw)
        sy = np.broadcast_to(np.clip(sy, 0, ah - 1), (sh, sw))
        sx = np.broadcast_to(np.clip(sx, 0, aw - 1), (sh, sw))
        return sy.copy(), sx.copy(), mask

    sy = np.clip((np.arange(sh) * ah) // max(sh, 1), 0, ah - 1)[:, None]
    sx = np.clip((np.arange(sw) * aw) // max(sw, 1), 0, aw - 1)[None, :]
    sy = np.broadcast_to(sy, (sh, sw)).copy()
    sx = np.broadcast_to(sx, (sh, sw)).copy()
    return sy, sx, np.ones((sh, sw), dtype=bool)


def _onset(ctx: Ctx) -> tuple[bool, float]:
    """Fast/slow envelope over the low end — a rising edge is a hit.

    ``ctx.energy`` is spring-smoothed (see widget.py) and would smear a hit
    across several frames if thresholded directly, the same reasoning
    ``Fireworks`` uses its own copy of in particles.py. Each
    caller rolling this itself rather than sharing one implementation across
    files is deliberate — it's four lines, and the alternative is a new
    render.py primitive for something this small.
    """
    st = ctx.scratch("flipbook_onset", lambda: {"fast": 0.0, "slow": 0.0})
    bass = ctx.range(0.0, 0.15)
    st["fast"] += (bass - st["fast"]) * min(1.0, ctx.dt / 0.03)
    st["slow"] += (bass - st["slow"]) * min(1.0, ctx.dt / 0.4)
    return (st["fast"] - st["slow"]) > 0.12, bass


def _dissolve_state(sh: int, sw: int) -> dict:
    rng = np.random.default_rng(97)
    ang = rng.uniform(0.0, 2 * math.pi, (sh, sw)).astype(np.float32)
    return {
        "alive": np.ones((sh, sw), dtype=np.float32),
        "oy": np.zeros((sh, sw), dtype=np.float32),
        "ox": np.zeros((sh, sw), dtype=np.float32),
        "fragility": rng.uniform(0.5, 1.5, (sh, sw)).astype(np.float32),
        "dir_y": np.sin(ang),
        "dir_x": np.cos(ang),
    }


@mode("Flipbook", group="scenes", blurb="your own ASCII animation, played by the music")
def flipbook(ctx: Ctx):
    w, h = ctx.w, ctx.h
    if w < 16 or h < 6:
        return empty(w, h)

    reel = asciiart.current()
    if reel is None:
        return empty(w, h)

    codes_all, density_all = reel.frames()
    n_frames, ah, aw = codes_all.shape

    hit, _bass = _onset(ctx)

    pos_st = ctx.scratch("flipbook_pos", lambda: {"pos": 0.0})
    pos_st["pos"] += ctx.dt * 12.0 * (0.25 + ctx.energy * 2.5)
    if hit:
        pos_st["pos"] += 2.5
    pos_st["pos"] %= max(n_frames, 1)
    frame_idx = int(pos_st["pos"]) % max(n_frames, 1)

    fit_st = ctx.scratch("flipbook_fit", lambda: {"key": None})
    key = (reel.name, ah, aw)
    if fit_st["key"] != key:
        sy, sx, mask = _fit(ah, aw, h, w)
        fit_st["key"], fit_st["sy"], fit_st["sx"], fit_st["mask"] = key, sy, sx, mask
    sy, sx, mask = fit_st["sy"], fit_st["sx"], fit_st["mask"]

    frame_codes = codes_all[frame_idx]
    frame_density = density_all[frame_idx]
    src_codes = np.where(mask, frame_codes[sy, sx], SPACE)
    src_density = np.where(mask, frame_density[sy, sx], 0.0).astype(np.float32)

    fx = asciiart.current_fx()

    if fx == "lit":
        row_energy = ctx.display_bands(h)[::-1]   # treble at the top, bass at the bottom
        codes = src_codes
        density = src_density * (0.35 + 0.9 * row_energy)[:, None]

    elif fx == "warp":
        bass = ctx.range(0.0, 0.2)
        treble = ctx.range(0.6, 1.0)
        rows = np.arange(h, dtype=np.float64)[:, None]
        cols = np.arange(w, dtype=np.float64)[None, :]
        amp = 1.2 + bass * 3.5
        dy = np.sin(cols * 0.15 + ctx.t * 1.6) * amp
        dx = np.sin(rows * 0.22 - ctx.t * 1.3) * amp * 0.6
        jitter = (noise((h, w), ctx.frame).astype(np.float64) - 0.5) * treble * 2.2
        dy = dy + jitter
        dx = dx + jitter * 0.6

        gy = np.clip(np.rint(sy.astype(np.float64) + dy).astype(np.int32), 0, ah - 1)
        gx = np.clip(np.rint(sx.astype(np.float64) + dx).astype(np.int32), 0, aw - 1)
        warped_codes = np.where(mask, frame_codes[gy, gx], SPACE)
        warped_density = np.where(mask, frame_density[gy, gx], 0.0)
        codes = warped_codes
        density = warped_density.astype(np.float32)

    else:  # "dissolve"
        st = ctx.scratch("flipbook_dissolve", lambda: _dissolve_state(h, w))

        decay = 0.15 + (1.0 - ctx.energy) * 0.6
        st["alive"] -= decay * st["fragility"] * ctx.dt
        if hit:
            st["alive"][:] = 1.0
            st["oy"][:] = 0.0
            st["ox"][:] = 0.0
        np.clip(st["alive"], 0.0, 1.0, out=st["alive"])

        gone = 1.0 - st["alive"]
        speed = gone * gone * 14.0
        st["oy"] += st["dir_y"] * speed * ctx.dt
        st["ox"] += st["dir_x"] * speed * ctx.dt

        rows = np.arange(h)[:, None]
        cols = np.arange(w)[None, :]
        dest_y = np.clip(np.rint(rows + st["oy"]).astype(np.int32), 0, h - 1)
        dest_x = np.clip(np.rint(cols + st["ox"]).astype(np.int32), 0, w - 1)

        codes = np.full((h, w), SPACE, dtype=np.int32)
        density = np.zeros((h, w), dtype=np.float32)
        visible = mask & (src_density > 0.0) & (st["alive"] > 0.02)
        ys, xs = dest_y[visible], dest_x[visible]
        # overlapping destinations are last-write-wins, which reads fine —
        # a genuine physical resolve isn't worth the cost here
        codes[ys, xs] = src_codes[visible]
        density[ys, xs] = src_density[visible] * st["alive"][visible]

    cidx = ctx.ramp(np.clip(density, 0.0, 1.0))
    return codes.astype(np.int32), cidx
