"""Cosmology modes — the sky, and things crossing it.

The first of a family, and they share a *scale* rather than a look: the
picture is mostly empty and mostly still, and the music arrives as events in
it rather than as a level being redrawn. That is a different bargain from the
rest of the app, where something is moving in every cell of every frame, and
it is why these get their own file. A mode that is 98% dark has to earn its
reactivity from timing instead of from area, so the onset detector is the mode
here rather than a garnish on it.

Everything draws into the braille dot grid. A terminal cell is twice as tall
as it is wide and braille puts four dot rows and two dot columns in it, so a
dot is square: a streak at 45 degrees is 45 degrees on screen, and none of the
geometry below wants an aspect correction. Modes elsewhere in this codebase
have halved a vertical velocity believing otherwise and been wrong twice over
— see the note in ``particles.fireworks``.
"""

from __future__ import annotations

import math

import numpy as np

from ..render import cell_max, pack_braille
from . import Ctx, empty, mode

#: Fixed stars per dot cell. Sparse on purpose: a sky is mostly nothing, and
#: past a certain density the eye stops reading stars and starts reading haze.
_STAR_DENSITY = 1.0 / 260.0

#: Meteors in flight at once. A shower is not a barrage — more than a handful
#: on screen stops reading as "something rare just happened", which is the
#: only thing a shooting star has to say.
_METEOR_CAP = 20


def _sky(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(19)
    n = int(np.clip(dr * dc * _STAR_DENSITY, 40, 2200))
    return {
        # Fixed stars. The positions never change: a sky that reshuffles
        # itself is a sky nobody believes, and ``Dune``'s grain texture
        # shimmering every frame already made that lesson expensive here.
        "sy": rng.integers(0, dr, n),
        "sx": rng.integers(0, dc, n),
        # Magnitude, skewed dim. Squaring a uniform draw gives many faint
        # stars and a few bright ones, which is roughly the real distribution
        # and — more to the point here — spreads the picture across the ramp
        # instead of bunching it into two or three steps.
        "mag": rng.uniform(0.10, 1.0, n) ** 2.2,
        "twf": rng.uniform(0.25, 1.1, n),
        "twp": rng.uniform(0.0, 2 * math.pi, n),

        # Meteors. y < 0 is the free-slot sentinel, the same convention Rain,
        # Snow, Bubbles, Fireworks and Ember all use.
        "my": np.full(_METEOR_CAP, -1.0),
        "mx": np.zeros(_METEOR_CAP),
        "mvy": np.zeros(_METEOR_CAP),
        "mvx": np.zeros(_METEOR_CAP),
        "mlen": np.zeros(_METEOR_CAP),
        "mbright": np.zeros(_METEOR_CAP),
        "mage": np.zeros(_METEOR_CAP),
        "mlife": np.ones(_METEOR_CAP),

        # The radiant: the point on the sky a shower appears to diverge from.
        # Real meteors travel parallel and only look otherwise, which is the
        # whole reason a shower has one. It drifts, because the sky turns.
        "rad": float(rng.uniform(0, 2 * math.pi)),
        "acc": 0.0,
        "rng": rng,
    }


@mode("Shooting Star", group="cosmos",
      blurb="a night sky, with meteors thrown from a drifting radiant on the beat")
def shooting_star(ctx: Ctx):
    """A sky that is mostly empty, and a beat that puts something across it.

    Every other mode answers to the current level: louder is taller, brighter,
    faster. This one answers to *events*. The stars do almost nothing — they
    twinkle, and the whole field lifts a little when the track is loud — and
    the music's job is to throw meteors, which is a thing that either happened
    or did not.

    So the base rate is deliberately low. There is one, because a silent
    passage should not be a still image, but it is slow enough that the eye
    goes on reading a streak as a beat rather than as weather. A mode built on
    events is ruined by a steady supply of them.

    A hard onset throws a brighter, longer, faster meteor from further out.
    That is ``ctx.onset_strength`` doing the work rather than ``ctx.energy``,
    because a fireball should answer to how sharp the hit was and not to how
    loud the bed underneath it is — the two come apart exactly where it
    matters, on a quiet track with a crisp snare.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("shooting_star", lambda: _sky(dr, dc))
    rng = st["rng"]
    field = np.zeros((dr, dc), dtype=np.float64)

    # ── the fixed stars ──
    # Twinkle is atmospheric, so it is slow and shallow and never switches a
    # star off. The loud-passage lift is a separate whole-sky term, so the
    # music is visible even in the stretches when nothing is crossing.
    tw = 0.80 + 0.20 * np.sin(ctx.t * st["twf"] + st["twp"])
    lift = 0.55 + 0.45 * min(1.0, ctx.energy * 1.8)
    field[st["sy"], st["sx"]] = np.clip(st["mag"] * tw * lift, 0.0, 1.0)

    # ── the radiant, drifting ──
    st["rad"] = (st["rad"] + ctx.dt * 0.035) % (2 * math.pi)
    # Kept well off-centre: a radiant in the middle of the screen throws
    # meteors symmetrically in every direction, which reads as an explosion
    # rather than as a shower.
    rx = dc * (0.5 + 0.42 * math.cos(st["rad"]))
    ry = dr * (0.5 + 0.42 * math.sin(st["rad"] * 0.7))

    # ── spawning ──
    st["acc"] += (0.22 + ctx.energy * 0.5) * ctx.dt
    want = int(st["acc"])
    if want:
        st["acc"] -= want
    if ctx.onsets:
        want += ctx.onsets + int(min(2, ctx.onset_strength * 2.2))

    if want:
        free = np.flatnonzero(st["my"] < 0.0)[:want]
        if free.size:
            k = free.size
            hard = float(np.clip(ctx.onset_strength, 0.0, 1.0))
            # Outward from the radiant, in a wedge rather than all round: a
            # shower seen from the ground covers part of the sky, not all of
            # it.
            ang = st["rad"] + rng.uniform(-0.9, 0.9, k)
            # Not from the radiant itself. A meteor only becomes visible some
            # way out from it, and spawning them all on one dot looks like a
            # leak rather than a shower.
            away = rng.uniform(0.05, 0.45, k) * min(dr, dc)
            st["my"][free] = ry + np.sin(ang) * away
            st["mx"][free] = rx + np.cos(ang) * away
            # Speed scales with the grid so the time to cross is the same on
            # any terminal, which is the same reason Ember and Rain do it.
            speed = (0.45 + 0.55 * hard) * dc * rng.uniform(0.8, 1.3, k)
            # Isotropic, and this is the one place it matters most. Halving
            # the vertical component here — the reflex this module's docstring
            # warns about — does not merely flatten the trajectory: the meteor
            # then travels along a different line from the one it spawned on,
            # so it stops radiating from the radiant and the whole conceit
            # goes with it. Dots are square; there is nothing to correct.
            st["mvy"][free] = np.sin(ang) * speed
            st["mvx"][free] = np.cos(ang) * speed
            st["mlen"][free] = (9.0 + 24.0 * hard) * rng.uniform(0.7, 1.4, k)
            st["mbright"][free] = 0.55 + 0.45 * hard
            st["mage"][free] = 0.0
            st["mlife"][free] = rng.uniform(0.45, 1.1, k)

    # ── flight ──
    alive = st["my"] >= 0.0
    if alive.any():
        st["my"][alive] += st["mvy"][alive] * ctx.dt
        st["mx"][alive] += st["mvx"][alive] * ctx.dt
        st["mage"][alive] += ctx.dt
        dead = alive & (
            (st["mage"] > st["mlife"])
            | (st["my"] < -4) | (st["my"] > dr + 4)
            | (st["mx"] < -4) | (st["mx"] > dc + 4)
        )
        st["my"][dead] = -1.0

    # ── the streaks ──
    live = np.flatnonzero(st["my"] >= 0.0)
    if live.size:
        # Bright on arrival and dimming as it burns up. The reverse — fading
        # in — reads as a light being switched on, which is not what this is.
        age = st["mage"][live] / np.maximum(st["mlife"][live], 1e-3)
        glow = st["mbright"][live] * np.clip(1.0 - age, 0.0, 1.0) ** 0.75

        speed = np.hypot(st["mvx"][live], st["mvy"][live])
        ux = st["mvx"][live] / np.maximum(speed, 1e-6)
        uy = st["mvy"][live] / np.maximum(speed, 1e-6)

        # The tail is stepped backwards along the flight path, a dot at a
        # time. One gather per step over every live meteor at once, so the
        # whole shower costs what the longest streak costs rather than what
        # the streaks cost added up.
        steps = int(np.clip(st["mlen"][live].max(), 3, 26))
        for s in range(steps):
            f = s / max(1, steps - 1)
            back = f * st["mlen"][live]
            py = np.rint(st["my"][live] - uy * back).astype(np.int32)
            px = np.rint(st["mx"][live] - ux * back).astype(np.int32)
            ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
            if not ok.any():
                continue
            # Falling away as the square is what makes the leading dot read as
            # the object and everything behind it as what it left.
            w = glow * (1.0 - f) ** 2
            np.maximum.at(field, (py[ok], px[ok]), w[ok])

    np.clip(field, 0.0, 1.0, out=field)
    dots = field > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx
