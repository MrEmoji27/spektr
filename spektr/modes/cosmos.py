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
        # Scintillation is updated in short, irregular bursts rather than with
        # a private sine wave per star. Most stars stay at their magnitude;
        # only a small random subset gets a brief lift.
        "tw": np.ones(n, dtype=np.float32),
        "tw_tick": -1,

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
        "mflare": np.zeros(_METEOR_CAP),

        # A meteor leaves a short-lived afterimage when it burns out. These
        # have their own arrays so a newly spawned meteor can reuse its slot
        # without erasing an afterglow from another slot.
        "ay": np.zeros(_METEOR_CAP),
        "ax": np.zeros(_METEOR_CAP),
        "avy": np.zeros(_METEOR_CAP),
        "avx": np.zeros(_METEOR_CAP),
        "alen": np.zeros(_METEOR_CAP),
        "aage": np.full(_METEOR_CAP, -1.0),

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
    field = np.zeros((dr, dc), dtype=np.float32)

    # ── the fixed stars ──
    # Twinkle is atmospheric, so it is slow and shallow and never switches a
    # star off. The loud-passage lift is a separate whole-sky term, so the
    # music is visible even in the stretches when nothing is crossing.
    tick = int(ctx.t * 10.0)
    if tick != st["tw_tick"]:
        # Ease old flashes back to normal, then start a few new ones. The
        # seeded generator keeps this deterministic while still reading as
        # irregular scintillation instead of a synchronized animation.
        st["tw"] += (1.0 - st["tw"]) * 0.48
        flash = rng.random(st["tw"].size) < 0.055
        if flash.any():
            st["tw"][flash] = rng.uniform(1.04, 1.28, int(flash.sum()))
        st["tw_tick"] = tick
    tw = st["tw"]
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
    st["acc"] += (0.10 + ctx.energy * 0.20) * ctx.dt
    want = int(st["acc"])
    if want:
        st["acc"] -= want
    # A meteor per onset is a barrage. It did not used to be: most of them
    # were thrown off the grid and retired before they were ever drawn, so
    # the rate above was tuned against a picture showing a fraction of what
    # it spawned. With the wedge aimed into the sky they all survive, and the
    # same numbers measured 2.3 meteors on screen at once with 5% of frames
    # empty — weather, not an event. So a beat throws one only if it wins a
    # strength-weighted draw: a hard hit usually does, an ordinary one
    # usually does not, and the sky is empty about two frames in three again.
    if ctx.onsets:
        want += int(rng.random() < 0.05 + 0.25 * min(1.0, ctx.onset_strength))

    if want:
        free = np.flatnonzero(st["my"] < 0.0)[:want]
        if free.size:
            k = free.size
            hard = float(np.clip(ctx.onset_strength, 0.0, 1.0))
            # Outward from the radiant, in a wedge rather than all round: a
            # shower seen from the ground covers part of the sky, not all of
            # it.
            # The wedge points from the radiant *into* the sky. It used to be
            # centred on the radiant's own drift phase, which is also what
            # places the radiant, so the two agreed: with the radiant on the
            # right of the screen the wedge pointed right, and the meteors it
            # threw left the grid on their first frame having never been
            # drawn. How badly depended on the radiant's random starting angle
            # and on nothing else — 85% of spawns landed on the grid for one
            # draw of it and 48% for another, a coin toss deciding how alive
            # the mode looks. Aiming at the middle of the sky costs the
            # picture nothing: a radiant is the point meteors diverge *from*,
            # so they have to travel away from it across the sky to read as
            # one at all.
            aim = math.atan2(dr * 0.5 - ry, dc * 0.5 - rx)
            ang = aim + rng.uniform(-0.9, 0.9, k)
            sa, ca = np.sin(ang), np.cos(ang)
            # Not from the radiant itself. A meteor only becomes visible some
            # way out from it, and spawning them all on one dot looks like a
            # leak rather than a shower.
            away = rng.uniform(0.05, 0.45, k) * min(dr, dc)
            st["my"][free] = ry + sa * away
            st["mx"][free] = rx + ca * away
            # Speed scales with the grid so the time to cross is the same on
            # any terminal, which is the same reason Ember and Rain do it.
            speed = (0.45 + 0.55 * hard) * dc * rng.uniform(0.8, 1.3, k)
            # Isotropic, and this is the one place it matters most. Halving
            # the vertical component here — the reflex this module's docstring
            # warns about — does not merely flatten the trajectory: the meteor
            # then travels along a different line from the one it spawned on,
            # so it stops radiating from the radiant and the whole conceit
            # goes with it. Dots are square; there is nothing to correct.
            st["mvy"][free] = sa * speed
            st["mvx"][free] = ca * speed
            st["mlen"][free] = (9.0 + 24.0 * hard) * rng.uniform(0.7, 1.4, k)
            st["mbright"][free] = 0.55 + 0.45 * hard
            st["mage"][free] = 0.0
            st["mlife"][free] = rng.uniform(0.45, 1.1, k)
            # A bolide is rare and reserved for the hardest hits. It is a
            # flare around the head, not another meteor or a denser shower.
            bolide = (hard > 0.86) & (rng.random(k) < 0.16)
            st["mflare"][free] = bolide * (0.78 + 0.22 * hard)

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
        if dead.any():
            st["ay"][dead] = st["my"][dead]
            st["ax"][dead] = st["mx"][dead]
            st["avy"][dead] = st["mvy"][dead]
            st["avx"][dead] = st["mvx"][dead]
            st["alen"][dead] = st["mlen"][dead] * 0.72
            st["aage"][dead] = 0.0
        st["my"][dead] = -1.0

    after = st["aage"] >= 0.0
    if after.any():
        st["aage"][after] += ctx.dt
        st["aage"][st["aage"] > 0.38] = -1.0

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

        # One sample per dot along the longest tail, plus one, so consecutive
        # samples land on adjacent dots. The cap used to be 26, which is under
        # the tail a hard onset produces — up to 46 dots — so the samples came
        # more than a dot apart and the streak was drawn as a dotted line:
        # over one dot of spacing on 80% of the frames a meteor was on screen.
        # A shooting star is a streak; a dashed one is a different object.
        steps = int(np.clip(st["mlen"][live].max() + 1.0, 3, 52))

        # Every sample of every streak at once, rather than a pass per step.
        # The arrays here are tiny — at most 52 steps by a handful of live
        # meteors — so a stepped loop spends its time in numpy's per-call
        # overhead and nothing else, and doubling the step count above would
        # otherwise have cost more than the whole mode saves.
        f = np.linspace(1.0, 0.0, steps)[:, None]         # furthest row first
        back = f * st["mlen"][live]
        py = np.rint(st["my"][live] - uy * back).astype(np.int32)
        px = np.rint(st["mx"][live] - ux * back).astype(np.int32)
        ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
        if ok.any():
            # Falling away as the square is what makes the leading dot read as
            # the object and everything behind it as what it left. The head is
            # the last row, so where a streak writes over itself the head wins.
            w = glow * (1.0 - f) ** 2
            sel = (py[ok], px[ok])
            field[sel] = np.maximum(field[sel], w[ok])

        # A hard onset occasionally makes a small flare around the head. It
        # stays sparse and is visually distinct from a merely long meteor.
        flare = st["mflare"][live]
        if np.any(flare > 0.0):
            head = flare > 0.0
            py = np.rint(st["my"][live][head]).astype(np.int32)
            px = np.rint(st["mx"][live][head]).astype(np.int32)
            val = flare[head] * glow[head] * 0.62
            for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                fy, fx = py + oy, px + ox
                ok = (fy >= 0) & (fy < dr) & (fx >= 0) & (fx < dc)
                if ok.any():
                    sel = (fy[ok], fx[ok])
                    field[sel] = np.maximum(field[sel], val[ok])

    # Afterimages are dim and short, but remain aligned with the old path long
    # enough to read as persistence rather than a second meteor.
    after = np.flatnonzero(st["aage"] >= 0.0)
    if after.size:
        aage = st["aage"][after]
        au = np.hypot(st["avx"][after], st["avy"][after])
        aux = st["avx"][after] / np.maximum(au, 1e-6)
        auy = st["avy"][after] / np.maximum(au, 1e-6)
        after_steps = 2 if dr * dc < 50000 else 4
        for s in range(after_steps):
            f = s / max(after_steps - 1, 1)
            back = f * st["alen"][after]
            py = np.rint(st["ay"][after] - auy * back).astype(np.int32)
            px = np.rint(st["ax"][after] - aux * back).astype(np.int32)
            ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
            if ok.any():
                val = 0.20 * np.clip(1.0 - aage / 0.38, 0.0, 1.0) * (1.0 - f) ** 1.5
                sel = (py[ok], px[ok])
                field[sel] = np.maximum(field[sel], val[ok])


    # Every contributor is bounded to 0..1 and every write above maxed
    # against what was already there, so no full-grid clip is needed.
    dots = field > 0.04
    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx
