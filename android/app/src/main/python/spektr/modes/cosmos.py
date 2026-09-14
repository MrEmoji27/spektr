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

from ..render import cell_max, noise, pack_braille
from . import Ctx, contrast_ramp, empty, mode

#: Fixed stars per dot cell. Sparse on purpose: a sky is mostly nothing, and
#: past a certain density the eye stops reading stars and starts reading haze.
_STAR_DENSITY = 1.0 / 260.0

#: Meteors in flight at once. A shower is not a barrage — more than a handful
#: on screen stops reading as "something rare just happened", which is the
#: only thing a shooting star has to say. Sized so a full beat cluster plus
#: the ambient strays fit without evicting anybody mid-flight.
_METEOR_CAP = 26

#: Onset strengths that throw something. At or above ``_METEOR_HARD`` a hit
#: always breaks a cluster loose; between the two it may throw one meteor;
#: below ``_METEOR_MEDIUM`` — the hats, the ghost notes — it throws nothing.
_METEOR_HARD = 0.70
_METEOR_MEDIUM = 0.50


def _sky_colour(ctx: Ctx, field: np.ndarray) -> np.ndarray:
    """Cell colours for a sky field, where the field value means *brightness*.

    Every mode here encodes faint and bright as a number — a dim star, a
    fading trail, the galactic band against a shell or a meteor head — and
    the ramp does not: it is a hue gradient, and on gruvbox its low end is the
    most visible colour it has. Run through ``ctx.ramp``, the faint things came
    out as loud as the events, and on screen the band was a yellow slab and a
    star-trail exposure a wall. ``contrast_ramp`` measures the ramp against the
    theme's background instead, so 0 lands on the quietest visible colour and
    1 on the loudest, on every theme.
    """
    return contrast_ramp(ctx.palette, cell_max(field))


def _star_twinkle(st: dict, ctx: Ctx) -> np.ndarray:
    """Advance the family's shared scintillation model, return the per-star gain.

    Twinkle is atmospheric, so it is slow and shallow and never switches a
    star off. Updates land in short, irregular bursts rather than with a
    private sine wave per star: most stars stay at their magnitude, and only
    a small random subset gets a brief lift. The rate rides ``ctx.drive`` a
    little, so percussive passages shimmer without turning the sky into
    static. Every sky mode in this file shares it, so they all scintillate
    the same way — one idea, stated once.
    """
    tick = int(ctx.t * 10.0)
    if tick != st["tw_tick"]:
        # Ease old flashes back to normal, then start a few new ones. The
        # seeded generator keeps this deterministic while still reading as
        # irregular scintillation instead of a synchronized animation.
        st["tw"] += (1.0 - st["tw"]) * 0.48
        flash = st["rng"].random(st["tw"].size) < 0.055 + 0.06 * ctx.drive
        if flash.any():
            st["tw"][flash] = st["rng"].uniform(1.04, 1.28, int(flash.sum()))
        st["tw_tick"] = tick
    return st["tw"]


def _sky_lift(ctx: Ctx) -> float:
    """How bright the whole sky sits this frame, 0..1.

    Two terms, both bounded before they are summed: how loud the track is,
    and a small beat-locked swell on top. The swell is what keeps a sky
    answering between events — the family's whole bargain is sparse events,
    so the bed it does show has to breathe with the music rather than sit at
    one level until something crosses it. ``pulse`` is 0.0 whenever no tempo
    is established, so silence and unknown tempo both stay still.
    """
    return min(1.0, 0.55 + 0.45 * min(1.0, ctx.energy * 1.8) + 0.10 * ctx.pulse)


def _edge_entry(rx: float, ry: float, ca, sa, dr: int, dc: int):
    """Where a meteor enters the display, given a direction of travel.

    ``(ca, sa)`` is the unit direction the meteor moves in. Walking backward
    from the radiant ``(rx, ry)`` along ``(-ca, -sa)`` until the boundary is
    crossed gives the edge point the meteor enters from: spawn it there and it
    crosses the whole screen instead of blinking into existence mid-sky. This
    is what keeps a shooting star a *crossing*, never a sudden dot in the
    middle of the frame. Vectorised over ``ca``/``sa``; ``rx``/``ry`` are the
    shared radiant.
    """
    vx, vy = -ca, -sa
    tx = np.full_like(vx, np.inf)
    tx[vx > 0] = (dc - 1 - rx) / vx[vx > 0]
    tx[vx < 0] = (0.0 - rx) / vx[vx < 0]
    ty = np.full_like(vy, np.inf)
    ty[vy > 0] = (dr - 1 - ry) / vy[vy > 0]
    ty[vy < 0] = (0.0 - ry) / vy[vy < 0]
    t = np.minimum(tx, ty)
    return rx + vx * t, ry + vy * t


def _blur_merge(acc: np.ndarray, amount: float) -> np.ndarray:
    """Smear the accumulated field toward its neighbours, in place.

    Motion blur for a spinning sky: blend each dot toward its four
    neighbours so bright trails lighten, widen and flow into one another.
    ``amount`` is 0..1; at 0 the field is left exactly as it is (a slow sky
    stays a pin-sharp set of arcs), and at 1 it becomes the local average.
    Done with strided slices rather than a convolution so it stays a handful
    of vectorised adds on the dot grid. Writes back into ``acc`` (the field
    is the long-lived exposure), so the smear accumulates from frame to frame.

    The quarter is not a taste knob and must not be nudged: the field is
    fed back into itself every frame, so the kernel has to conserve energy
    exactly. Each of the four neighbours carries ``amount / 4`` and the
    centre keeps ``1 - amount``, which sums to one. Give the neighbours a
    half each and the weights sum to ``1 + amount`` — a compounding gain
    that outruns the exposure's own fade, and a single lit dot floods the
    whole screen to full white inside ten seconds with no stars stamped
    at all.
    """
    if amount <= 0.0:
        return acc
    w = 0.25 * amount
    s = acc * (1.0 - amount)
    # horizontal smear
    s[:, 1:] += acc[:, :-1] * w
    s[:, :-1] += acc[:, 1:] * w
    # vertical smear
    s[1:, :] += acc[:-1, :] * w
    s[:-1, :] += acc[1:, :] * w
    acc[:] = s
    return acc


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
    matters, on a quiet track with a crisp snare. And since the rework, a
    winning beat does not throw one: it breaks a *cluster* loose — 3 to 6
    fragments of one body sharing a line, a parallax and a lifetime, laid
    out head to tail along their path with a little sideways scatter. The
    head burns brightest and the train dims behind it; they arrive together,
    cross together and burn out together. The harder the hit, the more
    fragments come away.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("shooting_star", lambda: _sky(dr, dc))
    rng = st["rng"]
    field = np.zeros((dr, dc), dtype=np.float32)

    # ── the fixed stars ──
    tw = _star_twinkle(st, ctx)
    lift = _sky_lift(ctx)
    field[st["sy"], st["sx"]] = np.clip(st["mag"] * tw * lift, 0.0, 1.0)

    # ── the radiant, drifting ──
    st["rad"] = (st["rad"] + ctx.dt * 0.035) % (2 * math.pi)
    # Kept well off-centre: a radiant in the middle of the screen throws
    # meteors symmetrically in every direction, which reads as an explosion
    # rather than as a shower.
    rx = dc * (0.5 + 0.42 * math.cos(st["rad"]))
    ry = dr * (0.5 + 0.42 * math.sin(st["rad"] * 0.7))

    # ── spawning ──
    # Base rate stays low and the energy term carries the loudness, but a
    # percussive passage that isn't necessarily loud throws more too: at
    # equal level a busy snare-and-hat groove should cross more sky than a
    # held pad, which is the difference between energy (how much) and drive
    # (how attacked).
    st["acc"] += (0.06 + ctx.energy * 0.14) * ctx.dt
    want = int(st["acc"])
    if want:
        st["acc"] -= want
    # Admission is still a strength-weighted draw — a beat throws something
    # only if it wins it, and the sky stays empty about two frames in three.
    # What a winning beat throws changed: not one meteor but a *cluster*,
    # fragments of one body breaking loose along a shared line. A single
    # meteor is an event; a cluster falling together is an event you tell
    # someone about, which is the whole register this mode lives in. The
    # harder the hit, the more fragments come away (3..6).
    # What a hit throws follows how hard it was, and a hard one always throws.
    # This used to be a strength-weighted coin toss for every onset, hats
    # included: in the live terminal a groove's kicks and snares went by with
    # the sky mostly empty, and whether any given hit had thrown something was
    # a guess. Now a hard hit (``_METEOR_HARD`` and up) always breaks a cluster
    # loose, a medium one throws a single meteor about half the time, and the
    # quiet onsets a hi-hat pattern lands several times a bar throw nothing —
    # they would turn a shower into weather. A short refractory keeps a flam
    # or a double-detected hit from throwing twice.
    beat = False
    lone = False
    if ctx.onsets and ctx.t - st.get("thrown", -9.0) >= 0.18:
        hit = float(ctx.onset_strength)
        if hit >= _METEOR_HARD:
            beat = True
        elif hit >= _METEOR_MEDIUM and rng.random() < 0.5:
            lone = True
        if beat or lone:
            st["thrown"] = ctx.t
    if lone:
        want += 1

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
            # Tighter than it used to be (±0.9 rad read as a fan). A real
            # shower concentrates: most members land within about twenty
            # degrees of the radiant's axis, with a few strays for variety.
            ang = aim + rng.uniform(-0.66, 0.66, k)
            sa, ca = np.sin(ang), np.cos(ang)
            # Not from the radiant itself, and never from the middle of the
            # sky: a meteor enters where its line of flight crosses the edge
            # of the display, then crosses the whole screen. Spawning any
            # distance out from the radiant put a meteor on screen already
            # partway through its run, which reads as something appearing
            # mid-frame rather than as a crossing.
            ex, ey = _edge_entry(rx, ry, ca, sa, dr, dc)
            st["my"][free] = ey
            st["mx"][free] = ex
            # One draw decides how *near* a meteor is, and speed, tail length
            # and brightness all follow it. They used to be three independent
            # rolls, which happily produced slow long bright meteors — a
            # combination nothing in the sky produces, and exactly the one the
            # eye reads as wrong. Correlated, they are parallax: near means
            # fast, long and bright together; far means the opposite.
            depth = rng.uniform(0.75, 1.35, k)
            # Speed scales with the grid so the time to cross is the same on
            # any terminal, which is the same reason Ember and Rain do it.
            speed = (0.45 + 0.55 * hard) * dc * depth
            # Isotropic, and this is the one place it matters most. Halving
            # the vertical component here — the reflex this module's docstring
            # warns about — does not merely flatten the trajectory: the meteor
            # then travels along a different line from the one it spawned on,
            # so it stops radiating from the radiant and the whole conceit
            # goes with it. Dots are square; there is nothing to correct.
            st["mvy"][free] = sa * speed
            st["mvx"][free] = ca * speed
            st["mlen"][free] = (9.0 + 24.0 * hard) * depth
            st["mbright"][free] = np.clip((0.55 + 0.45 * hard) * depth, 0.0, 1.0)
            st["mage"][free] = 0.0
            st["mlife"][free] = rng.uniform(0.45, 1.1, k)
            # A bolide is rare and reserved for the hardest hits. It is a
            # flare around the head, not another meteor or a denser shower.
            bolide = (hard > 0.86) & (rng.random(k) < 0.16)
            st["mflare"][free] = bolide * (0.78 + 0.22 * hard)

    if beat:
        hard = float(np.clip(ctx.onset_strength, 0.0, 1.0))
        fam = min(int(3 + round(hard * 3.0)), _METEOR_CAP)      # 3..6 fragments
        free = np.flatnonzero(st["my"] < 0.0)[:fam]
        if free.size:
            k = free.size
            # One family axis inside the wedge, members within about nine
            # degrees of it. The first cut allowed twelve per member and the
            # widest legal family fanned past half a radian end to end — at
            # that spread the eye files them as separate meteors that
            # happen to share a frame, not as one thing coming apart.
            axis = math.atan2(dr * 0.5 - ry, dc * 0.5 - rx) + rng.uniform(-0.24, 0.24)
            ang = axis + rng.uniform(-0.16, 0.16, k)
            sa, ca = np.sin(ang), np.cos(ang)
            # Fragments share one body, so they share one parallax draw:
            # speed, tail and brightness all hang off the same depth, with
            # only small per-member jitter. Independent draws produced
            # families whose members disagreed about how near they were,
            # and the group fell apart visually exactly because of it.
            depth = rng.uniform(0.75, 1.35)
            speed = (0.45 + 0.55 * hard) * dc * depth * rng.uniform(0.94, 1.06, k)
            # Staggered head to tail along a train that enters at the edge.
            # Every member's own line of flight crosses the screen boundary,
            # so it is anchored there; the head sits at the entry point and
            # each fragment behind it trails a little further off-screen
            # along the shared path, so the group crosses together rather
            # than materialising across the middle of the frame. The per-
            # member angle jitter above already gives the train its width.
            away = np.sort(rng.uniform(0.06, 0.40, k) * min(dr, dc))
            ex, ey = _edge_entry(rx, ry, ca, sa, dr, dc)
            trail = away[-1] - away
            st["my"][free] = ey - sa * trail
            st["mx"][free] = ex - ca * trail
            st["mvy"][free] = sa * speed
            st["mvx"][free] = ca * speed
            st["mlen"][free] = (9.0 + 24.0 * hard) * depth * rng.uniform(0.85, 1.15, k)
            # The train dims towards its tail: the head is the parent body
            # and each fragment behind it is a little poorer. ``away`` was
            # sorted ascending above, so train rank runs with the array —
            # the first cut wrote the ramp descending in slot order, which
            # attached the brightest value to whichever slot held the tail,
            # i.e. exactly backwards.
            fade = np.linspace(0.72, 1.0, k) if k > 1 else np.ones(1)
            st["mbright"][free] = np.clip(
                (0.55 + 0.45 * hard) * depth * fade * rng.uniform(0.90, 1.10, k),
                0.0, 1.0)
            # They burn up together: one life for the family, lightly
            # jittered. Independent draws had fragments winking out one by
            # one mid-fall, which dismantles the group as surely as any
            # angular spread.
            st["mage"][free] = 0.0
            life = rng.uniform(0.45, 1.1)
            st["mlife"][free] = life * rng.uniform(0.93, 1.07, k)
            # A bolide is rare and reserved for the hardest hits — and for
            # the head alone. The flare is the parent body's; scattered down
            # the train it is confetti, not an event.
            bolide = np.zeros(k, dtype=bool)
            bolide[-1] = bool(hard > 0.86) and bool(rng.random() < 0.16)
            st["mflare"][free] = bolide * (0.78 + 0.22 * hard)

    # ── flight ──
    alive = st["my"] >= 0.0
    if alive.any():
        st["my"][alive] += st["mvy"][alive] * ctx.dt
        st["mx"][alive] += st["mvx"][alive] * ctx.dt
        st["mage"][alive] += ctx.dt
        # Off the grid only counts as gone when the meteor is heading further
        # out. A cluster's trailing fragments are *placed* off-screen, behind
        # the entry point, so the train crosses together — and a plain bounds
        # test killed them on their first frame: a hard hit meant to break
        # three to six fragments loose put two on screen.
        dead = alive & (
            (st["mage"] > st["mlife"])
            | ((st["my"] < -4) & (st["mvy"] <= 0)) | ((st["my"] > dr + 4) & (st["mvy"] >= 0))
            | ((st["mx"] < -4) & (st["mvx"] <= 0)) | ((st["mx"] > dc + 4) & (st["mvx"] >= 0))
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

        # The head then gets a little more than the tail's own peak, so the
        # front of the streak is a point rather than wherever the gradient
        # happens to stop. Subtle by design — enough to aim the eye, not
        # enough to read as a second object riding the streak.
        hy, hx = py[-1], px[-1]
        hok = (hy >= 0) & (hy < dr) & (hx >= 0) & (hx < dc)
        if hok.any():
            hw = np.minimum(1.0, glow[hok] * 1.3)
            sel = (hy[hok], hx[hok])
            field[sel] = np.maximum(field[sel], hw)

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
    cidx = _sky_colour(ctx, field)
    return codes, cidx


# ── the rest of the family ───────────────────────────────────────────────────
#
# Three more skies, each answering a different question the first mode left
# open. Shooting Star is about *motion* — something crosses. These are about
# structure and accumulation: lines drawn between fixed points (Constellations),
# an exposure that builds up over seconds (Star Trails), and one rare event
# violent enough to own the screen for a while (Supernova). All three keep
# the family bargain — mostly empty, mostly still, music as events — and all
# three lean on the same shared sky: seeded stars, the scintillation model,
# the lift.

#: Stars per dot cell for Constellations. Denser than a bare night sky:
#: edges need anchors within reach of each other, or every beat ends in a
#: failed search instead of a line.
_CHART_DENSITY = 1.0 / 200.0

#: How long a drawn edge survives, seconds, and how many live at once.
#: The cap is sized for a busy build racing ahead of the fade, since the
#: whole point of the change is that music visibly outruns the sky.
_EDGE_TAU = 9.0
_EDGE_CAP = 36

#: Edge-drawing rate, edges per second. The base keeps a constellation on
#: screen even in silence — a chain is forever being built, just slowly —
#: and the energy/drive terms are what make music *speed it up*: a loud,
#: percussive passage draws several times as fast. An onset drops an extra
#: edge on top the moment it lands, which is the hard-hit burst.
_CHART_BASE = 0.12
_CHART_ENERGY = 0.35
#: Whole edges dropped the moment an onset lands, on top of the accumulator:
#: one for a medium hit, two for a hard one. Beats are what draw a figure; the
#: accumulator is only the slow hand that keeps a quiet sky from being bare.
_CHART_ONSET = 1
_CHART_ONSET_HARD = 0.75

#: A figure is a handful of stars, not a wire: it grows to at most this many
#: edges, only ever to one of the ``_CHART_NEAREST`` nearest stars it has not
#: already used, and then the next figure starts somewhere else. Before these
#: rules a chain hopped anywhere inside a 48-dot reach and revisited its own
#: stars freely; on screen it tangled into knots and long crossing wires, and
#: at the cap it stayed a knot for the whole fade.
_CHART_FIGURE_EDGES = 6
_CHART_NEAREST = 3

#: Seconds over which the first edge of a new figure fades in, so the
#: handoff from one constellation to the next is a dissolve rather than a
#: line popping into a far corner of the sky at full brightness.
_CHART_RAMP = 0.8


def _chart(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(47)
    n = int(np.clip(dr * dc * _CHART_DENSITY, 30, 1200))
    return {
        "sy": rng.integers(0, dr, n),
        "sx": rng.integers(0, dc, n),
        "mag": rng.uniform(0.15, 1.0, n) ** 1.6,
        "tw": np.ones(n, dtype=np.float32),
        "tw_tick": -1,
        # Edges as [x0, y0, x1, y1, born, strength, ramp]; oldest trimmed
        # past the cap, dead ones dropped by age. ``ramp`` marks the first
        # edge of a new figure so it can fade in. A list because it holds a
        # few dozen tiny rows, and array bookkeeping would outgrow the data.
        "edges": [],
        #: Index of the star the next edge grows from. A constellation is a
        #: *chain* — each beat extends the last figure — not independent
        #: pairs flashing at random across the sky. Seeded at creation so
        #: the very first accumulated mark draws a line immediately rather
        #: than spending its turn choosing a starting star.
        "tail": int(rng.integers(0, n)),
        #: Fraction of an edge to draw per second, accumulated across frames
        #: so a base rate can lay edges without waiting for an onset.
        "acc": 0.0,
        #: True when the next edge starts a brand-new figure, so it fades in.
        "fresh": True,
        #: Stars already used by the figure being drawn, and how many edges
        #: it has, so a figure never doubles back through itself.
        "used": set(),
        "fig_edges": 0,
        "rng": rng,
    }


@mode("Constellations", group="cosmos",
      blurb="beats draw lines between fixed stars, then the figures fade")
def constellations(ctx: Ctx):
    """The music as a surveyor of the sky.

    A chain of lines grows between fixed stars, one edge at a time, always
    in progress — there is always a constellation on screen, whether or not
    anything plays. Music is the *speed*: energy and drive multiply how fast
    the surveyor draws, so a quiet passage builds a figure slowly and a loud
    percussive one races through it. A landed onset adds a burst of edges on
    the beat. The figures fade over about nine seconds, so a long silence
    still dissolves back toward a sparse skeleton, but never to a bare sky.
    When a chain runs out of neighbours it restarts elsewhere, and the first
    edge of the new figure eases in — the subtle handoff between
    constellations rather than a line popping into a far corner.

    Onset strength sets how bright the new line burns in, on top of a floor
    that keeps the always-on figure readable. A figure that runs out of
    neighbours restarts rather than drawing one long jump, because a line
    across half the sky reads as an artifact, not a figure.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("constellations", lambda: _chart(dr, dc))
    field = np.zeros((dr, dc), dtype=np.float32)

    tw = _star_twinkle(st, ctx)
    lift = _sky_lift(ctx)
    field[st["sy"], st["sx"]] = np.clip(st["mag"] * tw * lift, 0.0, 1.0)

    # ── growth ──
    # Drawn from a continuous accumulator, not only on onset. A base rate
    # keeps a figure building even in silence (so the sky is never bare), and
    # energy and drive multiply it — that is what makes music speed the
    # surveyor up. A landed onset drops an extra edge straight away, a hard
    # burst on top of an already-fast groove. The chain rules are unchanged:
    # every edge extends the last, and a figure that runs out of neighbours
    # restarts elsewhere.
    rate = (_CHART_BASE + _CHART_ENERGY * ctx.energy) * ctx.dt
    st["acc"] += rate
    marks = int(st["acc"])
    st["acc"] -= marks
    if ctx.onsets and ctx.onset_strength >= 0.5:
        marks += _CHART_ONSET + int(ctx.onset_strength >= _CHART_ONSET_HARD)
    if marks:
        sy, sx = st["sy"], st["sx"]
        reach = min(dr, dc) * 0.22
        rng = st["rng"]
        for stamp in range(marks):
            t = st["tail"]
            done = st["fig_edges"] >= _CHART_FIGURE_EDGES
            if t is not None and not done:
                dx = sx.astype(np.int32) - sx[t]
                dy = sy.astype(np.int32) - sy[t]
                d2 = dx * dx + dy * dy
                cand = np.flatnonzero(d2 <= reach * reach)
                cand = np.array([c for c in cand if c != t and c not in st["used"]], dtype=np.int64)
                if cand.size:
                    # The nearest few, not anything in reach: nearest-neighbour
                    # steps keep a figure compact and stop it crossing itself.
                    cand = cand[np.argsort(d2[cand])[:_CHART_NEAREST]]
                    k = int(rng.choice(cand))
                    # Strength is the onset's own, but never below a floor: a
                    # figure drawn by the quiet accumulator still has to read.
                    s = max(0.55, float(np.clip(ctx.onset_strength, 0.0, 1.0)))
                    st["edges"].append([
                        float(sx[t]), float(sy[t]), float(sx[k]), float(sy[k]),
                        ctx.t, s, float(st["fresh"]),
                    ])
                    st["used"].add(k)
                    st["tail"] = k
                    st["fresh"] = False
                    st["fig_edges"] += 1
                    continue
            # The figure is finished, or cannot grow: start the next one at a
            # star away from everything still on screen, so figures sit apart
            # instead of knotting into each other.
            live = st["edges"]
            pick = int(rng.integers(0, sy.size))
            if live:
                ex = np.array([[e[0], e[1]] for e in live] + [[e[2], e[3]] for e in live])
                trial = rng.integers(0, sy.size, 12)
                far = [((ex[:, 0] - sx[i]) ** 2 + (ex[:, 1] - sy[i]) ** 2).min() for i in trial]
                pick = int(trial[int(np.argmax(far))])
            st["tail"] = pick
            st["used"] = {pick}
            st["fig_edges"] = 0
            st["fresh"] = True

    # ── fade and draw ──
    if st["edges"]:
        st["edges"] = [e for e in st["edges"] if ctx.t - e[4] < _EDGE_TAU]
        del st["edges"][:-_EDGE_CAP]
        for x0, y0, x1, y1, born, s, ramp in st["edges"]:
            age = ctx.t - born
            w = s * (1.0 - age / _EDGE_TAU) ** 1.3 * 0.85
            # The first line of a new constellation eases in rather than
            # appearing: the transition between figures is a dissolve. The
            # ramp is a per-edge flag, so only the figure's opening edge
            # brightens slowly, and only for its first moment on screen.
            if ramp and age < _CHART_RAMP:
                w *= age / _CHART_RAMP
            steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            ts = np.linspace(0.0, 1.0, steps)
            px = np.rint(x0 + (x1 - x0) * ts).astype(np.int32)
            py = np.rint(y0 + (y1 - y0) * ts).astype(np.int32)
            ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
            if ok.any():
                sel = (py[ok], px[ok])
                field[sel] = np.maximum(field[sel], w)
            # The endpoints get a little more than the line, so vertices read
            # as stars the figure is hung on rather than places a line passes.
            for vy, vx in ((y0, x0), (y1, x1)):
                if 0 <= vy < dr and 0 <= vx < dc:
                    field[int(vy), int(vx)] = max(
                        field[int(vy), int(vx)], min(1.0, w + 0.20))

    dots = field > 0.04
    codes = pack_braille(dots)
    cidx = _sky_colour(ctx, field)
    return codes, cidx


#: Exposure persistence, seconds — the *ceiling*, not the value used. The
#: accumulated buffer *is* the picture, and this tau decides how long an
#: arc's tail stays on the film after the star has moved on.
#:
#: It also decides whether the mode reads as arcs at all: the angle a star
#: sweeps while its mark survives is ``omega * tau``, and when that exceeds
#: the gap between neighbouring stars the arcs fuse into a filled annulus —
#: measured, not hypothetical. A fixed tau cannot hold that line, because
#: omega is not fixed: the sidereal floor is 0.055 rad/s but drive, pulse,
#: kick and beat together carry it past 0.6 on ordinary material, which at
#: this tau is a 110-degree arc per star and a screen of solid white. So tau
#: is a ceiling that :func:`star_trails` shortens as the sky speeds up —
#: which is what a photographer does anyway: a faster subject wants a
#: shorter shutter. See ``_TRAIL_FILL``.
_TRAIL_TAU = 3.2

#: How much of the gap between neighbouring stars an arc may fill. This is
#: the whole invariant: the exposure is cut short so ``omega * tau`` never
#: exceeds ``_TRAIL_FILL * gap``, whatever the music does. At 1.0 an arc may
#: reach the next star's trail and never pass it, which is exactly the line
#: between a sky of arcs and a filled annulus. Measured across silence,
#: pad, groove and loud beats at several sizes: the silent sky is unchanged,
#: and a driven one goes from 85% of the screen lit to 39%, still reading as
#: separate curves. Lower it for a sparser sky; above ~1.4 the arcs start to
#: touch and the picture closes up again. Held at 0.65: at 1.0 a driven sky
#: still covered about 40% of the cells in the live terminal and pad, groove and
#: drop were indistinguishable walls of arcs.
_TRAIL_FILL = 0.65

#: Most sub-steps the swept arc is sampled at in one frame. A star is stamped
#: along the arc it travelled this frame rather than at the single point it
#: ended on, because a point stamp draws a *dashed* arc the moment the star
#: moves more than a dot between frames — which it does at any of the lower
#: frame rates the app offers. The cap bounds the cost on a huge terminal
#: spun hard; past it the arc dashes again, but only in a corner of the
#: parameter space nothing reaches in practice.
_TRAIL_SUBSTEPS = 8

#: Sidereal rate plus what percussion adds, rad/s. The base is a real sky
#: turning — slow enough that a still track still lives — and ``drive`` is
#: the timelapse knob: percussive material visibly spins the heavens up.
#: Bumped so a beat-heavy groove reads as an actual acceleration rather than
#: a gentle drift.
_TRAIL_OMEGA = 0.055
_TRAIL_DRIVE = 0.30

#: Beat-locked and per-onset spin additions, rad/s. ``pulse`` keeps the whole
#: exposure shivering on the tempo between onsets; ``kick`` is the lurch from
#: a single hit. ``_TRAIL_BEAT`` is the per-onset spin the sky keeps building
#: toward — each beat nudges it up, it coasts back down.
_TRAIL_PULSE = 0.06
_TRAIL_BEAT = 0.06

#: Motion blur, as a rate *per second*, scaled by how fast the sky is
#: turning. This is what makes a fast spin read as speed rather than as a
#: crisp ring: the accumulated field is smeared toward its neighbours, so
#: trails lighten, widen and merge into a smooth glow, and because the ramp
#: is a function of brightness, a softened field reads as *blended* colours
#: instead of separate hard arcs. Near zero on a slow sky, so a still night
#: stays pin-sharp.
#:
#: Per second, not per frame, and the distinction is the whole point. The
#: smear is fed back into the exposure every frame, so a per-frame amount
#: compounds with the frame rate: the same music rendered at 240 fps came
#: out twice as dense as at 24 fps, which made the picture a property of the
#: terminal rather than of the track. Everything else here already scales by
#: ``ctx.dt``; this now does too.
#:
#: There is no blur at the sidereal floor. There used to be 0.9 per second of
#: it, which over a three-second exposure diffused every arc about a dot and a
#: half either side; with the marks max-stamped on top, the smeared edges stayed
#: above the film's threshold and in the live terminal every arc was three or
#: four dots thick — pad, groove and drop all read as the same wall of rings,
#: and it was still a wall five seconds after the music stopped.
_TRAIL_BLUR_BASE = 0.0        # the sidereal floor's own blur, per second
_TRAIL_BLUR_GAIN = 3.5        # blur per second, per rad/s above the floor

#: Star count scales with the *radius* available rather than the area: what
#: separates arcs from each other is angular spacing, and spacing comes from
#: how many stars share a circle, not from how big the screen is. Roughly
#: one star per fifteen dots of circumference at the rim.
_TRAIL_STAR_FROM_RADIUS = True


def _angular_gap(r: np.ndarray, th: np.ndarray) -> float:
    """Typical angular spacing between stars that share a radius, radians.

    What decides whether a long exposure reads as separate arcs is how far a
    star can sweep before it runs into the trail of its neighbour — and only
    a neighbour at nearly the same radius counts, because arcs a couple of
    dots apart in ``r`` are separate lines on the dot grid however long they
    both get. So the spacing is measured inside a shell two dots deep, and
    the median is taken: the odd tight pair fusing is what a real sky does,
    a fused majority is the failure. Called once per layout, so the O(n^2)
    pass costs nothing on the frame path.
    """
    dr_ = np.abs(r[:, None] - r[None, :])
    dth = np.abs(th[:, None] - th[None, :])
    dth = np.minimum(dth, 2 * math.pi - dth)
    shell = (dr_ < 2.0) & (dth > 1e-9)
    dth = np.where(shell, dth, np.inf)
    near = dth.min(axis=1)
    near = near[np.isfinite(near)]
    # A layout so sparse that no star shares a shell with any other cannot
    # fuse at all, so it gets no ceiling beyond a full turn.
    return float(np.median(near)) if near.size else 2 * math.pi


def _exposure(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(73)
    # The pole sits close to centre so the rotation reads as a wheel rather
    # than a target. r_max is the *circumscribed* radius — the distance to the
    # nearest corner — so the outermost stars actually reach the edges of the
    # terminal and the trails sweep the whole screen instead of stopping at
    # an inscribed circle that leaves the corners dark.
    px = dc * 0.5
    py = dr * 0.5
    r_max = max(2.0, math.hypot(max(px, dc - 1 - px), max(py, dr - 1 - py)) * 1.02)
    # More stars than the inscribed layout needed: the arc length a star can
    # sweep grows with its radius, so a bigger field wants roughly one star
    # per ten dots of rim circumference rather than fifteen, to keep the
    # separate trails from thinning into a handful of sparse curves.
    n = int(np.clip(2 * math.pi * r_max / 10.0, 18, 420))
    r = rng.uniform(0.14, 1.0, n) ** 0.85 * r_max
    th = rng.uniform(0.0, 2 * math.pi, n)
    return {
        # Inner radius above zero: stars huddling on the pole draw circles so
        # small they stack into one bright blob beside Polaris, and the pole
        # star stops reading as the still point everything else rounds.
        "r": r,
        "th": th,
        # The spacing the exposure length is held against. Measured within a
        # shell rather than across the whole disc: two stars only fuse if
        # they share a radius, since arcs a couple of dots apart in r sweep
        # past each other without ever touching. Computed once here — it is
        # a property of the layout, and the layout only changes on resize.
        "gap": _angular_gap(r, th),
        # Skewed faint, harder than the other skies: a trail photo is a
        # handful of bright arcs over many barely-there ones, and every star
        # here becomes a line rather than a dot, so the dim end has to stay
        # near the film's threshold or the whole exposure turns to mush.
        "mag": rng.uniform(0.30, 1.0, n) ** 2.5,
        "pole_x": px,
        "pole_y": py,
        # The outermost star's radius, which is the one that moves furthest
        # per frame and so decides how finely the sweep has to be sampled.
        "r_max": float(r.max()),
        "acc": np.zeros((dr, dc), dtype=np.float32),
        "kick": 0.0,
        "flare": 0.0,
        #: The exposure length actually used last frame, seconds. Written
        #: every frame by :func:`star_trails` purely so the ceiling it
        #: enforces can be read back and checked.
        "tau": _TRAIL_TAU,
        #: Spin accumulated over recent onsets, rad/s. Each beat nudges it up
        #: and it decays, so a run of hits winds the sky up and a quiet passage
        #: lets it settle back to the sidereal floor.
        "beat": 0.0,
        "rng": rng,
    }


@mode("Star Trails", group="cosmos",
      blurb="a long exposure: the sky turns, and the arcs it leaves are the show")
def star_trails(ctx: Ctx):
    """Astrophotography as a visualiser.

    The camera shutter is the accumulator: every frame stamps where the
    stars are now onto a field that fades over several seconds, so motion
    leaves arcs behind it the way light burns film. Nothing here is drawn as
    a shape — the streaks are simply where things have been, which is why
    the picture keeps its composure however busy the music gets: more drive
    means longer arcs, never more clutter. The pole sits near centre and the
    star field reaches the corners, so the trails fill the whole screen.

    The spin has several parts. A sidereal floor, so the sky turns whether or
    not anything plays (a frozen night sky is a poster); ``ctx.drive`` as
    the timelapse term; a beat-locked pulse shiver on the tempo; a kick per
    onset; and a ``beat`` term that builds while the hits keep coming, so a
    busiest stretch genuinely winds the sky up and a quiet one lets it coast
    back. One star does not move: the pole sits still while everything wheels
    round it, and is the anchor that makes the rotation read as rotation.

    As the sky spins faster it blurs: the accumulated exposure is smeared
    toward its neighbours in proportion to how fast it is turning, so a fast
    arc softens and merges into a smooth glow — and because the ramp is a
    function of brightness, the softened field reads as *blended* colours
    flowing together rather than as separate hard streaks. A slow sky turns
    the blur off entirely, so it stays a pin-sharp set of arcs.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("star_trails", lambda: _exposure(dr, dc))
    acc = st["acc"]

    if ctx.onsets:
        st["kick"] = min(0.5, st["kick"] + 0.35 * float(ctx.onset_strength))
        st["flare"] = min(1.0, st["flare"] + float(ctx.onset_strength))
        # Each hit adds a little to the standing spin, so a run of beats
        # winds the sky up rather than each one being a transient.
        st["beat"] = min(0.5, st["beat"] + _TRAIL_BEAT * (0.5 + ctx.drive))
    st["flare"] *= math.exp(-ctx.dt / 0.30)
    st["kick"] *= math.exp(-ctx.dt / 1.10)
    st["beat"] *= math.exp(-ctx.dt / 1.60)

    omega = (_TRAIL_OMEGA + _TRAIL_DRIVE * ctx.drive
             + _TRAIL_PULSE * ctx.pulse + st["kick"] + st["beat"])
    dth = omega * ctx.dt                # the sky turns; direction is taste
    st["th"] -= dth

    # Blur scales with how far the spin has left the sidereal floor: a still
    # sky stays sharp, a fast one smears. Applied to the accumulated field so
    # the exposure itself softens and the colours merge. The rate is per
    # second and ``dt`` turns it into this frame's share, so the amount of
    # smear an arc collects over its life is the same at 24 fps and at 240.
    blur = min(1.0, (_TRAIL_BLUR_BASE + _TRAIL_BLUR_GAIN
                     * max(0.0, omega - _TRAIL_OMEGA)) * ctx.dt)
    acc = _blur_merge(acc, blur)
    # The exposure is cut short in proportion to the spin, so the arc a star
    # sweeps before its mark fades stays just inside the gap to its
    # neighbour however fast the sky turns. Without this the mode has no
    # ceiling at all: omega rides past 0.6 rad/s on ordinary percussive
    # material, every star draws a 110-degree arc, and the exposure fuses
    # into the filled annulus the whole design exists to avoid.
    tau = min(_TRAIL_TAU, _TRAIL_FILL * st["gap"] / max(omega, 1e-6))
    st["tau"] = tau                     # kept so the ceiling is observable
    acc *= math.exp(-ctx.dt / tau)
    # The smear averages neighbours, so a bright dot can momentarily read
    # above 1.0; clip it so the ramp always gets a well-defined 0..1 norm and
    # no NaN/inf survives into the colour lookup.
    np.clip(acc, 0.0, 1.0, out=acc)
    lift = _sky_lift(ctx)

    # Stamp the arc the star swept this frame, not the point it stopped on.
    # A single point per frame is only continuous while a star moves less
    # than a dot between frames; at 24 fps the rim stars move two or three,
    # and the exposure records a dotted line instead of a trail. Sampling
    # the sweep finely enough that consecutive marks touch makes the picture
    # a property of the music rather than of the frame rate — at 24 fps the
    # sky was a third emptier than the same passage at 240.
    nsub = int(np.clip(math.ceil(dth * st["r_max"]), 1, _TRAIL_SUBSTEPS))
    # Each frame covers its own sweep down to, but not including, where the
    # next frame starts, so the marks tile the arc without doubling up.
    steps = 1.0 - np.arange(nsub, dtype=np.float32) / nsub
    th = st["th"][None, :] + dth * steps[:, None]
    cx = st["pole_x"] + st["r"][None, :] * np.cos(th)
    cy = st["pole_y"] + st["r"][None, :] * np.sin(th)
    px = np.rint(cx).astype(np.int32).ravel()
    py = np.rint(cy).astype(np.int32).ravel()
    val = np.broadcast_to(
        st["mag"] * (0.60 + 0.40 * lift) + 0.25 * st["flare"] * st["mag"],
        (nsub, st["mag"].size)).ravel()
    ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
    if ok.any():
        # ``maximum.at`` rather than ``acc[sel] = maximum(...)``: the sweep
        # puts several marks on one dot routinely, and plain fancy-index
        # assignment resolves a repeated index by last-write-wins, so a
        # bright star could be overwritten by a faint one landing after it.
        np.maximum.at(acc, (py[ok], px[ok]), np.clip(val[ok], 0.0, 1.0))

    # Polaris. Not subject to the fade: it is stamped onto the picture, not
    # exposed onto it, and its stillness is the reference everything else
    # moves against.
    pole_val = 0.75 + 0.25 * lift
    ix, iy = int(st["pole_x"]), int(st["pole_y"])
    # A slightly higher film threshold than the other skies: every star here
    # becomes a line, so the faintest tails are what turn a set of arcs into
    # haze, and they are the first thing to let go.
    dots = acc > 0.07
    if 0 <= iy < dr and 0 <= ix < dc:
        dots[iy, ix] = True
        acc[iy, ix] = max(acc[iy, ix], pole_val)

    codes = pack_braille(dots)
    cidx = _sky_colour(ctx, acc)
    return codes, cidx


#: Stars per dot cell behind Supernova. Sparse even by this file's standards:
#: the flash needs darkness to be violent, and the band below exists to give
#: the quiet stretches some depth without competing with it.
_NOVA_DENSITY = 1.0 / 300.0

#: Flash duration, shell lifetime, and remnant decay. The shell is the show
#: (~5.5 s); the core it leaves behind glows on for about twice that,
#: which is what makes the sky remember the event after it has passed.
_NOVA_FLASH_S = 0.30
_NOVA_SHELL_S = 5.5
_NOVA_LIFE_S = 12.0

#: A nova fires on a hard onset — but only sometimes, and only when the sky
#: is clear. Rarity is the entire effect: a supernova every bar is fireworks.
#: The threshold and draw are tuned so the hardest hits land a catastrophe
#: noticeably more often than the bare floor, without becoming predictable.
_NOVA_STRENGTH = 0.66
_NOVA_ODDS = 0.75


def _night(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(101)
    n = int(np.clip(dr * dc * _NOVA_DENSITY, 24, 900))
    # The band: a galactic plane through the field, dimmest thing in the
    # picture by design. It exists so the mode has structure before anything
    # happens and depth behind the shell when it does — pure black makes the
    # wait read as a broken mode rather than a held breath. Static on
    # purpose: it is geography, not weather. Two measured guards keep it a
    # suggestion rather than a wall: the falloff stays narrow (an earlier
    # sigma ate half the screen), and roughly half its dots are dropped to
    # grain — a solid stripe of lit braille reads as fog, not as sky.
    ang = rng.uniform(0.0, math.pi)
    cy, cx = dr / 2.0, dc / 2.0
    yy, xx = np.mgrid[0:dr, 0:dc].astype(np.float32)
    dperp = np.abs((xx - cx) * math.sin(ang) - (yy - cy) * math.cos(ang))
    sigma = 0.055 * min(dr, dc) + 1.5
    band = np.exp(-(dperp * dperp) / (2.0 * sigma * sigma))
    # Grain keeps under a fifth of the band's dots, over a narrower band. At
    # a half the band lit a quarter of every cell on screen, and in the live
    # terminal it was a slab across the sky rather than a suggestion of one.
    # Colour alone cannot push it back: on a theme like gruvbox even the
    # quietest colour on the ramp stands well clear of the background, so the
    # band has to recede by carrying less ink.
    grain = noise((dr, dc), 556) < 0.18
    band = np.where(grain, band * (0.35 + 0.65 * noise((dr, dc), 555)), 0.0)
    band = (band * 0.11).astype(np.float32)
    return {
        "sy": rng.integers(0, dr, n),
        "sx": rng.integers(0, dc, n),
        "mag": rng.uniform(0.15, 1.0, n) ** 2.2,
        "tw": np.ones(n, dtype=np.float32),
        "tw_tick": -1,
        "band": band,
        # The first idle event is a long way off. It used to be four to eight
        # seconds, so the mode's first catastrophe landed on its own clock —
        # in the middle of a pad, or of silence — before any hit had asked
        # for one, which is exactly the thing the mode exists not to do.
        "next": float(rng.uniform(25.0, 40.0)),
        "ev": None,
        "rng": rng,
    }


@mode("Supernova", group="cosmos",
      blurb="a mostly dark sky, waiting for the one hit worth a catastrophe")
def supernova(ctx: Ctx):
    """One event, and everything else waits.

    This is the family bargain taken to its limit. Shooting Star spends its
    budget on frequent small motions; Supernova spends all of it on a single
    occurrence — a hard hit tears a hole in the sky, a shell expands across
    it for five-odd seconds, and the core it leaves glows for several more
    while the field drifts back to sleep.

    Two guards keep it rare enough to mean something. Strength: only hits
    the detector scores above ``_NOVA_STRENGTH`` may fire one, and even then
    only win a draw. Concurrency: one event at a time — a second blast
    during the first shell would halve both, and the mode says nothing the
    rest of the time anyway. The idle clock that also fires one is a long
    fallback — tens of seconds — so a catastrophe answers a hit, not a timer.
    Between events the sky is not idle filler:
    variable stars twinkle on their own clock, and a faint galactic band
    gives the shell somewhere to expand across.

    The shell is ragged through stable per-event noise, so it reads as
    ejecta ploughing through uneven medium rather than a clean ring from a
    circle renderer — geometry supplies the circle, texture supplies the
    object.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("supernova", lambda: _night(dr, dc))
    field = np.where(st["band"] > 0.04, st["band"], np.float32(0.0))

    tw = _star_twinkle(st, ctx)
    lift = _sky_lift(ctx)
    field[st["sy"], st["sx"]] = np.clip(st["mag"] * tw * lift, 0.0, 1.0)

    # ── scheduling ──
    ev = st["ev"]
    hard = float(np.clip(ctx.onset_strength, 0.0, 1.0))
    if ev is None:
        due = ctx.t >= st["next"]
        asked = bool(ctx.onsets) and hard >= _NOVA_STRENGTH \
            and st["rng"].random() < _NOVA_ODDS
        if due or asked:
            cy0 = dr * float(st["rng"].uniform(0.18, 0.82))
            cx0 = dc * float(st["rng"].uniform(0.18, 0.82))
            yy, xx = np.mgrid[0:dr, 0:dc].astype(np.float32)
            dy, dx = yy - cy0, xx - cx0
            ev = {
                "t0": ctx.t,
                # Idle-scheduled events still carry real weight: a nova that
                # fires in a quiet passage at the bare floor would read as a
                # smudge, and the mode's whole promise is catastrophe.
                "s": max(hard, 0.55 + 0.25 * float(st["rng"].random())),
                "cy": cy0, "cx": cx0,
                "dy": dy, "dx": dx,
                "dist": np.sqrt(dx * dx + dy * dy),
                # Stable per event, so the shell's ragged holes travel with
                # the ring instead of boiling frame to frame — built once
                # here rather than per frame, which measured as most of the
                # mode's cost.
                "gate": noise((dr, dc), int(ctx.t * 1000.0) & 0xFFFF) < 0.72,
            }
            st["ev"] = ev
            # A busy passage invites the next event sooner: drive (attack)
            # shortens the idle, and a tempo-locked pulse does too, so a
            # groove that keeps landing beats fires more often than a held
            # drone without ever coming close to a fixed bar. Floor kept at
            # 0.20 so even the busiest stretch stays a wait, not a metronome.
            st["next"] = ctx.t + st["rng"].uniform(25.0, 45.0) \
                * max(0.35, 1.35 - ctx.drive - 0.35 * ctx.pulse)

    # ── the event ──
    if ev is not None:
        age = ctx.t - ev["t0"]
        if age > _NOVA_LIFE_S:
            st["ev"] = None
        else:
            s, dist = ev["s"], ev["dist"]

            # Flash: a core too bright to look at, with diffraction spikes
            # along both axes — the one place the mode allows itself theatre.
            f = max(0.0, 1.0 - age / _NOVA_FLASH_S) if age < _NOVA_FLASH_S else 0.0
            if f > 0.0:
                core = dist <= 1.5 + 11.0 * (age / _NOVA_FLASH_S)
                field[core] = np.maximum(field[core], min(1.0, (0.85 + 0.15 * s)))
                span = dc * 0.09 * f
                spike_h = (np.abs(ev["dx"]) < span) & (np.abs(ev["dy"]) < 0.9)
                spike_v = (np.abs(ev["dy"]) < span) & (np.abs(ev["dx"]) < 0.9)
                both = spike_h | spike_v
                field[both] = np.maximum(field[both], 0.70 * f * (0.6 + 0.4 * s))
                # The whole sky catches the light for a blink, held at one.
                np.minimum(field + 0.10 * f, 1.0, out=field)

            # Shell: expanding, thinning, ragged.
            if age < _NOVA_SHELL_S:
                frac = age / _NOVA_SHELL_S
                radius = 0.085 * min(dr, dc) * age
                width = max(1.1, 2.6 * (1.0 - frac))
                ring = np.abs(dist - radius) <= width
                val = s * (1.0 - frac) ** 1.4 * 0.95
                sel = ring & ev["gate"]
                field[sel] = np.maximum(field[sel], val)

            # Remnant: the core cools for three times the shell's life.
            glow = 0.9 * math.exp(-age / 6.0) * s
            if glow > 0.04:
                remnant = dist <= 2.2
                field[remnant] = np.maximum(field[remnant], glow)

    dots = field > 0.04
    codes = pack_braille(dots)
    cidx = _sky_colour(ctx, field)
    return codes, cidx
