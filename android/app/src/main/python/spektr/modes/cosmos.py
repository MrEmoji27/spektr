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
from . import Ctx, empty, mode

#: Fixed stars per dot cell. Sparse on purpose: a sky is mostly nothing, and
#: past a certain density the eye stops reading stars and starts reading haze.
_STAR_DENSITY = 1.0 / 260.0

#: Meteors in flight at once. A shower is not a barrage — more than a handful
#: on screen stops reading as "something rare just happened", which is the
#: only thing a shooting star has to say. Sized so a full beat cluster plus
#: the ambient strays fit without evicting anybody mid-flight.
_METEOR_CAP = 26


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
    st["acc"] += (0.10 + ctx.energy * 0.20) * ctx.dt
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
    beat = False
    if ctx.onsets:
        beat = bool(rng.random() < 0.05 + 0.25 * min(1.0, ctx.onset_strength))

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
            # Not from the radiant itself. A meteor only becomes visible some
            # way out from it, and spawning them all on one dot looks like a
            # leak rather than a shower.
            away = rng.uniform(0.05, 0.45, k) * min(dr, dc)
            st["my"][free] = ry + sa * away
            st["mx"][free] = rx + ca * away
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
            # Staggered along the shared line, sorted so the train runs head
            # to tail. Sideways scatter scales with the screen: fixed dots
            # vanish on a tablet and swamp a phone.
            away = np.sort(rng.uniform(0.06, 0.40, k) * min(dr, dc))
            jitter = rng.uniform(-1.0, 1.0, k) * min(dr, dc) * 0.02
            st["my"][free] = ry + sa * away + ca * jitter
            st["mx"][free] = rx + ca * away - sa * jitter
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
    cidx = ctx.ramp(cell_max(field))
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
_EDGE_TAU = 12.0
_EDGE_CAP = 26


def _chart(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(47)
    n = int(np.clip(dr * dc * _CHART_DENSITY, 30, 1200))
    return {
        "sy": rng.integers(0, dr, n),
        "sx": rng.integers(0, dc, n),
        "mag": rng.uniform(0.15, 1.0, n) ** 1.6,
        "tw": np.ones(n, dtype=np.float32),
        "tw_tick": -1,
        # Edges as [x0, y0, x1, y1, born, strength]; oldest trimmed past the
        # cap, dead ones dropped by age. A list because it holds a few dozen
        # tiny rows, and array bookkeeping would outgrow the data.
        "edges": [],
        #: Index of the star the next edge grows from. A constellation is a
        #: *chain* — each beat extends the last figure — not independent
        #: pairs flashing at random across the sky.
        "tail": None,
        "rng": rng,
    }


@mode("Constellations", group="cosmos",
      blurb="beats draw lines between fixed stars, then the figures fade")
def constellations(ctx: Ctx):
    """The music as a surveyor of the sky.

    Every beat connects two stars with a thin line, growing outward from
    wherever the previous beat stopped; the figures so drawn fade over about
    twelve seconds, so a quiet passage dissolves back into bare stars and a
    dense one builds a web. The stars themselves never move — the thing the
    mode accumulates is *structure*, which is what separates it from every
    other event mode here: Shooting Star answers a hit with motion, this one
    answers it with a mark that stays.

    Onset strength sets how bright the new line burns in, not whether one is
    drawn — admission is what the cap and the fade are for. A chain that runs
    out of neighbours restarts elsewhere rather than drawing one long jump,
    because a line across half the sky reads as an artifact, not a figure.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("constellations", lambda: _chart(dr, dc))
    field = np.zeros((dr, dc), dtype=np.float32)

    tw = _star_twinkle(st, ctx)
    lift = _sky_lift(ctx)
    field[st["sy"], st["sx"]] = np.clip(st["mag"] * tw * lift, 0.0, 1.0)

    # ── growth, one edge per beat ──
    if ctx.onsets:
        sy, sx = st["sy"], st["sx"]
        reach = min(dr, dc) * 0.30
        if st["tail"] is None:
            st["tail"] = int(st["rng"].integers(0, sy.size))
        else:
            t = st["tail"]
            dx = sx.astype(np.int32) - sx[t]
            dy = sy.astype(np.int32) - sy[t]
            near = np.flatnonzero((dx * dx + dy * dy <= reach * reach))
            near = near[near != t]
            if near.size:
                k = int(st["rng"].choice(near))
                st["edges"].append([
                    float(sx[t]), float(sy[t]), float(sx[k]), float(sy[k]),
                    ctx.t, float(np.clip(ctx.onset_strength, 0.0, 1.0)),
                ])
                st["tail"] = k
            else:
                # Nowhere to grow from here. Restart the chain at a random
                # star rather than leaping: the silence between the two marks
                # is what makes the next line read as a new figure instead
                # of a wire strung across the chart.
                st["tail"] = int(st["rng"].integers(0, sy.size))

    # ── fade and draw ──
    if st["edges"]:
        st["edges"] = [e for e in st["edges"] if ctx.t - e[4] < _EDGE_TAU]
        del st["edges"][:-_EDGE_CAP]
        for x0, y0, x1, y1, born, s in st["edges"]:
            age = ctx.t - born
            w = s * (1.0 - age / _EDGE_TAU) ** 1.3 * 0.85
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
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx


#: Exposure persistence, seconds. The accumulated buffer *is* the picture;
#: this tau decides how long an arc's tail stays on the film after the star
#: has moved on. It also decides whether the mode reads as arcs at all: the
#: angle a star sweeps while its mark survives is ``omega * tau``, and when
#: that exceeds the gap between neighbouring stars the arcs fuse into a
#: filled annulus — measured, not hypothetical. Tau keeps the sweep just
#: under typical spacing, so the trails stay lines.
_TRAIL_TAU = 3.2

#: Sidereal rate plus what percussion adds, rad/s. The base is a real sky
#: turning — slow enough that a still track still lives — and ``drive`` is
#: the timelapse knob: percussive material visibly spins the heavens up.
_TRAIL_OMEGA = 0.055
_TRAIL_DRIVE = 0.16

#: Star count scales with the *radius* available rather than the area: what
#: separates arcs from each other is angular spacing, and spacing comes from
#: how many stars share a circle, not from how big the screen is. Roughly
#: one star per fifteen dots of circumference at the rim.
_TRAIL_STAR_FROM_RADIUS = True


def _exposure(dr: int, dc: int) -> dict:
    rng = np.random.default_rng(73)
    # The celestial pole sits off-centre on purpose: centred, the trails form
    # concentric rings around mid-screen and the picture reads as a target.
    px = dc * 0.57
    py = dr * 0.36
    r_max = max(2.0, min(px, dc - 1 - px, py, dr - 1 - py) * 0.98)
    n = int(np.clip(2 * math.pi * r_max / 14.0, 18, 240))
    return {
        # Inner radius above zero: stars huddling on the pole draw circles so
        # small they stack into one bright blob beside Polaris, and the pole
        # star stops reading as the still point everything else rounds.
        "r": rng.uniform(0.14, 1.0, n) ** 0.85 * r_max,
        "th": rng.uniform(0.0, 2 * math.pi, n),
        # Skewed faint, harder than the other skies: a trail photo is a
        # handful of bright arcs over many barely-there ones, and every star
        # here becomes a line rather than a dot, so the dim end has to stay
        # near the film's threshold or the whole exposure turns to mush.
        "mag": rng.uniform(0.30, 1.0, n) ** 2.5,
        "pole_x": px,
        "pole_y": py,
        "acc": np.zeros((dr, dc), dtype=np.float32),
        "kick": 0.0,
        "flare": 0.0,
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
    means longer arcs, never more clutter.

    The spin has three parts. A sidereal floor, so the sky turns whether or
    not anything plays (a frozen night sky is a poster); ``ctx.drive`` as
    the timelapse term; and a kick per onset that decays in about a second,
    so a drum hit visibly lurches the heavens and lets them coast back. One
    star does not move: the pole sits still while everything wheels round
    it, and is the anchor that makes the rotation read as rotation.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 12 or dc < 16:
        return empty(ctx.w, ctx.h)

    st = ctx.scratch("star_trails", lambda: _exposure(dr, dc))
    acc = st["acc"]

    if ctx.onsets:
        st["kick"] = min(0.5, st["kick"] + 0.35 * float(ctx.onset_strength))
        st["flare"] = min(1.0, st["flare"] + float(ctx.onset_strength))
    st["flare"] *= math.exp(-ctx.dt / 0.30)
    st["kick"] *= math.exp(-ctx.dt / 1.10)

    omega = _TRAIL_OMEGA + _TRAIL_DRIVE * ctx.drive + st["kick"]
    st["th"] -= omega * ctx.dt          # the sky turns; direction is taste

    acc *= math.exp(-ctx.dt / _TRAIL_TAU)
    lift = _sky_lift(ctx)
    cx = st["pole_x"] + st["r"] * np.cos(st["th"])
    cy = st["pole_y"] + st["r"] * np.sin(st["th"])
    px = np.rint(cx).astype(np.int32)
    py = np.rint(cy).astype(np.int32)
    ok = (py >= 0) & (py < dr) & (px >= 0) & (px < dc)
    if ok.any():
        val = st["mag"][ok] * (0.60 + 0.40 * lift) + 0.25 * st["flare"] * st["mag"][ok]
        sel = (py[ok], px[ok])
        acc[sel] = np.maximum(acc[sel], np.clip(val, 0.0, 1.0))

    # Polaris. Not subject to the fade: it is stamped onto the picture, not
    # exposed onto it, and its stillness is the reference everything else
    # moves against.
    pole_val = 0.75 + 0.25 * lift
    ix, iy = int(st["pole_x"]), int(st["pole_y"])
    dots = acc > 0.04
    if 0 <= iy < dr and 0 <= ix < dc:
        dots[iy, ix] = True
        acc[iy, ix] = max(acc[iy, ix], pole_val)

    codes = pack_braille(dots)
    cidx = ctx.ramp(cell_max(acc))
    return codes, cidx


#: Stars per dot cell behind Supernova. Sparse even by this file's standards:
#: the flash needs darkness to be violent, and the band below exists to give
#: the quiet stretches some depth without competing with it.
_NOVA_DENSITY = 1.0 / 300.0

#: Flash duration, shell lifetime, and remnant decay. The shell is the show
#: (~5.5 s); the core it leaves behind glows on for about three times that,
#: which is what makes the sky remember the event after it has passed.
_NOVA_FLASH_S = 0.30
_NOVA_SHELL_S = 5.5
_NOVA_LIFE_S = 18.0

#: A nova fires on a hard onset — but only sometimes, and only when the sky
#: is clear. Rarity is the entire effect: a supernova every bar is fireworks.
_NOVA_STRENGTH = 0.72
_NOVA_ODDS = 0.65


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
    sigma = 0.07 * min(dr, dc) + 1.5
    band = np.exp(-(dperp * dperp) / (2.0 * sigma * sigma))
    grain = noise((dr, dc), 556) < 0.55
    band = np.where(grain, band * (0.35 + 0.65 * noise((dr, dc), 555)), 0.0)
    band = (band * 0.11).astype(np.float32)
    return {
        "sy": rng.integers(0, dr, n),
        "sx": rng.integers(0, dc, n),
        "mag": rng.uniform(0.15, 1.0, n) ** 2.2,
        "tw": np.ones(n, dtype=np.float32),
        "tw_tick": -1,
        "band": band,
        "next": float(rng.uniform(4.0, 8.0)),
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
    it for five-odd seconds, and the core it leaves glows for twenty more
    while the field drifts back to sleep.

    Two guards keep it rare enough to mean something. Strength: only hits
    the detector scores above ``_NOVA_STRENGTH`` may fire one, and even then
    only win a draw. Concurrency: one event at a time — a second blast
    during the first shell would halve both, and the mode says nothing the
    rest of the time anyway. Between events the sky is not idle filler:
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
            st["next"] = ctx.t + st["rng"].uniform(9.0, 20.0) * (1.25 - ctx.drive)

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
    cidx = ctx.ramp(cell_max(field))
    return codes, cidx
