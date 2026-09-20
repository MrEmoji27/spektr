"""Full-field modes: waterfall, plasma, and level meters."""

from __future__ import annotations

import math

import numpy as np

from ..render import (
    cell_hilo,
    pack_octant_smooth,
)
from . import Ctx, empty, mode

_UPPER_HALF = ord("▀")


#: Resolution of the Kaleidoscope source patch — the little chamber of glass
#: the mirrors look at, sampled as a (radius, wedge-angle) table.
#:
#: ``_KAL_NU`` is the angular resolution and has to stay a power of two: the
#: per-dot gather masks with ``& (nu - 1)`` rather than taking a modulus. It
#: is sized against the fraction of the chamber one wedge actually shows
#: (:data:`_KAL_SECTOR_K`) rather than against the dot grid: a wedge spans
#: ``_KAL_SECTOR_K / k`` of the patch, so at the busiest mirror count it still
#: gets ``512 * 2 / 20`` = 51 samples, which is the number that has to beat the
#: dots competing for them.
_KAL_NU, _KAL_NR = 512, 128

#: How much of the chamber's circumference one mirror wedge looks at.
#:
#: Getting this wrong is what a first attempt at the rebuild did, and the
#: symptom was unmistakable: mapping the *whole* source disc onto one wedge
#: means a screen wedge sweeps all 2*pi of the glass, so every fragment is
#: compressed into a thin radial sliver and the rosette reads as a starburst
#: of rays rather than as pieces of anything. Fragments need an aspect ratio
#: close to one to read as fragments, and that is set here.
#:
#: Proportional to the wedge, not fixed, and the reasoning that made it fixed
#: was exactly backwards.
#:
#: It used to be a sixth of the circumference whatever the mirror count, on the
#: argument that the physically exact ``1 / k`` would thin the fragments every
#: time the spectrum pushed the count up. The opposite happens. A screen wedge
#: spans ``2*pi / k``; showing ``s`` of the glass inside it compresses the
#: source tangentially by ``s * k``, so a *fixed* sector squeezes harder the
#: more mirrors there are. Measured as the median piece's width against its
#: height on the rendered grid, cell aspect corrected — 1.0 being a chunk and
#: below it a radial sliver:
#:
#:     mirrors      fixed 1/6      2/k
#:           8           0.99     0.99
#:          12           0.78     0.92
#:          16           0.68     0.85
#:          20           0.63     0.83
#:
#: Tying the sector to the wedge is what the real object does, and it holds the
#: piece shape steady across the whole range the centroid can ask for. What a
#: high mirror count now costs is *how much* glass is in view — 91 pieces at 8
#: mirrors against 49 at 20 — which is the honest trade, and the one a real
#: tube makes: more mirrors, more repeats of less glass.
_KAL_SECTOR_K = 2.0

#: How many straight cuts shatter the glass, and how many different chambers
#: are kept ready to swap between.
#:
#: The cuts are what make this a kaleidoscope rather than an iris. See
#: :func:`_kal_glass` for the construction.
#:
#: The count is set by the *worst* wedge, not the average one. What a wedge
#: shows is a slice of the chamber, and with too few cuts the odds are decent
#: that a given slice at a given rotation falls almost entirely inside one
#: fragment — at which point the frame is a flat wash with a thin
#: figure in it. Swept over six chambers and twelve rotations, measuring how
#: much of the wedge the largest visible fragment takes:
#:
#:     cuts   fragments   visible/wedge   largest    worst case
#:       12          58            10.8       42%           89%
#:       16          98            16.1       33%           80%
#:       22         180            25.5       25%           48%
#:       32         368            45.9       17%           37%
#:
#: 22 was chosen off that table and it was still too few. The table measures
#: the largest fragment as a share of one *wedge*; what the eye judges is its
#: share of the *screen*, and by that measure 22 cuts put a third of the frame
#: inside a single piece of glass — one flat region with a few slivers around
#: it, which is what "I can't see the shapes" describes. Measured over the
#: rendered cell grid at 80x24, largest fragment as a share of the screen:
#:
#:     cuts   fragments   visible   largest   400x100 total   segments
#:       22         187        42    26-34%         7.3 ms       5200
#:       40         540       ~90    13-20%         9.7 ms       7200
#:       48         794       111    13-20%        10.0 ms       8700
#:
#: The count then has to be read together with :data:`_KAL_SECTOR_K`, because
#: what costs milliseconds is not how finely the glass is cut but how many
#: pieces land on screen, and the sector decides how much glass is in view.
#: Against a ``2/k`` sector, measured through the real mode over four terminal
#: sizes and two spectra — largest piece as a share of the frame, and total
#: cost at 400x100:
#:
#:     cuts   colours   largest   400x100
#:       28     18-25    16-31%    8.0 ms
#:       32     19-27    15-23%    7.8 ms
#:       36     19-28    17-21%    7.9 ms
#:       40     19-26    15-22%    8.9 ms
#:
#: 36 is where no piece dominates at *any* mirror count — 28 was fine at eight
#: mirrors and let a piece take a third of the frame at twenty, where the
#: sector is narrowest and the glass in view is magnified most. Past 40 the
#: average fragment drops below a few cells at ordinary terminal sizes, where
#: the reduction to half-blocks starts dropping pieces rather than drawing them.
_KAL_CUTS = 36
_KAL_CHAMBERS = 4

#: Brightness levels the picture is graded to, applied after the field has been
#: stretched over the range that is actually on screen.
#:
#: The old value was eight, applied *before* that stretch — so eighths of a
#: scale the picture only ever occupied the bottom quarter of. Three colours
#: survived out of sixty-four, and the tiling was invisible. Thirty-two after
#: the stretch merges only fragments the eye was not going to separate, which
#: is what keeps the strip builder's segment count near where it was.
_KAL_LEVELS = 32

#: Screen radius that maps to the rim of the chamber.
#:
#: The dot geometry normalises so ``r == 1`` at the nearer edge, which puts
#: the corners at ``sqrt(2)``. Mapping 0..1 onto the patch therefore clamped
#: everything outside the inscribed circle to the outermost ring of samples —
#: one fragment smeared across all four corners, which at 78x11 was most of
#: the frame and read as a flat surround with a small figure in it. Dividing
#: by the corner distance puts the whole terminal inside the glass at every
#: aspect ratio.
_KAL_RIM = 1.45

#: Where in the chamber the screen centre sits, as a source radius.
#:
#: Nonzero, and this is the last thing standing between the rebuild and the
#: iris it replaced. Sampling a pie slice that runs from the chamber's own
#: centre out to its rim means screen radius *is* source radius, so any
#: fragment spanning a range of source radii lands as an arc band and the
#: rosette organises itself into concentric rings — the exact read the
#: gaussians used to produce, arrived at from a different direction.
#:
#: Looking at an annular sector instead removes the shared centre: the glass
#: in view has no radial structure relative to the screen, because the point
#: everything is radial about is not in the picture.
#:
#: 0.20 rather than the 0.45 it started at. The core does two things at once
#: and they pull opposite ways — it keeps the singular centre of the source
#: disc out of frame, where every cut converges and the glass would shatter
#: into slivers too fine to draw, and it decides how much of the chamber's
#: radius is in view. At 0.45 only 0.55 of the radius was, magnified onto the
#: whole screen, and the magnification is what made a single piece cover a
#: sixth of the frame. Widening to 0.80 of the radius brings enough glass into
#: view that no piece dominates (largest 11-17% against 15-16%) and rounds the
#: pieces out, while 0.20 is still clear of the convergence at the centre.
_KAL_CORE = 0.20

def _kal_glass(seed: int) -> tuple[np.ndarray, int]:
    """One chamber of shattered glass, as a fragment index per patch sample.

    **This is the whole difference between a kaleidoscope and an iris**, and
    the previous version of this mode was the iris. It drew twelve soft 2-D
    gaussians at staggered radii inside the wedge, which has two consequences
    that between them name the shape: the blobs have no edges, so nothing
    reads as a *piece* of anything; and mirroring a radial chain of blobs
    around a circle stacks them into concentric rings, which is an aperture.
    No amount of retuning widths or radii escapes that — the arrangement is
    the problem.

    Real stained glass in a mirror tube is the opposite on both counts. The
    fragments are hard-edged, flat in colour, and they *tile*: they meet each
    other along straight seams and cover the whole field, with no privileged
    centre and no radial banding. So build exactly that.

    :data:`_KAL_CUTS` random lines are drawn across the source disc. Every
    sample records which side of each line it fell on, giving one bit per cut;
    samples sharing a code are on the same side of every cut, which is
    precisely the definition of one convex cell of the arrangement. Those cells are the fragments. The
    edges are hard because the code changes discontinuously at a line, which
    is free — there is no anti-aliasing to switch off and no width to tune.

    The cuts are placed in the *Cartesian* plane of the patch, not in
    ``(radius, angle)``, and that matters for a reason easy to miss: the
    sampler wraps the angular axis with ``& (nu - 1)`` once the spin is added,
    so the patch has to be periodic in u or there is a visible seam at the
    wrap. Laid out as chords of a disc, periodicity is automatic — a line in
    the plane is a closed curve in ``(r, phi)`` — where any pattern authored
    directly in u would have to be made periodic by hand.

    Returns the fragment index per sample and the fragment count. Both are
    functions of the seed alone, so chambers are built once at import and
    cost nothing at any terminal size.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(_KAL_NU, dtype=np.float32) + 0.5) * np.float32(1.0 / _KAL_NU)
    r = (np.arange(_KAL_NR, dtype=np.float32) + 0.5) * np.float32(1.0 / _KAL_NR)
    phi = u * np.float32(2.0 * math.pi)
    px = r[:, None] * np.cos(phi)[None, :]
    py = r[:, None] * np.sin(phi)[None, :]

    code = np.zeros((_KAL_NR, _KAL_NU), dtype=np.int32)
    for _ in range(_KAL_CUTS):
        th = float(rng.uniform(0.0, math.pi))
        # Offsets kept well inside the disc: a chord that grazes the rim
        # splits off a sliver too thin to survive the reduction to half-block
        # cells, and spends a fragment index on something invisible.
        d = float(rng.uniform(-0.62, 0.62))
        side = (px * np.float32(math.cos(th))
                + py * np.float32(math.sin(th))) > np.float32(d)
        code = (code << 1) | side

    # Dense-remap the sparse sign codes to 0..M-1 so the per-frame value array
    # is M long rather than 128 long with holes.
    uniq, flat = np.unique(code.ravel(), return_inverse=True)
    return flat.reshape(_KAL_NR, _KAL_NU).astype(np.int32), int(uniq.size)


#: The chambers, built once at import. Size-independent by construction, so
#: this is not scratch and never rebuilds — the whole point of authoring the
#: glass in patch space rather than on the dot grid.
_KAL_GLASS = [_kal_glass(0x6C1A55 + i) for i in range(_KAL_CHAMBERS)]

#: Per-fragment constants: which band lights a fragment, how much light it
#: passes, and where it sits in its own slow shimmer. Fixed, because a piece
#: of glass does not change colour — the light behind it changes.
#:
#: Sized from the chambers actually built rather than from ``1 << _KAL_CUTS``.
#: Forty cuts have an upper bound of a million million sign codes and produce
#: about three hundred and seventy regions; allocating for the bound is not
#: merely wasteful but impossible.
_KAL_MAX_CELLS = max(n for _, n in _KAL_GLASS)
_KAL_RNG = np.random.default_rng(0x91A55)
_KAL_FRAG_BAND = _KAL_RNG.integers(0, 8, _KAL_MAX_CELLS).astype(np.int32)
_KAL_FRAG_TONE = _KAL_RNG.uniform(0.30, 1.0, _KAL_MAX_CELLS).astype(np.float32)
_KAL_FRAG_PHASE = _KAL_RNG.uniform(0.0, 2.0 * math.pi, _KAL_MAX_CELLS).astype(np.float32)


def _kaleido(ctx: Ctx, cells: str):
    """A mirrored tube looking at a chamber of broken glass.

    Three modes share this body, differing only in how a text cell is filled.
    The geometry, the glass, the spin and the colour grading are identical in
    all three.

    ``"half"`` is the original: every cell is a two-colour ``▀`` pair, one
    colour per half-row — 1x2 subcells.

    ``"octant"`` draws each cell as one of 256 Unicode 16 octant glyphs — 2x4
    subcells at the same two colours and very nearly the same strip cost,
    because the extra detail rides in the glyph rather than in the colour runs
    the strip builder charges for.

    ``"ultra"`` is the same 2x4 grid, antialiased. Resolution is not what is
    left to win — 2x4 is the ceiling text offers, and the next step up is a
    raster protocol that costs 115 ms a frame to encode. What is left is the
    *threshold*: a subcell is on or off, so a fragment seam lands as a hard
    step whatever the resolution. :func:`render.octant_smooth` scatters that
    step into a stipple with an ordered threshold and colours each side of it
    by the mean of the subcells actually on that side, rather than by the
    cell's two extremes. The seams stop being staircases and the glass stops
    reading as pixels.

    Three parts, and they map one-to-one onto the physical object: a chamber
    of coloured fragments (:func:`_kal_glass`), a ring of mirrors around it,
    and the fact that turning the tube rotates the glass behind fixed mirror
    seams.

    **The mirrors.** A ring of 8, 12 or 16 of them, eased by the spectral
    centroid and snapped on a beat, each showing the same source slice
    reflected left-right alternately. The slice narrows as the count rises
    (:data:`_KAL_SECTOR_K`), which is what a real tube does and what keeps a
    piece of glass the same shape whether it is repeated eight times or sixteen. Every dot's angle is wrapped into its
    sector, then even sectors read the source forward and odd sectors read it
    reversed, so adjacent sectors mirror across the shared boundary and the
    picture is symmetric about every mirror line. Only a multiple of four puts
    a mirror line on the vertical axis, which is why the count snaps rather
    than running through every integer.

    The mirror lines are fixed in screen space; the spin rotates the *source*
    inside the wedge. The fold coordinate stays put and only the lookup shifts
    by the accumulated phase, so the picture stays bilaterally symmetric at
    every angle of rotation — a property that falls out of the construction
    rather than being close enough for the eye.

    That construction is the important part. The fold runs on a grid of
    absolute columns: the geometry is evaluated on ``|x|`` measured from the
    centre line between the two middle dot columns, so a dot and its L/R
    mirror compute exactly the same angle, radius and folded coordinate, and
    any function of those coordinates is bit-for-bit identical between the
    two. No averaging, no tolerance: the rendered frame is symmetric by
    construction. The gather then runs on the left half only and is mirrored
    for the right, which halves the per-dot path without changing a bit.

    **The glass.** Hard-edged convex fragments tiling the whole source disc,
    cut once at import and never rebuilt. What changes per frame is only how
    brightly each fragment is lit: one value per fragment, a few hundred of
    them, then a single gather turns that into the source table. That is a much
    smaller per-frame job than the twelve gaussians this used to evaluate over
    a 128x512 table, and it is the reason the mode now costs less than the
    version that looked worse.

    Each fragment is lit by one band, scaled by its own fixed transmittance
    and a slow shimmer on its own phase — a piece of glass does not change
    colour, the light behind it does. Flat colour within a fragment is not a
    simplification: it is what glass looks like, and it is also what the
    run-length encoder in the strip builder wants, so the look and the render
    cost agree for once.

    **Shaking the tube.** A beat swaps the whole chamber for a different one,
    round-robin through four cut at import. That is the one gesture a real
    kaleidoscope has that rotation cannot give you — the fragments tumble into
    a new arrangement — and swapping a precomputed index array costs nothing,
    where re-cutting the glass would cost a sort over the patch. The same beat
    kicks the rotation and snaps the sector count.

    The rotation rate is re-integrated through ``ctx.dt`` rather than read
    from ``ctx.t``, so pausing the audio locks the rotation in place without
    disturbing the phase. ``beat_phase`` drives a gentle brightening between
    hits — gated on ``tempo_bpm``, which is 0.0 until a tempo is established
    and takes the phase to 0.0 with it, i.e. a permanent on-the-beat swell if
    read ungated.

    Colour: every cell renders as a solid two-colour ``▀`` pair — the top half
    from the max over the cell's top two dot rows, the bottom half from its
    bottom two. The field is then stretched over the range that is actually in
    view and graded to :data:`_KAL_LEVELS`, in that order. Doing it the other
    way round is what made this mode unreadable: the values were quantised
    against an absolute scale they occupied the bottom quarter of, so three
    colours reached the screen out of sixty-four and the tiling — and the
    theme's gradient with it — was invisible.
    """
    dr, dc = ctx.dot_rows, ctx.dot_cols
    if dr < 8 or dc < 8:
        return empty(ctx.w, ctx.h)

    from ..render import frac, pack_octant

    # Geometry on the |x|-folded grid: dot (x, y) and its L/R mirror
    # (dc - 1 - x, y) compute identical turn and radius, so any function of
    # them is bit-for-bit symmetric. The pole sits on the boundary between the
    # two centre dot columns, so no dot straddles the mirror axis.
    #
    # Only the left half is kept — everything downstream reads the half and
    # mirrors the result — and the full-width brightness buffer rides in the
    # same entry, since it is per-size and has exactly this lifetime. The
    # audit caps a mode at four scratch keys.
    def geo():
        cx, cy = (dc - 1) / 2.0, (dr - 1) / 2.0
        hw = dc // 2
        x_scale = cy / max(cx, 1.0)
        ax = np.abs(np.arange(hw, dtype=np.float32) - np.float32(cx)) * np.float32(x_scale)
        ys = np.arange(dr, dtype=np.float32) - np.float32(cy)
        dx = ax[None, :]
        dy = ys[:, None]
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        ang = np.arctan2(dy, dx).astype(np.float32)
        ang = np.where(ang < 0, ang + np.float32(2 * math.pi), ang).astype(np.float32)
        turn = (ang / np.float32(2 * math.pi)).astype(np.float32)
        r = (dist / max(cy - 1.0, 1.0)).astype(np.float32)
        # Radius index into the source patch, fixed for the size. Clamped
        # rather than wrapped: past the rim of the glass there is no glass.
        rr = np.minimum(r * np.float32(1.0 / _KAL_RIM), np.float32(1.0))
        rr = rr * np.float32(1.0 - _KAL_CORE) + np.float32(_KAL_CORE)
        ir = np.minimum((rr * np.float32(_KAL_NR)).astype(np.int32), _KAL_NR - 1)
        return turn, ir, np.empty((dr, dc), dtype=np.float32)

    turn, ir, bright = ctx.scratch("kaleido_geo", geo)
    hw = dc // 2

    bands8 = ctx.display_bands(8).astype(np.float32)
    total = float(bands8.sum())
    centroid = float((bands8 * np.arange(8)).sum() / total / 7.0) if total > 1e-9 else 0.0
    bass = ctx.range(0.0, 0.18)

    # ``beat_phase`` is continuous and available between hits, where the onset
    # effects are not, so it carries the pulse on material the detector is
    # sparse about. Gated on the tempo, never on the phase.
    breathe = (1.0 - float(ctx.beat_phase)) ** 2 if ctx.tempo_bpm > 0.0 else 0.0

    st = ctx.scratch("kaleido", lambda: {
        "kc": 8.0, "spin": 0.0, "k": None, "folded": None,
        "chamber": 0, "shimmer": 0.0,
        # eased bounds of the visible field, for the stretch at the end
        "lo": 0.10, "hi": 1.0,
    })
    # ctx.onsets, not a private difference of ctx.onset_seq. Scratch survives
    # a mode switch, so differencing here would replay every beat that played
    # while the mode was not drawing, all in a single frame.
    onsets = ctx.onsets

    # The mirror count eases toward the centroid and snaps on a beat. It only
    # ever lands on a multiple of four — that is what keeps a mirror line on
    # the vertical axis — and the reachable set is 8, 12, 16, 20.
    st["kc"] += (8.0 + centroid * 10.0 - st["kc"]) * min(1.0, ctx.dt / 1.2)
    if onsets:
        st["kc"] = float(8.0 + 10.0 * min(1.0, bass * 1.5))
        # Shake the tube: the glass tumbles into a different arrangement.
        st["chamber"] = (st["chamber"] + onsets) % _KAL_CHAMBERS
    k = 4 * int(round(st["kc"] / 4.0))
    k = max(8, min(k, 20))

    # Spin moves the SOURCE inside the wedge; the mirror lines never move.
    st["spin"] = (st["spin"] + (0.22 + ctx.energy * 0.8) * max(ctx.dt, 0.0)) % (2 * math.pi)
    if onsets:
        st["spin"] = (st["spin"] + 0.3 * min(onsets, 3)) % (2 * math.pi)
    spin = st["spin"]
    st["shimmer"] = (st["shimmer"] + 1.7 * max(ctx.dt, 0.0)) % (2 * math.pi)

    # The fold: wrap the angle into its sector, then read even sectors forward
    # and odd sectors backward, so the array alternates orientation around the
    # ring exactly like physical mirror tubes. It depends only on the sector
    # count, which takes four values, so it is cached; the per-frame path is
    # just the phase shift and the gather below.
    #
    # The fold is pre-scaled by the sector while it is being cached, because the
    # sector is now a function of ``k`` and so changes at exactly the same
    # moments the fold does. That keeps the per-frame path one multiply-free
    # add, as it was when the sector was a constant.
    if st["k"] != k:
        wedge = turn * np.float32(k)
        m = np.floor(wedge).astype(np.int32)
        u = wedge - np.float32(m)
        u = np.where((m & 1) == 0, u, np.float32(1.0) - u)
        st["folded"] = (u * np.float32(min(0.5, _KAL_SECTOR_K / k))).astype(np.float32)
        st["k"] = k
    folded = st["folded"]

    # ── how brightly each fragment is lit ──
    # One value per fragment — a few hundred numbers — then a single gather
    # builds the whole source table. The old build evaluated twelve 2-D gaussians
    # over the table every frame; this is the same table for a fraction of the
    # arithmetic, and it comes out with hard edges instead of soft ones.
    cell_id, n_cells = _KAL_GLASS[st["chamber"]]
    band = _KAL_FRAG_BAND[:n_cells]
    tone = _KAL_FRAG_TONE[:n_cells]
    shim = 0.80 + 0.20 * np.sin(_KAL_FRAG_PHASE[:n_cells] + np.float32(st["shimmer"]))

    # Transmittance sets the fragment apart from its neighbour; the band decides
    # how hard the light behind it is pushed. The two were multiplied, which
    # means a fragment on a quiet band went to zero *whatever* its glass was
    # like — and with a pink spectrum most bands are quiet, so most fragments
    # collapsed onto the same value and the tiling disappeared into a wash.
    # Forty fragments were visible and eight colours were drawn. Keeping a
    # floor under the band term leaves every fragment separated by its own
    # glass at all times, and still lets the spectrum pick which ones flare.
    lit = (np.float32(0.35) + np.float32(0.65) * bands8[band] * shim) * tone

    # Spread the fragments across the ramp before anything else touches them.
    #
    # ``lit`` is a product of three factors that are each well under one most
    # of the time — a band level, a transmittance averaging 0.66, a shimmer —
    # so on ordinary material it lands in a narrow strip near the bottom of
    # 0..1. Scaling that by level and quantising it to eighths, which is what
    # this did, put 187 fragments into three colours spanning 24 of the 64 ramp
    # steps: the tiling was invisible, and so was the theme, because neither
    # end of its gradient was ever asked for. Normalising against the frame's
    # own range makes a fragment's colour mean "brighter than its neighbour"
    # rather than "some fraction of an absolute scale nothing reaches".
    #
    # A silent frame has no range to normalise — every fragment is unlit — and
    # falls through to a flat dark chamber, which is the correct picture for
    # it.
    lo = float(lit.min())
    span = max(float(lit.max()) - lo, 1e-4)
    spread = (lit - np.float32(lo)) * np.float32(1.0 / span)

    # Structure only — no level in here. Level arrives at the very end, after
    # the picture has been stretched over the ramp, so that dimming the rosette
    # cannot flatten it. The floor keeps unlit glass reading as glass rather
    # than as a hole: a dark fragment is a dark fragment, not an absence of one.
    val = np.float32(0.10) + np.float32(0.90) * spread

    table = val[cell_id]

    # The source is sampled on the LEFT half of the |x|-symmetric grid and
    # mirrored: dot (x, y) and its mirror gather the same table cell
    # bit-for-bit, so the right half is a reversed copy of the left. That
    # halves the frac, the index arithmetic and the gather without averaging,
    # and the rendered frame stays symmetric by construction.
    src = frac(folded + np.float32(spin / (2 * math.pi)))
    fu = src * np.float32(_KAL_NU)
    iu = fu.astype(np.int32) & (_KAL_NU - 1)
    flat = table.ravel()
    if cells == "ultra":
        # Read the source *between* table entries rather than snapping to one.
        #
        # This is what antialiasing a mirror tube actually needs, and it is not
        # what the dither in octant_smooth can do on its own: the glass is flat
        # within a fragment, so a cell straddling a seam holds exactly two
        # values, and for a two-valued cell every threshold in (0, 1) picks the
        # same subcells. The staircase is geometric — the boundary can only
        # land on a subcell edge — so no thresholding rule moves it.
        #
        # Interpolating along the angular axis puts a real gradient across the
        # seam, one table cell wide, and the ordered threshold downstream turns
        # that gradient into a coverage-proportional stipple. The boundary then
        # reads as falling *between* subcells. Angular only: seams in a mirror
        # tube run radially, so that is the axis they cross.
        t_u = fu - np.floor(fu)
        base = ir * _KAL_NU
        b_half = flat[base + iu]
        edge = b_half * (np.float32(1.0) - t_u) + flat[
            base + ((iu + 1) & (_KAL_NU - 1))
        ] * t_u
        soft = ctx.scratch("kaleido_soft", lambda: np.empty((dr, dc), dtype=np.float32))
        soft[:, :hw] = edge
        soft[:, hw:] = edge[:, ::-1]
    else:
        b_half = flat[ir * _KAL_NU + iu]
        soft = None
    bright[:, :hw] = b_half
    bright[:, hw:] = b_half[:, ::-1]

    # Two colour samples per text cell. Both reductions run on the same
    # |x|-symmetric dot grid, so both are symmetric bit for bit — every sample
    # equals its mirror.
    oct_codes = None
    if cells == "ultra":
        # Shape from the interpolated field, colour from the flat one. The
        # gradient exists to place the boundary between subcells; letting it
        # near the palette as well is what doubled the colour runs — 6,284 to
        # 12,102 at 400x100, and the frame with them.
        lo_cell, hi_cell = cell_hilo(bright)
        oct_codes = pack_octant_smooth(soft)
        top, bot = hi_cell, lo_cell
    elif cells == "octant":
        # The cell's range rather than two half-row maxima: the foreground
        # takes the brightest subcell and the background the darkest, and the
        # glyph says which of the eight subcells are on which side of the
        # midpoint. Four times the vertical detail of the half-block path for
        # one extra pass over the same array.
        lo_cell, hi_cell = cell_hilo(bright)
        top, bot = hi_cell, lo_cell
    else:
        # One colour per half-row: the top half is the max over the cell's top
        # two dot rows, the bottom half over its bottom two.
        top = np.maximum(bright[0::4, 0::2], bright[1::4, 0::2])
        top = np.maximum(top, bright[0::4, 1::2])
        top = np.maximum(top, bright[1::4, 1::2])
        bot = np.maximum(bright[2::4, 0::2], bright[3::4, 0::2])
        bot = np.maximum(bot, bright[2::4, 1::2])
        bot = np.maximum(bot, bright[3::4, 1::2])

    field = np.empty((2 * top.shape[0], top.shape[1]), dtype=np.float32)
    field[0::2] = top
    field[1::2] = bot

    # Stretch what is actually on screen over the ramp.
    #
    # A wedge sees a slice of the chamber across part of its radius — fifty to
    # ninety fragments of the several hundred, depending on the mirror count,
    # and no reason for those to span the full range the chamber does. Normalising the fragment vector alone therefore still left
    # five colours on screen out of sixty-four, most of the frame in one of
    # them. This is the same normalisation applied where the question is
    # settled: the half-row field, which is both the smallest array in the mode
    # and exactly what gets ramped.
    #
    # The bounds are eased rather than taken raw. A chamber swap or a fast spin
    # changes which fragments are visible between one frame and the next, and
    # rescaling instantly on that reads as the whole picture flinching.
    flo, fhi = float(field.min()), float(field.max())
    ease = min(1.0, max(ctx.dt, 0.0) * 3.0)
    st["lo"] += (flo - st["lo"]) * ease
    st["hi"] += (fhi - st["hi"]) * ease
    field -= np.float32(st["lo"])
    field *= np.float32(1.0 / max(st["hi"] - st["lo"], 0.05))
    np.clip(field, 0.0, 1.0, out=field)

    # Quantise *after* the stretch, not before it.
    #
    # The strip builder pays per colour boundary, and full grading over eight
    # hundred fragments costs about 8700 segments a frame at 400x100 against
    # 5200 for the old flat wash. Rounding here merges only fragments that are
    # already within a thirty-second of each other, which the eye was not going
    # to separate anyway, and it does so on the normalised range — where a
    # thirty-second means a thirty-second of what is *on screen*. That is the
    # difference from the old eighths, which were a thirty-second of a scale
    # the picture never reached.
    field *= np.float32(_KAL_LEVELS)
    np.round(field, 0, out=field)
    field *= np.float32(1.0 / _KAL_LEVELS)

    # Level, last: it scales a picture that already has its contrast, so a
    # quiet passage dims the rosette without collapsing it into one colour.
    field *= np.float32(
        (0.34 + 0.66 * min(1.0, ctx.energy * 1.7)) * (1.0 + 0.14 * breathe)
    )

    # No quantisation here. It used to happen on this grid, which is both more
    # work than quantising the fragments (above) and lossier: the reduction
    # from dots to half-rows can only ever return a value some fragment already
    # had, so rounding the grid rounds the same numbers again, one per cell
    # instead of one per fragment.
    #
    # No radial vignette either. The old version blended the cell radius in at
    # half weight, which is a concentric gradient laid over everything and one
    # of the two things making the mode read as an iris; the other was the
    # gaussians. Both are gone.
    idx = ctx.ramp(field)
    # The mask is taken from the raw dot grid, not the graded field: the
    # threshold is each cell's own midpoint, and every step between here and
    # there — stretch, quantise, level — is monotonic, so grading first would
    # produce the same eight bits after more arithmetic and one more chance to
    # lose them to a flat quantisation bucket.
    if oct_codes is not None:
        codes = oct_codes
    elif cells == "octant":
        codes = pack_octant(bright, lo_cell, hi_cell)
    else:
        codes = np.full((ctx.h, ctx.w), _UPPER_HALF, dtype=np.int32)
    return codes, idx[0::2], idx[1::2]


@mode("Kaleidoscope", group="fields",
      blurb="a mirrored tube of stained glass — the wedge count follows the spectrum, beats shake the chamber")
def kaleidoscope(ctx: Ctx):
    return _kaleido(ctx, "half")


@mode("Kaleidoscope (o)", hidden=True, after="Kaleidoscope", group="fields",
      blurb="the same tube at four times the vertical detail — needs a terminal that draws Unicode 16 octants")
def kaleidoscope_fine(ctx: Ctx):
    """Kaleidoscope on octant cells.

    Kept as a separate mode rather than a switch on the original because the
    glyphs are Unicode 16 (2024) and a terminal or font without them shows a
    grid of tofu — which is a thing to opt into, not something to discover
    when the original mode stops working. Everything else is identical, so the
    two can be compared directly by switching between them.
    """
    return _kaleido(ctx, "octant")


@mode("Kaleidoscope Ultra (o)", hidden=True, after="Kaleidoscope (o)", group="fields",
      blurb="the tube with its seams antialiased — the smoothest a terminal gets")
def kaleidoscope_ultra(ctx: Ctx):
    """Kaleidoscope with the staircase taken out of its seams.

    The same 2x4 subcells as Fine — that is the ceiling text offers, and the
    only thing past it is a raster protocol that costs 115 ms a frame to
    encode at this size. What Ultra removes is not a resolution limit but a
    quantisation one: a hard on/off threshold puts every fragment boundary on
    a subcell edge, and a boundary snapped to a grid is what the eye reads as
    pixels. An ordered threshold scatters it and a coverage-weighted colour
    softens it, so the seam falls *between* subcells as far as the eye is
    concerned.
    """
    return _kaleido(ctx, "ultra")
