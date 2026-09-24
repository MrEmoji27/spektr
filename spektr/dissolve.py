"""Dissolve one mode's frame into another's.

Shuffle used to cut. This mixes the two frames instead, dot by dot, in the
order :mod:`spektr.bluenoise` lays down: at progress ``p`` a dot belongs to the
incoming frame when its rank is below ``p``. Blue noise is what keeps that
even — with random ranks the change arrives in clumps and reads as tearing.

Before the dots swap, both pictures are moved onto each other: every column of
ink slides and stretches so its centre and spread land where the other's is,
and the columns themselves are shuffled across, so ink that moved sideways
travels across the frame instead of sliding up and down in place. Two
pictures the same shape as each other are what make the swap underneath read
as one picture becoming the other rather than as dots being replaced.

How the change gets there is a separate matter from where it is going, and it
is all in :class:`Style`: the travel sweeps across the frame rather than
setting off everywhere at once, lands a little past its target and settles,
leans out of the straight line on the way, and swaps its dots behind a
wavefront that the music has a say in. Each of those is a strength that can be
turned down to zero on its own, and the widget decides what each is worth for
the switch being made.

Frame kinds do not mix in one output. A braille frame is ``(codes, cidx)``,
where every cell is eight dots packed into one codepoint, so a cell can hold
dots from both frames at once and the dissolve happens inside it. Every other
frame is a single glyph per cell — a half-block or octant ``(codes, fg, bg)``,
or a bar mode's ``(codes, cidx)``, which is block glyphs and not braille at
all — and there the whole cell switches at once, which at the sizes these
modes run is still fine-grained enough to read as a dissolve rather than a
wipe. Two frames of different kinds never share a frame: the incoming picture
grows out of the outgoing one's shape instead (see :func:`_grow`).

Nothing here is per-frame expensive: the mask is cached, and each call is a
handful of whole-array operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import bluenoise
from .render import BRAILLE_BASE, BRAILLE_BITS, SPACE, pack_braille

#: How long a morph lasts. Long enough to be seen as a change of scene —
#: half a second reads as a glitch when the two modes look nothing alike —
#: and short enough that a fifteen-second shuffle still shows the mode.
SECONDS = 0.9

#: How long a morph between two modes of the same family lasts. Shorter,
#: because there is less to say: both are drawing the same kind of picture,
#: and the change is one of shape rather than of scene. A morph the user
#: asked for by hand runs for this long too, whatever the two modes are:
#: a keypress has to feel like it did something.
FAMILY_SECONDS = 0.5

#: How far the picture gathers in on itself at the half-way point, as a share
#: of its own height. The travel alone is invisible between two modes that
#: happen to fill the frame the same way; drawing in and blooming back out is
#: what makes a change of family read as one picture becoming another.
GATHER = 0.22

#: The share of the morph the shapes have to reach each other's geometry by.
#: The handover finishes later, so the colours land after the shapes have
#: stopped moving: dots swapping under a picture that is still travelling is
#: what the eye catches, and the last of the old ink is the louder change.
TRAVEL = 0.80

#: When the handover from old ink to new runs, as a share of the morph. The
#: first part is the old picture bending into the new shape; the swap runs
#: from there to the very end. It used to stop at 0.9 of an already eased
#: clock, and with the travel done by then too, the last third of every
#: cross-family morph was one frame held still.
HANDOVER = (0.22, 0.98)

#: What one beat adds to the travel, and the most a morph's beats can add
#: between them. A hit that lands mid-morph shoves the shape on towards the
#: incoming picture, so a change lands with the music rather than to its own
#: schedule. Only the travel: the handover stays on its own clock, or a beat
#: would swap the colours before the shapes had met.
PUSH = 0.06
PUSH_MAX = 0.25

#: The least a column map may move anything, as a share of the frame's width,
#: before it is worth moving at all. Two pictures whose ink already sits in
#: the same place come out with a map that shifts a column or so, which cannot
#: be seen — and a gather over the whole grid to do it can. Under a quarter of
#: a percent, and never under a whole column, the map is dropped.
SLACK = 0.0025

#: How far past the edge of the frame, as a share of the size of the grid
#: being moved, ink goes on drawing as it leaves. Nothing is drawn off the
#: frame, so without this a dot is dropped the instant it crosses the boundary
#: and the picture reads as clipped rather than as one being replaced.
EDGE = 0.04

#: Size of the mask tiled over the frame. 32 is enough that the pattern does
#: not repeat visibly at terminal sizes, and it builds in about half a second.
MASK = 32

# ── what gives a morph its life ─────────────────────────────────────────────
#
# Everything below is manner rather than content: none of it changes which
# picture the morph is heading for, only the way it gets there. Each one is a
# strength and each one can be taken out on its own by setting it to zero —
# that is the switch for keeping the ones that work and dropping the rest.
# They are read per call rather than bound as defaults, so a running app can
# be talked out of any of them without a restart.

#: How much of a morph the sweep across the frame takes: the leading column
#: sets off when the morph does, the trailing one this much later. Zero moves
#: every column together — the change happening everywhere at once, which is
#: what reads as mechanical however well it is timed.
SWEEP = 0.15

#: The share of that sweep a morph inside one family uses. The two pictures
#: are nearly the same shape already and its morph is the short one, where the
#: full delay would be most of the change.
SWEEP_FAMILY = 0.35

#: How far past its target the travel lands, as a share of the way it came,
#: before settling back onto it. Zero eases flat into place, which is correct
#: and lifeless: nothing with weight stops dead on the mark.
OVERSHOOT = 0.03

#: Where the travel has arrived, as a share of its own window, when there is a
#: spring in the landing. The rest of the window is the settle, so the shapes
#: come to rest on their target instead of reaching it at the last instant.
ARRIVE = 0.85

#: How far the picture leans sideways at the top of its flight, as a share of
#: the frame's width. Nothing at either end, so it starts and ends straight
#: and the ink is carried through an arc rather than slid down the screen.
SWAY = 0.02

#: The share of a morph the handover's wavefront takes to cross the frame.
#: Zero swaps dots everywhere at once — an even scatter, which the eye reads
#: as a flicker — and anything larger lets it watch the change travel.
WAVEFRONT = 0.15

#: How much the frame's own bands reorder a sweep, as a share of it either
#: way: a loud column sets off early, a quiet one waits, so the front runs
#: through the loud parts of the picture. Zero sweeps by position alone.
MUSIC = 0.5

# ── entrances ────────────────────────────────────────────────────────────────
#
# How the incoming picture arrives, by the family it belongs to. Every switch
# used to hand over in one order -- a blue-noise scatter behind a single
# left-to-right front -- so a bar mode arrived exactly the way a starfield or a
# fluid did. An entrance is an order field over the frame, 0 where the new
# picture lands first and 1 where it lands last, and the blue-noise rank still
# dithers every cell inside its own window, so the front is a soft band of
# mixed cells rather than a slideshow wipe.

#: The gesture each family arrives with. A family missing here arrives by the
#: plain sweep; ``tests/test_morph_entrances.py`` checks none is missing.
ENTRANCES = {
    "spectrum": "rise",     # bars grow up out of the floor
    "jp": "rise",
    "stereo": "rise",
    "terrain": "rise",      # a sea comes up to its horizon
    "lofi": "fall",         # rain, snow and embers come down from the top
    "scope": "draw",        # a trace is drawn across, the way a scope sweeps
    "particles": "burst",   # thrown out of the centre
    "cosmos": "burst",
    "fields": "ripple",     # rings running outward through the field
    "scenes": "dive",       # the view closes in from the edges
}
ENTRANCE_NAMES = ("rise", "fall", "draw", "burst", "ripple", "dive")

#: How much of the handover the front takes to cross the frame. The rest of
#: the window is each cell's own dither, which is how soft the front's edge
#: is. It was 0.6, and at the height of a morph two thirds of the frame was
#: a speckle of old and new cells, which is what read as messy: a gesture
#: needs an edge you can follow. At 0.85 the mixed band is about a sixth of
#: the frame -- soft enough not to be a slideshow wipe, narrow enough to be
#: one front moving.
ENTRANCE_FRONT = 0.85

#: How much of the morph a beat moves the front on by, per unit of ``push``.
#: The travel already takes the beats; this lets the front take them too, so a
#: hit mid-morph visibly throws the new picture further in.
ENTRANCE_PUSH = 0.5

#: Ripple's rings: how many run across the frame, and how deep each is as a
#: share of the whole order. Shallow enough that the rings ride on a front
#: moving outward rather than breaking it into separate bands.
RIPPLE_RINGS = 3.0
RIPPLE_DEPTH = 0.12

#: Between two kinds of frame, how far the incoming picture travels out of
#: the old shape on its own before the entrance takes over. All the way, and
#: it had reached its own form by 0.8 of the morph and the entrance had
#: nothing left to reveal; this far, the picture is still visibly in the old
#: shape when the front reaches it, and the front is what sets it free.
GROW_HOLD = 0.55


@dataclass(frozen=True)
class Style:
    """How a morph moves, as opposed to what it changes.

    Built once per morph and handed to every frame of it — the widget shortens
    the sweep for a switch inside one family and for a fast tempo, and passes
    the bands in as they are. A test wanting one of these out of the way turns
    that one to zero and leaves the others alone.
    """

    #: Share of the morph the sweep across the frame takes. 0: all at once.
    sweep: float = SWEEP
    #: Share of the morph the handover's wavefront takes to cross it.
    wavefront: float = WAVEFRONT
    #: How far past its target the travel lands, of the way it came.
    overshoot: float = OVERSHOOT
    #: How far the picture leans sideways in flight, of the frame's width.
    sway: float = SWAY
    #: The frame's bands, for the sweep to be reordered by. None: in order.
    levels: np.ndarray | None = None
    #: How much of the sweep those bands may reorder.
    music: float = MUSIC
    #: How the incoming picture arrives, one of :data:`ENTRANCE_NAMES`.
    #: None: the plain left-to-right sweep.
    entrance: str | None = None
    #: Whether the two pictures are bent onto each other's shape as they
    #: change. The widget turns it off: with the entrance's front doing the
    #: moving, two pictures also sliding and stretching underneath it was
    #: most of the cost of a morph and most of the mess. Off, the old picture
    #: holds still ahead of the front and the new one is itself behind it.
    travel: bool = True


def ease(p):
    """Smoothstep: start and finish gently, travel quickly in the middle.

    Takes an array as readily as a number: once the swap has a wavefront in it
    the handover has a threshold per column rather than one for the frame.
    """
    p = np.clip(p, 0.0, 1.0)
    return p * p * (3.0 - 2.0 * p)


def _curve(u, over: float = -1.0):
    """Where the travel has got to at ``u``, its own clock from 0 to 1.

    Eased, and with a little spring in the landing: the shapes are at their
    target by ``ARRIVE`` of the way through and then settle onto it from
    ``over`` past it, which is what anything with weight does. Zero ``over``
    is the plain ease, arriving at the last instant.

    ``over`` is how far past the target it lands, as a share of the way it
    came; below zero it is read from ``OVERSHOOT``, so that the constant stays
    a switch a running app can be talked out of.
    """
    over = OVERSHOOT if over < 0.0 else over
    if over <= 0.0:
        return ease(u)
    # Rise to the target early, then bow out and back. The sine squared
    # settles at both ends, so the picture leaves the rise and lands on the
    # target without a step in speed at either join.
    settle = np.clip((u - ARRIVE) / (1.0 - ARRIVE), 0.0, 1.0)
    return ease(np.minimum(u, ARRIVE) / ARRIVE) + over * np.sin(np.pi * settle) ** 2


def _sweep(cols: int, levels: np.ndarray | None = None,
           music: float = MUSIC) -> np.ndarray:
    """Where each column sits in a sweep: 0 leads, 1 trails.

    With ``levels`` — the bands the frame was drawn from — the sweep is
    reordered by what the music is doing, loud columns setting off before
    quiet ones, so the front runs through the loud parts of the picture. The
    levels are read against their own mean, which keeps the pattern at any
    volume and leaves a silent frame sweeping in order of position.
    """
    q = np.arange(cols, dtype=np.float32) / max(1, cols - 1)
    if levels is None or music <= 0.0 or len(levels) == 0:
        return q
    mean = float(levels.mean())
    if mean <= 1e-6:
        return q
    bands = len(levels)
    band = np.minimum((q * bands).astype(np.intp), bands - 1)
    loud = np.asarray(levels, dtype=np.float32) / mean
    return np.clip(q - music * (loud[band] - 1.0), 0.0, 1.0)


def _travel(progress: float, push: float = 0.0, sweep: float = 0.0,
            delay: np.ndarray | None = None, over: float = -1.0):
    """Progress through the shapes' move: done before the colours are.

    ``TRAVEL`` finishes the travel with the morph still going, and the
    handover runs past it, so the picture has stopped moving by the time the
    last dot of the new one lands. ``push`` — beats that landed earlier in the
    morph — moves the whole thing along; it is spent, not borrowed, so the
    travel never runs backwards except for the settle the overshoot puts at
    the end of it, which is the picture coming to rest rather than setting off
    again.

    ``delay`` is where each column sits in the sweep, 0 leading and 1
    trailing (None: one clock for the whole frame). The trailing column still
    lands by ``TRAVEL``, so however wide the sweep is the shapes are all in
    place before the handover is over; the columns ahead of it arrive early
    and wait.
    """
    if delay is None:
        return _curve(min(1.0, progress / TRAVEL + push), over)
    span = TRAVEL - sweep
    return _curve(
        np.clip((progress - sweep * delay) / span + push, 0.0, 1.0), over
    )


def _handover(progress: float, wavefront: float = 0.0,
              delay: np.ndarray | None = None):
    """Progress through the swap itself, held back until the ink has moved.

    ``delay`` gives each column the same head start the sweep gives it, so the
    swap crosses the frame behind the front rather than arriving everywhere at
    once. The trailing column keeps the plain window, so the colours still
    finish where they finished before.
    """
    lo, hi = HANDOVER
    if delay is None or wavefront <= 0.0:
        return ease((progress - lo) / (hi - lo))
    return ease((progress + wavefront * (1.0 - delay) - lo) / (hi - lo))


def _order(rows: int, cols: int, entrance: str, aspect: float,
           levels: np.ndarray | None = None, music: float = MUSIC) -> np.ndarray:
    """Where each cell sits in an entrance, 0 first to 1 last.

    ``aspect`` is how tall one unit of the grid is against how wide: a cell
    is about twice as tall as it is wide, a braille dot about as tall. The
    round entrances need it, or a burst from the centre is an ellipse lying
    on its side.

    The straight ones are reordered by the music the way the sweep is: a loud
    column rises (or falls, or is drawn) ahead of a quiet one, so the front is
    shaped by the frame's own spectrum rather than being a ruled line.
    """
    if entrance == "draw":
        return np.broadcast_to(_sweep(cols, levels, music)[None, :],
                               (rows, cols)).astype(np.float32)
    base = _order_base(rows, cols, entrance, aspect)
    if entrance in ("rise", "fall"):
        # How far the music moves each column's front, from the sweep's own
        # reordering of it: a loud column is ahead, a quiet one behind.
        plain = np.arange(cols, dtype=np.float32) / max(1, cols - 1)
        lead = _sweep(cols, levels, music) - plain
        return np.clip(base + 0.2 * lead[None, :], 0.0, 1.0)
    return base


@lru_cache(maxsize=16)
def _order_base(rows: int, cols: int, entrance: str, aspect: float) -> np.ndarray:
    """The part of an entrance that depends on the frame's size alone.

    Built once per size: at 400x100 in braille dots it was a sixth of every
    morph frame, rebuilt sixty times a second to the same answer.
    """
    y, x = np.indices((rows, cols), dtype=np.float32)
    if entrance in ("rise", "fall"):
        fy = y / max(1, rows - 1)
        return (1.0 - fy if entrance == "rise" else fy) * np.float32(0.8) + np.float32(0.1)
    dy = (y - (rows - 1) / 2.0) * aspect
    dx = x - (cols - 1) / 2.0
    d = np.hypot(dy, dx)
    d /= max(float(d.max()), 1e-6)
    if entrance == "burst":
        return d
    if entrance == "dive":
        return 1.0 - d
    if entrance == "ripple":
        ring = np.sin(d * np.float32(np.pi * 2.0 * RIPPLE_RINGS)) * np.float32(RIPPLE_DEPTH)
        return np.clip(d * np.float32(1.0 - RIPPLE_DEPTH) + ring
                       + np.float32(RIPPLE_DEPTH / 2), 0.0, 1.0)
    raise ValueError(f"unknown entrance {entrance!r}")


def _arrival(progress: float, style, shape: tuple[int, int], aspect: float,
             push: float = 0.0):
    """Per cell of ``shape``, the rank below which it has handed over.

    With an entrance, each cell has its own window inside the handover,
    placed by its order and ``ENTRANCE_FRONT`` of the way across it; without,
    the plain sweep along the columns.
    """
    rows, cols = shape
    if style.entrance is None:
        return _handover(progress, style.wavefront,
                         _sweep(cols, style.levels, style.music))
    lo, hi = HANDOVER
    order = _order(rows, cols, style.entrance, aspect, style.levels, style.music)
    span = hi - lo
    t = progress + push * ENTRANCE_PUSH - lo - span * ENTRANCE_FRONT * order
    return ease(t / (span * (1.0 - ENTRANCE_FRONT)))


def _rank(rows: int, cols: int) -> np.ndarray:
    """Per cell, its place in the reveal order, in ``[0, 1)``."""
    return bluenoise.tile(bluenoise.mask(MASK), rows, cols)


def _braille_dots(codes: np.ndarray) -> np.ndarray:
    """Unpack braille codepoints into a (rows*4, cols*2) boolean dot grid."""
    h, w = codes.shape
    bits = np.where(
        (codes >= BRAILLE_BASE) & (codes < BRAILLE_BASE + 256),
        codes - BRAILLE_BASE,
        0,
    ).astype(np.int32)
    dots = np.zeros((h * 4, w * 2), dtype=bool)
    for dy in range(4):
        for dx in range(2):
            dots[dy::4, dx::2] = (bits & int(BRAILLE_BITS[dy, dx])) != 0
    return dots


def _braille(codes: np.ndarray) -> bool:
    """Whether every glyph in a frame is a braille codepoint.

    A two-array frame is not braille by virtue of its shape: the bar modes
    draw block glyphs, one to a cell, in exactly the same ``(codes, cidx)``.
    Reading those as braille finds no dots at all, which would erase the
    outgoing picture on the first frame of a morph.
    """
    lit = codes != SPACE
    if not lit.any():
        return False  # nothing to read: a cell frame, so nothing is lost
    in_range = (codes >= BRAILLE_BASE) & (codes < BRAILLE_BASE + 256)
    return bool(np.all(in_range | ~lit))


def _kind(frame: tuple) -> tuple[int, bool]:
    """What two frames have to agree on to be mixed dot for dot."""
    return len(frame), len(frame) == 2 and _braille(frame[0])


def _ink(frame: tuple, dots: bool) -> np.ndarray:
    """A frame's ink, on the grid its picture is moved on."""
    if dots:
        return _braille_dots(frame[0])
    return frame[0] != SPACE


def _ink_onto(frame: tuple, kind: tuple[int, bool], dots: bool) -> np.ndarray:
    """A frame's ink, read on the grid another frame is drawn on.

    A cross-kind morph has to measure both pictures on one grid to compare
    their columns, and that grid is the incoming frame's: dots are the finer
    of the two, so cells blow up to fill them rather than the other way
    round, which would throw away the dots the incoming frame is made of.
    """
    _, braille = kind
    if dots:
        if braille:
            return _braille_dots(frame[0])
        return np.repeat(np.repeat(frame[0] != SPACE, 4, axis=0), 2, axis=1)
    if braille:
        return frame[0] != BRAILLE_BASE
    return frame[0] != SPACE


def _dot_count(dots: np.ndarray) -> np.ndarray:
    """Per cell, how many of its eight dots are set.

    Eight strided adds rather than a reshape and a two-axis reduction: the
    same elements either way, but reducing along contiguous axes instead of
    across them is what ``pack_braille`` is written that way for, and this
    runs on the dot grid twice per morph.
    """
    h, w = dots.shape
    total = np.zeros((h // 4, w // 2), dtype=np.uint8)
    for dy in range(4):
        for dx in range(2):
            total += dots[dy::4, dx::2]
    return total


def _profile(lit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each column's ink sits: how much, its centre, and its spread.

    Three numbers a column at a time is enough to move one picture onto
    another: a bar that is tall on the left and short on the right becomes a
    wave by sliding and stretching each column, which is what the eye reads as
    the picture changing shape rather than being replaced.

    The two moments come out of one matrix-vector product each rather than a
    broadcast multiply and a reduction: on the dot grid that is eight times
    fewer passes over 300,000 elements, and this runs twice per picture per
    morph. The spread is the mean square less the square of the mean, which at
    these sizes is stable to well under a row — and a column whose ink is one
    dot thick has no spread to lose anyway, since one dot-row is the floor.
    """
    rows = lit.shape[0]
    ink = lit.astype(np.float32)
    ys = np.arange(rows, dtype=np.float32)[:, None]
    count = ink.sum(axis=0)
    safe = np.maximum(count, 1.0)
    centre = np.dot(ys.T, ink)[0] / safe
    second = np.dot((ys * ys).T, ink)[0] / safe
    spread = np.sqrt(np.maximum(second - centre * centre, 0.0))
    # An empty column has no centre of its own; it keeps the grid's, so ink
    # arriving into it grows from the middle instead of leaping in from row 0.
    empty = count < 1.0
    centre = np.where(empty, (rows - 1) * 0.5, centre)
    # A single lit dot has no spread; one dot-row keeps the scale finite.
    spread = np.maximum(spread, 1.0)
    return count, centre.astype(np.float32), spread.astype(np.float32)


def _span(count: np.ndarray) -> tuple[float, float, bool]:
    """Where a picture's ink sits across the frame, and whether it has any.

    The same two numbers a column gets in y — centre and spread — taken once
    over the whole frame, so the picture can be slid and scaled across it as
    one. Per-column ink totals, summed up rather than matched column for
    column: matching the columns themselves means stretching the ink over
    every part of the other picture that has none, which reads as a smear
    where a picture whose mass moved sideways should read as one travelling.
    """
    total = float(count.sum())
    cols = count.size
    if total < 1.0:
        return (cols - 1) * 0.5, cols * 0.5, False
    xs = np.arange(cols, dtype=np.float64)
    centre = float((count * xs).sum() / total)
    var = float((count * (xs - centre) ** 2).sum() / total)
    # A picture one column wide has no spread; half a column keeps the scale
    # finite, the way one dot-row does for a column in y.
    return centre, max(float(np.sqrt(max(var, 0.0))), 0.5), True


def _across(cols: int, centre: float, spread: float,
            centre_t: float, spread_t: float) -> np.ndarray | None:
    """Source columns to move a picture's ink to ``centre_t``/``spread_t``.

    The x half of the same move the columns get in y: a picture whose mass
    sits on the left of the frame and one whose mass sits on the right travel
    across to meet, where matching column for column would only slide each
    one's ink up and down where it stands. Returns one source column per
    column out, or ``None`` when the map is the identity — which is the
    common case, since most modes spread their ink across the frame evenly
    and never need moving at all.
    """
    columns = np.arange(cols, dtype=np.float32)
    src = centre + (columns - centre_t) * (spread / spread_t)
    idx = np.rint(np.clip(src, 0.0, cols - 1.0)).astype(np.int32)
    if np.array_equal(idx, columns.astype(np.int32)):
        return None
    if float(np.abs(src - columns).max()) < max(1.0, SLACK * cols):
        return None
    return src[None, :]


def _reached(src_y: np.ndarray, src_x: np.ndarray | None,
             rank: np.ndarray | None) -> np.ndarray:
    """Which cells of a move land on the frame, and which slide off it.

    A cell whose source sits past an edge is not simply gone: it holds its
    place at the edge for a few rows' or columns' worth of travel, thinning
    out as it goes, which is what ink leaving a picture looks like. Dropped
    outright, the whole dot disappears on the frame it crosses the boundary
    and the picture reads as clipped rather than as one being replaced.
    """
    rows = src_y.shape[0]
    over = np.maximum(-src_y, src_y - (rows - 1.0)) / max(2.0, EDGE * rows)
    if src_x is not None:
        cols = src_x.shape[1]
        over = np.maximum(
            over, np.maximum(-src_x, src_x - (cols - 1.0)) / max(2.0, EDGE * cols)
        )
    if rank is None:
        return over <= 0.0
    # Only the cells that did leave are asked about their fade: on a full dot
    # grid a threshold for every cell would be a bigger array than the move.
    outside = np.flatnonzero((over > 0.0).ravel())
    keep = np.ones(src_y.shape, dtype=bool)
    keep.ravel()[outside] = over.ravel()[outside] <= rank.ravel()[outside]
    return keep


def _warp(arrays: tuple, centre: np.ndarray, spread: np.ndarray,
          centre_t: np.ndarray, spread_t: np.ndarray,
          across: np.ndarray | None = None,
          rank: np.ndarray | None = None) -> tuple:
    """Resample each cell so its column's ink sits at ``centre_t``/``spread_t``.

    ``across`` moves the columns sideways in the same pass — source columns
    from :func:`_across`, one per column out — and ``rank`` is the blue-noise
    grid whose values decide how long ink that has left the frame keeps
    drawing (see :func:`_reached`).

    Returns the moved arrays and the mask of cells the move actually reached.
    What to put outside that mask is the caller's business: blank for ink,
    but a colour or a glyph code has no zero that means "nothing", and filling
    those with 0 paints real ramp colour and invalid codepoints into the gap.
    """
    rows = arrays[0].shape[0]
    ys = np.arange(rows, dtype=np.float32)[:, None]
    src = centre + (ys - centre_t) * (spread / spread_t)
    idx = np.rint(np.clip(src, 0.0, rows - 1.0)).astype(np.int32)
    near = None
    if across is not None:
        near = np.rint(np.clip(across, 0.0, arrays[0].shape[1] - 1.0)).astype(np.int32)
    keep = _reached(src, across, rank)
    moved = []
    for a in arrays:
        out = np.take_along_axis(a, idx, axis=0)
        if near is not None:
            out = np.take_along_axis(out, near, axis=1)
        moved.append(out)
    return tuple(moved), keep


def _morph(old: tuple, new: tuple, lit_old: np.ndarray, lit_new: np.ndarray,
           progress: float, gather: float, push: float, rank: np.ndarray,
           style: Style) -> tuple:
    """Both pictures, each moved a share of the way to the other's shape.

    A column with ink on only one side has nothing to move towards, so it
    keeps its own geometry and simply fades. Without that, ink heading into an
    empty column gets squeezed onto a single row on its way out, which looks
    like the picture collapsing rather than changing.
    """
    n_old, c_old, s_old = _profile(lit_old)
    n_new, c_new, s_new = _profile(lit_new)
    lonely = (n_old < 1.0) | (n_new < 1.0)
    c_new = np.where(lonely, c_old, c_new)
    s_new = np.where(lonely, s_old, s_new)
    x_old, w_old, ink_old = _span(n_old)
    x_new, w_new, ink_new = _span(n_new)
    if not ink_old:  # neither has a shape to lend the other
        x_old, w_old = x_new, w_new
    elif not ink_new:
        x_new, w_new = x_old, w_old
    cols = n_old.size
    t = _travel(progress, push, style.sweep,
                _sweep(cols, style.levels, style.music), style.overshoot)
    # The gather and the sideways lean both belong to the flight, and the
    # flight is over by the time the travel passes its target: they ride the
    # travel up to 1 and stop there, so only the shapes settle.
    held = np.clip(t, 0.0, 1.0)
    c_t = c_old + (c_new - c_old) * t
    s_t = (s_old + (s_new - s_old) * t) * (
        1.0 - gather * np.sin(np.pi * held)
    )
    # Across the frame the two pictures move as one, on the average of the
    # column clocks: there is one width to share between them, so it cannot
    # sweep. The lean bows it out and back, which is what carries the ink
    # through an arc instead of down a straight line.
    mid = float(t.mean())
    in_flight = float(np.sin(np.pi * min(mid, 1.0)))
    x_t = x_old + (x_new - x_old) * mid + style.sway * cols * in_flight
    w_t = w_old + (w_new - w_old) * mid
    return (_warp(old, c_old, s_old, c_t, s_t,
                  _across(cols, x_old, w_old, x_t, w_t), rank),
            _warp(new, c_new, s_new, c_t, s_t,
                  _across(cols, x_new, w_new, x_t, w_t), rank))


def _settle(arrays: tuple, lit_old: np.ndarray, lit_new: np.ndarray,
            progress: float, gather: float, push: float, rank: np.ndarray,
            style: Style) -> tuple:
    """The incoming picture alone, on its way out of the outgoing one's shape.

    Where :func:`_morph` moves both pictures towards a shape between them,
    this leaves the incoming one where it belongs and starts it in the
    outgoing one's geometry — each column's centre and spread, and the ink's
    centre and spread across the frame — so a picture of a different kind can
    arrive as something growing rather than as a cut.
    """
    n_old, c_old, s_old = _profile(lit_old)
    n_new, c_new, s_new = _profile(lit_new)
    # A column with ink on one side only has no parting shape to settle out
    # of, so the incoming picture keeps its own there and simply arrives.
    lonely = (n_old < 1.0) | (n_new < 1.0)
    c_from = np.where(lonely, c_new, c_old)
    s_from = np.where(lonely, s_new, s_old)
    x_old, w_old, ink_old = _span(n_old)
    x_new, w_new, _ = _span(n_new)
    if not ink_old:
        x_old, w_old = x_new, w_new
    cols = n_new.size
    t = _travel(progress, push, style.sweep,
                _sweep(cols, style.levels, style.music), style.overshoot)
    held = np.clip(t, 0.0, 1.0)
    c_t = c_from + (c_new - c_from) * t
    s_t = (s_from + (s_new - s_from) * t) * (
        1.0 - gather * np.sin(np.pi * held)
    )
    mid = float(t.mean())
    in_flight = float(np.sin(np.pi * min(mid, 1.0)))
    x_t = x_old + (x_new - x_old) * mid + style.sway * cols * in_flight
    w_t = w_old + (w_new - w_old) * mid
    return _warp(arrays, c_new, s_new, c_t, s_t,
                 _across(cols, x_new, w_new, x_t, w_t), rank)


def _grow(old: tuple, new: tuple, kind_old: tuple[int, bool],
          kind_new: tuple[int, bool], progress: float, gather: float,
          push: float, style: Style) -> tuple:
    """The incoming picture, starting out in the outgoing picture's shape.

    Two frames of different kinds cannot share one, so past the first instant
    only one of them is drawn: the incoming one, which is where the change is
    going. It settles out of the outgoing picture's geometry as the morph
    runs, so the new picture grows out of the old shape instead of appearing
    beside it or cutting over at the half-way point.
    """
    rows, cols = new[0].shape
    cells_old = _ink_onto(old, kind_old, False)
    # With an entrance, the travel out of the old shape only goes part way:
    # the rest of the change is the front arriving (see GROW_HOLD).
    travel = progress * GROW_HOLD if style.entrance is not None else progress
    if kind_new == (2, True):
        # A braille frame is moved on two grids at once: the dots that make
        # the picture, and one colour per cell for where they are.
        dots_new = _ink(new, True)
        (moved,), _ = _settle((dots_new,), _ink_onto(old, kind_old, True),
                              dots_new, travel, gather, push,
                              _rank(*dots_new.shape), style)
        (moved_cidx,), keep = _settle((new[1],), cells_old,
                                      new[0] != BRAILLE_BASE, travel,
                                      gather, push, _rank(rows, cols), style)
        cidx = np.where(keep, moved_cidx, new[1])
        if style.entrance is None:
            return pack_braille(moved), cidx
        # Along the entrance, the picture lets go of the old shape and snaps
        # to its own: the dots that have arrived are the new frame's own.
        rank = _rank(*dots_new.shape)
        arrived = rank < _arrival(progress, style, rank.shape, 1.0, push)
        dots = np.where(arrived, dots_new, moved)
        cidx = np.where(_dot_count(arrived) >= 4, new[1], cidx)
        return pack_braille(dots), cidx
    moved, keep = _settle(new, cells_old, _ink(new, False), travel, gather,
                          push, _rank(rows, cols), style)
    blanks = (SPACE,) + new[1:]
    grown = tuple(np.where(keep, m, f) for m, f in zip(moved, blanks))
    if style.entrance is None:
        return grown
    rank = _rank(rows, cols)
    arrived = rank < _arrival(progress, style, rank.shape, 2.0, push)
    return tuple(np.where(arrived, n, g) for g, n in zip(grown, new))


def _reveal(old: tuple, new: tuple, kind_old: tuple[int, bool],
            kind_new: tuple[int, bool], progress: float, push: float,
            style: Style) -> tuple:
    """The new picture behind the front, the old one still ahead of it.

    No bending of either picture: the front is the whole of the motion. Two
    braille frames mix dot by dot, so the front's edge is a dither of single
    dots; anything else mixes cell by cell, in the incoming frame's kind --
    a braille cell and a block cell are both just a glyph and a colour, and
    ahead of the front a cell keeps whatever it was drawing.
    """
    if kind_old == kind_new == (2, True):
        dots_old = _braille_dots(old[0])
        dots_new = _braille_dots(new[0])
        rank = _rank(*dots_new.shape)
        arrived = rank < _arrival(progress, style, rank.shape, 1.0, push)
        codes = pack_braille(np.where(arrived, dots_new, dots_old))
        return codes, np.where(_dot_count(arrived) >= 4, new[1], old[1])
    rows, cols = new[0].shape
    rank = _rank(rows, cols)
    cells = rank < _arrival(progress, style, rank.shape, 2.0, push)
    if len(new) == len(old):
        return tuple(np.where(cells, n, o) for o, n in zip(old, new))
    codes = np.where(cells, new[0], old[0])
    fg = np.where(cells, new[1], old[1])
    if len(new) == 2:
        return codes, fg
    # The old frame had no background of its own: ahead of the front its
    # glyphs are drawn over the background the new picture is arriving with.
    return codes, fg, new[2]


def _refit(frame: tuple, shape: tuple[int, int]) -> tuple:
    """An old frame resampled onto a new cell shape, after a resize.

    Nearest neighbour, which for a codepoint or a ramp index is the only
    thing it could be: there is no average of two glyphs. It is the old
    picture stretched to the size the new one is drawn at, which then morphs
    into it — the picture is being redrawn to a new size either way, so the
    morph rides along with it rather than cutting to the incoming frame.
    """
    rows, cols = shape
    h, w = frame[0].shape
    ry = np.minimum((np.arange(rows) * h) // rows, h - 1)
    rx = np.minimum((np.arange(cols) * w) // cols, w - 1)
    return tuple(a[ry][:, rx] for a in frame)


def _dots(old: tuple, new: tuple, progress: float, gather: float,
          push: float, style: Style) -> tuple:
    """Both frames braille, so the dissolve happens inside the cells too."""
    codes_old, codes_new = old[0], new[0]
    rows, cols = codes_old.shape
    dots_old = _braille_dots(codes_old)
    dots_new = _braille_dots(codes_new)
    rank = _rank(rows * 4, cols * 2)
    ((moved_old,), keep_old), ((moved_new,), keep_new) = _morph(
        (dots_old,), (dots_new,), dots_old, dots_new, progress, gather, push,
        rank, style,
    )
    # Ink the move pushed off the frame, or faded out at the edge, is not
    # there any more: the mask says which cells the move reached at all.
    warped_old = moved_old & keep_old
    warped_new = moved_new & keep_new
    arrived = rank < _arrival(progress, style, rank.shape, 1.0, push)
    mixed = np.where(arrived, warped_new, warped_old)
    codes = pack_braille(mixed)
    # Colour travels with the shape: the ramp indices are moved by the same
    # per-column slide, judged on where each cell's ink is, and a cell takes
    # the incoming colour once most of its dots have arrived.
    cells_old = codes_old != BRAILLE_BASE
    cells_new = codes_new != BRAILLE_BASE
    ((moved_cidx_old,), cell_keep_old), ((moved_cidx_new,), cell_keep_new) = _morph(
        (old[1],), (new[1],), cells_old, cells_new, progress, gather, push,
        _rank(rows, cols), style,
    )
    # A cell the move did not reach keeps its own colour rather than
    # taking ramp index 0, which is a colour like any other.
    cidx_old = np.where(cell_keep_old, moved_cidx_old, old[1])
    cidx_new = np.where(cell_keep_new, moved_cidx_new, new[1])
    new_share = _dot_count(arrived)
    return codes, np.where(new_share >= 4, cidx_new, cidx_old)


def _cells(old: tuple, new: tuple, progress: float, gather: float,
           push: float, style: Style) -> tuple:
    """Frames of one glyph per cell: columns move, the handover is per cell."""
    codes_old, codes_new = old[0], new[0]
    rows, cols = codes_old.shape
    lit_old = codes_old != SPACE
    lit_new = codes_new != SPACE
    rank = _rank(rows, cols)
    (moved_old, keep_old), (moved_new, keep_new) = _morph(
        old, new, lit_old, lit_new, progress, gather, push, rank, style,
    )
    # Outside the move: a blank cell where the glyph was, and the cell's own
    # colour everywhere else.
    warped_old = tuple(np.where(keep_old, m, f)
                       for m, f in zip(moved_old, (SPACE,) + old[1:]))
    warped_new = tuple(np.where(keep_new, m, f)
                       for m, f in zip(moved_new, (SPACE,) + new[1:]))
    cells = rank < _arrival(progress, style, rank.shape, 2.0, push)
    return tuple(np.where(cells, n, o) for o, n in zip(warped_old, warped_new))


def blend(old: tuple, new: tuple, progress: float,
          gather: float = GATHER, push: float = 0.0,
          style: Style | None = None) -> tuple:
    """The frame part way from ``old`` to ``new``.

    ``progress`` is 0 at the start and 1 at the end, eased by the caller or
    by :func:`ease`.

    ``gather`` is how far the picture draws in on itself half way through.
    Pass 0 between two modes of the same family: there the two pictures are
    already the same kind of thing, both still reacting to the music, and
    gathering them would interrupt motion the eye is following.

    ``push`` is what beats landing during this morph have added to the travel
    so far, 0 to ``PUSH_MAX``: the shapes move on with the music, while the
    handover and the morph itself keep to their own clock.

    ``style`` is how it moves rather than what it changes — the sweep, the
    landing, the lean, the wavefront and the music's say in them. None is the
    house style; see :class:`Style` for taking any one of them out.

    Two frames of different kinds — a braille pair against a half-block or
    octant triple, a block-glyph pair against either — are not mixed at all;
    the incoming one is drawn in the outgoing one's shape and settles out of
    it. See the module docstring.
    """
    if progress <= 0.0:
        return old
    if progress >= 1.0:
        return new

    if style is None:
        style = Style()
    codes_old, codes_new = old[0], new[0]
    if codes_old.shape != codes_new.shape:
        # A resize part way through. Nothing can be blended against a frame of
        # another shape, but the morph does not have to die here: the outgoing
        # picture is resampled onto the new one's size and carries on, so the
        # change finishes as a morph at the size it is now being drawn at,
        # rather than cutting to the incoming frame on the frame it resized.
        old = _refit(old, codes_new.shape)

    kind_old, kind_new = _kind(old), _kind(new)
    if not style.travel:
        return _reveal(old, new, kind_old, kind_new, progress, push, style)
    if kind_old != kind_new:
        return _grow(old, new, kind_old, kind_new, progress, gather, push, style)
    if kind_new == (2, True):
        return _dots(old, new, progress, gather, push, style)
    return _cells(old, new, progress, gather, push, style)
