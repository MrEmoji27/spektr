"""The JP family's music machines: four pieces of hardware the song plays.

Where :mod:`spektr.modes.jp` is the level meter on a head unit's faceplate,
these are the boxes on a producer's desk, each driven by what the 0.6.0
analysis hears rather than by the spectrum alone:

- ``JP Sequencer``, a clip launcher's pad grid, every drum playing its own
  light show across it;
- ``JP Chords``, a synth's chord display, naming the chord it hears;
- ``JP Panel``, a drum machine's front panel, the drums lighting the step
  they landed on;
- ``JP Tracker``, the song scrolling past as notes and drum lanes.

They keep the family's ink weights -- a lit bulb, a peak, a trail, an unlit
dot -- and its honesty about what it does not know: no step one is claimed
without a bar, no chord is named that does not fit, and without a beat the
lamps run dim on their own time.
"""
from __future__ import annotations

import numpy as np

from ..audio.drums import named
from ..render import SPACE
from . import Ctx, empty, mode
from .jp import _LED, _OFF, _PEAK, _TRAIL, recede_index, zone

#: Steps across a bar: sixteenths of four beats.
_STEPS = 16

#: The rows, top to bottom, and where each sits on the theme's ramp: the hats
#: hot at the top, the kick deep at the bottom, the way a mixer strip reads.
_ROWS = (("hat", 0.95), ("snare", 0.65), ("kick", 0.35))

#: Below this a row's level draws nothing under the playhead.
_LEVEL_FLOOR = 0.08

#: How sure the bar tracker has to be before step one is called one.
_SURE = 0.4

#: One bar of the playhead when there is no tempo at all, in seconds.
_IDLE_BAR_S = 2.0

#: How long a pad flashes after it is written, in seconds. A hard on and off
#: rather than a fade: a fade smeared the hit across the frames after it, and
#: the grid changed as much between the beats as on them.
_FLASH_S = 0.06


def _playhead(ctx: Ctx, st: dict) -> tuple[float, bool, bool]:
    """Where the playhead is across the bar, 0..1; whether that is the bar
    (step one is known); and whether there is a beat at all."""
    if ctx.bar_confidence >= _SURE and ctx.tempo_bpm > 0.0:
        return float(ctx.bar_phase) % 1.0, True, True
    if ctx.tempo_bpm > 0.0:
        # A beat but no bar: count four beats from wherever the count began,
        # and do not call any of them one.
        if ctx.beat_phase < st["last_beat"] - 0.5:
            st["count"] = (st["count"] + 1) % 4
        st["last_beat"] = ctx.beat_phase
        return (st["count"] + float(ctx.beat_phase)) / 4.0, False, True
    st["idle"] = (st["idle"] + max(ctx.dt, 0.0) / _IDLE_BAR_S) % 1.0
    return st["idle"], False, False


# ── shared by the modes below ────────────────────────────────────────────────

#: Pitch classes, C first, as ``ctx.chroma`` orders them.
_PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

#: A pitch class counts as a note when it is at least this share of the
#: loudest one, and a frame gives at most this many: the chord's notes, not
#: the haze of overtones round them.
_NOTE_SHARE = 0.6
_NOTE_MAX = 3

#: How sure the key has to be before a panel names it.
_KEY_SURE = 0.5


def _notes(ctx: Ctx) -> list[int]:
    """The pitch classes sounding now, loudest first; none in silence."""
    if ctx.silent:
        return []
    chroma = np.asarray(ctx.chroma, dtype=np.float32)
    top = float(chroma.max()) if chroma.size else 0.0
    if top <= 0.0:
        return []
    order = np.argsort(chroma)[::-1][:_NOTE_MAX]
    return [int(p) for p in order if chroma[p] >= _NOTE_SHARE * top]


def _pitch_heat(p: int) -> float:
    """Where pitch class ``p`` sits on the theme's ramp: C low, B high."""
    return 0.25 + 0.7 * p / 11.0


def _tonic(ctx: Ctx) -> int | None:
    """The key's tonic as a pitch class, when the key is known well enough."""
    if not ctx.key or ctx.key_uncertain or ctx.key_confidence < _KEY_SURE:
        return None
    name = ctx.key.split()[0]
    return _PITCHES.index(name) if name in _PITCHES else None


def _text(codes, cidx, row: int, col: int, s: str, colour: int) -> None:
    """Write ``s`` into the frame at ``row``, ``col``, clipped to the frame."""
    h, w = codes.shape
    if not 0 <= row < h:
        return
    for i, ch in enumerate(s):
        c = col + i
        if 0 <= c < w and ch != " ":
            codes[row, c] = ord(ch)
            cidx[row, c] = colour


# ── JP Sequencer: a grid of light-up pads ────────────────────────────────────

#: Pads a side, the way a clip launcher's grid is.
_PADS = 8

#: How fast a lit pad fades back, in seconds (to about a third). Long enough
#: that a show drags a short tail behind its front, short enough that the
#: grid is dark again before the next beat.
_PAD_TAU = 0.12

#: How fast a light show travels across the grid, in pads a second.
_SHOW_SPEED = 12.0

#: Light shows in flight at once. A new one with no free slot takes the oldest's.
_SHOWS = 16

#: How far each show travels before it is done, in pads.
_SHOW_REACH = {"ring": 5.5, "square": 4.5, "cross": 8.0, "x": 8.0, "wipe": 15.0}

#: A hit at least this hard sends its show twice, the second this far behind.
_DOUBLE_AT = 0.85
_DOUBLE_S = 0.1

_GRID_R, _GRID_C = np.mgrid[0:_PADS, 0:_PADS]
_MID = (_PADS - 1) / 2
_FROM_CENTRE = np.hypot(_GRID_R - _MID, _GRID_C - _MID)
_SQUARE = np.maximum(np.abs(_GRID_R - _MID), np.abs(_GRID_C - _MID))

#: The four corners a wipe can start from, taken in turn.
_CORNERS = ((0, 0), (0, _PADS - 1), (_PADS - 1, _PADS - 1), (_PADS - 1, 0))

#: Between the shows, the unlit pads glow faintly in rings from the centre,
#: the bass in the middle and the top end at the edge: the grid still shows
#: the music when nothing is hitting. Each pad's ring, and the level a ring
#: has to reach to glow.
_RING_OF = np.minimum(_FROM_CENTRE.astype(int), 3)
_AMBIENT = 0.45


def _show_front(kind: str, r: int, c: int, reach: float) -> np.ndarray:
    """The pads a show lights at ``reach`` pads from where it began."""
    if kind == "ring":
        return np.abs(_FROM_CENTRE - reach) < 0.75
    if kind == "square":
        return np.abs(_SQUARE - 0.5 - reach) < 0.5
    dr, dc = np.abs(_GRID_R - r), np.abs(_GRID_C - c)
    if kind == "cross":
        return ((dr == 0) | (dc == 0)) & (np.maximum(dr, dc) <= reach)
    if kind == "x":
        return (dr == dc) & (dr <= reach)
    # a wipe: a diagonal front from the corner at (r, c)
    return np.abs((dr + dc) - reach) < 0.75


@mode("JP Sequencer", group="jp",
      blurb="a grid of light-up pads: every drum plays its own light show across it")
def jp_sequencer(ctx: Ctx):
    """Eight by eight pads, and a light show for every hit.

    A clip launcher's pad grid, played by the drums it hears. Every hit sends
    a show across the grid, its front flashing hot and a short tail of the
    drum's colour behind it. A kick rings out from the centre, a snare
    throws a cross through a pad and the next one an X, a hat blinks a few
    pads, and a hit the drums cannot name wipes the grid from each corner in
    turn. A hard hit sends its show twice, and the first beat of every bar
    opens a square out from the middle. Lit pads fade fast, so a busy beat
    keeps the grid moving and a quiet one lets it go dark, the unlit pads
    glowing faintly with the spectrum.

    A row of lamps along the top runs across the bar in eighths.
    """
    rows, w = ctx.h, ctx.w
    gr = 1 if rows >= 2 + _PADS * 2 + (_PADS - 1) else 0
    pad_h = (rows - 2 - gr * (_PADS - 1)) // _PADS
    gc = 2 if w >= 60 else 1
    avail = w - gc * (_PADS - 1)
    # about square on a terminal's tall cells, as wide as there is room for
    pad_w = min(avail // _PADS, 2 * pad_h + 2)
    if pad_h < 1 or pad_w < 1:
        return empty(w, rows)

    st = ctx.scratch("jp_pads", lambda: {
        "glow": np.zeros((_PADS, _PADS)), "heat": np.zeros((_PADS, _PADS)),
        "shows": [], "rng": np.random.default_rng(8), "snares": 0, "wipes": 0,
        "last": 0.0, "count": 0, "last_beat": 0.0, "idle": 0.0,
    })
    before = st["last"]
    phase, known, beating = _playhead(ctx, st)
    st["last"] = phase
    st["glow"] *= np.exp(-max(ctx.dt, 0.0) / _PAD_TAU)
    glow, heat, rng = st["glow"], st["heat"], st["rng"]
    front = np.zeros((_PADS, _PADS), bool)

    def paint(mask, amp, h):
        on = mask & (amp >= glow)
        glow[on] = amp
        heat[on] = h
        front[mask] = True

    def launch(kind, r, c, amp, h, delay=0.0):
        if len(st["shows"]) >= _SHOWS:
            st["shows"].pop(0)
        st["shows"].append((kind, r, c, ctx.t + delay, amp, h))

    if ctx.onsets:
        amp = 0.55 + 0.45 * min(1.0, float(ctx.onset_strength))
        hit = named(ctx.drums or {})
        heats = dict(_ROWS)
        for name in hit or ["?"]:
            h = heats.get(name, 0.8)
            if name == "hat":
                n = 2 + int(2 * amp)
                mask = np.zeros((_PADS, _PADS), bool)
                mask.flat[rng.choice(_PADS * _PADS, n, replace=False)] = True
                paint(mask, amp, h)
                continue
            if name == "kick":
                kind, r, c = "ring", 0, 0
            elif name == "snare":
                kind = ("cross", "x")[st["snares"] % 2]
                st["snares"] += 1
                r, c = (int(v) for v in rng.integers(0, _PADS, 2))
            else:
                kind = "wipe"
                r, c = _CORNERS[st["wipes"] % 4]
                st["wipes"] += 1
            launch(kind, r, c, amp, h)
            if amp >= _DOUBLE_AT:
                launch(kind, r, c, amp * 0.8, h, _DOUBLE_S)
    if beating and not ctx.silent and phase < before - 0.5:
        # the first beat of the bar
        launch("square", 0, 0, 1.0, 1.0)

    alive = []
    for show in st["shows"]:
        kind, r, c, born, amp, h = show
        reach = (ctx.t - born) * _SHOW_SPEED
        if reach > _SHOW_REACH[kind]:
            continue
        if reach >= 0.0:
            paint(_show_front(kind, r, c, reach), amp, h)
        alive.append(show)
    st["shows"] = alive

    codes = np.full((rows, w), SPACE, dtype=np.int32)
    recede = recede_index(ctx.palette)
    cidx = np.full((rows, w), recede, dtype=np.int32)
    hot = int(np.asarray(ctx.ramp(np.float32(1.0))))
    # a tail cools down the ramp as it fades, from the drum's colour
    colour = np.asarray(ctx.ramp((heat * (0.35 + 0.65 * glow)).astype(np.float32)))
    rings = np.array([ctx.range(k / 4, (k + 1) / 4) for k in range(4)])
    ambient = (rings[_RING_OF] > _AMBIENT) & (not ctx.silent)
    grid_w = _PADS * pad_w + gc * (_PADS - 1)
    grid_h = _PADS * pad_h + gr * (_PADS - 1)
    x0 = (w - grid_w) // 2
    y0 = 2 + (rows - 2 - grid_h) // 2

    for r in range(_PADS):
        top = y0 + r * (pad_h + gr)
        for c in range(_PADS):
            left = x0 + c * (pad_w + gc)
            cells = (slice(top, top + pad_h), slice(left, left + pad_w))
            g = glow[r, c]
            if front[r, c] or g > 0.55:
                codes[cells] = _LED
            elif g > 0.25:
                codes[cells] = _PEAK
            elif g > 0.08:
                codes[top + pad_h - 1, left:left + pad_w] = _TRAIL
            elif ambient[r, c]:
                codes[top + pad_h - 1, left:left + pad_w] = _TRAIL
                continue
            else:
                codes[top + (pad_h - 1) // 2, left + pad_w // 2] = _OFF
                continue
            # the show's front flashes hot; its tail is the drum's colour
            cidx[cells] = hot if front[r, c] else int(colour[r, c])

    # the lamps across the top: a light running across the bar in eighths
    eighth = int(phase * _PADS) % _PADS
    warm = int(np.asarray(ctx.ramp(np.float32(0.8))))
    for c in range(_PADS):
        mid = x0 + c * (pad_w + gc) + pad_w // 2
        here = c == eighth
        codes[y0 - 2, mid] = (_LED if beating else _TRAIL) if here else _OFF
        cidx[y0 - 2, mid] = (hot if c == 0 and known else warm) if here else recede
    return codes, cidx


# ── JP Chords: a synth's chord display ───────────────────────────────────────

#: The chords it can name, simplest first so a tie goes to the plainer name:
#: the suffix and the notes above the root, in semitones.
_QUALITIES = (
    ("", (0, 4, 7)), ("m", (0, 3, 7)), ("5", (0, 7)),
    ("7", (0, 4, 7, 10)), ("m7", (0, 3, 7, 10)), ("maj7", (0, 4, 7, 11)),
    ("dim", (0, 3, 6)), ("sus4", (0, 5, 7)),
)


def _templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows, names = [], []
    for suffix, steps in _QUALITIES:
        for root in range(12):
            t = np.zeros(12, np.float32)
            t[[(root + s) % 12 for s in steps]] = 1.0
            rows.append(t / np.linalg.norm(t))
            names.append((root, suffix))
    return np.stack(rows), names


_CHORD_T, _CHORD_NAMES = _templates()

#: How well the notes have to fit a chord before it is named, as the cosine
#: between what is heard and the chord's notes. Below it the display says
#: no chord, which is what a chord display says over a drum break.
_CHORD_FIT = 0.75

#: A new chord has to win for this long, in seconds, before the display
#: changes: a passing note is not a chord change.
_CHORD_HOLD_S = 0.35

#: How quickly the heard notes are smoothed, in seconds.
_CHROMA_TAU = 0.2

#: How much better a new chord has to fit than the one showing before it can
#: take over, so two near-equal readings do not trade places.
_CHORD_MARGIN = 0.03

#: How long a new chord shows hot before it settles into its colour, in seconds.
_CHORD_NEW_S = 0.25

#: Chords kept in the line at the foot.
_HISTORY = 4

#: The display's letters: a small pixel font, drawn in half blocks. Widths
#: vary; every glyph is five pixels tall.
_FONT = {
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": (".##", "#..", "#.#", "#.#", ".##"),
    "N": ("#..#", "##.#", "#.##", "#..#", "#..#"),
    "#": ("#.#", "###", "#.#", "###", "#.#"),
    "m": (".....", ".....", "####.", "#.#.#", "#.#.#"),
    "a": ("...", "...", ".##", "#.#", ".##"),
    "j": ("..#", "...", "..#", "..#", "##."),
    "d": ("..#", "..#", ".##", "#.#", ".##"),
    "i": ("#", ".", "#", "#", "#"),
    "s": ("...", ".##", "#..", "..#", "##."),
    "u": ("...", "...", "#.#", "#.#", ".##"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "##.", "..#", "##."),
    "7": ("###", "..#", ".#.", ".#.", ".#."),
    ".": (".", ".", ".", ".", "#"),
}

#: The widest name the display has to fit, so the letters keep one size
#: whatever chord is showing.
_WIDEST = "C#maj7"

#: The keyboard: which pitch classes are white keys, and which white key
#: each black key sits after.
_WHITE = (0, 2, 4, 5, 7, 9, 11)
_BLACK = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}


def _bitmap(text: str) -> np.ndarray:
    """``text`` in the display font, one pixel a cell, a pixel's gap between."""
    cols = []
    for i, ch in enumerate(text):
        g = np.array([[p == "#" for p in row] for row in _FONT[ch]], bool)
        if i:
            cols.append(np.zeros((5, 1), bool))
        cols.append(g)
    return np.hstack(cols) if cols else np.zeros((5, 0), bool)


def _half_blocks(bm: np.ndarray) -> np.ndarray:
    """A bitmap as half-block characters, two pixel rows to a screen row."""
    if bm.shape[0] % 2:
        bm = np.vstack((bm, np.zeros((1, bm.shape[1]), bool)))
    top, bot = bm[0::2], bm[1::2]
    out = np.full(top.shape, SPACE, np.int32)
    out[top & bot] = ord("█")
    out[top & ~bot] = ord("▀")
    out[~top & bot] = ord("▄")
    return out


def _chord_name(root: int, suffix: str) -> str:
    return _PITCHES[root] + suffix


@mode("JP Chords", group="jp", after="JP Sequencer",
      blurb="a synth's chord display: the chord it hears in big letters, lit on a keyboard, and the ones before it")
def jp_chords(ctx: Ctx):
    """The chord the music is playing, the way a synth's display shows it.

    The notes heard (``ctx.chroma``) are matched against the chords it knows
    -- major, minor, sevenths, power chords, diminished and suspended -- and
    the best fit is named in big letters, in the colour of its root. A chord
    has to hold for a moment, and fit clearly better than the one showing,
    before the display changes, so a passing note is not a chord change; over
    a drum break or anything without a clear chord it says N.C., no chord. A
    new chord arrives lit hot and settles into its colour.

    The name sits on a line that breathes with how loud the music is and
    brightens on every hit. Under it, a keyboard lights the chord's notes, the
    root hottest, and under that the last few chords and the key.
    """
    rows, w = ctx.h, ctx.w
    widest = _bitmap(_WIDEST).shape[1]
    kb_rows = 3 if rows >= 20 else 2
    room = rows - kb_rows - 6
    scale = max(1, min((w - 4) // widest, (room * 2) // 5))
    big = (5 * scale + 1) // 2
    kw = max(2, min(7, (w - 10) // 7))
    if room < 3 or w < widest + 4 or 7 * (kw + 1) - 1 > w:
        return empty(w, rows)

    st = ctx.scratch("jp_chords", lambda: {
        "chroma": np.zeros(12, np.float32), "chord": None, "since": 0.0,
        "cand": None, "hist": [], "hit": -99.0, "changed": -99.0,
    })
    dt = max(ctx.dt, 0.0)
    heard = np.zeros(12, np.float32) if ctx.silent else np.asarray(ctx.chroma, np.float32)
    k = 1.0 - np.exp(-dt / _CHROMA_TAU) if dt else 1.0
    st["chroma"] += (heard - st["chroma"]) * np.float32(k)
    c = st["chroma"]
    norm = float(np.linalg.norm(c))
    best = None
    if norm > 0.05:
        fit = _CHORD_T @ (c / norm)
        i = int(np.argmax(fit))
        if fit[i] >= _CHORD_FIT:
            best = _CHORD_NAMES[i]
            now = st["chord"]
            if now is not None and best != now \
                    and fit[i] < fit[_CHORD_NAMES.index(now)] + _CHORD_MARGIN:
                best = now      # not clearly better than what is showing
    if best != st["chord"]:
        if best != st["cand"]:
            st["cand"], st["since"] = best, ctx.t
        if ctx.t - st["since"] >= _CHORD_HOLD_S or st["chord"] is None:
            if st["chord"] is not None and st["hist"][-1:] != [st["chord"]]:
                st["hist"] = (st["hist"] + [st["chord"]])[-_HISTORY:]
            st["chord"] = best
            if best is not None:
                st["changed"] = ctx.t
    else:
        st["cand"] = best
    if ctx.onsets:
        st["hit"] = ctx.t

    codes = np.full((rows, w), SPACE, dtype=np.int32)
    recede = recede_index(ctx.palette)
    cidx = np.full((rows, w), recede, dtype=np.int32)
    hot = int(np.asarray(ctx.ramp(np.float32(1.0))))
    colours = [int(np.asarray(ctx.ramp(np.float32(_pitch_heat(p))))) for p in range(12)]
    kb_w = 7 * (kw + 1) - 1
    y = (rows - (big + kb_rows + 6)) // 2
    name_top = y
    line_row = name_top + big
    kb_top = line_row + 2
    labels, foot = kb_top + kb_rows, kb_top + kb_rows + 2
    kx = (w - kb_w) // 2

    # ── the chord's name, in big letters ─────────────────────────────────────
    chord = st["chord"]
    text = _chord_name(*chord) if chord else "N.C."
    bm = _bitmap(text).repeat(scale, axis=0).repeat(scale, axis=1)
    glyphs = _half_blocks(bm)
    gh, gw = glyphs.shape
    x = (w - gw) // 2
    ink = glyphs != SPACE
    codes[name_top:name_top + gh, x:x + gw][ink] = glyphs[ink]
    fresh = ctx.t - st["changed"] < _CHORD_NEW_S
    colour = recede if chord is None else (hot if fresh else colours[chord[0]])
    cidx[name_top:name_top + gh, x:x + gw][ink] = colour

    # ── the line under it: as wide as the music is loud, bright on a hit ─────
    level = 0.0 if ctx.silent else min(1.0, float(ctx.range(0.0, 1.0)) * 1.6)
    half = round(level * kb_w / 2)
    if half:
        mid = kx + kb_w // 2
        codes[line_row, mid - half:mid + half] = ord("▔")
        # a meter spreading from the middle, cool at the centre and hot at
        # its ends, all of it hot for a moment on a hit
        out = np.abs(np.arange(-half, half) + 0.5) / (kb_w / 2)
        hit = ctx.t - st["hit"] < _FLASH_S * 2
        cidx[line_row, mid - half:mid + half] = hot if hit else np.asarray(
            ctx.ramp(zone(out)))

    # ── the keyboard ─────────────────────────────────────────────────────────
    tones = set()
    if chord:
        root, suffix = chord
        tones = {(root + s) % 12 for s in dict(_QUALITIES)[suffix]}
    upper = kb_rows - 1
    bw = max(1, kw // 2) | 1

    def key_ink(p: int, unlit: int) -> tuple[int, int]:
        if p in tones:
            return ord("█"), hot if p == chord[0] else colours[p]
        return unlit, recede

    centres = {}
    for i, p in enumerate(_WHITE):
        left = kx + i * (kw + 1)
        ch, col = key_ink(p, ord("░"))
        codes[kb_top:kb_top + kb_rows, left:left + kw] = ch
        cidx[kb_top:kb_top + kb_rows, left:left + kw] = col
        centres[p] = left + kw // 2
    for p, after in _BLACK.items():
        mid = kx + (after + 1) * (kw + 1) - 1
        left = mid - bw // 2
        ch, col = key_ink(p, SPACE)
        codes[kb_top:kb_top + upper, left:left + bw] = ch
        cidx[kb_top:kb_top + upper, left:left + bw] = col
        centres[p] = mid

    # the chord's notes by name, under their keys
    for p in sorted(tones, key=lambda p: centres[p]):
        _text(codes, cidx, labels, centres[p] - (len(_PITCHES[p]) - 1) // 2,
              _PITCHES[p], colours[p])

    # ── the foot: the last few chords on the left, the key on the right ──────
    key = f"key of {ctx.key}" if _tonic(ctx) is not None and ctx.key else ""
    names = [_chord_name(*h) for h in st["hist"]]
    line = "  ".join(names)
    while names and len(line) + len(key) + 3 > kb_w:
        names = names[1:]
        line = "  ".join(names)
    _text(codes, cidx, foot, kx, line, recede)
    _text(codes, cidx, foot, kx + kb_w - len(key), key, recede)
    return codes, cidx


# ── JP Panel: a drum machine's front panel ───────────────────────────────────

#: Where each group of four pads' colour tab sits on the ramp, the way a
#: drum machine's pads come in coloured groups of four.
_GROUP_HEAT = (0.3, 0.5, 0.7, 0.95)

#: The drums' letters, same order as :data:`_ROWS`.
_DRUM_LETTER = ("H", "S", "K")


@mode("JP Panel", group="jp", after="JP Chords",
      blurb="a drum machine's front panel: sixteen pads, a running light, and the drums lighting the pads they land on")
def jp_panel(ctx: Ctx):
    """One row of sixteen big pads, a light chasing across them once a bar.

    Each pad lights in the colour of the drum that landed on its step -- a
    kick deep, a snare in the middle, a hat hot -- and a step two drums share
    is split between them. The bar before stays as a line at the foot of its
    pads until the new bar writes over it. Above the pads, a display shows
    the tempo, the key when it is known, and four beat lamps; under them, the
    drums on each step by letter and the step numbers, the fours marked the
    way a panel prints them.
    """
    rows, w = ctx.h, ctx.w
    gap = 1
    pad_w = (w - gap * (_STEPS - 1)) // _STEPS
    if rows < 8 or pad_w < 2:
        return empty(w, rows)
    # tall enough to read as a button, not so tall it reads as a bar
    pad_h = min(12, rows - 8) if rows >= 11 else rows - 6
    boxed = pad_w >= 3 and pad_h >= 3

    st = ctx.scratch("jp_panel", lambda: {
        "cur": np.zeros((len(_ROWS), _STEPS), np.float32),
        "prev": np.zeros((len(_ROWS), _STEPS), np.float32),
        "flash": np.full(_STEPS, -99.0),
        "last": 0.0, "count": 0, "last_beat": 0.0, "idle": 0.0,
    })
    phase, known, beating = _playhead(ctx, st)
    if phase < st["last"] - 0.5:
        st["prev"] = st["cur"].copy()
        st["cur"][:] = 0.0
    st["last"] = phase
    step = int(phase * _STEPS) % _STEPS

    if ctx.onsets:
        strength = 0.55 + 0.45 * min(1.0, float(ctx.onset_strength))
        hit = named(ctx.drums or {})
        for r, (name, _) in enumerate(_ROWS):
            if name in hit:
                st["cur"][r, step] = max(st["cur"][r, step], strength)
                st["flash"][step] = ctx.t

    codes = np.full((rows, w), SPACE, dtype=np.int32)
    recede = recede_index(ctx.palette)
    cidx = np.full((rows, w), recede, dtype=np.int32)
    hot = int(np.asarray(ctx.ramp(np.float32(1.0))))
    drum_c = [int(np.asarray(ctx.ramp(np.float32(heat)))) for _, heat in _ROWS]
    group_c = [int(np.asarray(ctx.ramp(np.float32(g)))) for g in _GROUP_HEAT]
    x0 = (w - (pad_w * _STEPS + gap * (_STEPS - 1))) // 2
    x1 = x0 + pad_w * _STEPS + gap * (_STEPS - 1)
    level = min(1.0, float(ctx.range(0.0, 1.0)))
    y = (rows - (6 + pad_h)) // 2
    disp, lamp, pad_top = y, y + 2, y + 3
    label, numbers = pad_top + pad_h, pad_top + pad_h + 1

    # ── the display: tempo, key, and a lamp per beat ─────────────────────────
    tempo = f"TEMPO {ctx.tempo_bpm:3.0f}" if ctx.tempo_bpm > 0.0 else "TEMPO ---"
    _text(codes, cidx, disp, x0, tempo, hot if beating else recede)
    if _tonic(ctx) is not None and ctx.key:
        key = ctx.key.upper()
        _text(codes, cidx, disp, (x0 + x1 - len(key)) // 2, key, group_c[2])
    beat = int(phase * 4) % 4
    for b in range(4):
        c = x1 - 1 - (3 - b) * 2
        on = beating and b == beat
        codes[disp, c] = _LED if on else _OFF
        cidx[disp, c] = (hot if b == 0 and known else group_c[b]) if on else recede

    for s in range(_STEPS):
        left = x0 + s * (pad_w + gap)
        right = left + pad_w - 1
        mid = left + pad_w // 2
        tab = group_c[s // 4]
        here = s == step

        # the step's lamp
        codes[lamp, mid] = (_LED if beating else _TRAIL) if here else _OFF
        cidx[lamp, mid] = hot if here else recede

        # the pad: a frame, its group's colour along the foot
        foot = pad_top + pad_h - 1
        if boxed:
            codes[pad_top, left + 1:right] = ord("─")
            codes[foot, left + 1:right] = ord("─")
            codes[pad_top + 1:foot, left] = ord("│")
            codes[pad_top + 1:foot, right] = ord("│")
            codes[pad_top, left], codes[pad_top, right] = ord("┌"), ord("┐")
            codes[foot, left], codes[foot, right] = ord("└"), ord("┘")
            cidx[pad_top:foot, left:right + 1] = hot if here else recede
            r0, r1, c0, c1 = pad_top + 1, foot, left + 1, right
        else:
            codes[foot, left:right + 1] = ord("▔")
            r0, r1, c0, c1 = pad_top, foot, left, right + 1
        cidx[foot, left:right + 1] = tab

        written = [r for r in range(len(_ROWS)) if st["cur"][r, s] > 0.0]
        if written:
            # split the pad between the drums that share its step, top to
            # bottom as the rows read: hat, snare, kick
            flash = (ctx.t - st["flash"][s]) < _FLASH_S * 2
            span = r1 - r0
            for i, r in enumerate(written):
                a = r0 + span * i // len(written)
                b = r0 + span * (i + 1) // len(written)
                codes[a:b, c0:c1] = _LED
                cidx[a:b, c0:c1] = hot if flash else drum_c[r]
        else:
            last = [r for r in range(len(_ROWS)) if st["prev"][r, s] > 0.0]
            if last and r1 > r0:
                codes[r1 - 1, c0:c1] = _TRAIL
                cidx[r1 - 1, c0:c1] = drum_c[last[-1]]
            if here and level > _LEVEL_FLOOR and r1 > r0:
                # the pad under the light fills with how loud it is: the panel
                # breathes with the music between the hits
                up = max(1, round(level * (r1 - r0)))
                codes[r1 - up:r1, c0:c1] = _PEAK
                cidx[r1 - up:r1, c0:c1] = np.asarray(ctx.ramp(
                    np.linspace(min(1.0, 0.2 + 0.8 * level), 0.2, up, dtype=np.float32)))[:, None]

        # under the pad: its drums by letter, then its number
        letters = "".join(_DRUM_LETTER[r] for r in written)[:pad_w]
        for i, ch in enumerate(letters):
            _text(codes, cidx, label, mid - len(letters) // 2 + i, ch,
                  drum_c[_DRUM_LETTER.index(ch)])
        num = str(s + 1)
        _text(codes, cidx, numbers, mid - (len(num) - 1) // 2, num,
              tab if s % 4 == 0 else recede)
    return codes, cidx


# ── JP Tracker: the song scrolling past ──────────────────────────────────────

#: Screen columns a beat takes as the timeline scrolls; the pace follows the
#: tempo, so a bar is always the same width, and runs at 120 BPM's with none.
_COLS_PER_BEAT = 8
_FREE_BPM = 120.0

#: The drum lanes' labels, same order as :data:`_ROWS`.
_LANE_NAMES = ("HAT", "SNR", "KCK")


@mode("JP Tracker", group="jp", after="JP Panel",
      blurb="the song scrolling past like a tracker: the notes on top, the drums in their lanes, the beats marked")
def jp_tracker(ctx: Ctx):
    """A timeline running right to left, the present at the right edge.

    Top: a note roll of the pitch classes heard, C at the bottom and B at the
    top, so a held chord is a stack of lines and a melody steps up and down.
    Below it: a lane each for hat, snare and kick, a block wherever that drum
    hit. Dots mark the beats and a line marks each bar when the bar is known,
    and the pace follows the tempo, so a bar is always the same width and a
    steady beat draws a steady pattern.
    """
    rows, w = ctx.h, ctx.w
    gutter = 4 if w >= 40 else 0
    n = w - gutter
    lane_h = 2 if rows >= 24 else 1
    roll = rows - (3 * lane_h + 2) - 1
    if n < 8 or roll < 6:
        return empty(w, rows)

    st = ctx.scratch("jp_tracker", lambda: {
        "notes": np.zeros((12, n), np.float32),
        "drums": np.zeros((len(_ROWS), n), np.float32),
        "grid": np.zeros(n, np.int8),
        "acc": 0.0, "beat": 0.0, "bar": 0.0,
    })
    bpm = ctx.tempo_bpm if ctx.tempo_bpm > 0.0 else _FREE_BPM
    st["acc"] += max(ctx.dt, 0.0) * bpm / 60.0 * _COLS_PER_BEAT
    k = min(int(st["acc"]), n)
    st["acc"] -= int(st["acc"])
    if k:
        for key in ("notes", "drums", "grid"):
            st[key] = np.roll(st[key], -k, axis=-1)
            st[key][..., -k:] = 0

    # the present column
    level = min(1.0, 0.45 + 1.5 * float(ctx.energy))
    for i, p in enumerate(_notes(ctx)):
        st["notes"][p, -1] = max(st["notes"][p, -1], level * (1.0 if i == 0 else 0.7))
    if ctx.onsets:
        strength = 0.55 + 0.45 * min(1.0, float(ctx.onset_strength))
        hit = named(ctx.drums or {})
        for r, (name, _) in enumerate(_ROWS):
            if name in hit:
                st["drums"][r, -1] = max(st["drums"][r, -1], strength)
    if ctx.tempo_bpm > 0.0:
        if ctx.beat_phase < st["beat"] - 0.5:
            st["grid"][-1] = max(st["grid"][-1], 1)
        if ctx.bar_confidence >= _SURE and ctx.bar_phase < st["bar"] - 0.5:
            st["grid"][-1] = 2
    st["beat"], st["bar"] = float(ctx.beat_phase), float(ctx.bar_phase)

    codes = np.full((rows, w), SPACE, dtype=np.int32)
    recede = recede_index(ctx.palette)
    cidx = np.full((rows, w), recede, dtype=np.int32)
    body_c = codes[:, gutter:]
    body_i = cidx[:, gutter:]
    beat_cols = st["grid"] == 1
    bar_cols = st["grid"] == 2

    # ── the note roll ────────────────────────────────────────────────────────
    tonic = _tonic(ctx)
    for p in range(12):
        # the twelve lines spread down the whole roll, B at the top
        line = round((11 - p) * (roll - 1) / 11)
        colour = int(np.asarray(ctx.ramp(np.float32(_pitch_heat(p)))))
        v = st["notes"][p]
        body_c[line, beat_cols] = _OFF
        body_c[line, v > 0.2] = _PEAK
        body_c[line, v > 0.5] = _LED
        body_i[line, v > 0.2] = colour
        if gutter:
            _text(codes, cidx, line, 0, _PITCHES[p], colour if p == tonic else recede)

    # ── the drum lanes ───────────────────────────────────────────────────────
    for r, (_, heat) in enumerate(_ROWS):
        top = roll + 1 + r * (lane_h + 1)
        colour = int(np.asarray(ctx.ramp(np.float32(heat))))
        v = st["drums"][r]
        # a hit is a block two columns wide, so a single one reads at a glance
        shown = np.maximum(v, np.concatenate((v[1:], [0.0]))) > 0.0
        body_c[top + lane_h - 1, beat_cols] = _OFF
        body_c[top:top + lane_h, shown] = _LED
        body_i[top:top + lane_h, shown] = colour
        if gutter:
            _text(codes, cidx, top + lane_h - 1, 0, _LANE_NAMES[r], recede)

    # bar lines through the whole height, under anything written
    body_c[(body_c == SPACE) & bar_cols[None, :]] = ord("│")
    return codes, cidx
