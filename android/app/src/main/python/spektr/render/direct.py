"""Writing the picture straight to the terminal.

Textual is a widget framework: every frame it composes the screen, pushes each
line of our strips through the styles cache, and re-encodes the result for the
terminal. Measured at 200x50 on ``Bars``, that is about a fifth of a core for a
picture that changes every frame — ``Compositor.render_update`` 19% of the wall
and the styles cache inside it 11% — against well under one percent for
encoding the same cells ourselves. The visualiser is one full-screen widget
whose every line is dirty every frame, so none of that machinery has anything
to reuse; it is pure overhead on the way to the same bytes.

This is the encoder that replaces it while the picture is the only thing on
screen. It keeps the cells the strips would have held, diffs them against the
last frame, and writes only the runs that changed, in a synchronized-output
bracket. What it draws is the same picture ``make_strips`` would have handed
Textual — the same packed colour key, the same glyphs, the same rule for a
cleared background (see :func:`_pack`) — which a test pins cell for cell.

It is deliberately not a terminal abstraction: no cursor bookkeeping, no
scrolling, no input, nothing but "these cells, please". Anything else on screen
— a picker, a settings panel, a notification — is Textual's, and the widget
hands the screen back for as long as one is up.
"""
from __future__ import annotations

import numpy as np

from .palette import RAMP_STEPS
from .render import SPACE

#: Every codepoint a mode can draw is below this, so a cell's glyph rides in
#: the low bits of its key without ever colliding with another cell's colour.
GLYPH = 0x110000

#: Nothing is a real background, and None is a background of the terminal's:
#: the two have to be told apart in a cache that outlives a frame.
UNSET = object()

#: Reset, then name the foreground; the pair form names the background too. A
#: cleared cell names no background of its own — the terminal's own shows
#: through — so it resets that instead of leaving the last one in place.
FG = "\x1b[0;38;2;{};{};{}m"
PAIR = "\x1b[0;38;2;{};{};{};48;2;{};{};{}m"
CLEARED = "\x1b[0;49;38;2;{};{};{}m"
#: Cursor to a cell, 1-based: rows first, as every terminal wants it.
MOVE = "\x1b[{};{}H"


def _rgb(hexes: list[str], i: int) -> tuple[int, int, int]:
    h = hexes[i]
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _pack(codes, cidx, bidx, palette, clear):
    """The per-cell colour key, and the glyphs, exactly as ``make_strips``.

    Returns ``(key, base, pair, glyphs)``: ``key`` is the packed colour index —
    ``fg * base + bg``, or just ``fg`` when a frame carries one colour — and
    ``pair`` says whether a background was packed in at all.
    """
    base = RAMP_STEPS
    if bidx is None:
        return cidx.astype(np.int32, copy=False), base, False, codes
    if palette.transparent:
        if clear is None:
            clear = bidx == 0
        if clear.any():
            codes = np.where(clear & (cidx == bidx), np.int32(SPACE), codes)
            bidx = np.where(clear, np.int32(RAMP_STEPS), bidx)
            base = RAMP_STEPS + 1
    return (cidx * base + bidx).astype(np.int32, copy=False), base, True, codes


def _runs(row: np.ndarray, first: int, last: int) -> list[tuple[int, int]]:
    """Stretches of one colour inside ``[first, last)``, as (start, end)."""
    edges = (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist()
    starts = [0, *edges]
    ends = [*edges, row.size]
    return [
        (max(s, first), min(e, last))
        for s, e in zip(starts, ends)
        if e > first and s < last
    ]


class Screen:
    """One widget's worth of cells, diffed frame to frame.

    ``frame`` returns the escapes for the cells that changed since the last
    call, or ``""`` when nothing did.
    """

    __slots__ = ("_base", "_bg", "_keys", "_pair", "_sgr", "cols", "rows", "x", "y")

    def __init__(self, cols: int = 0, rows: int = 0, x: int = 0, y: int = 0):
        self.cols, self.rows, self.x, self.y = cols, rows, x, y
        self._keys: np.ndarray | None = None
        self._sgr: dict[int, str] = {}
        self._base = RAMP_STEPS
        self._pair = False
        self._bg = UNSET

    def place(self, cols: int, rows: int, x: int = 0, y: int = 0) -> None:
        """Where the widget's cells are, in screen coordinates.

        A move or a resize invalidates the buffer: the cells that were on
        screen are not the cells this widget is about to draw.
        """
        if (cols, rows, x, y) != (self.cols, self.rows, self.x, self.y):
            self.forget()
        self.cols, self.rows, self.x, self.y = cols, rows, x, y

    def forget(self) -> None:
        """Draw everything next frame.

        For anything that puts something else over these cells — a resize, a
        panel, Textual repainting for its own reasons — and for a palette whose
        colours moved, which the keys alone cannot see.
        """
        self._keys = None
        self._sgr.clear()
        self._bg = UNSET

    def frame(self, codes, cidx, bidx, palette, clear=None) -> str:
        """The escapes for this frame: only what changed, positioned."""
        codes = np.asarray(codes)
        h, w = codes.shape
        if h == 0 or w == 0:
            return ""
        if (w, h) != (self.cols, self.rows):
            self.place(w, h, self.x, self.y)
        key, base, pair, glyphs = _pack(codes, cidx, bidx, palette, clear)
        if (base, pair) != (self._base, self._pair):
            # A different packing (a frame that carries a background meeting
            # one that does not): the cached escapes are keyed the old way.
            self._sgr.clear()
            self._base, self._pair = base, pair
        text = glyphs.astype("<u4", copy=False).tobytes().decode(
            "utf-32-le", errors="replace"
        )
        if len(text) != h * w:
            # Not one glyph a cell, so there is no column mapping to trust.
            self.forget()
            return ""
        cells = key.astype(np.int64) * GLYPH + glyphs
        prev, self._keys = self._keys, cells
        if prev is None or prev.shape != cells.shape:
            rows = range(h)
            changed = None
        else:
            changed = cells != prev
            rows = np.flatnonzero(changed.any(axis=1)).tolist()
            if not rows:
                return ""
        out: list[str] = []
        for y in rows:
            if changed is None:
                first, last = 0, w
            else:
                row_changed = changed[y]
                first = int(np.argmax(row_changed))
                last = w - int(np.argmax(row_changed[::-1]))
            out.append(MOVE.format(self.y + y + 1, self.x + first + 1))
            row = key[y]
            base_at = y * w
            for start, end in _runs(row, first, last):
                out.append(self._escape(int(row[start]), base, pair, palette))
                out.append(text[base_at + start:base_at + end])
        return "".join(out)

    def _escape(self, key: int, base: int, pair: bool, palette) -> str:
        """The escape that names one cell's colours, cached per key."""
        seq = self._sgr.get(key)
        if seq is None:
            f, b = divmod(key, base) if pair else (key, RAMP_STEPS)
            fg = _rgb(palette.hexes, f)
            if b == RAMP_STEPS:
                bg = self._theme_bg(palette)
                seq = PAIR.format(*fg, *bg) if bg is not None else CLEARED.format(*fg)
            else:
                seq = PAIR.format(*fg, *_rgb(palette.hexes, b))
            self._sgr[key] = seq
        return seq

    def _theme_bg(self, palette):
        """The background a style with no colour of its own carries.

        The palette hands its foreground styles the theme's own background, so
        that a cell which names no colour still lands on the theme rather than
        on whatever the terminal is set to. With a transparent theme it names
        none, and then so does this.
        """
        if self._bg is UNSET:
            colour = palette.styles[0].bgcolor
            self._bg = (
                None if colour is None else tuple(colour.get_truecolor()[:3])
            )
        return self._bg
