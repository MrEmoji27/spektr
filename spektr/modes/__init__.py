"""Render mode registry.

Every mode is a function ``(ctx) -> (codes, cidx)`` — a ``(h, w)`` array of
Unicode codepoints and a matching array of palette ramp indices. Modes never
touch Rich, never build strings, and never see a colour value. That keeps them
short (most are 20-40 lines), makes them uniformly fast, and means a new mode
can't accidentally reintroduce the per-cell Segment problem.

Plugins register through exactly the same decorator. Everything a plugin is
allowed to touch is re-exported from :mod:`spektr.api`, which is the stable
surface — this module is free to move around underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from ..analysis import resample_bands
from ..palette import Palette
from ..render import SPACE

Codes = np.ndarray
Result = "tuple[np.ndarray, np.ndarray]"


@dataclass
class Ctx:
    """Everything a mode is allowed to know about the current frame.

    This is a published API — plugins receive it and it must stay compatible
    within an ``API_VERSION``. Two notes for anyone writing against it:

    * ``bands`` is at spektr's *internal* resolution, which is deliberately not
      promised to stay at 32. Write resolution-independent code: ask for the
      count you want with ``display_bands(n)``, or read ``n_bands`` rather than
      hard-coding a length or slicing fixed indices.
    * Arrays handed to you are the live smoothing buffers. Read them freely;
      don't write into them. Anything you need to keep between frames goes in
      ``scratch()``, which is per-mode and cleared on resize.
    """

    w: int
    h: int
    bands: np.ndarray            # smoothed, n_bands wide, each 0..1
    peaks: np.ndarray
    bands_l: np.ndarray
    bands_r: np.ndarray
    wave: np.ndarray             # smoothed mono trace, roughly -1..1
    stereo: np.ndarray           # raw (N, 2) L/R pairs
    frame: int
    t: float                     # seconds since start
    dt: float                    # seconds since the previous frame
    energy: float                # mean band level, 0..1
    silent: bool                 # True when the noise gate is shut
    palette: Palette
    state: dict = field(default_factory=dict)
    #: How many bars the user asked for, or 0 for "fit the terminal". Read
    #: through :attr:`n_display`; modes should not consult it directly.
    bars: int = 0

    # ── derived geometry ──
    @property
    def size(self) -> tuple[int, int]:
        return self.w, self.h

    @property
    def dot_rows(self) -> int:
        """Braille dot rows — four per text row."""
        return self.h * 4

    @property
    def dot_cols(self) -> int:
        """Braille dot columns — two per text column."""
        return self.w * 2

    @property
    def n_bands(self) -> int:
        """Length of ``bands``. Read this instead of assuming 32."""
        return len(self.bands)

    @property
    def n_display(self) -> int:
        """How many bars to draw.

        Defaults to fitting the terminal, so a 200-column window doesn't end up
        with ten 20-cell slabs. A user setting overrides it, still bounded by
        what the width can actually show — asking for 64 bars in an 80-column
        terminal would give you one-cell bars and a worse picture.
        """
        if self.bars:
            return int(max(4, min(self.bars, self.n_bands, max(4, self.w // 2))))
        return int(max(8, min(self.n_bands, self.w // 7)))

    def range(self, lo: float, hi: float) -> float:
        """Mean level across a slice of the spectrum, given as fractions.

        ``ctx.range(0, 0.2)`` is the bottom fifth — the kick drum — whatever
        the internal band count happens to be. This exists so plugins never
        need to write ``ctx.bands[:6]``.
        """
        n = self.n_bands
        a = int(max(0.0, min(1.0, lo)) * n)
        b = int(max(0.0, min(1.0, hi)) * n)
        if b <= a:
            b = min(n, a + 1)
        return float(self.bands[a:b].mean())

    def display_bands(self, n: int | None = None) -> np.ndarray:
        n = self.n_display if n is None else n
        return resample_bands(self.bands, n)

    def display_peaks(self, n: int | None = None) -> np.ndarray:
        n = self.n_display if n is None else n
        return resample_bands(self.peaks, n)

    def ramp(self, norm) -> np.ndarray:
        return self.palette.indices(np.asarray(norm, dtype=np.float64))

    def scratch(self, key: str, factory):
        """Per-mode persistent state, keyed and invalidated on resize.

        Stale entries for other sizes are dropped rather than the whole dict
        being cleared, so a mode holding several buffers doesn't lose the
        others every time one of them is rebuilt.
        """
        full = (key, self.w, self.h)
        got = self.state.get(full)
        if got is None:
            for stale in [k for k in self.state if k[1:] != (self.w, self.h)]:
                del self.state[stale]
            got = factory()
            self.state[full] = got
        return got


@dataclass(frozen=True)
class Mode:
    name: str
    fn: Callable[[Ctx], tuple[np.ndarray, np.ndarray]]
    group: str = "spectrum"
    blurb: str = ""
    #: None for built-ins, otherwise the plugin that registered it
    plugin: str | None = None

    @property
    def is_plugin(self) -> bool:
        return self.plugin is not None


MODES: list[Mode] = []
_BY_NAME: dict[str, Mode] = {}

#: Set by the plugin loader while a plugin module is executing, so modes get
#: attributed to their source without the author having to declare it.
_LOADING: str | None = None


class ModeNameTaken(Exception):
    """Raised when a plugin tries to shadow an existing mode."""


def mode(name: str, group: str = "spectrum", blurb: str = ""):
    """Register a render mode.

    ``name`` must be unique — a plugin cannot silently replace a built-in or
    another plugin's mode, because the picker and the saved settings key off
    it. Collisions raise, and the loader turns that into a readable message.
    """

    def wrap(fn):
        if name in _BY_NAME:
            owner = _BY_NAME[name].plugin or "spektr"
            raise ModeNameTaken(f"mode {name!r} is already registered by {owner}")
        m = Mode(name=name, fn=fn, group=group, blurb=blurb, plugin=_LOADING)
        # keep "None" last in the cycle, however late a plugin arrives
        if MODES and MODES[-1].name == "None":
            MODES.insert(len(MODES) - 1, m)
        else:
            MODES.append(m)
        _BY_NAME[name] = m
        return fn

    return wrap


def get(name: str) -> Mode | None:
    return _BY_NAME.get(name)


def names() -> list[str]:
    return [m.name for m in MODES]


def unregister_plugin(plugin: str) -> list[str]:
    """Drop every mode a plugin registered. Returns the names removed."""
    doomed = [m for m in MODES if m.plugin == plugin]
    for m in doomed:
        MODES.remove(m)
        _BY_NAME.pop(m.name, None)
    return [m.name for m in doomed]


# ── shared layout helpers ────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def band_columns(w: int, n: int) -> tuple:
    """Map each terminal column to a band index, with 1-column gutters.

    Returns ``(col_band, active)`` where ``col_band`` is an int array of length
    w and ``active`` is False on gutter columns. Cached, because this only
    changes when the terminal is resized.
    """
    gaps = n - 1
    usable = max(n, w - gaps)
    base, extra = divmod(usable, n)

    col_band = np.zeros(w, dtype=np.int32)
    active = np.zeros(w, dtype=bool)
    x = 0
    for b in range(n):
        width = base + (1 if b < extra else 0)
        end = min(w, x + width)
        col_band[x:end] = b
        active[x:end] = True
        x = end + 1          # leave a gutter
        if x >= w:
            break
    return col_band, active


@lru_cache(maxsize=64)
def smooth_columns(w: int, n: int) -> tuple:
    """Continuous column -> band position, for gapless interpolated modes."""
    pos = np.linspace(0.0, n - 1, w)
    idx = np.floor(pos).astype(np.int32)
    idx = np.clip(idx, 0, n - 1)
    nxt = np.clip(idx + 1, 0, n - 1)
    frac = pos - idx
    # cosine blend reads smoother than linear across a coarse band set
    frac = (1.0 - np.cos(frac * np.pi)) * 0.5
    return idx, nxt, frac


def spread(levels: np.ndarray, w: int) -> np.ndarray:
    """Band levels -> one interpolated level per terminal column."""
    n = len(levels)
    idx, nxt, frac = smooth_columns(w, n)
    return levels[idx] * (1.0 - frac) + levels[nxt] * frac


def empty(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((h, w), SPACE, dtype=np.int32),
        np.zeros((h, w), dtype=np.int32),
    )


# importing the mode modules registers them, in menu order
from . import spectrum   # noqa: E402,F401
from . import scope      # noqa: E402,F401
from . import particles  # noqa: E402,F401
from . import scenes     # noqa: E402,F401
from . import fields     # noqa: E402,F401


@mode("None", group="off", blurb="nothing at all")
def _off(ctx: Ctx):
    return empty(ctx.w, ctx.h)
