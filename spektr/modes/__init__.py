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

import importlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np

from ..analysis import resample_bands
from ..palette import Palette

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

    # ── rhythm ──
    # Passed through from the analyser verbatim. Every field here reads as its
    # "nothing known" value (all zeros) during silence, under a sustained
    # drone, and before a tempo is established — that is a real runtime state,
    # not a placeholder — so modes must treat them as optional, never required.
    #: Monotonic count of detected onsets since start. Never resets, including
    #: across silence. Key on this *changing* (compare with the last value
    #: seen) rather than on ``onset_strength > 0`` — a single drum hit can
    #: fire the detector repeatedly, and a reader that keys on the counter
    #: sees exactly one signal per hit.
    onset_seq: int = 0
    #: How many onsets landed since *this mode's previous frame* — 0 or 1
    #: normally, 2+ at a low frame rate or on fast drums. ``if ctx.onsets:``
    #: is the short answer to "did anything hit", and it is the field to reach
    #: for; :attr:`onset_seq` is the raw counter underneath it.
    #:
    #: The widget differences the counter once per frame so that 44 modes do
    #: not each keep their own ``last_seq`` in scratch. Per-mode differencing
    #: also gets the answer wrong after a mode switch: scratch survives, so a
    #: mode returned to after a minute away sees every beat that played while
    #: it was not drawing, and reports them all as having just happened.
    onsets: int = 0
    #: 0..1 strength of the most recently detected onset. It persists after
    #: that onset rather than clearing, so it answers "how hard was the last
    #: hit", not "how hard was the hit this frame" — gate it on
    #: :attr:`onsets` when you mean the latter.
    onset_strength: float = 0.0
    #: 0..1 raw onset-detection-function value for this frame. Continuous, so
    #: it is safe to read at any frame rate; useful for "how percussive is
    #: right now" rather than discrete hits.
    flux: float = 0.0
    #: Estimated tempo in BPM. 0.0 means unknown — never divide by this
    #: without checking.
    tempo_bpm: float = 0.0
    #: 0..1 position within the current beat, 0.0 on the beat. 0.0 whenever
    #: :attr:`tempo_bpm` is 0.0.
    beat_phase: float = 0.0

    # ── derived geometry ──
    @property
    def size(self) -> tuple[int, int]:
        return self.w, self.h

    # ── rhythm, in the forms a mode actually wants ───────────────────────────
    #
    # The raw fields above are the honest ones and they stay. These two are
    # what almost every caller was going to build out of them, and the reason
    # they exist is that the raw fields turned out to be easy to get wrong and
    # therefore mostly went unused: of 52 modes, 40 read no rhythm field at
    # all, ``flux`` had one reader and ``tempo_bpm`` and ``beat_phase`` two
    # each — for analysis the widget computes 187 times a second regardless.
    #
    # Both are safe to read at any frame rate, in silence, under a drone and
    # before a tempo is established. That is the whole point: the trap they
    # remove is real and was hit twice while writing the modes that do use
    # them.

    @property
    def pulse(self) -> float:
        """0..1 beat-locked swell — 1.0 on the beat, decaying through the bar.

        ``beat_phase`` with its footgun removed. The footgun: ``beat_phase``
        is 0.0 whenever ``tempo_bpm`` is 0.0, and 0.0 is *on the beat*, so the
        obvious ``1 - ctx.beat_phase`` reads as a permanent full-strength swell
        during silence, under a drone, and for the first seconds of every
        track — exactly the states where the mode should be doing nothing.
        Gating on the tempo rather than the phase is the fix, and it has to be
        remembered at every call site.

        Squared, so the swell sits on the beat and decays through the bar
        rather than sweeping linearly between them.

        This is the field to reach for when a mode looks twitchy on sparse
        material. Onsets are discrete and only exist on the frames where the
        peak picker committed; a mode driven only by them coasts in between,
        which on a slow track reads as a still picture jerking four times a
        bar. ``pulse`` is continuous and fills those gaps.
        """
        if self.tempo_bpm <= 0.0:
            return 0.0
        return (1.0 - float(self.beat_phase)) ** 2

    @property
    def drive(self) -> float:
        """0..1 how percussive the signal is right now.

        ``flux`` clamped, which is all most callers need. Continuous and
        computed per hop, so unlike :attr:`onsets` it is safe to read at any
        frame rate and does not depend on the peak picker committing to
        anything — it answers "how much attack is in the signal" rather than
        "was there a hit this frame".

        Use it for rates: how fast something scrolls, spins or spawns. On
        material the detector is deliberately conservative about — brushed
        drums, a dense mix where nothing clears the adaptive threshold — the
        onset path stays quiet while this does not, so the picture keeps
        answering to music a listener hears as busy.
        """
        return float(min(1.0, max(0.0, self.flux)))

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

    @cached_property
    def flatness(self) -> float:
        """Spectral flatness, 0..1 — how noise-like the spectrum is.

        The geometric mean of the band magnitudes over their arithmetic mean.
        A pure tone puts everything in one band and scores near 0; white noise
        or a wall of distortion spreads energy evenly and scores near 1.

        This is the only *timbre* measure in the app — everything else here is
        level and rough band position, which cannot tell a clean guitar from a
        distorted one at the same volume. Clipping and percussive noise both
        drive it up, a sustained tone drives it down. ``Chladni Extreme`` uses
        it to tell a drum-led groove from a pad.

        Cached per frame: it is cheap over 32 bands but several modes may want
        it, and a ``Ctx`` is built fresh every frame so the cache cannot go
        stale.
        """
        b = np.asarray(self.bands, dtype=np.float64) + 1e-6
        return float(np.exp(np.log(b).mean()) / b.mean())

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
    #: Kept out of the picker and out of shuffle, but still registered, still
    #: tested, and still reachable by name — ``spektr --mode "Chladni"``, or by
    #: writing it into the config. Superseded modes live here: the octant
    #: variants replaced their originals in the menu, and putting the pair
    #: side by side there is a choice nobody wants to make forty times.
    hidden: bool = False

    @property
    def is_plugin(self) -> bool:
        return self.plugin is not None


MODES: list[Mode] = []
_BY_NAME: dict[str, Mode] = {}
_PENDING: set[str] = set()

#: Set by the plugin loader while a plugin module is executing, so modes get
#: attributed to their source without the author having to declare it.
_LOADING: str | None = None


class ModeNameTaken(Exception):
    """Raised when a plugin tries to shadow an existing mode."""


def mode(name: str, group: str = "spectrum", blurb: str = "",
         after: str | None = None, hidden: bool = False):
    """Register a render mode.

    ``name`` must be unique — a plugin cannot silently replace a built-in or
    another plugin's mode, because the picker and the saved settings key off
    it. Collisions raise, and the loader turns that into a readable message.

    ``after`` places the mode directly behind an already-registered one in the
    picker, instead of at the end. The cycle order is otherwise the order the
    mode modules are imported and the order the functions appear inside them,
    which puts a variant wherever its *implementation* happens to live rather
    than next to the mode it is a variant of: ``Dither Storm`` shares a file
    with nothing and landed twenty places from ``Dither``, and ``Tunnel In``
    sat four modes past ``Tunnel`` because ``scenes.py`` grew in between.
    Moving the functions would drag their private helpers across modules to fix
    a menu ordering, which is the wrong thing to refactor for.

    An unknown ``after`` appends as usual — a plugin naming a built-in that a
    later version renames should still load.

    ``hidden`` registers the mode without listing it: it stays out of the
    picker and out of shuffle, and stays reachable by name. That is what an
    opt-in variant wants — the code is still there, still tested, still the
    thing a plugin or a config file can ask for, without a menu that offers
    the same picture twice.
    """

    def wrap(fn):
        if name in _BY_NAME:
            if name in _PENDING and _LOADING is None:
                # The catalogue seeded this entry so the picker could list the
                # mode without importing it. Now the real module has loaded,
                # its own decorator is the truth: copy every field across, not
                # just the function, or editing a mode's group, blurb or
                # hidden flag in source would change nothing at all.
                pending = _BY_NAME[name]
                object.__setattr__(pending, "fn", fn)
                object.__setattr__(pending, "group", group)
                object.__setattr__(pending, "blurb", blurb)
                object.__setattr__(pending, "hidden", hidden)
                _PENDING.remove(name)
                return fn
            owner = _BY_NAME[name].plugin or "spektr"
            raise ModeNameTaken(f"mode {name!r} is already registered by {owner}")
        m = Mode(name=name, fn=fn, group=group, blurb=blurb, plugin=_LOADING,
                 hidden=hidden)
        # keep "None" last in the cycle, however late a plugin arrives
        end = len(MODES) - 1 if MODES and MODES[-1].name == "None" else len(MODES)
        at = end
        if after is not None and after in _BY_NAME:
            at = min(MODES.index(_BY_NAME[after]) + 1, end)
        MODES.insert(at, m)
        _BY_NAME[name] = m
        return fn

    return wrap


#: What the subcell variants were called before the (o)/(q) suffix.
#:
#: A saved config, a preset or a ``--mode`` flag written against an earlier
#: version still names them this way, and a mode that stops resolving is a
#: silent fall back to Bars rather than an error anyone sees.
_RENAMED = {
    "Scope Fine": "Scope (o)",
    "ECG Fine": "ECG (o)",
    "Radial Fine": "Radial (o)",
    "Sonar Fine": "Sonar (o)",
    "Plasma Fine": "Plasma (o)",
    "Chladni Fine": "Chladni (o)",
    "Chladni Flow Fine": "Chladni Flow (o)",
    "Chladni Extreme Fine": "Chladni Extreme (o)",
    "Kaleidoscope Fine": "Kaleidoscope (o)",
    "Kaleidoscope Ultra": "Kaleidoscope Ultra (o)",
    "Valentine Fine": "Valentine (o)",
    "Maelstrom Fine": "Maelstrom (o)",
}

#: The suffix each cell geometry is shown with.
CELL_SUFFIX = {"octant": "(o)", "quadrant": "(q)"}


class _LazyFn:
    """Callable placeholder that imports its catalogue module on first use."""

    def __init__(self, name: str, module: str):
        self.name = name
        self.module = module

    @property
    def __module__(self) -> str:
        return f"spektr.modes.{self.module}"

    def __call__(self, ctx: Ctx):
        loaded = ensure_loaded(self.name)
        if loaded is None or loaded.fn is self:
            raise LookupError(f"mode {self.name!r} did not register from {self.module}")
        return loaded.fn(ctx)


def _seed_catalogue() -> None:
    from ..catalog import CATALOG

    for name, group, blurb, hidden, module in CATALOG:
        lazy = _LazyFn(name, module)
        registered = Mode(name, lazy, group, blurb, None, hidden)
        MODES.append(registered)
        _BY_NAME[name] = registered
        _PENDING.add(name)


def ensure_loaded(name: str) -> Mode | None:
    """Import the built-in implementation for ``name`` and return its mode."""
    registered = _BY_NAME.get(name)
    if registered is None or registered.plugin is not None or name not in _PENDING:
        return registered
    from ..catalog import module_for

    module = module_for(name)
    if module == "modes":
        return registered
    if module is None:
        return None
    importlib.import_module(f"{__name__}.{module}")
    return _BY_NAME.get(name)


def load_all() -> None:
    """Load every built-in implementation for generators and audit tools."""
    for item in tuple(MODES):
        ensure_loaded(item.name)


def get(name: str) -> Mode | None:
    """The mode by name, accepting every name it has ever answered to.

    Registered under ``(o)`` because octants are the default and what these
    modes are designed around — but the geometry is a *setting*, so the same
    mode draws quadrants when ``cells`` says so, and someone reading ``(q)``
    off the picker has every reason to type it back. Both resolve here, along
    with the ``Fine`` names they carried before.
    """
    resolved = name
    m = _BY_NAME.get(resolved)
    if m is not None:
        return ensure_loaded(resolved)
    alt = _RENAMED.get(name)
    if alt is None and name.endswith(" (q)"):
        alt = name[:-4] + " (o)"
    return ensure_loaded(alt) if alt is not None else None


def label(name: str) -> str:
    """The name as the interface should show it, for the live cell geometry.

    The registered name says ``(o)``; in quadrant mode the mode really is
    drawing quadrants, and a picker that insists otherwise is lying about the
    one thing the suffix exists to report.
    """
    if not name.endswith(" (o)"):
        return name
    from ..render import CELL_MODE

    return name[:-4] + " " + CELL_SUFFIX.get(CELL_MODE, "(o)")


def names() -> list[str]:
    return [m.name for m in MODES]


def listed() -> list[Mode]:
    """Every mode the interface offers — everything except the hidden ones.

    The picker, the cycle keys and shuffle all go through here, so an
    opt-in mode disappears from all three at once while staying available to
    ``--mode`` and to anything that already has its name.
    """
    return [m for m in MODES if not m.hidden]


def unregister_plugin(plugin: str) -> list[str]:
    """Drop every mode a plugin registered. Returns the names removed."""
    doomed = [m for m in MODES if m.plugin == plugin]
    for m in doomed:
        MODES.remove(m)
        _BY_NAME.pop(m.name, None)
    return [m.name for m in doomed]


_SHARED_NAMES = {
    "angular_bands",
    "angular_lut",
    "band_columns",
    "bg_contrast",
    "contrast_ramp",
    "empty",
    "polar_grid",
    "smooth_columns",
    "spread",
}


def __getattr__(name):
    if name in _SHARED_NAMES:
        from . import _shared

        return getattr(_shared, name)
    raise AttributeError(f"module 'spektr.modes' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(_SHARED_NAMES))


_seed_catalogue()


@mode("None", group="off", blurb="nothing at all")
def _off(ctx: Ctx):
    from ._shared import empty

    return empty(ctx.w, ctx.h)
