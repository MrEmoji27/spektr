"""ASCII animation loading for the ``Flipbook`` mode.

Frames come from ``<config>/ascii/`` — either a subfolder of numbered ``.txt``
files (a real animation) or a single bare ``.txt`` (a still). A couple of
built-in reels are generated at import time so the mode has something to show
before you've dropped anything in; they are plain Python, not data files, so
packaging (``pyproject.toml``, ``packaging/spektr.spec``) needs no changes for
this feature.

Nothing here does rendering — see ``spektr/modes/flipbook.py`` for how a
reel's frames turn into a picture. This module only answers "what reels exist"
and "what are this reel's frames", as cheaply as it can:

* Discovery (``reels()``) lists the directory and counts files — no file
  content is read. A folder's frame count is just how many ``.txt`` files it
  has, so the settings row can show it without decoding anything.
* Decoding (``Reel.frames()``) reads and packs the actual glyphs, and only
  happens the first time a reel is actually rendered. Once decoded, a reel
  keeps its packed arrays until the next explicit reload.

Caps (``_MAX_FRAMES``, ``_MAX_W``, ``_MAX_H``) exist because this is the one
place spektr renders content it did not generate — an unbounded reel is an
unbounded amount of memory read from disk into a mode's scratch state.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import palette
from .render import BLOCKS_UP, SHADES, SPACE

_MAX_FRAMES = 240
_MAX_W = 400
_MAX_H = 200

_VALID_FX = ("warp", "dissolve", "lit")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_DIGIT_RE = re.compile(r"(\d+)")


def _natural_key(name: str) -> list:
    """Sort key where ``frame2`` comes before ``frame10``."""
    return [int(p) if p.isdigit() else p.lower() for p in _DIGIT_RE.split(name)]


# ── glyph -> ink density ─────────────────────────────────────────────────────
# Precomputed once per reel (see _pack), not per render frame — a mode's warp
# and dissolve effects move this array alongside the codepoints instead of
# re-deriving it every frame.

_DEFAULT_DENSITY = 0.55


def _build_density_table() -> dict[int, float]:
    table: dict[int, float] = {SPACE: 0.0}
    for i, ch in enumerate(SHADES):
        table[ord(ch)] = i / (len(SHADES) - 1)
    for i, ch in enumerate(BLOCKS_UP):
        table[ord(ch)] = i / (len(BLOCKS_UP) - 1)
    for ch in ".,'`:;~-":
        table.setdefault(ord(ch), 0.15)
    for ch in "\"^_/\\|()[]{}<>?!ilj+":
        table.setdefault(ord(ch), 0.4)
    for ch in "oOaAcCeEsSzZ0123456789*":
        table.setdefault(ord(ch), 0.65)
    for ch in "@#%&$8BMW":
        table.setdefault(ord(ch), 0.9)
    return table


_DENSITY = _build_density_table()


def _density_of(codes: np.ndarray) -> np.ndarray:
    d = np.full(codes.shape, _DEFAULT_DENSITY, dtype=np.float32)
    for cp, val in _DENSITY.items():
        d[codes == cp] = val
    return d


# ── text -> packed frames ────────────────────────────────────────────────────

def _parse_text(raw: str) -> list[str]:
    raw = _ANSI_RE.sub("", raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.expandtabs(8).split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # a trailing newline shouldn't add a blank frame row
    return lines


def _pack(frame_lines: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged per-frame text -> ``(n, h, w)`` codepoints + density, space-padded."""
    h = min(_MAX_H, max((len(lines) for lines in frame_lines), default=1)) or 1
    w = min(_MAX_W, max((len(line) for lines in frame_lines for line in lines), default=1)) or 1
    n = len(frame_lines)

    codes = np.empty((n, h, w), dtype=np.int32)
    for i, lines in enumerate(frame_lines):
        rows = []
        for r in range(h):
            line = lines[r] if r < len(lines) else ""
            rows.append(line[:w].ljust(w))
        text = "".join(rows)
        codes[i] = (
            np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
            .astype(np.int32)
            .reshape(h, w)
        )
    return codes, _density_of(codes)


def _load_paths(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    frame_lines: list[list[str]] = []
    for p in paths[:_MAX_FRAMES]:
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        frame_lines.append(_parse_text(raw))
    if not frame_lines:
        frame_lines = [["(no readable frames)"]]
    return _pack(frame_lines)


# ── reels ────────────────────────────────────────────────────────────────────

@dataclass
class Reel:
    name: str
    source: str
    n_frames: int
    _paths: tuple = field(default=(), repr=False)
    _cache: tuple | None = field(default=None, repr=False, compare=False)

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        """``(codes, density)``, each ``(n_frames, h, w)``. Decoded on first call."""
        if self._cache is None:
            self._cache = _load_paths(self._paths) if self._paths else _pack([["(empty)"]])
        return self._cache


def ascii_dir(config_dir: Path | None = None) -> Path:
    root = config_dir if config_dir is not None else palette.config_dir()
    return root / "ascii"


def _discover(config_dir: Path | None = None) -> list[Reel]:
    root = ascii_dir(config_dir)
    out: list[Reel] = []
    try:
        if not root.is_dir():
            return out
        entries = sorted(root.iterdir(), key=lambda p: _natural_key(p.name))
    except OSError:
        return out

    for entry in entries:
        try:
            if entry.is_dir():
                txts = sorted(
                    (f for f in entry.iterdir() if f.is_file() and f.suffix.lower() == ".txt"),
                    key=lambda f: _natural_key(f.name),
                )
                if not txts:
                    continue
                txts = txts[:_MAX_FRAMES]
                out.append(Reel(name=entry.name, source=str(entry), n_frames=len(txts), _paths=tuple(txts)))
            elif entry.is_file() and entry.suffix.lower() == ".txt":
                out.append(Reel(name=entry.stem, source=str(entry), n_frames=1, _paths=(entry,)))
        except OSError:
            continue
    return out


def _builtin(name: str, frame_lines: list[list[str]]) -> Reel:
    r = Reel(name=name, source="builtin", n_frames=len(frame_lines))
    r._cache = _pack(frame_lines)
    return r


def _make_orbit() -> list[list[str]]:
    size = 9
    centre = size // 2
    ring = [(0, 4), (1, 6), (4, 8), (7, 6), (8, 4), (7, 2), (4, 0), (1, 2)]
    frames = []
    for i in range(len(ring)):
        grid = [[" "] * size for _ in range(size)]
        grid[centre][centre] = "+"
        for j, (r, c) in enumerate(ring):
            grid[r][c] = "*" if j == i else "."
        frames.append(["".join(row) for row in grid])
    return frames


def _make_wave() -> list[list[str]]:
    w, h = 24, 7
    ramp = " .:-=+*#%@"
    frames = []
    for phase in range(16):
        grid = [[" "] * w for _ in range(h)]
        for x in range(w):
            y = (h - 1) / 2 + math.sin(x * (2 * math.pi / w) + phase * 0.35) * (h - 1) / 2.2
            top = max(0, min(h - 1, int(round(y))))
            for row in range(top, h):
                grid[row][x] = ramp[min(len(ramp) - 1, row - top + 1)]
        frames.append(["".join(row) for row in grid])
    return frames


def _builtin_reels() -> list[Reel]:
    return [
        _builtin("Orbit", _make_orbit()),
        _builtin("Wave", _make_wave()),
    ]


_reels_cache: list[Reel] | None = None


def reels(config_dir: Path | None = None) -> list[Reel]:
    """Every discovered + built-in reel. Cached until :func:`reload`."""
    global _reels_cache
    if _reels_cache is None:
        _reels_cache = _discover(config_dir) + _builtin_reels()
    return _reels_cache


def reload() -> None:
    """Drop the reel list and every reel's decoded frames — picks up disk edits."""
    global _reels_cache
    _reels_cache = None


# ── selection ────────────────────────────────────────────────────────────────
# Module-level: modes only ever receive a Ctx (see spektr/modes/__init__.py),
# with no other channel to a user setting, the same reason the plugin trust
# store and the theme cache live at module scope rather than on some instance.

_selected_reel_name = ""
_selected_fx = "warp"


def restore(reel: str, fx: str) -> None:
    """Set the starting selection from saved config. Never touches disk."""
    global _selected_reel_name, _selected_fx
    _selected_reel_name = reel or ""
    _selected_fx = fx if fx in _VALID_FX else "warp"


def current_fx() -> str:
    return _selected_fx


def step_fx(delta: int) -> str:
    global _selected_fx
    i = (_VALID_FX.index(_selected_fx) + delta) % len(_VALID_FX)
    _selected_fx = _VALID_FX[i]
    return _selected_fx


def current(config_dir: Path | None = None) -> Reel | None:
    all_reels = reels(config_dir)
    if not all_reels:
        return None
    for r in all_reels:
        if r.name == _selected_reel_name:
            return r
    return all_reels[0]


def step_reel(delta: int, config_dir: Path | None = None) -> Reel | None:
    global _selected_reel_name
    all_reels = reels(config_dir)
    if not all_reels:
        return None
    names = [r.name for r in all_reels]
    try:
        i = names.index(_selected_reel_name)
    except ValueError:
        i = 0
    i = (i + delta) % len(names)
    _selected_reel_name = names[i]
    return all_reels[i]
