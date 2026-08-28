"""Named sets of modes, saved next to config.json.

This replaces ``presets.py``, which stored named snapshots of mode + theme +
fps + bands + sensitivity + gate. Those were four kinds of thing in one file
and three of them were already settings you could see and change in the
panel, so recalling one moved things you had not asked it to move. A loadout
is one kind of thing: a name and the modes it offers. Everything about the
look stays where it is when you load one.

An old ``presets.json`` is left alone rather than migrated or deleted — the
entries carry a single ``mode``, not a set, so there is nothing in one that
answers "which modes should be offered", and silently inventing a one-mode
loadout out of each would be worse than ignoring them.

Same guarded shape as :mod:`spektr.config`: a missing file, broken JSON or a
malformed entry all end at "no loadouts", never at a crash, because a file
you can hand-edit is exactly as disposable as the config beside it.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import palette


def _path(config_dir: Path | None = None) -> Path:
    root = config_dir if config_dir is not None else palette.config_dir()
    return root / "loadouts.json"


def load(config_dir: Path | None = None) -> dict[str, list[str]]:
    """Read the loadouts file, dropping anything malformed rather than failing.

    Every entry is normalised the way :meth:`config.Settings.clamp` normalises
    the active loadout — strings only, order kept, duplicates collapsed — so
    the rest of the app can treat a loaded name as a clean list without
    checking it again. An entry that survives that as empty is dropped: an
    empty loadout means "no restriction", and a *named* one meaning that is
    a trap rather than a feature.

    ``config_dir`` overrides where the file lives; None means the platform
    default from :func:`palette.config_dir`.
    """
    try:
        raw = json.loads(_path(config_dir).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list[str]] = {}
        for name, names in raw.items():
            if not (isinstance(name, str) and name):
                continue
            if not isinstance(names, (list, tuple)):
                continue
            seen: set[str] = set()
            clean = [
                n for n in names
                if isinstance(n, str) and n and not (n in seen or seen.add(n))
            ]
            if clean:
                out[name] = clean
        return out
    except Exception:
        return {}


def save(loadouts: dict[str, list[str]], config_dir: Path | None = None) -> None:
    try:
        path = _path(config_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(loadouts, indent=2), encoding="utf-8")
    except Exception:
        pass  # a read-only home should never take the visualiser down
