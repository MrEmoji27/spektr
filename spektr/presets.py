"""Named snapshots of mode + theme + settings.

Saved as JSON next to config.json — same directory, same guarded load/save
shape, because a preset file is exactly as disposable as the main config: a
malformed one should cost you your presets, not the app.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import palette
from .config import _clamp_number

#: What a preset captures. Not chrome or shuffle — those are session/display
#: choices, not part of "the look", and forcing them on preset load would
#: mean recalling a preset could hide your header or start auto-cycling
#: without asking.
FIELDS = ("mode", "theme", "fps", "bands", "sensitivity", "gate")

#: Backfilled onto a hand-edited entry that only bothered with mode/theme, so
#: the apply path in app.py can assume every field is always present rather
#: than defending against a KeyError on every lookup. Mirrors config.Settings'
#: own defaults.
_DEFAULTS = {"fps": 60, "bands": 16, "sensitivity": 1.0, "gate": 8e-5}


def _path(config_dir: Path | None = None):
    root = config_dir if config_dir is not None else palette.config_dir()
    return root / "presets.json"


def load(config_dir: Path | None = None) -> dict[str, dict]:
    """Read the presets file, dropping anything malformed rather than failing.

    Same posture as :func:`config.load`: a missing file, broken JSON, or a
    preset entry with the wrong shape all end at "no preset there", not a
    crash. Unknown fields in an entry are dropped rather than kept, so a
    hand-edited file can't smuggle in something :func:`app.py`'s apply-preset
    path isn't expecting.

    ``config_dir`` overrides where the presets file lives; None means the
    platform default from :func:`palette.config_dir`.
    """
    try:
        raw = json.loads(_path(config_dir).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict] = {}
        for name, values in raw.items():
            if not (isinstance(name, str) and name and isinstance(values, dict)):
                continue
            entry = {k: values[k] for k in FIELDS if k in values}
            if not (isinstance(entry.get("mode"), str) and isinstance(entry.get("theme"), str)):
                continue
            for key, default in _DEFAULTS.items():
                entry.setdefault(key, default)
            # a hand-edited file is the only way one of these goes bad — the
            # app only ever writes values it already validated on the way in
            entry["fps"] = int(_clamp_number(entry["fps"], 15, 120, 60))
            bands = int(_clamp_number(entry["bands"], 0, 64, 16))
            entry["bands"] = bands if bands == 0 else max(8, bands)
            entry["sensitivity"] = _clamp_number(entry["sensitivity"], 0.15, 8.0, 1.0)
            entry["gate"] = _clamp_number(entry["gate"], 1e-6, 2e-3, 8e-5)
            out[name] = entry
        return out
    except Exception:
        return {}


def save(presets: dict[str, dict], config_dir: Path | None = None) -> None:
    try:
        path = _path(config_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(presets, indent=2), encoding="utf-8")
    except Exception:
        pass  # a read-only home should never take the visualiser down
