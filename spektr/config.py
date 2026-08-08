"""Settings that survive a restart.

Everything was previously lost on exit — mode, palette, sensitivity, gate. That
is a small thing that makes the tool feel disposable, so it's worth the forty
lines. Written as JSON rather than TOML because the standard library can write
JSON and cannot write TOML.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from . import palette

#: Band counts offered in the settings panel. 0 means "fit the terminal",
#: which is what every mode did unconditionally before this was settable.
BAND_CHOICES = (0, 8, 12, 16, 24, 32, 48, 64)

#: ``fps`` value meaning "run as fast as the display can show". Stored as the
#: sentinel rather than as a resolved number so the preference survives moving
#: the terminal to a different monitor: what was saved is the *intent*.
#: :func:`spektr.display.unlimited_fps` turns it into a real rate at startup.
FPS_UNLIMITED = 0

#: Highest explicit frame rate accepted, from ``--fps`` or the panel. Raised
#: from 120 because high-refresh panels are ordinary now and the pacer has no
#: opinion about the number.
FPS_MAX = 240

#: Shuffle scopes offered in the settings panel, in cycle order. ``off`` is one
#: of them rather than a separate switch, so the row reads as a single decision.
SHUFFLE_OFF = "off"
SHUFFLE_CHOICES = (SHUFFLE_OFF, "modes", "themes", "both")

#: What ``s`` turns shuffle *on* to, when there is no previous scope to restore.
SHUFFLE_DEFAULT = "both"

#: Frame rates offered in the settings panel. Anything in 15..240 is valid via
#: ``--fps``; these are just the useful stops — 24 and 48 for people who want
#: the film-ish look, 30/60 for the obvious ones, 90/120/144 for high-refresh.
#: Unlimited sits at the end, past every explicit stop.
FPS_CHOICES = (24, 30, 36, 48, 56, 60, 72, 90, 120, 144, FPS_UNLIMITED)


@dataclass
class Settings:
    mode: str = "Bars"
    theme: str = "classic"
    sensitivity: float = 1.0
    gate: float = 8e-5
    fps: int = 60
    chrome: bool = True
    #: How many bars to draw. 0 fits the terminal width; the default is a fixed
    #: 16 so the picture is consistent whatever the window size. Above the
    #: analyser's native 32 this rebuilds the band plan for real resolution
    #: rather than interpolating, which is why it lives here and not in the
    #: widget.
    bands: int = 16
    #: Screensaver-style auto-cycling of mode and theme. Remembered across
    #: restarts like everything else here — if you left it on, you wanted it
    #: on, not a surprise burst of quiet the next time you open a terminal.
    #: What shuffle cycles: ``off``, ``modes``, ``themes`` or ``both``. This is
    #: the on/off switch as well as the scope — a separate boolean plus a scope
    #: would have made "on, shuffling nothing" representable, which is a state
    #: with no meaning. Older configs stored a bool here; :meth:`clamp` migrates
    #: those, so ``true`` still means what it used to.
    shuffle: str = SHUFFLE_OFF
    #: Flipbook's selected reel and effect. Empty string means "unresolved,
    #: pick the first one" — see asciiart.restore(), which never touches disk
    #: itself, only records these names for the mode to resolve on first use.
    ascii_reel: str = ""
    ascii_fx: str = "warp"

    def clamp(self) -> "Settings":
        """Force every field back into range, replacing junk with the default.

        A hand-edited or half-written config can hold ``null``, a string, or a
        list where a number belongs, and ``float(None)`` raises — which used to
        propagate out of :func:`load` and take startup down. Nothing in a
        settings file is worth crashing over, so a value that will not convert
        is simply the default value.
        """
        self.sensitivity = _clamp_number(self.sensitivity, 0.15, 8.0, 1.0)
        self.gate = _clamp_number(self.gate, 1e-6, 2e-3, 8e-5)
        # The sentinel is recognised *before* clamping, not after. Clamping
        # first and then testing for 0 turns every negative number in a
        # hand-edited config into "unlimited", because the low bound floors
        # them onto the sentinel — ``fps: -3`` should be junk, not a request to
        # uncap. ``is not True/False`` because JSON booleans survive ``== 0``.
        raw = self.fps
        if (
            isinstance(raw, (int, float))
            and raw is not True
            and raw is not False
            and raw == FPS_UNLIMITED
        ):
            self.fps = FPS_UNLIMITED
        else:
            self.fps = int(_clamp_number(raw, 15, FPS_MAX, 60))
        bands = int(_clamp_number(self.bands, 0, 64, 16))
        self.bands = bands if bands == 0 else max(8, bands)
        self.mode = self.mode if isinstance(self.mode, str) and self.mode else "Bars"
        self.theme = self.theme if isinstance(self.theme, str) and self.theme else "classic"
        self.chrome = bool(self.chrome)
        # Migrates the old boolean. `is True` / `is False` rather than a
        # truthiness test, and checked before the membership test, because
        # `True == 1` and a bare `in` would not catch it.
        if self.shuffle is True:
            self.shuffle = SHUFFLE_DEFAULT
        elif self.shuffle is False or self.shuffle not in SHUFFLE_CHOICES:
            self.shuffle = SHUFFLE_OFF
        self.ascii_reel = self.ascii_reel if isinstance(self.ascii_reel, str) else ""
        self.ascii_fx = self.ascii_fx if self.ascii_fx in ("warp", "dissolve", "lit") else "warp"
        return self


def _clamp_number(value, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number != number:  # NaN survives float() and passes every comparison
        return float(default)
    return float(min(high, max(low, number)))


def _path():
    return palette.config_dir() / "config.json"


def load() -> Settings:
    """Read the settings file, falling back to defaults on anything unusable.

    Both the parse and the coercion are guarded: a missing file, malformed
    JSON, a top-level array instead of an object, and a field of the wrong
    type all end at the same place — a usable :class:`Settings`.
    """
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return Settings()
        known = {f.name for f in fields(Settings)}
        return Settings(**{k: v for k, v in raw.items() if k in known}).clamp()
    except Exception:
        return Settings()


def save(settings: Settings) -> None:
    try:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    except Exception:
        pass  # a read-only home should never take the visualiser down
