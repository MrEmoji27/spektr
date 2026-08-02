"""The stable surface for plugin authors.

Everything a visualiser plugin needs is re-exported here, and nothing else is
promised. Import from ``spektr.api`` and the internals stay free to move::

    from spektr.api import mode, pack_braille, cell_max

Importing from ``spektr.modes`` or ``spektr.render`` directly will usually work
and may break on any release.

A mode is a function taking a :class:`Ctx` and returning two ``(h, w)`` int
arrays: Unicode codepoints, and indices into the active palette ramp
(``0..RAMP_STEPS-1``, cool to hot). Optionally a third array of background
ramp indices, for modes that colour whole cells.

    @mode("Heartbeat", blurb="pulses on the kick")
    def heartbeat(ctx):
        level = ctx.range(0.0, 0.15)
        codes = np.full((ctx.h, ctx.w), ord("*"), dtype=np.int32)
        cidx = np.full((ctx.h, ctx.w), ctx.palette.index(level), dtype=np.int32)
        return codes, cidx

See ``docs/plugins.md`` for the longer version.
"""

from __future__ import annotations

#: Bumped when anything below changes incompatibly. A plugin may declare
#: ``SPEKTR_API = 1`` at module level; the loader refuses to run it if the
#: major version no longer matches, rather than letting it fail obscurely.
API_VERSION = 1

from .analysis import resample_bands
from .modes import (
    Ctx,
    Mode,
    ModeNameTaken,
    band_columns,
    empty,
    mode,
    spread,
)
from .palette import RAMP_STEPS, Palette
from .render import (
    BLOCKS_LEFT,
    BLOCKS_UP,
    BRAILLE_BASE,
    SHADES,
    SPACE,
    blocks_from_levels,
    cell_max,
    cell_mean,
    noise,
    pack_braille,
)

__all__ = [
    "API_VERSION",
    # registration
    "mode",
    "Ctx",
    "Mode",
    "ModeNameTaken",
    # palette
    "Palette",
    "RAMP_STEPS",
    # dot-grid primitives
    "pack_braille",
    "cell_max",
    "cell_mean",
    "noise",
    "BRAILLE_BASE",
    # block characters
    "blocks_from_levels",
    "BLOCKS_UP",
    "BLOCKS_LEFT",
    "SHADES",
    "SPACE",
    # band layout
    "resample_bands",
    "band_columns",
    "spread",
    "empty",
]
