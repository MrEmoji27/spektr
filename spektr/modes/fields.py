"""Compatibility surface for the field-mode families.

The implementations live in focused modules. Attribute access stays live so
tests, plugins, and embedders that patch an old private path still reach the
object the renderer uses.
"""

from . import field_chladni as _chladni
from . import field_dither as _dither
from . import field_hearts as _hearts
from . import field_kaleidoscope as _kaleidoscope
from . import field_meters as _meters
from . import field_spectral as _spectral

_FAMILIES = (_spectral, _chladni, _meters, _kaleidoscope, _dither, _hearts)


def __getattr__(name):
    for module in _FAMILIES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'spektr.modes.fields' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(*(vars(module) for module in _FAMILIES)))
