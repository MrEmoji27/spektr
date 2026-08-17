"""Every glyph the Android picker can reach must exist in a font on the device.

The desktop runs in a terminal, so the font is the user's problem and a
missing glyph is visibly the terminal's doing. The APK is not: it ships one
font, DejaVu Sans, and a mode emitting a codepoint DejaVu lacks renders as a
grid of tofu boxes — which on a visualiser reads as a crash rather than as a
missing glyph, and which no amount of looking at the Python output would
reveal.

That is not hypothetical. ``Matrix`` draws entirely in halfwidth katakana
(U+FF71-FF9D) and DejaVu Sans has none of it, so the first build of the mode
picker would have shipped one mode in fifty-two that was forty rows of boxes.
The renderer now falls back to the platform's own fallback chain for anything
DejaVu is missing; this file is what decides which codepoints that is, and
fails if a mode reaches for something *nothing* can draw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spektr.analysis import N_BANDS          # noqa: E402
from spektr.modes import Ctx                 # noqa: E402
from spektr.palette import BUILTIN, Palette   # noqa: E402
import spektr.modes as M                     # noqa: E402

TTLib = pytest.importorskip("fontTools.ttLib")

FONT = ROOT / "android" / "app" / "src" / "main" / "res" / "font" / "dejavu_sans.ttf"

#: The tablet is 2560x1536 and the renderer targets ~40 rows, clamped to
#: 8..400 by 8..200. Several shapes because modes branch on size and aspect.
SIZES = ((120, 40), (200, 60), (60, 24), (400, 100))

#: Codepoints DejaVu Sans does not have, which the renderer draws with a second
#: Paint on Typeface.MONOSPACE — that resolves through the platform fallback
#: chain, which reaches Noto Sans CJK. Listed rather than merely tolerated: a
#: mode that starts needing the fallback should be a decision, not a surprise,
#: because the fallback draws one cell at a time and is the slower path.
FALLBACK = frozenset(range(0xFF71, 0xFF9E))    # halfwidth katakana, for Matrix


def _codepoints(mode, w: int, h: int, *, silent: bool, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    pal = Palette(BUILTIN["gruvbox"])
    state: dict = {}
    seen: set[int] = set()
    for i in range(24):
        if silent:
            bands = np.zeros(N_BANDS, dtype=np.float32)
            wave = np.zeros(1024, dtype=np.float32)
        else:
            bands = rng.random(N_BANDS).astype(np.float32)
            wave = rng.standard_normal(1024).astype(np.float32) * 0.4
        out = mode.fn(
            Ctx(
                w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
                wave=wave, stereo=np.stack([wave, np.roll(wave, 3)], axis=1),
                frame=i, t=i / 60.0, dt=1 / 60.0,
                energy=float(bands.mean()), silent=silent,
                palette=pal, state=state, bars=N_BANDS,
                onset_seq=i // 6, onsets=1 if i % 6 == 0 else 0,
                onset_strength=float(rng.random()), flux=float(rng.random()),
                tempo_bpm=124.0, beat_phase=(i % 30) / 30.0,
            )
        )
        seen.update(int(c) for c in np.unique(out[0]))
    return seen


@pytest.fixture(scope="module")
def emitted() -> dict[str, set[int]]:
    """Every codepoint each offered mode produces, over a spread of conditions."""
    return {
        m.name: set().union(
            *(
                _codepoints(m, w, h, silent=s, seed=seed)
                for (w, h) in SIZES
                for s, seed in ((False, 1), (True, 2))
            )
        )
        for m in M.listed()
    }


@pytest.fixture(scope="module")
def cmap() -> set[int]:
    assert FONT.is_file(), f"the APK's grid font is missing: {FONT}"
    font = TTLib.TTFont(FONT)
    covered: set[int] = set()
    for table in font["cmap"].tables:
        covered.update(table.cmap)
    return covered


def test_the_font_covers_every_offered_mode_or_the_fallback_does(emitted, cmap):
    """Nothing may need a glyph that is neither in DejaVu nor in the fallback set."""
    stranded: dict[str, set[int]] = {}
    for name, cps in emitted.items():
        missing = {c for c in cps if c and c not in cmap and c not in FALLBACK}
        if missing:
            stranded[name] = missing
    assert not stranded, "modes emitting codepoints nothing can draw:\n" + "\n".join(
        f"  {n}: " + ", ".join(f"U+{c:04X}" for c in sorted(cs)) for n, cs in stranded.items()
    )


def test_the_fallback_set_is_still_needed_and_still_enough(emitted, cmap):
    """Keeps FALLBACK honest in both directions.

    If a mode stops needing katakana the slow per-cell path should go with it;
    if the set grows stale in the other direction the test above starts
    failing. Either way the constant tracks something real rather than being
    a comment about 2026.
    """
    needed = {c for cps in emitted.values() for c in cps if c and c not in cmap}
    assert needed, "nothing needs the fallback — delete it and the second Paint"
    assert needed <= FALLBACK, (
        "the fallback set is missing codepoints modes emit: "
        + ", ".join(f"U+{c:04X}" for c in sorted(needed - FALLBACK))
    )


def test_matrix_is_the_reason_and_says_so(emitted, cmap):
    """The concrete case, pinned so a rename cannot quietly drop the coverage."""
    assert "Matrix" in emitted, "Matrix left the offered list — is FALLBACK still needed?"
    assert emitted["Matrix"] - cmap, "Matrix no longer needs the fallback"


def test_the_hidden_modes_are_hidden_for_a_reason_the_font_agrees_with(cmap):
    """They draw through Unicode 16 octants, which is why Android does not list them.

    Asserted rather than assumed: if a future font gains U+1CD00 this test
    fails, and that is the moment to reconsider offering the twelve rather
    than a year later by accident.
    """
    hidden = [m for m in M.MODES if m.hidden]
    assert hidden
    octants = set().union(*(_codepoints(m, 120, 40, silent=False, seed=3) for m in hidden))
    beyond = {c for c in octants if c >= 0x1CD00}
    assert beyond, "the hidden modes stopped emitting octants — why are they hidden?"
    assert not (beyond & cmap), "the shipped font now has octants; the twelve could be offered"
