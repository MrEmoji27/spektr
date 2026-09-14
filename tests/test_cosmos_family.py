"""The cosmos family: a mostly dark sky, and the music as events in it.

Inspected live, the four modes each broke that bargain in their own way, and
the family shared one cause underneath:

* faintness was a low ramp index, and on gruvbox a low index is the brightest
  colour on the ramp, so faint stars, the galactic band and fading trails came
  out as loud as the events they were meant to sit behind;
* Shooting Star threw on a coin toss for every onset, hats included, so
  whether a given hit had thrown anything was a guess;
* Constellations hopped anywhere inside a wide reach and revisited its own
  stars, tangling into knots that sat at the edge cap long after the music;
* Star Trails let every arc fill the whole gap to its neighbour and blurred
  even at rest, so every section of a song was the same wall of rings;
* Supernova's first catastrophe came off a four-to-eight-second timer, before
  any hit had asked for one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
import spektr.modes.cosmos as cosmos  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx, bg_contrast, contrast_ramp  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import BRAILLE_BASE, SPACE  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])
DT = 1 / 60


def _ctx(state, t, energy=0.0, onset=0, strength=0.0, drive=0.0, w=120, h=40):
    bands = np.full(N_BANDS, energy)
    return Ctx(w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
               wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=int(round(t / DT)), t=t,
               dt=DT, energy=energy, silent=energy < 0.01, palette=PAL, state=state,
               onsets=onset, onset_strength=strength if onset else 0.0, flux=drive)


def _state(state, key):
    return next(v for k, v in state.items() if k[0] == key)


# ── colour ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theme", sorted(BUILTIN))
def test_faint_sky_is_quieter_than_bright_sky_on_every_theme(theme):
    """Brightness in a sky field must be brightness on screen, measured."""
    pal = Palette(BUILTIN[theme])
    cr = bg_contrast(pal)
    idx = contrast_ramp(pal, np.linspace(0.0, 1.0, 11))
    c = cr[idx]
    assert c[0] >= 1.5 or c[0] == cr.min(), f"the faint end is invisible on {theme} ({c[0]:.2f})"
    assert c[-1] == cr.max()
    assert np.all(np.diff(c) >= -0.05 * (c[-1] - c[0])), f"contrast steps backwards on {theme}"


@pytest.mark.parametrize("theme", ["gruvbox", "flexoki-light", "ethereal", "sapphire"])
@pytest.mark.parametrize("name", ["Shooting Star", "Constellations", "Star Trails", "Supernova"])
def test_a_quiet_sky_is_drawn_at_the_quiet_end_of_the_theme(name, theme):
    """The modes must use it: a silent sky is faint things, drawn faint.

    Through ``ctx.ramp`` the galactic band, the dim stars and the fading arcs
    took low indices, which on gruvbox are the loudest colours on the ramp.
    """
    pal = Palette(BUILTIN[theme])
    fn = M.get(name).fn
    state: dict = {}
    for f in range(int(4 / DT)):
        bands = np.zeros(N_BANDS)
        ctx = Ctx(w=120, h=40, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
                  wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=f, t=f * DT, dt=DT,
                  energy=0.0, silent=True, palette=pal, state=state)
        codes, cidx = fn(ctx)[:2]
    cr = bg_contrast(pal)
    lit = (codes != BRAILLE_BASE) & (codes != SPACE)
    faint, loud = cr[contrast_ramp(pal, 0.0)], cr.max()
    median = float(np.median(cr[cidx[lit]]))
    assert median <= faint + 0.5 * (loud - faint), (
        f"{name} on {theme}: a silent sky's median contrast is {median:.2f}, "
        f"nearer the loudest colour ({loud:.2f}) than the quietest ({faint:.2f})")


# ── Shooting Star ────────────────────────────────────────────────────────────

def test_a_hard_hit_always_throws_and_a_hat_never_does():
    fn = M.get("Shooting Star").fn
    for trial in range(12):
        # Each trial plays a different length of quiet sky first, so the
        # generator is at a different point when the hits land; a fresh state
        # alone would replay one identical draw twelve times.
        state: dict = {}
        t = 0.0
        for f in range(1 + 37 * trial):
            t = f * DT
            fn(_ctx(state, t))
        st = _state(state, "shooting_star")
        st["my"][:] = -1.0
        fn(_ctx(state, t + DT, energy=0.4, onset=1, strength=0.35))
        assert (st["my"] >= 0).sum() <= 1, "a hi-hat onset broke a cluster loose"
        st["my"][:] = -1.0
        fn(_ctx(state, t + 0.5, energy=0.4, onset=1, strength=0.9))
        assert (st["my"] >= 0).sum() >= 3, "a hard hit threw nothing"


def test_a_silent_sky_stays_nearly_empty_of_meteors():
    fn = M.get("Shooting Star").fn
    state: dict = {}
    busy = 0
    for f in range(int(20 / DT)):
        fn(_ctx(state, f * DT))
        busy += int((_state(state, "shooting_star")["my"] >= 0).any())
    assert busy / (20 / DT) < 0.15, f"meteors crossed a silent sky {100 * busy / (20 / DT):.0f}% of the time"


# ── Constellations ───────────────────────────────────────────────────────────

def _play_groove(fn, state, secs, t0=0.0):
    for f in range(int(secs / DT)):
        t = t0 + f * DT
        beat = (t % 0.5) < DT
        fn(_ctx(state, t, energy=0.55, onset=int(beat), strength=0.85, drive=0.6))


def test_a_figure_never_doubles_back_through_its_own_stars():
    fn = M.get("Constellations").fn
    state: dict = {}
    _play_groove(fn, state, 30.0)
    st = _state(state, "constellations")
    figure, figures = [], []
    for e in st["edges"]:
        if e[6] and figure:
            figures.append(figure)
            figure = []
        figure.append(((e[0], e[1]), (e[2], e[3])))
    figures.append(figure)
    for fig in figures:
        assert len(fig) <= cosmos._CHART_FIGURE_EDGES
        heads = [b for _, b in fig]
        assert len(heads) == len(set(heads)), "a figure revisited one of its own stars"


def test_figures_fade_once_the_music_stops():
    fn = M.get("Constellations").fn
    state: dict = {}
    _play_groove(fn, state, 20.0)
    at_stop = len(_state(state, "constellations")["edges"])
    for f in range(int(10 / DT)):
        fn(_ctx(state, 20.0 + f * DT))
    after = len(_state(state, "constellations")["edges"])
    assert after < at_stop / 2, f"{after} of {at_stop} edges still up ten seconds after the music"


# ── Star Trails ──────────────────────────────────────────────────────────────

def _lit(codes):
    return float(((codes != BRAILLE_BASE) & (codes != SPACE)).mean())


def test_a_driven_sky_stays_a_sky_of_arcs():
    fn = M.get("Star Trails").fn
    state: dict = {}
    lit = []
    for f in range(int(20 / DT)):
        t = f * DT
        codes = fn(_ctx(state, t, energy=0.8, onset=int((t % 0.5) < DT), strength=0.9, drive=0.85))[0]
        if t > 6.0:
            lit.append(_lit(codes))
    # 21% as tuned; letting each arc fill its whole gap again measured 30%
    assert np.mean(lit) < 0.25, f"a driven exposure covered {100 * np.mean(lit):.0f}% of the cells"


def test_a_resting_sky_does_not_blur():
    """No smear at the sidereal floor: the arcs of a quiet sky stay pin-sharp."""
    assert cosmos._TRAIL_BLUR_BASE == 0.0
    fn = M.get("Star Trails").fn
    state: dict = {}
    for f in range(int(8 / DT)):
        fn(_ctx(state, f * DT))
    acc = _state(state, "star_trails")["acc"]
    lit = acc > 0.07
    # a one-dot-wide arc has at most two lit 4-neighbours along its length;
    # a blurred one fills in across its width
    nb = np.zeros_like(acc, dtype=np.int32)
    nb[1:, :] += lit[:-1, :]
    nb[:-1, :] += lit[1:, :]
    nb[:, 1:] += lit[:, :-1]
    nb[:, :-1] += lit[:, 1:]
    thick = (lit & (nb >= 4)).sum() / max(lit.sum(), 1)
    assert thick < 0.05, f"{100 * thick:.0f}% of a resting sky's arc dots are buried inside a smear"


# ── Supernova ────────────────────────────────────────────────────────────────

def test_no_catastrophe_before_a_hit_asks_for_one():
    fn = M.get("Supernova").fn
    state: dict = {}
    for f in range(int(20 / DT)):
        fn(_ctx(state, f * DT, energy=0.3))
        # every frame, not only the last: an event that fired early is over
        # and gone well before twenty seconds
        assert _state(state, "supernova")["ev"] is None,             f"a nova fired off the idle clock at {f * DT:.1f} s, before any hit"


def test_a_hard_hit_on_a_clear_sky_can_start_one():
    fn = M.get("Supernova").fn
    fired = 0
    for trial in range(10):
        state: dict = {}
        fn(_ctx(state, 1.0))
        for k in range(6):
            fn(_ctx(state, 1.1 + 0.5 * k, energy=0.7, onset=1, strength=0.95))
        fired += _state(state, "supernova")["ev"] is not None
    assert fired >= 9, f"hard hits on a clear sky started a nova in only {fired} of 10 tries"
