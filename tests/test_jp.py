"""The JP family has to go quiet when the music does.

Every one of these was a picture that did not match its audio, and none of
them showed up in the audit, because the audit asks whether a mode moves and
responds — not whether it *stops*.

* JP Pulse lit a full hot ring at its hub in silence: a peak of 0 floors
  to bulb 0, and bulb 0 was drawn as a peak. It also launched chasers off a
  timer, so the ring was flared by beats that were not there.
* JP Drift only ever gained sand. After half a minute of bass-heavy music
  the emptiest column was 55% lit, and when the music stopped the panel froze
  full — nothing ever took sand out. The first fix, a fixed drain, was a
  cutoff instead: every band steadily under ≈ 0.47 sank to nothing.
* The flat panel lit its bottom bulb at any positive level, and bands decay
  toward zero without reaching it, so every column kept one lit bulb for as
  long as the app stayed open after a track ended.
* Drift's first and last bands spilled into themselves on a collapse.

And one that is about legibility rather than timing: unlit bulbs were drawn in
ramp index 0, which on half the themes is the loudest colour on the ramp and on
most of the rest is nearly the background. They are now drawn in a colour
chosen by measured contrast, and that has to stay visible on every theme.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
import spektr.modes.jp as K  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette, contrast_ratio, rgb_to_hex  # noqa: E402
from spektr.render import SPACE  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])
FLAT = ["JP Bars", "JP Drift"]
FAMILY = FLAT + ["JP Pulse"]
DT = 1 / 60
_X = np.linspace(0.0, 1.0, N_BANDS)


def _bass(t: float) -> np.ndarray:
    """A kick on every half second over a quiet top end."""
    kick = np.exp(-(t % 0.5) / 0.12)
    lo = np.clip(0.35 + 0.6 * kick, 0, 1) * np.exp(-(_X / 0.18) ** 2)
    mid = 0.28 * np.exp(-((_X - 0.45) / 0.2) ** 2)
    return np.clip(lo + mid + 0.08, 0.0, 1.0)


def _ctx(w, h, state, t, bands, peaks=None, onsets=0, pal=PAL):
    if peaks is None:
        peaks = bands
    return Ctx(
        w=w, h=h, bands=bands, peaks=peaks, bands_l=bands, bands_r=bands,
        wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=int(t / DT), t=t,
        dt=DT, energy=float(np.mean(bands)), silent=float(np.max(bands)) < 0.02,
        palette=pal, state=state, onsets=onsets,
        onset_strength=0.8 if onsets else 0.0,
    )


def _play(name, w, h, state, t0, secs, source, onset_every=None, pal=PAL):
    """Render ``secs`` of ``source(t)``, smoothed and peak-held like the app."""
    fn = M.get(name).fn
    sm = np.zeros(N_BANDS)
    pk = np.zeros(N_BANDS)
    out = None
    for f in range(int(round(secs / DT))):
        t = t0 + f * DT
        raw = source(t)
        sm = sm + (raw - sm) * np.where(raw > sm, 0.6, 0.12)
        pk = np.maximum(pk - 0.6 * DT, sm)
        onset = int(onset_every is not None and (t % onset_every) < DT)
        out = fn(_ctx(w, h, state, t, sm.copy(), pk.copy(), onset, pal))
    return out


def _silence(t):
    return np.zeros(N_BANDS)


def _quiet_picture(name, codes, cidx, pal=PAL):
    """What is left on screen that is not an unlit bulb."""
    recede = K.recede_index(pal)
    if name in FLAT:
        return int(np.count_nonzero((codes != SPACE) & (codes != K._OFF)))
    return int(np.count_nonzero((codes != SPACE) & (cidx != recede)))


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
@pytest.mark.parametrize("name", FAMILY)
def test_silence_draws_only_unlit_bulbs(name, size):
    state: dict = {}
    codes, cidx = _play(name, *size, state, 0.0, 3.0, _silence)[:2]
    assert _quiet_picture(name, codes, cidx) == 0
    assert np.count_nonzero(codes != SPACE), "the panel itself vanished in silence"


@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
@pytest.mark.parametrize("name", FAMILY)
def test_every_variant_clears_after_the_music_stops(name, size):
    """Twenty seconds of bass, then silence: dark again within four seconds.

    The source after the stop is a decaying tail rather than zeros, because
    that is what the analyser hands over — a level that approaches zero and
    never reaches it is what kept the bottom bulb lit.
    """
    state: dict = {}
    _play(name, *size, state, 0.0, 20.0, _bass, onset_every=0.5)
    codes, cidx = _play(name, *size, state, 20.0, 4.0, lambda t: _bass(20.0) * 1e-7)[:2]
    assert _quiet_picture(name, codes, cidx) == 0


def test_pulse_launches_chasers_only_on_onsets():
    state: dict = {}
    steady = lambda t: np.full(N_BANDS, 0.5)  # noqa: E731
    _play("JP Pulse", 120, 40, state, 0.0, 5.0, steady)
    born = next(v for k, v in state.items() if k[0] == "jp_pulse")["born"]
    assert np.all(born < 0), "a chaser launched with no onset"

    _play("JP Pulse", 120, 40, state, 5.0, 0.2, steady, onset_every=0.1)
    assert np.any(born >= 5.0), "an onset no longer launches a chaser"


def test_pulse_draws_no_peak_at_the_hub_in_silence():
    """The specific failure: a peak of 0 landing on bulb 0 of every spoke."""
    state: dict = {}
    fn = M.get("JP Pulse").fn
    z = np.zeros(N_BANDS)
    for f in range(30):
        codes, cidx = fn(_ctx(120, 40, state, f * DT, z, z))[:2]
    assert np.all(cidx[codes != SPACE] == K.recede_index(PAL))


def _drift_pile(state):
    return next(v for k, v in state.items() if k[0] == "jp_drift")["h"]


def _steady_drift(level, secs=12.0):
    """Hold one level on every band; return mean shown height and collapses/s."""
    state: dict = {}
    fn = M.get("JP Drift").fn
    lv = np.full(N_BANDS, level)
    heights, drops, prev = [], 0, None
    for f in range(int(secs / DT)):
        fn(_ctx(120, 40, state, f * DT, lv))
        pile = _drift_pile(state).copy()
        if prev is not None:
            drops += int(np.sum(prev - pile > 0.3))
        prev = pile
        if f * DT > secs - 5.0:
            heights.append(np.clip(pile, 0.0, 1.0).mean())
    return float(np.mean(heights)), drops / len(pile) / secs


def test_drift_answers_quiet_input_without_accumulating():
    """No cutoff at the bottom, no pile-up at the top.

    A fixed drain sank every band under ≈ 0.47 to an empty column; no drain
    let every band climb to the angle of repose. Height-dependent drainage
    has to do neither: quiet input holds a small height, heights rise with
    the level, and only loud input keeps collapsing.
    """
    quiet, quiet_av = _steady_drift(0.2)
    medium, medium_av = _steady_drift(0.5)
    loud, loud_av = _steady_drift(0.9)
    assert 0.05 < quiet < 0.2, f"steady 0.2 settled at {quiet:.2f}"
    assert quiet < medium < 0.8, f"steady 0.5 settled at {medium:.2f}"
    assert quiet_av == 0 and medium_av == 0, "a band below the avalanche level collapsed"
    assert loud_av > 1.0, f"loud input avalanches only {loud_av:.2f} times a second per band"
    assert loud < 0.95, f"loud input accumulated to {loud:.2f}"


def test_drift_still_avalanches_on_bass():
    state: dict = {}
    fn = M.get("JP Drift").fn
    drops, prev = 0, None
    sm = np.zeros(N_BANDS)
    for f in range(int(20.0 / DT)):
        raw = _bass(f * DT)
        sm = sm + (raw - sm) * np.where(raw > sm, 0.6, 0.12)
        fn(_ctx(120, 40, state, f * DT, sm.copy()))
        pile = _drift_pile(state).copy()
        if prev is not None and f * DT > 5.0:
            drops += int(np.sum(prev - pile > 0.3))
        prev = pile
    assert drops > 5, f"{drops} collapses in 15 s of kicks"
    # the top quarter is fed 0.08: it should hold a trace, not a column
    top = _drift_pile(state)[3 * len(prev) // 4:]
    assert top.max() < 0.1, f"the quiet top of the spectrum filled to {top.max():.2f}"


def _loud_drift(level_fn, secs=10.0):
    """Collapse statistics under loud input, after a two-second settle."""
    state: dict = {}
    fn = M.get("JP Drift").fn
    prev = prev_hit = None
    hits = chained = flashed = band_frames = reversals = 0
    crests: list = []
    for f in range(int(secs / DT)):
        fn(_ctx(120, 40, state, f * DT, level_fn(f * DT)))
        st = next(v for k, v in state.items() if k[0] == "jp_drift")
        pile = st["h"].copy()
        crest = K.crest_bulb(np.clip(pile, 0.0, 1.0), 40)
        if prev is not None and f * DT > 2.0:
            hit = (prev - pile) > 0.3
            hits += int(hit.sum())
            if prev_hit is not None:
                near = np.zeros_like(hit)
                near[1:] |= prev_hit[:-1]
                near[:-1] |= prev_hit[1:]
                chained += int((hit & near).sum())
            prev_hit = hit
            flashed += int((st["flash"] > 0.3).sum())
            band_frames += len(pile)
            if len(crests) == 2:
                reversals += int(((crests[0] == crest) & (crests[1] != crest)).sum())
        crests = (crests + [crest])[-2:]
        prev = pile
    n = len(prev)
    return {
        "per_band_s": hits / n / (secs - 2.0),
        "chained": chained / max(hits, 1),
        "flashed": flashed / max(band_frames, 1),
        "reversals_per_band_s": reversals / n / (secs - 2.0),
    }


@pytest.mark.parametrize("shape", ["uniform", "spectrum"])
def test_drift_loud_avalanches_stay_distinct(shape):
    """Loud input: collapses keep happening, but as separate events.

    Before, every band sat at the angle of repose and a collapse tipped its
    neighbours over on the very next frame: 68–77% of collapses were that
    ping-pong, column tops reversed within two frames, and the flash was
    retriggered faster than it faded, so the panel stayed red. And a row of
    bands collapsing on the same frame refilled each other straight back over
    the edge, which left steady full-scale input a solid, motionless panel.
    """
    if shape == "uniform":
        level = lambda t: np.full(N_BANDS, 0.95)  # noqa: E731
    else:
        level = lambda t: np.clip(0.95 * np.exp(-((_X - 0.35) / 0.6) ** 2), 0, 1)  # noqa: E731
    r = _loud_drift(level)
    assert r["per_band_s"] > 0.8, f"loud input barely avalanches: {r['per_band_s']:.2f}/band/s"
    assert r["chained"] < 0.1, f"{100 * r['chained']:.0f}% of collapses chained off the previous frame"
    assert r["reversals_per_band_s"] < 0.1, f"column tops reversed {r['reversals_per_band_s']:.2f}/band/s"
    assert r["flashed"] < 0.5, f"{100 * r['flashed']:.0f}% of the panel sat flashed"


@pytest.mark.parametrize("edge", [0, -1])
def test_drift_edge_spill_leaves_the_panel(edge):
    """A collapsing edge band feeds its one neighbour, never itself."""
    state: dict = {}
    fn = M.get("JP Drift").fn
    z = np.zeros(N_BANDS)
    fn(_ctx(120, 40, state, 0.0, z))
    st = next(v for k, v in state.items() if k[0] == "jp_drift")
    pile = st["h"]
    pile[:] = 0.0
    pile[edge] = 1.25
    fn(_ctx(120, 40, state, DT, z))

    drained = (K._DRIFT_DRAIN_LIN + K._DRIFT_DRAIN_SQ * 1.25) * 1.25 * DT
    excess = (1.25 - drained) - 0.55
    inner = 1 if edge == 0 else len(pile) - 2
    assert pile[edge] <= 0.55 + 0.03 + 1e-9, "the collapsing band took its own spill back"
    assert pile[inner] == pytest.approx(excess * 0.35)
    assert np.count_nonzero(pile) == 2


def test_peaks_and_trails_never_cover_a_lit_bulb():
    """Lit bulbs are the heaviest ink and nothing is drawn over them."""
    state: dict = {}
    fn = M.get("JP Bars").fn
    sm = np.zeros(N_BANDS)
    pk = np.zeros(N_BANDS)
    for f in range(600):
        t = f * DT
        raw = _bass(t)
        sm = sm + (raw - sm) * np.where(raw > sm, 0.6, 0.12)
        pk = np.maximum(pk - 0.6 * DT, sm)
        ctx = _ctx(120, 40, state, t, sm.copy(), pk.copy())
        codes = fn(ctx)[0]
        col_band, active = M.band_columns(120, ctx.n_display)
        levels = np.where(active, ctx.display_bands(ctx.n_display)[col_band], 0.0)
        lit_rows = K.crest_bulb(levels, 40) + 1              # bulbs lit per column
        for c in np.flatnonzero(active & (lit_rows > 0)):
            rws = K.bulb_row_index(40)[: lit_rows[c]]
            assert np.all(codes[rws, c] == K._LED), f"column {c} lost a lit bulb at frame {f}"


@pytest.mark.parametrize("theme", sorted(BUILTIN))
def test_unlit_bulbs_stay_visible_on_every_theme(theme):
    """Visible, not necessarily quiet: on about thirty built-ins every ramp
    entry is well above the 2.2 target, so there the dots recede by glyph size
    alone. What must never happen is the old failure on the dark ramps — a
    colour at contrast 1.2 that is simply not there."""
    pal = Palette(BUILTIN[theme])
    rgb = np.clip(np.rint(pal.rgb), 0, 255).astype(int)
    ratios = [contrast_ratio(rgb_to_hex(c), pal.theme.bg) for c in rgb]
    got = ratios[K.recede_index(pal)]
    assert got >= min(1.8, max(ratios)), f"unlit bulbs at contrast {got:.2f} on {theme}"
    assert got <= max(ratios)


def test_the_cut_modes_resolve_to_the_family_meter():
    bars = M.get("JP Bars")
    assert M.get("JP Keys") is bars
    assert M.get("JP Sweep") is bars
    assert "JP Keys" not in M.names()
