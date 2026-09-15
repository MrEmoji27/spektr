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


def _drift_state(state):
    return next(v for k, v in state.items() if k[0] == "jp_drift")


def _drift_run(level_fn, secs=12.0, beat_every=None, settle=2.0):
    """Play Drift a passage; return avalanche statistics after ``settle`` s.

    An avalanche is counted where it starts — a band whose pour goes from
    empty to non-empty — because it now takes a fifth of a second to pour
    out, and a per-frame drop test no longer sees one.
    """
    state: dict = {}
    fn = M.get("JP Drift").fn
    sm = np.zeros(N_BANDS)
    prev_pour = None
    last_beat = -9.0
    starts = on_beat = tinted = band_frames = reversals = 0
    heights, crests = [], []
    for f in range(int(secs / DT)):
        t = f * DT
        onset = int(beat_every is not None and (t % beat_every) < DT)
        if onset:
            last_beat = t
        raw = level_fn(t)
        sm = sm + (raw - sm) * np.where(raw > sm, 0.6, 0.12)
        fn(_ctx(120, 40, state, t, sm.copy(), onsets=onset))
        st = _drift_state(state)
        pile, pour = st["h"], st["pour"]
        crest = K.crest_bulb(np.clip(pile, 0.0, 1.0), 40)
        if prev_pour is not None and t > settle:
            new = (prev_pour <= 0.0) & (pour > 0.0)
            starts += int(new.sum())
            if t - last_beat <= K._DRIFT_ARMED_S + DT:
                on_beat += int(new.sum())
            tinted += int((st["flash"] > 0.3).sum())
            band_frames += len(pile)
            heights.append(float(np.clip(pile, 0.0, 1.0).mean()))
            if len(crests) == 2:
                reversals += int(((crests[0] == crest) & (crests[1] != crest)).sum())
        crests = (crests + [crest])[-2:]
        prev_pour = pour.copy()
    n = len(prev_pour)
    span = secs - settle
    return {
        "starts": starts,
        "per_band_s": starts / n / span,
        "on_beat": on_beat / max(starts, 1),
        "tinted": tinted / max(band_frames, 1),
        "reversals_per_band_s": reversals / n / span,
        "height": float(np.mean(heights)),
    }


def _steady(level):
    return lambda t: np.full(N_BANDS, level)


def _loud_spectrum(t):
    return np.clip(0.95 * np.exp(-((_X - 0.35) / 0.6) ** 2), 0, 1)


def test_drift_answers_quiet_input_without_accumulating():
    """No cutoff at the bottom, no pile-up at the top.

    A fixed drain sank every band under ≈ 0.47 to an empty column; no drain
    let every band climb to the angle of repose. Height-dependent drainage
    has to do neither: quiet input holds a small height, heights rise with
    the level, and only loud input keeps collapsing.
    """
    quiet = _drift_run(_steady(0.2))
    medium = _drift_run(_steady(0.5))
    loud = _drift_run(_steady(0.9))
    assert 0.05 < quiet["height"] < 0.2, f"steady 0.2 settled at {quiet['height']:.2f}"
    assert quiet["height"] < medium["height"] < 0.8, f"steady 0.5 settled at {medium['height']:.2f}"
    assert quiet["starts"] == 0 and medium["starts"] == 0, "a band below the avalanche level collapsed"
    assert loud["per_band_s"] > 0.3, f"loud input avalanches only {loud['per_band_s']:.2f}/band/s"
    assert loud["height"] < 0.9, f"loud input left the columns standing at {loud['height']:.2f}"


def test_drift_avalanches_land_on_the_beat():
    """A column past the edge topples on a hit, so avalanches keep time.

    Tested on the onset frame alone this produced no avalanches at all on
    bass: the onset is detected on a kick's first frame and the sand it feeds
    crosses the edge a moment later. Hence the short window after each hit.
    """
    bass = _drift_run(_bass, secs=20.0, beat_every=0.5, settle=5.0)
    assert bass["starts"] > 5, f"bass triggered {bass['starts']} avalanches in 15 s"
    assert bass["on_beat"] > 0.8, f"only {100 * bass['on_beat']:.0f}% of bass avalanches landed on a beat"
    loud = _drift_run(_loud_spectrum, beat_every=0.5)
    assert loud["on_beat"] > 0.8, f"only {100 * loud['on_beat']:.0f}% of loud avalanches landed on a beat"


def test_drift_quiet_top_stays_low_under_bass():
    state: dict = {}
    fn = M.get("JP Drift").fn
    sm = np.zeros(N_BANDS)
    for f in range(int(20.0 / DT)):
        raw = _bass(f * DT)
        sm = sm + (raw - sm) * np.where(raw > sm, 0.6, 0.12)
        fn(_ctx(120, 40, state, f * DT, sm.copy(), onsets=int((f * DT) % 0.5 < DT)))
    pile = _drift_state(state)["h"]
    # the top quarter is fed 0.08: it should hold a trace, not a column
    top = pile[3 * len(pile) // 4:]
    assert top.max() < 0.1, f"the quiet top of the spectrum filled to {top.max():.2f}"


@pytest.mark.parametrize("shape", ["uniform", "spectrum"])
@pytest.mark.parametrize("beats", [None, 0.5])
def test_drift_loud_avalanches_stay_distinct(shape, beats):
    """Loud input: collapses keep happening, but as separate, readable events.

    Before, a collapse tipped its neighbours over on the very next frame and
    they tipped it back, column tops reversed within two frames, the flash was
    retriggered faster than it faded so the panel stayed red, and a row
    collapsing on one frame refilled itself straight back over the edge.
    """
    level = _steady(0.95) if shape == "uniform" else _loud_spectrum
    r = _drift_run(level, beat_every=beats)
    assert r["per_band_s"] > 0.3, f"loud input barely avalanches: {r['per_band_s']:.2f}/band/s"
    assert r["reversals_per_band_s"] < 0.1, f"column tops reversed {r['reversals_per_band_s']:.2f}/band/s"
    assert r["tinted"] < 0.35, f"{100 * r['tinted']:.0f}% of the panel sat tinted"
    assert r["height"] < 0.9, f"the columns stood at {r['height']:.2f} on average"


def test_an_avalanche_pours_out_rather_than_teleporting():
    """A collapse takes several frames and only ever goes down while it does."""
    state: dict = {}
    fn = M.get("JP Drift").fn
    z = np.zeros(N_BANDS)
    fn(_ctx(120, 40, state, 0.0, z))
    st = _drift_state(state)
    st["h"][:] = 0.0
    st["h"][8] = 1.25
    heights = [1.25]
    for f in range(1, 40):
        fn(_ctx(120, 40, state, f * DT, z, onsets=int(f == 1)))
        heights.append(float(st["h"][8]))
    steps = np.diff(heights)
    falling = np.flatnonzero(steps < -1e-9)
    assert falling.size >= 8, f"the avalanche was over in {falling.size} frames"
    assert np.all(steps[: falling.max() + 1] <= 1e-9), "the column rose again while it poured out"
    assert heights[-1] < 0.6


@pytest.mark.parametrize("edge", [0, -1])
def test_drift_edge_spill_leaves_the_panel(edge):
    """A collapsing edge band pours into its one neighbour, never itself."""
    state: dict = {}
    fn = M.get("JP Drift").fn
    z = np.zeros(N_BANDS)
    fn(_ctx(120, 40, state, 0.0, z))
    pile = _drift_state(state)["h"]
    pile[:] = 0.0
    pile[edge] = 1.25
    n = len(pile)
    inner = 1 if edge == 0 else n - 2
    far = [i for i in range(n) if i not in (edge % n, inner)]
    for f in range(1, 30):
        fn(_ctx(120, 40, state, f * DT, z, onsets=int(f == 1)))
    assert pile[edge] <= 0.58, "the collapsing band took its own spill back"
    assert pile[inner] > 0.05, "the neighbour received nothing"
    assert np.all(pile[far] == 0.0), "sand landed beyond the one neighbour"


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
