"""Crosscurrent: two ring streams in one wireframe, and the music where they meet.

What has to hold, measured on the rendered braille where it is about the
picture and on the mode's state where it is about timing:

* both directions are on screen at once — round rings travel out, octagonal
  rings travel in, in the same frames;
* the whole wireframe turns, faster with a tempo and a busy track, easing up
  to speed rather than jumping, and slowing to a crawl in silence;
* each stream answers its own half of the spectrum;
* a spoke pulse is local to the sector whose band jumped, and a strong hit is
  a sweep that spreads from one sector rather than every spoke at once;
* a pulsing spoke flickers — dots actually go dark, in dashes that move — and
  a spoke that is not pulsing is never touched;
* silence settles everything;
* under a busy passage it keeps the wireframe tunnels' line quality: no specks,
  no solid blocks, straight spokes and a clean centre.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from test_tunnel import _components, _dots  # noqa: E402

PAL = Palette(BUILTIN["hackerman"])
NAME = "Crosscurrent"
DT = 1 / 60
_X = np.linspace(0.0, 1.0, N_BANDS)


def _ctx(state, f, bands, onset=0, strength=0.0, w=120, h=40, pal=PAL, tempo=0.0):
    return Ctx(w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
               wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=f, t=f * DT, dt=DT,
               energy=float(np.mean(bands)), silent=float(np.max(bands)) < 0.02, palette=pal,
               state=state, onsets=onset, onset_strength=strength, tempo_bpm=tempo)


def _groove(f):
    """A kick every half second, a hat every quarter, a mid pad: ordinary busy music."""
    t = f * DT
    beat, hat = t % 0.5, t % 0.25
    bands = np.clip((0.3 + 0.55 * math.exp(-beat / 0.1)) * np.exp(-(_X / 0.25) ** 2)
                    + 0.3 * np.exp(-((_X - 0.45) / 0.25) ** 2)
                    + (0.1 + 0.25 * math.exp(-hat / 0.04)) * (_X > 0.65), 0, 1)
    onset = int(beat < DT or hat < DT)
    return bands, onset, (0.9 if beat < DT else 0.45) if onset else 0.0


def _state(state, key):
    return next(v for k, v in state.items() if k[0] == key)


def _walls(state, shape, drawn=True):
    """The spoke dots of the last frame: as drawn, or as they would be unflickered."""
    st = _state(state, "crosscurrent")
    outer, inner, _ = st["spokes"]
    m = np.zeros(shape[0] * shape[1], dtype=bool)
    m[st["drawn"] if drawn else outer] = True
    m[inner] = True
    return m.reshape(shape)


def _run(frames, source, w=120, h=40, pal=PAL, keep=False):
    fn = M.get(NAME).fn
    state: dict = {}
    out = []
    for f in range(frames):
        bands, onset, strength = source(f)
        codes = fn(_ctx(state, f, bands, onset, strength, w, h, pal))[0]
        if keep:
            out.append(codes)
    return state, out


# ── line quality under a busy passage ────────────────────────────────────────

@pytest.mark.parametrize("size", [(188, 50), (120, 40), (60, 20)])
def test_busy_music_keeps_thin_clean_lines(size):
    _, frames = _run(150, _groove, *size, keep=True)
    for codes in frames[-60::6]:
        d = _dots(codes)
        sizes = _components(d)
        assert int((sizes < 6).sum()) == 0, f"{size}: {int((sizes < 6).sum())} specks of under six dots"
        win = sliding_window_view(np.pad(d, 2), (5, 5)).sum(axis=(-1, -2))
        heavy = float((win[d] >= 18).mean())
        assert heavy < 0.10, f"{size}: {100 * heavy:.0f}% of lit dots in solid blocks"


def test_rings_stay_out_of_the_unresolvable_centre():
    state, frames = _run(90, _groove, keep=True)
    xc = _state(state, "crosscurrent_geo")
    d = _dots(frames[-1])
    d &= ~_walls(state, d.shape)
    dr, dc = d.shape
    yy, xx = np.mgrid[0:dr, 0:dc]
    cy, cx = dr / 2.0, dc / 2.0
    dist = np.hypot((xx - cx) * (cy / cx), yy - cy)
    max_r = max(1.0, cy - 1.0)
    cut = math.sqrt(2 * 3.5 * 0.55 * max_r)
    assert dist[d].min() >= cut * 0.85, "a ring was drawn inside the radius two streams cannot resolve"
    assert abs(xc["depth_cut"] - 0.55 * max_r / cut) < 1e-6


# ── both directions, at once ─────────────────────────────────────────────────

def test_both_streams_visibly_travel_their_own_way_in_the_same_frames():
    """Split the rendered ring dots by which stream's ring they sit on, then
    follow each stream's radial profile three frames later."""
    fn = M.get(NAME).fn
    state: dict = {}
    classes = []
    for f in range(160):
        bands, onset, strength = _groove(f)
        codes = fn(_ctx(state, f, bands, onset, strength))[0]
        if f < 100:
            continue
        geo = _state(state, "tunnel_geo")
        xc = _state(state, "crosscurrent_geo")
        st = _state(state, "crosscurrent")
        d = _dots(codes)
        d &= ~_walls(state, d.shape)
        fo = geo["depth055"] + np.float32(st["out"] % 1.0)
        fo = np.abs(fo - np.rint(fo))
        steps = len(xc["oct_table"])
        spin_idx = int(round(st["spin"] * steps)) & (steps - 1)
        oct_scale = xc["oct_table"][(xc["turn_idx"] - spin_idx) & (steps - 1)].reshape(d.shape)
        max_r = max(1.0, d.shape[0] / 2.0 - 1.0)
        depth_oct = np.float32(0.55 * max_r) / np.maximum(xc["dist"].reshape(d.shape) * oct_scale, np.float32(0.9))
        fi = depth_oct - np.float32(st["in"] % 1.0)
        fi = np.abs(fi - np.rint(fi))
        classes.append((d & (fo <= fi), d & (fi < fo)))
    dr, dc = classes[0][0].shape
    yy, xx = np.mgrid[0:dr, 0:dc]
    cy, cx = dr / 2.0, dc / 2.0
    rad = np.hypot((xx - cx) * (cy / cx), yy - cy)
    bins = np.arange(0, rad.max() + 1, 0.5)
    moved = {0: [], 1: []}
    for a, b in zip(classes[:-3], classes[3:]):
        for j in (0, 1):
            ha = np.histogram(rad[a[j]], bins=bins)[0].astype(float)
            hb = np.histogram(rad[b[j]], bins=bins)[0].astype(float)
            if ha.sum() < 20 or hb.sum() < 20:
                continue
            lags = list(range(-12, 13))
            sc = [np.dot(ha[max(0, -l):len(ha) - max(0, l)], hb[max(0, l):len(hb) - max(0, -l)]) for l in lags]
            moved[j].append(lags[int(np.argmax(sc))])
    out, inn = np.array(moved[0]), np.array(moved[1])
    assert len(out) > 30 and len(inn) > 30, "one of the streams is missing from the picture"
    assert np.mean(out > 0) > 0.8, f"round rings moved outward in only {100 * np.mean(out > 0):.0f}% of steps"
    assert np.mean(inn < 0) > 0.8, f"octagonal rings moved inward in only {100 * np.mean(inn < 0):.0f}% of steps"


def test_each_stream_answers_its_own_half_of_the_spectrum():
    def speeds(level_lo, level_hi, frames=360):
        fn = M.get(NAME).fn
        state: dict = {}
        mark = None
        for f in range(frames):
            lo_part = level_lo if f > 240 else 0.2
            hi_part = level_hi if f > 240 else 0.2
            bands = np.where(_X < 0.45, lo_part, hi_part)
            fn(_ctx(state, f, bands))
            if f == 240:
                st = _state(state, "crosscurrent")
                mark = (st["out"], st["in"])
        st = _state(state, "crosscurrent")
        span = (frames - 241) * DT
        return (st["out"] - mark[0]) / span, (st["in"] - mark[1]) / span

    base_out, base_in = speeds(0.2, 0.2)
    low_out, low_in = speeds(0.6, 0.2)
    high_out, high_in = speeds(0.2, 0.6)
    assert low_out > 1.4 * base_out and abs(low_in - base_in) < 0.15 * base_in, "a lower-half swell did not push only the outbound stream"
    assert high_in > 1.4 * base_in and abs(high_out - base_out) < 0.15 * base_out, "an upper-half swell did not pull only the inbound stream"


# ── spokes carry transients ──────────────────────────────────────────────────

def test_a_band_jump_flicks_only_the_spokes_in_its_sector():
    fn = M.get(NAME).fn
    state: dict = {}
    quiet = np.full(N_BANDS, 0.15)
    for f in range(120):
        fn(_ctx(state, f, quiet))
    jump = quiet.copy()
    jump[: N_BANDS // 9] = 0.8          # the lowest sector: the horizontal spokes
    fn(_ctx(state, 120, jump))
    flash = _state(state, "crosscurrent")["flash"]
    assert flash[0] > 0.5 and flash[8] > 0.5, "the bass jump did not flick the horizontal spokes"
    assert flash[4] < 0.1 and flash[12] < 0.1, "the bass jump lit the vertical spokes too"
    assert int((flash > 0.3).sum()) <= 6, f"{int((flash > 0.3).sum())} spokes lit for one band"


def test_a_broadband_attack_still_lights_only_part_of_the_wireframe():
    """Every band jumping at once must not light every spoke at once."""
    fn = M.get(NAME).fn
    state: dict = {}
    quiet = np.full(N_BANDS, 0.15)
    for f in range(120):
        fn(_ctx(state, f, quiet))
    jump = quiet + np.linspace(0.55, 0.35, N_BANDS)   # everything rises, the low end most
    fn(_ctx(state, 120, jump))
    lit = int((_state(state, "crosscurrent")["flash"] > 0.3).sum())
    assert 0 < lit <= 8, f"{lit} of 16 spokes lit by one broadband attack"


def test_a_strong_hit_is_a_sweep_not_a_strobe():
    fn = M.get(NAME).fn
    state: dict = {}
    steady = np.full(N_BANDS, 0.3)
    for f in range(120):
        fn(_ctx(state, f, steady))
    lit_counts = []
    surge = 0.0
    hit_bands = steady.copy()
    hit_bands[: N_BANDS // 9] = 0.9
    for k in range(20):
        onset = int(k == 0)
        fn(_ctx(state, 120 + k, hit_bands if k < 3 else steady, onset, 1.0 if onset else 0.0))
        lit_counts.append(int((_state(state, "crosscurrent")["flash"] > 0.3).sum()))
        surge = max(surge, _state(state, "crosscurrent")["surge"])
    assert lit_counts[0] <= 4, f"{lit_counts[0]} spokes lit on the hit's own frame"
    assert max(lit_counts) >= 5, "the hit never spread past its own sector"
    assert max(lit_counts) < 12, f"{max(lit_counts)} of 16 spokes lit at once"
    assert surge > 0.9, "the hit did not surge the streams"


def test_pulsing_spokes_flicker_and_the_rest_are_never_touched():
    """Under a groove, a pulsing spoke must really lose dots on its dark
    strobes, not just change colour, in dashes that move from one strobe to
    the next; a spoke that is not pulsing keeps every dot and a dashed one
    keeps most of them."""
    fn = M.get(NAME).fn
    state: dict = {}
    flickered = 0
    gaps = []
    for f in range(240):
        bands, onset, strength = _groove(f)
        codes = fn(_ctx(state, f, bands, onset, strength, tempo=120.0))[0]
        if f < 60:
            continue
        d = _dots(codes)
        drawn = _walls(state, d.shape)
        assert np.array_equal(d & drawn, drawn), "a drawn spoke dot is missing from the picture"
        st = _state(state, "crosscurrent")
        outer, _, spoke_of = st["spokes"]
        dropped = np.isin(outer, st["drawn"], invert=True)
        if not dropped.any():
            continue
        assert (st["flash"][spoke_of[dropped]] > 0.2).all(), "a spoke that was not pulsing lost dots"
        for k in np.unique(spoke_of[dropped]):
            kept = 1.0 - dropped[spoke_of == k].mean()
            assert kept >= 0.55, f"a flickering spoke kept only {100 * kept:.0f}% of its dots"
        flickered += 1
        gaps.append(frozenset(outer[dropped].tolist()))
    assert flickered >= 20, f"spokes visibly flickered in only {flickered} of 180 frames of a groove"
    assert len(set(gaps)) >= flickered // 3, "the dashes do not move between strobes"


# ── the whole tunnel turns with the music ────────────────────────────────────

def _spin_run(source, frames, tempo):
    fn = M.get(NAME).fn
    state: dict = {}
    trace = []
    for f in range(frames):
        bands, onset, strength = source(f)
        fn(_ctx(state, f, bands, onset, strength, tempo=tempo))
        trace.append(_state(state, "crosscurrent")["spin_v"])
    return np.array(trace)


def _pad(f):
    return np.full(N_BANDS, 0.25), 0, 0.0


def test_the_tunnel_turns_faster_with_tempo_and_activity_and_eases_up_to_speed():
    def settled(v):
        return float(np.mean(v[-60:]))

    pad = _spin_run(_pad, 300, 0.0)
    groove_free = _spin_run(_groove, 300, 0.0)
    groove_beat = _spin_run(_groove, 300, 120.0)
    groove_fast = _spin_run(_groove, 300, 170.0)
    assert settled(groove_free) > 2.0 * settled(pad), "a busy track does not turn the tunnel faster than a steady pad"
    assert settled(groove_fast) > 1.25 * settled(groove_beat), "a faster tempo does not turn it faster"
    assert settled(groove_fast) < 0.5, f"the tunnel spins at {settled(groove_fast):.2f} turns a second, too fast to follow"
    # it accelerates into the groove rather than jumping to speed
    assert groove_beat[5] < 0.5 * settled(groove_beat), "the turn jumped straight to speed"
    assert groove_beat[90] > 0.7 * settled(groove_beat), "the turn takes too long to reach speed"


def test_the_picture_turns_by_the_spin():
    """Follow the spokes' angles on the rendered grid three frames apart:
    they must have turned by what the spin says."""
    fn = M.get(NAME).fn
    state: dict = {}
    snaps = {}
    for f in range(154):
        bands, onset, strength = _groove(f)
        codes = fn(_ctx(state, f, bands, onset, strength, tempo=120.0))[0]
        if f in (150, 153):
            snaps[f] = (_walls(state, _dots(codes).shape, drawn=False), _state(state, "crosscurrent")["spin"])
    dr, dc = snaps[150][0].shape
    yy, xx = np.mgrid[0:dr, 0:dc]
    cy, cx = dr / 2.0, dc / 2.0
    turn = (np.arctan2(yy - cy, (xx - cx) * (cy / cx)) / (2 * math.pi)) % 1.0
    far = np.hypot((xx - cx) * (cy / cx), yy - cy) > 0.4 * cy
    bins = 1024
    ha = np.histogram(turn[snaps[150][0] & far], bins=bins, range=(0, 1))[0].astype(float)
    hb = np.histogram(turn[snaps[153][0] & far], bins=bins, range=(0, 1))[0].astype(float)
    lags = np.arange(-bins // 32, bins // 32 + 1)       # within half a spoke gap either way
    best = lags[int(np.argmax([np.dot(ha, np.roll(hb, -lag)) for lag in lags]))] / bins
    expected = (snaps[153][1] - snaps[150][1] + 1 / 32) % (1 / 16) - 1 / 32   # spokes repeat every sixteenth
    assert abs(expected) > 3 / bins, "the tunnel barely turned in three frames of a groove"
    assert abs(best - expected) <= 3 / bins, f"the picture turned {best:.4f} but the spin says {expected:.4f}"


def test_silence_lets_it_settle():
    fn = M.get(NAME).fn
    state: dict = {}
    for f in range(240):
        bands, onset, strength = _groove(f)
        fn(_ctx(state, f, bands, onset, strength))
    z = np.zeros(N_BANDS)
    for f in range(240, 420):
        fn(_ctx(state, f, z))
    st = _state(state, "crosscurrent")
    assert st["flash"].max() < 0.01, "spokes still flickering after three seconds of silence"
    assert st["surge"] < 0.02 and st["kick"] < 0.01
    before = (st["out"], st["in"])
    for f in range(420, 480):
        fn(_ctx(state, f, z))
    v_out = (st["out"] - before[0]) / (60 * DT)
    v_in = (st["in"] - before[1]) / (60 * DT)
    assert 0.05 < v_out < 0.2 and 0.05 < v_in < 0.2, f"silence drifts at {v_out:.2f}/{v_in:.2f}, not a slow settle"
    assert 0.0 < st["spin_v"] < 0.02, f"silence still turns the tunnel at {st['spin_v']:.3f} turns a second"


# ── the shared centre ────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [(120, 40), (60, 20)])
def test_the_centre_converges_like_the_tunnels(size):
    state, frames = _run(60, _groove, *size, keep=True)
    d = _dots(frames[-1])
    walls = _walls(state, d.shape, drawn=False)
    spin = _state(state, "crosscurrent")["spin"]
    dr, dc = d.shape
    cx, cy = dc / 2.0, dr / 2.0
    s = cy / cx
    yy, xx = np.mgrid[0:dr, 0:dc]
    core = np.hypot(xx - cx, yy - cy) <= 6.0
    assert d[core].mean() < 0.45, f"{size}: {100 * d[core].mean():.0f}% of the centre is lit"
    pad = np.pad(walls, 1)
    near = walls | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    for q in range(4):                  # the four main spokes, wherever the turn has carried them
        theta = 2 * math.pi * (spin + q / 4)
        ux, uy = math.cos(theta) / s, math.sin(theta)
        norm = math.hypot(ux, uy)
        ray = [(int(round(cy + uy / norm * r)), int(round(cx + ux / norm * r))) for r in np.arange(4.0, 9.0, 0.5)]
        assert all(near[y, x] for y, x in ray), f"{size}: a main spoke stops short of the centre"
