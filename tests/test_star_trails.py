"""Star Trails is a long exposure, and a long exposure can flood.

The mode accumulates into a buffer that is fed back into itself every frame,
which makes it the one mode here where a small per-frame error compounds
instead of washing out. Three separate ways of getting that wrong all landed
in the same symptom — a screen of solid white — and none of them showed up in
any existing test, because every existing test asks whether a mode *runs*.

* The motion-blur kernel gave each of four neighbours ``amount / 2``, so its
  weights summed to ``1 + amount``. A gain of a few percent per frame beat the
  exposure's own fade, and one lit dot filled a 40x60 field to full white in
  ten seconds with no stars stamped at all.
* The exposure length was a constant. The angle a star sweeps before its mark
  fades is ``omega * tau``, and omega is not constant: drive, pulse, kick and
  beat carry it past 0.6 rad/s on ordinary percussive material, which at the
  old tau is a 110-degree arc per star and an annulus rather than a sky.
* The blur was applied per *frame* rather than per second, so the same passage
  came out twice as dense at 240 fps as at 24 — the picture was a property of
  the terminal instead of the track.

These pin the invariants rather than the pixels: energy in, energy out; the
sweep stays inside the star spacing; and the frame rate does not change what
the music looks like.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.modes as M  # noqa: E402
import spektr.modes.cosmos as cosmos  # noqa: E402
from spektr.analysis import N_BANDS  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, Palette  # noqa: E402
from spektr.render import BRAILLE_BASE  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])
TRAILS = M.get("Star Trails").fn


def _run(w=80, h=24, secs=25.0, fps=60.0, energy=0.85, beat_hz=3.0):
    """Play the mode a synthetic passage; return the last frame and lit history.

    Twenty-five seconds because the failures this guards against are all
    *accumulating* ones — the old blur gain needed about ten to reach full
    white, and a two-frame smoke test sees none of it.
    """
    state: dict = {}
    dt = 1.0 / fps
    last_beat = -1.0
    lit: list[float] = []
    codes = cidx = None
    for f in range(int(secs * fps)):
        t = f * dt
        bands = np.full(N_BANDS, energy, dtype=np.float32)
        wave = (np.sin(np.linspace(0.0, 40.0, 512)) * energy).astype(np.float32)
        onset = 0
        if beat_hz and t - last_beat >= 1.0 / beat_hz:
            onset, last_beat = 1, t
        ctx = Ctx(
            w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
            wave=wave, stereo=np.stack([wave, wave], 1), frame=f, t=t, dt=dt,
            energy=energy, silent=False, palette=PAL, state=state,
            onsets=onset, onset_strength=0.8 if onset else 0.0,
        )
        codes, cidx = TRAILS(ctx)
        # The first few seconds are the film filling up; judge the steady state.
        if t > 3.0:
            lit.append(float((codes != BRAILLE_BASE).mean()))
    return codes, cidx, np.array(lit), state[("star_trails", w, h)]


# ── the smear ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("amount", [0.02, 0.15, 0.5, 1.0])
def test_the_blur_neither_gains_nor_loses_light(amount):
    """A flat interior comes back exactly flat, at every strength.

    This is the whole property. The kernel is applied to a buffer that is its
    own input next frame, so weights summing to anything but one compound: at
    ``1 + amount`` the field saturates, at ``1 - amount`` the sky fades out
    from under the exposure. Only the interior is checked — an edge dot has
    fewer neighbours to borrow from, and dimming at the frame edge is what a
    real exposure does.
    """
    field = np.full((40, 60), 0.5, dtype=np.float32)
    cosmos._blur_merge(field, amount)
    assert np.allclose(field[1:-1, 1:-1], 0.5, atol=1e-6), (
        f"weights sum to {field[20, 30] / 0.5:.4f}, not 1 — the smear "
        f"{'gains' if field[20, 30] > 0.5 else 'loses'} light every frame")


def test_the_blur_alone_cannot_light_the_screen():
    """One dot, no stars, no music: the exposure must fade, never spread out.

    The regression in its purest form. With the old half-weight kernel this
    reached a fully white 40x60 field in about ten seconds of frames, purely
    by feeding the smear back into itself.
    """
    field = np.zeros((40, 60), dtype=np.float32)
    field[20, 30] = 1.0
    for _ in range(600):                       # ten seconds at 60 fps
        cosmos._blur_merge(field, 0.30)
        field *= math.exp(-(1.0 / 60.0) / cosmos._TRAIL_TAU)
        np.clip(field, 0.0, 1.0, out=field)
    assert field.max() < 0.04, "the smear is generating light instead of moving it"


# ── the exposure ─────────────────────────────────────────────────────────────

def test_the_sky_does_not_fill_in_however_hard_it_is_driven():
    """Loud, percussive, twenty-five seconds — and still a sky of arcs.

    The number is a ceiling, not a target: what it rules out is the failure,
    which measured at 85% of cells lit and read as a solid white wall. A sky
    of separate curves lands around 40%.
    """
    _, _, lit, _ = _run(energy=0.85, beat_hz=3.0)
    assert lit.max() < 0.60, f"the exposure fused into a filled disc ({lit.max():.2f} lit)"


def test_the_exposure_never_outruns_the_star_spacing():
    """``omega * tau`` stays inside the gap, over a real driven passage.

    This is the thing the design rests on: once a star's arc reaches its
    neighbour's there are no arcs left to see, only an annulus. Read from the
    exposure length the mode *used*, recorded frame by frame — recomputing
    the formula here instead would pass whether or not the mode applies it.
    """
    sweeps, omegas = [], []
    state: dict = {}
    dt, last_beat = 1.0 / 60.0, -1.0
    for f in range(int(20.0 * 60.0)):
        t = f * dt
        bands = np.full(N_BANDS, 0.85, dtype=np.float32)
        wave = (np.sin(np.linspace(0.0, 40.0, 512)) * 0.85).astype(np.float32)
        onset = 0
        if t - last_beat >= 1.0 / 3.0:
            onset, last_beat = 1, t
        ctx = Ctx(
            w=80, h=24, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
            wave=wave, stereo=np.stack([wave, wave], 1), frame=f, t=t, dt=dt,
            energy=0.85, silent=False, palette=PAL, state=state,
            onsets=onset, onset_strength=0.8 if onset else 0.0,
        )
        TRAILS(ctx)
        st = state[("star_trails", 80, 24)]
        omega = (cosmos._TRAIL_OMEGA + cosmos._TRAIL_DRIVE * ctx.drive
                 + cosmos._TRAIL_PULSE * ctx.pulse + st["kick"] + st["beat"])
        omegas.append(omega)
        sweeps.append(omega * st["tau"])

    st = state[("star_trails", 80, 24)]
    # Re-measured from the star layout rather than read back from the cached
    # value the mode divides by, so a mode that stopped capping — or that
    # capped against a spacing it had talked itself into — is still caught.
    ceiling = cosmos._TRAIL_FILL * cosmos._angular_gap(st["r"], st["th"])
    # The passage has to actually drive the sky, or the ceiling is never tested.
    assert max(omegas) > 4.0 * cosmos._TRAIL_OMEGA, "the run never span the sky up"
    assert max(sweeps) <= ceiling + 1e-6, (
        f"a star swept {max(sweeps):.3f} rad, past the {ceiling:.3f} rad "
        f"gap to its neighbour — the arcs have fused")


def test_a_silent_sky_still_turns_and_stays_sparse():
    """Nothing playing is a slow sky, not a blank one and not a bright one."""
    _, _, lit, _ = _run(energy=0.02, beat_hz=0)
    assert 0.02 < lit.mean() < 0.25, f"a silent sky reads wrong ({lit.mean():.3f} lit)"


# ── the frame rate ───────────────────────────────────────────────────────────

def test_the_same_music_looks_the_same_at_every_frame_rate():
    """24 fps is an offered setting, not a degraded mode.

    Two things used to make the picture a property of the terminal: the blur
    was a per-frame amount, and each star was stamped at a point rather than
    along the arc it had just swept — at 24 fps a rim star jumps two or three
    dots a frame, so the exposure recorded a dotted line. Together they made
    the same passage twice as dense at 240 fps as at 24.
    """
    lit = {fps: round(float(_run(fps=fps)[2].mean()), 3)
           for fps in (24, 30, 60, 120, 240)}
    lo, hi = min(lit.values()), max(lit.values())
    assert hi - lo < 0.06, f"the frame rate is changing the picture: {lit}"
