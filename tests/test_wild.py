"""The wild family: they draw, they answer a hit, and they survive any size."""
from __future__ import annotations

import numpy as np
import pytest

import spektr.modes as M
from spektr.analysis import N_BANDS
from spektr.modes import Ctx
from spektr.palette import BUILTIN, Palette

PAL = Palette(BUILTIN["gruvbox"])
DT = 1 / 60
WILD = ("Riptide", "Twin Storms", "Arc Storm", "Shatter")


def ctx(state, t=0.0, level=0.4, w=80, h=24, **kw) -> Ctx:
    bands = np.full(N_BANDS, level)
    base = dict(
        w=w, h=h, bands=bands, peaks=bands, bands_l=bands, bands_r=bands,
        wave=np.zeros(512), stereo=np.zeros((512, 2)), frame=int(t / DT), t=t,
        dt=DT, energy=float(level), silent=level < 0.02, palette=PAL, state=state,
    )
    base.update(kw)
    return Ctx(**base)


def run(name, frames, hit_at=None, **kw):
    st: dict = {}
    out = None
    for i in range(frames):
        hit = dict(onsets=1, onset_strength=1.0) if i == hit_at else {}
        out = M.get(name).fn(ctx(st, t=i * DT, **hit, **kw))
    return out


def test_they_are_the_wild_family_in_a_row():
    names = [m.name for m in M.MODES]
    at = names.index(WILD[0])
    assert names[at:at + len(WILD)] == list(WILD)
    assert all(M.get(n).group == "wild" and not M.get(n).hidden for n in WILD)


@pytest.mark.parametrize("name", WILD)
def test_it_draws_something_to_music(name):
    codes, _ = run(name, 20)
    assert (codes != 0x2800).sum() > 50


@pytest.mark.parametrize("name", WILD)
def test_a_hit_changes_the_picture(name):
    calm, _ = run(name, 12)
    hit, _ = run(name, 12, hit_at=9)
    assert (calm != hit).sum() > 20


@pytest.mark.parametrize("name", WILD)
@pytest.mark.parametrize("size", [(1, 1), (4, 2), (5, 3), (17, 7), (400, 100)])
def test_any_size(name, size):
    w, h = size
    codes, cidx = run(name, 3, hit_at=1, w=w, h=h)
    assert codes.shape == (h, w) and cidx.shape == (h, w)
