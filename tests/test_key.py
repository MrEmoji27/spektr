"""Naming the key, and refusing to when there isn't one."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spektr.audio import chroma, key  # noqa: E402

SR = 48000
N = 4096

#: Six harmonics at 1/n, which is a sawtooth and close enough to most
#: instruments. Pure sines would be the wrong test: real notes carry a strong
#: third harmonic, that harmonic is what pushed every estimate up a fifth, and
#: a corpus without it would have declared the bug fixed before it was.
HARMONICS = 6


def tone(hz: float) -> np.ndarray:
    t = np.arange(N) / SR
    return sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, HARMONICS + 1))


def hz_of(note: str, octave: int = 3) -> float:
    midi = 12 * (octave + 1) + chroma.NOTES.index(note)
    return 440.0 * 2 ** ((midi - 69) / 12)


def chord(notes) -> np.ndarray:
    x = sum(tone(hz_of(n)) for n in notes)
    return chroma.fold(np.abs(np.fft.rfft(x * np.hanning(N))), SR)


def listen(progression, repeats: int = 3, dt: float = 0.05, steps: int = 20):
    """Play ``progression`` a few times through a fresh estimator."""
    est = key.KeyEstimator()
    for _ in range(repeats):
        for notes in progression:
            folded = chord(notes)
            for _ in range(steps):
                est.feed(folded, dt)
    return est


#: A cadence in each key: tonic, subdominant, dominant, tonic. Enough of the
#: scale to place it, and the shape almost all tonal music is built on.
PROGRESSIONS = {
    "C major": [["C", "E", "G"], ["F", "A", "C"], ["G", "B", "D"], ["C", "E", "G"]],
    "D major": [["D", "F#", "A"], ["G", "B", "D"], ["A", "C#", "E"], ["D", "F#", "A"]],
    "G major": [["G", "B", "D"], ["C", "E", "G"], ["D", "F#", "A"], ["G", "B", "D"]],
    "F major": [["F", "A", "C"], ["A#", "D", "F"], ["C", "E", "G"], ["F", "A", "C"]],
    "A# major": [["A#", "D", "F"], ["D#", "G", "A#"], ["F", "A", "C"], ["A#", "D", "F"]],
    "A minor": [["A", "C", "E"], ["D", "F", "A"], ["E", "G", "B"], ["A", "C", "E"]],
    "E minor": [["E", "G", "B"], ["A", "C", "E"], ["B", "D", "F#"], ["E", "G", "B"]],
}


@pytest.mark.parametrize("expected", sorted(PROGRESSIONS))
def test_a_cadence_names_its_own_key(expected):
    est = listen(PROGRESSIONS[expected])
    assert est.key == expected, f"heard {est.key}"
    assert not est.uncertain


def test_every_note_would_otherwise_vote_for_its_own_fifth():
    """The bug this module's correction exists for.

    Without :func:`key.deharmonise` each note's third harmonic lands a fifth
    up, so the accumulated chroma of a piece in C looks like a piece in G.
    It is not noise — it is systematic, and it always points the same way.
    """
    est = listen(PROGRESSIONS["C major"])
    raw = key.score(est._avg).ravel()
    row, tonic = divmod(int(np.argmax(raw)), 12)
    assert key.name(tonic, bool(row)) == "G major", (
        "the harmonic bias this correction removes is no longer present, so "
        "either the chroma changed or this test has stopped measuring it"
    )
    # ...and with the correction, the same evidence names the right key.
    assert est.key == "C major"


def test_a_chromatic_run_is_uncertain_rather_than_wrong():
    """Every pitch class equally: no key fits, and noise must not be named.

    Note that a margin test cannot catch this — measured, a chromatic run beat
    its runner-up by *more* than a clean cadence in C did. What gives it away
    is that nothing fits well.
    """
    est = listen([["C", "C#", "D"], ["D#", "E", "F"],
                  ["F#", "G", "G#"], ["A", "A#", "B"]])
    assert est.key is None
    assert est.uncertain


def test_one_note_is_not_a_key():
    est = listen([["C"]] * 4)
    assert est.key is None
    assert est.uncertain


def test_nothing_is_claimed_before_enough_has_been_heard():
    est = listen(PROGRESSIONS["C major"], repeats=1, steps=2, dt=0.05)
    assert est.key is None
    assert est.confidence == 0.0
    assert not est.uncertain, "too early is not the same as ambiguous"


def test_silence_is_not_evidence():
    est = key.KeyEstimator()
    for _ in range(500):
        est.feed(np.zeros(12), 0.05)
    assert est.key is None and est.confidence == 0.0 and not est.uncertain


def test_reset_forgets_the_last_track():
    est = listen(PROGRESSIONS["C major"])
    assert est.key is not None
    est.reset()
    assert est.key is None and est.confidence == 0.0 and not est.uncertain


def test_the_answer_does_not_depend_on_loudness():
    quiet = listen(PROGRESSIONS["G major"])
    est = key.KeyEstimator()
    for _ in range(3):
        for notes in PROGRESSIONS["G major"]:
            folded = chord(notes) * 1000.0
            for _ in range(20):
                est.feed(folded, 0.05)
    assert est.key == quiet.key


def test_a_relative_pair_is_named_but_not_confidently():
    """A minor and C major share every note. Whichever wins is the better
    reading, and the confidence is what says how thin the win was."""
    minor = listen(PROGRESSIONS["A minor"])
    clear = listen(PROGRESSIONS["E minor"])
    assert minor.key == "A minor"
    assert minor.confidence < clear.confidence


def test_the_profiles_are_the_published_ones():
    assert len(key.MAJOR) == len(key.MINOR) == 12
    # Krumhansl-Kessler: the tonic is the strongest degree in both.
    assert key.MAJOR.index(max(key.MAJOR)) == 0
    assert key.MINOR.index(max(key.MINOR)) == 0


def test_nonsense_input_is_survivable():
    assert key.score(np.zeros(12)).shape == (2, 12)
    assert not key.score(np.zeros(5)).any()
    assert not key.deharmonise(np.zeros(3)).any()
    assert (key.deharmonise(np.zeros(12)) >= 0).all()


# ── through the real analyser ────────────────────────────────────────────────
#
# Above, the estimator is fed chroma directly. This plays actual audio into the
# actual analyser and reads the key off the published frame, so the wiring --
# the fold, the hop timing, the gate, the silence handling -- is covered too.

from spektr.analysis import HOP, Analyser  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402


def play_progression(progression, seconds_each=1.0, loops=4):
    """Audio for a progression, looped long enough to establish a key."""
    parts = []
    for _ in range(loops):
        for notes in progression:
            t = np.arange(int(SR * seconds_each)) / SR
            x = sum(
                sum(np.sin(2 * np.pi * hz_of(n) * k * t) / k
                    for k in range(1, HARMONICS + 1))
                for n in notes
            ) * 0.2
            parts.append(np.stack((x, x), axis=1).astype(np.float32))
    return np.concatenate(parts)


def last_frame_for(progression):
    signal = play_progression(progression)
    ring = RingBuffer(1 << 16)
    now = [0.0]
    an = Analyser(ring, lambda: SR, clock=lambda: now[0])
    an._ensure_plan(SR)
    for start in range(0, signal.shape[0] - HOP + 1, HOP):
        ring.push(signal[start:start + HOP])
        now[0] = (start + HOP) / SR
        an._analyse_once()
    return an._frame


@pytest.mark.parametrize("expected", ["C major", "D major", "E minor"])
def test_the_analyser_publishes_the_key(expected):
    frame = last_frame_for(PROGRESSIONS[expected])
    assert frame.key == expected, f"published {frame.key}"
    assert frame.key_confidence > 0.0
    assert not frame.key_uncertain


def test_the_analyser_publishes_uncertainty_rather_than_a_guess():
    frame = last_frame_for([["C", "C#", "D"], ["D#", "E", "F"],
                            ["F#", "G", "G#"], ["A", "A#", "B"]])
    assert frame.key is None
    assert frame.key_uncertain


# ── the reported key has to hold still ───────────────────────────────────────

def keep_playing(est, progression, bars, dt=0.05, steps=20):
    """Play into an existing estimator; return the key after every frame."""
    seen = []
    for _ in range(bars):
        for notes in progression:
            folded = chord(notes)
            for _ in range(steps):
                est.feed(folded, dt)
                seen.append(est.key)
    return seen


def test_a_borrowed_chord_does_not_change_the_key():
    """One bar from somewhere else is a borrowed chord, not a modulation. A
    mode recolouring on the key would strobe if this were not held."""
    est = listen(PROGRESSIONS["C major"], repeats=6)
    assert est.key == "C major"
    seen = keep_playing(est, [["A#", "D", "F"]], 1)
    assert set(seen) == {"C major"}, f"flipped to {set(seen) - {'C major'}}"


def test_a_real_modulation_does_change_the_key():
    est = listen(PROGRESSIONS["C major"], repeats=6)
    assert est.key == "C major"
    keep_playing(est, PROGRESSIONS["E minor"], 10)
    assert est.key == "E minor"
    assert est.confidence > 0.5


def test_the_key_does_not_flicker_once_it_has_settled():
    est = listen(PROGRESSIONS["G major"], repeats=6)
    settled = est.key
    seen = keep_playing(est, PROGRESSIONS["G major"], 6)
    assert set(seen) == {settled}, f"flickered across {set(seen)}"


def test_hysteresis_waits_for_a_full_memory_before_defending_a_key():
    """Early on the average is still filling and its first winner is not
    worth defending — holding one cost the right answer on a cadence in C,
    which locked onto its relative minor before the evidence was in."""
    from spektr.audio.key import MEMORY_S
    est = listen(PROGRESSIONS["C major"], repeats=6)
    assert est._heard >= MEMORY_S
    assert est.key == "C major"


def test_the_index_helper_round_trips_every_key():
    from spektr.audio.key import _index_of
    for tonic in range(12):
        for minor in (False, True):
            label = key.name(tonic, minor)
            assert _index_of(label) == (12 if minor else 0) + tonic
