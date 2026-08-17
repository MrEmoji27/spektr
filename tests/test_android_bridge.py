"""The Android bridge, checked against the engine it actually wraps.

This exists because the bridge rotted silently. It was written on 2026-08-15
against ``Analyser.snapshot()`` and a ``Frame`` carrying ``peaks``, ``energy``
and ``bars``; none of those are real, and by the time the port was picked up
again the engine had moved 101 commits. Every mode raised ``AttributeError``
on the first render and nothing said so, because no test imported the file.

The bridge is Python that ships inside an Android app, so it cannot be checked
by running the desktop app. It has to be checked here or not at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "android" / "app" / "src" / "main" / "python"))

spektr_android = pytest.importorskip("spektr_android")

RATE = 48000


def _tone(seconds: float = 4096 / RATE) -> np.ndarray:
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (0.4 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)


def _pcm(seconds: float = 4096 / RATE) -> bytes:
    """Interleaved float32 stereo — what ENCODING_PCM_FLOAT hands over.

    This said 16-bit and packed ``<i2`` for its first version, which is not
    what ``push`` reads. Nothing failed: ``frombuffer`` reinterpreted the same
    bytes as float32 and the modes drew whatever that happened to be. A format
    mismatch that renders is worse than one that raises, so
    ``test_pushed_audio_lands_in_the_ring_unchanged`` now pins the dtype.
    """
    sig = _tone(seconds)
    return np.stack([sig, sig], axis=1).ravel().astype("<f4").tobytes()


@pytest.fixture
def engine():
    return spektr_android.Engine(samplerate=RATE)


def _every_registered_mode() -> list[str]:
    """Every mode ``render`` will accept, hidden ones included.

    Not ``mode_names()``: that is the picker's list and deliberately shorter.
    The wire format has to hold for anything renderable, so this coverage must
    not quietly shrink because the interface stopped offering something.
    """
    import spektr.modes as M

    return M.names()


def test_the_engine_builds_and_lists_what_the_desktop_has(engine):
    import spektr.modes as M
    from spektr.palette import BUILTIN

    assert len(engine.mode_names()) == len(M.listed())
    assert len(engine.theme_names()) == len(BUILTIN)


def test_the_picker_is_not_offered_modes_android_cannot_draw(engine):
    """The hidden variants draw through Unicode 16 octants (U+1CD00 and up).

    Those landed in 2024 and no font on the device has them, so offering the
    twelve would put twelve entries in the picker that render as tofu — which
    on a visualiser reads as a crash, not as a missing glyph. What the shipped
    font does cover is checked in ``test_android_font.py``.
    """
    import spektr.modes as M

    offered = engine.mode_names()
    hidden = {m.name for m in M.MODES if m.hidden}
    assert hidden, "nothing is hidden — has the flag stopped being applied?"
    assert not (hidden & set(offered)), "a hidden mode reached the picker"
    assert set(offered) == {m.name for m in M.listed()}


def test_a_hidden_mode_is_still_renderable_by_name(engine):
    """A saved selection naming one must keep working, exactly as on desktop."""
    import spektr.modes as M

    engine.push(_pcm())
    for m in M.MODES:
        if m.hidden:
            engine.render(m.name, 60, 20)


def test_every_mode_renders_through_the_wire_format(engine):
    """The failure this file was written for: all 64 raised on the first call."""
    pcm = _pcm()
    bad = []
    for name in _every_registered_mode():
        try:
            for _ in range(6):
                engine.push(pcm)
                engine.render(name, 80, 24)
        except Exception as exc:                     # noqa: BLE001 — reporting
            bad.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not bad, "modes failed through the bridge:\n" + "\n".join(bad[:10])


def test_the_buffer_is_exactly_the_size_the_header_describes(engine):
    pcm = _pcm()
    for _ in range(6):
        engine.push(pcm)
    for w, h in ((80, 24), (37, 11), (200, 50)):
        buf = engine.render("Bars", w, h)
        magic, ver, planes, bw, bh = spektr_android._HEADER.unpack_from(buf, 0)
        assert magic == spektr_android._MAGIC
        assert ver == spektr_android.WIRE_VERSION
        assert (bw, bh) == (w, h)
        # codes are int32, each colour plane one byte a cell
        want = spektr_android._HEADER.size + w * h * 4 + w * h * (planes - 1)
        assert len(buf) == want, f"{w}x{h}: {len(buf)} bytes, header describes {want}"


def test_some_modes_carry_three_planes_and_the_count_says_so():
    """The trap the design document called out.

    Modes drawing with the half-block trick return a background ramp index as
    well. Kotlin that assumed two planes would not crash — it would render
    every one of them half-wrong, which is the worst kind of failure to ship.
    """
    engine = spektr_android.Engine(samplerate=RATE)
    pcm = _pcm()
    counts = {2: 0, 3: 0}
    for name in _every_registered_mode():
        for _ in range(4):
            engine.push(pcm)
            buf = engine.render(name, 40, 12)
        planes = spektr_android._HEADER.unpack_from(buf, 0)[2]
        assert planes in (2, 3), f"{name}: {planes} planes"
        counts[planes] += 1
    assert counts[3] > 0, "no mode reported a background plane — is bidx dropped?"
    assert counts[2] > 0


def test_colour_indices_stay_inside_the_ramp(engine):
    """They are packed as one byte; a value past the ramp would wrap silently."""
    from spektr.palette import RAMP_STEPS

    pcm = _pcm()
    w, h = 60, 20
    for name in _every_registered_mode():
        for _ in range(4):
            engine.push(pcm)
            buf = engine.render(name, w, h)
        planes = spektr_android._HEADER.unpack_from(buf, 0)[2]
        off = spektr_android._HEADER.size + w * h * 4
        for p in range(planes - 1):
            plane = np.frombuffer(buf, np.uint8, w * h, off + p * w * h)
            assert plane.max() < RAMP_STEPS, f"{name}: ramp index {plane.max()}"


def test_switching_mode_does_not_hand_over_the_last_one_s_scratch(engine):
    """Same rule the desktop widget follows, and for the same reason."""
    pcm = _pcm()
    engine.push(pcm)
    engine.render("Tunnel", 60, 20)
    engine.render("Chladni", 60, 20)          # different shapes in scratch
    engine.render("Tunnel", 60, 20)           # must not read Chladni's arrays


def test_a_resize_mid_session_is_just_two_different_numbers(engine):
    pcm = _pcm()
    engine.push(pcm)
    for w, h in ((80, 24), (200, 60), (40, 10), (80, 24)):
        buf = engine.render("Bars", w, h)
        assert spektr_android._HEADER.unpack_from(buf, 0)[3:] == (w, h)


def test_silence_renders_rather_than_raising(engine):
    """No audio pushed at all — the app opens before the user consents."""
    engine.render("Bars", 80, 24)


def test_an_unknown_mode_is_a_keyerror_not_a_wrong_picture(engine):
    with pytest.raises(KeyError):
        engine.render("No Such Mode", 80, 24)


def test_the_engine_smooths_like_the_desktop_widget():
    """The motion layer is between the analyser and the modes, and easy to miss.

    ``Analyser`` publishes raw bands and no peaks at all; every bit of the
    easing that makes spektr look like spektr lives in the widget. A port that
    forwards the Frame straight to the modes renders the same shapes, jittering
    — so this asserts the springs are actually in the path, by checking that
    one loud block does not arrive at full height in one frame.
    """
    engine = spektr_android.Engine(samplerate=RATE)
    engine.push(_pcm())
    engine.render("Bars", 80, 24)
    first = float(engine._spring.x.max())
    for _ in range(30):
        engine.push(_pcm())
        engine.render("Bars", 80, 24)
    settled = float(engine._spring.x.max())
    assert first < settled, "bands arrived instantly — the spring is not in the path"
    assert engine._peaks.value is not None


def test_the_vendored_engine_matches_the_one_the_desktop_runs():
    """The APK ships a *copy* of ``spektr/``, and a copy can drift.

    It is vendored rather than pip-installed because pip would resolve the
    desktop-only dependencies (sounddevice, soundcard, winrt, dbus-next), none
    of which have Android wheels. ``android/scripts/sync-python.ps1`` refreshes
    it — but a script nobody is forced to run is how this branch got 101
    commits behind in the first place, so forgetting has to fail here instead
    of shipping an APK that renders a stale engine.
    """
    real = ROOT / "spektr"
    shipped = ROOT / "android" / "app" / "src" / "main" / "python" / "spektr"
    assert shipped.is_dir(), "the vendored engine is missing from the APK tree"

    def files(base):
        # Compared as lines, not as bytes. The two copies were committed from a
        # Windows checkout at different times and git stored different line
        # endings for them, so a byte comparison passes on Windows — where
        # checkout normalises both — and fails on a Linux runner, which is
        # exactly how this first went red. What has to match is the code.
        return {
            p.relative_to(base).as_posix(): p.read_text(encoding="utf-8").splitlines()
            for p in sorted(base.rglob("*.py"))
            if "__pycache__" not in p.parts
        }

    a, b = files(real), files(shipped)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    differs = sorted(k for k in set(a) & set(b) if a[k] != b[k])

    assert not missing, f"not shipped in the APK: {missing}  (run sync-python.ps1)"
    assert not extra, f"in the APK but not in spektr/: {extra}  (run sync-python.ps1)"
    assert not differs, f"vendored copy is stale: {differs}  (run sync-python.ps1)"


def test_pushed_audio_lands_in_the_ring_unchanged(engine):
    """Pins the sample format, which nothing else did.

    ``push`` reads float32 because that is what ``ENCODING_PCM_FLOAT`` gives.
    Hand it 16-bit and nothing raises — ``frombuffer`` reinterprets the same
    bytes and the modes cheerfully draw noise. A wrong format that renders is
    worse than one that crashes, so the bytes are checked rather than the fact
    that a frame came out.
    """
    sig = _tone()
    engine.push(np.stack([sig, sig], axis=1).ravel().astype("<f4").tobytes())

    got = engine._ring.latest(sig.size)
    assert got is not None, "nothing reached the ring"
    assert got.shape == (sig.size, 2), f"ring holds {got.shape}, expected stereo pairs"
    assert np.allclose(got[:, 0], sig, atol=1e-6), "samples came back changed — wrong dtype?"


def test_mono_is_widened_to_two_columns(engine):
    sig = _tone()
    engine.push(sig.astype("<f4").tobytes(), channels=1)
    got = engine._ring.latest(sig.size)
    assert got is not None and got.shape == (sig.size, 2)
    assert np.allclose(got[:, 0], got[:, 1]), "mono should land identically in both columns"


def test_the_colours_kotlin_asks_for_come_back_as_flat_lists(engine):
    """The contract that replaced Kotlin reaching into Python's objects.

    ``PyEngine.create`` used to read ``BUILTIN[name]`` and ``Palette.hexes``
    across the boundary. Chaquopy's ``PyObject.get`` is *attribute* access, so
    asking a dict for ``.get("gruvbox")`` returned null and the ``!!`` after it
    threw a NullPointerException before the first frame — which is exactly what
    the tablet showed. These are plain lists reached by method call, which is
    unambiguous from Kotlin.
    """
    from spektr.palette import RAMP_STEPS

    ramp = engine.ramp_hexes()
    assert isinstance(ramp, list) and len(ramp) == RAMP_STEPS
    assert all(isinstance(h, str) for h in ramp), "Kotlin calls toString on each"

    chrome = engine.chrome_hexes()
    assert isinstance(chrome, list) and len(chrome) == 2

    for hexes, what in ((ramp, "ramp"), (chrome, "chrome")):
        for h in hexes:
            assert h.startswith("#") and len(h) == 7, f"{what}: {h!r} is not #rrggbb"
            int(h[1:], 16)          # Kotlin parses these as three hex pairs


def test_set_theme_reports_success_and_actually_changes_the_ramp(engine):
    """Kotlin now throws when this returns False, so the bool has to be right."""
    before = engine.ramp_hexes()
    assert engine.set_theme("nord") is True
    after = engine.ramp_hexes()
    assert after != before, "set_theme said yes and changed nothing"
    assert engine.chrome_hexes() != [], "chrome went missing with the theme"
    assert engine.set_theme("no such theme") is False
    assert engine.ramp_hexes() == after, "a rejected theme still moved the ramp"


def test_stats_report_the_interval_they_cover_and_then_reset(engine):
    """The debug build's only window onto how the engine is behaving."""
    engine.push(_pcm())
    for _ in range(5):
        engine.render("Bars", 60, 20)

    fps, dt_ms, energy, onsets, band, sample = engine.stats()
    assert fps > 0, "frames were rendered but the rate came back zero"
    assert dt_ms > 0
    assert 0.0 <= energy <= 1.5
    assert onsets >= 0.0
    assert band >= 0.0
    assert sample > 0.0, "audio was pushed but no sample peak was seen"

    # Reset on read, or every line describes the whole session instead of its
    # own second and a burst stops being visible.
    assert engine.stats()[0] == 0.0


def test_stats_see_the_level_of_what_was_pushed(engine):
    """The number Fireworks' launch rate is a function of.

    ``launch_acc += (0.35 + energy * 7.0) * dt``, so if a platform hands the
    engine hotter audio than another the sky fills faster on the same music —
    a difference no test of the mode itself would show.
    """
    quiet = spektr_android.Engine(samplerate=RATE)
    loud = spektr_android.Engine(samplerate=RATE)
    sig = _tone()
    for e, mul in ((quiet, 0.05), (loud, 1.0)):
        e.push(np.stack([sig * mul] * 2, axis=1).ravel().astype("<f4").tobytes())
        for _ in range(5):
            e.render("Bars", 60, 20)

    assert loud.stats()[5] > quiet.stats()[5] * 10, "the sample peak did not follow the input"


def test_use_theme_switches_and_yields_every_colour_in_one_call(engine):
    """One crossing, so a theme switch cannot half-apply.

    ``set_theme`` plus ``ramp_hexes`` plus ``chrome_hexes`` is three calls that
    can fail independently: a theme that switches and then fails to yield its
    ramp leaves the grid drawn in the old colours on the new background, and
    nothing in the app would say so.
    """
    from spektr.palette import RAMP_STEPS

    before = engine.ramp_hexes()
    got = engine.use_theme("nord")
    assert got is not None
    assert len(got) == RAMP_STEPS + 2, "expected bg + fg + the whole ramp"
    assert got[2:] == engine.ramp_hexes() != before
    assert got[:2] == engine.chrome_hexes()
    for h in got:
        assert h.startswith("#") and len(h) == 7, f"{h!r} is not #rrggbb"
        int(h[1:], 16)                       # Kotlin parses three hex pairs


def test_use_theme_refuses_an_unknown_name_without_moving_anything(engine):
    """What a selection saved by a newer build looks like to an older one."""
    engine.use_theme("nord")
    ramp = engine.ramp_hexes()
    assert engine.use_theme("no such theme") is None
    assert engine.ramp_hexes() == ramp, "a rejected theme still moved the ramp"


def test_every_theme_has_a_swatch_the_picker_can_paint(engine):
    """Fifty-four names is a list, not a choice — the picker draws colours."""
    rows = engine.theme_swatches()
    names = engine.theme_names()
    assert [r[0] for r in rows] == names, "swatches and names are out of step"
    for row in rows:
        assert len(row) == 3 + 6, f"{row[0]}: {len(row)} fields"
        for h in row[1:]:
            assert h.startswith("#") and len(h) == 7, f"{row[0]}: {h!r}"
            int(h[1:], 16)


def test_a_swatch_shows_more_than_one_colour(engine):
    """Taking the first n of a ramp gives six near-background smudges.

    Which makes every theme look the same in the picker — the exact failure
    the swatch exists to prevent, and invisible to a test that only checks the
    fields parse.
    """
    flat = [r[0] for r in engine.theme_swatches() if len(set(r[3:])) < 3]
    assert not flat, f"themes whose swatch is nearly one colour: {flat}"


def test_swatches_are_built_once(engine):
    """They cost ~70 ms of ramp interpolation; that must never land on a frame."""
    first = engine.theme_swatches()
    assert engine.theme_swatches() is first


def test_every_builtin_theme_survives_the_kotlin_path(engine):
    """One bad theme would be a crash on a picker that does not exist yet."""
    bad = []
    for name in engine.theme_names():
        assert engine.set_theme(name) is True, name
        try:
            ramp, chrome = engine.ramp_hexes(), engine.chrome_hexes()
            assert len(ramp) == 64 and len(chrome) == 2
            for h in ramp + chrome:
                assert h.startswith("#") and len(h) == 7
                int(h[1:], 16)
        except Exception as exc:                     # noqa: BLE001 — reporting
            bad.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not bad, "themes Kotlin could not colour:\n" + "\n".join(bad)
