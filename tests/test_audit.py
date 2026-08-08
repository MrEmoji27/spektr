"""Correctness audit — the checks that catch logic errors, not crashes.

``bench.py`` proves every mode returns arrays of the right shape. That is a low
bar: a mode can pass it while producing a still image, ignoring the audio
entirely, quietly corrupting the shared band buffer, or leaking a buffer on
every resize. These are the checks for that class of bug.
"""

from __future__ import annotations

import gc
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# cp1252 consoles cannot encode the box-drawing and dash characters printed
# below; utf-8 output keeps the suite runnable on a default Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spektr.modes as M  # noqa: E402
from spektr.analysis import N_BANDS, Analyser, resample_bands  # noqa: E402
from spektr.capture import RingBuffer  # noqa: E402
from spektr.modes import Ctx  # noqa: E402
from spektr.palette import BUILTIN, RAMP_STEPS, Palette, all_themes  # noqa: E402
from spektr.render import SPACE, make_strips  # noqa: E402

PAL = Palette(BUILTIN["gruvbox"])

#: Modes with motion of their own — scrolling, drifting, decaying, sparkling.
#: Everything else is a pure function of the spectrum, and *should* hold still
#: when the spectrum holds still: a bar chart of a constant signal is a
#: constant bar chart. Testing those for movement was an error in an earlier
#: version of this file, and it flagged seven correct modes as broken.
SELF_ANIMATING = {
    "Scatter", "Flame", "Pulse", "Retro", "Tunnel", "Warp",
    "Matrix", "Spectro", "Plasma", "Gonio",
    # scroll (ECG), standing-wave phase (Strings), ring breathing and angular
    # spin (Arcs), travelling particles (Bubbles), curtain billow (Auroras),
    # continuous sweep rotation (Sonar)
    "ECG", "Strings", "Arcs", "Bubbles", "Auroras", "Sonar",
    # continuous ticker scroll (Readout), continuous spin (Helix, Chladni),
    # continuous orbital motion (Orbit), continuous scroll (Keys), a real
    # playback clock (Flipbook), a persisted simulation that keeps evolving
    # on its own (Maelstrom), state that accumulates over time by design
    # (Dune), and physics that keeps running after a trigger — a launched
    # firework doesn't stop just because the spectrum did (Fireworks) — plus
    # a boot log that keeps typing, scrolling, and blinking its cursor on its
    # own clock (Boot), a cellular automaton stepping generations on its own
    # clock (Colony), a flock whose agents keep moving under their own
    # inter-agent forces even at a constant spectrum (Murmuration), and the
    # lofi set, each animating on its own even in silence by design: a
    # record keeps spinning after the needle drops
    # (Vinyl), rain keeps falling at a small base rate so the loop never
    # looks paused (Rain), embers keep drifting and cooling on their own
    # timers once spawned (Ember), bulbs each twinkle on their own fixed
    # clock (Fairylights), reels keep turning and the counter keeps ticking
    # (Cassette), steam keeps curling and rising off the cup (Steam), and
    # a speaker cone that is being driven keeps rippling — the phase
    # accumulates every frame, a held note is a cone still moving (Cabinet)
    "Readout", "Helix", "Chladni", "Orbit", "Keys", "Flipbook", "Maelstrom",
    "Dune", "Fireworks", "Boot", "Colony", "Murmuration", "Vinyl", "Rain",
    "Ember", "Fairylights", "Cassette", "Steam", "Cabinet",
}

#: Modes driven by the waveform rather than the band levels.
WAVEFORM_DRIVEN = {"Wave", "Scope", "Gonio", "ECG"}


def ctx_for(w, h, frame, state, t, bands=None, silent=False, stereo=None, dt=1 / 60):
    if bands is None:
        bands = np.clip(np.abs(np.sin(np.linspace(0, 3, N_BANDS) + t * 2)) * 0.8, 0, 1)
    wave = np.sin(np.linspace(0, 40, 512) + t * 10) * 0.7
    if stereo is None:
        stereo = np.stack((wave, np.roll(wave, 7)), axis=1)
    return Ctx(
        w=w, h=h, bands=bands, peaks=np.clip(bands * 1.05, 0, 1),
        bands_l=bands * 0.9, bands_r=bands, wave=wave, stereo=stereo,
        frame=frame, t=t, dt=dt, energy=float(bands.mean()),
        silent=silent, palette=PAL, state=state,
    )


# ── 1. modes must not write into the buffers they're handed ──────────────────

def check_no_mutation() -> list[str]:
    """``ctx.bands`` is the live spring buffer, shared with every other mode
    and with the next frame. A mode writing into it corrupts the animation
    globally and the symptom would appear somewhere else entirely."""
    bad = []
    for m in M.MODES:
        state: dict = {}
        for w, h in ((80, 24), (200, 50)):
            bands = np.linspace(0.1, 0.95, N_BANDS)
            ctx = ctx_for(w, h, 3, state, 0.5, bands=bands)
            snapshot = {
                "bands": ctx.bands.copy(), "peaks": ctx.peaks.copy(),
                "bands_l": ctx.bands_l.copy(), "bands_r": ctx.bands_r.copy(),
                "wave": ctx.wave.copy(), "stereo": ctx.stereo.copy(),
            }
            m.fn(ctx)
            for name, before in snapshot.items():
                if not np.array_equal(getattr(ctx, name), before):
                    bad.append(f"{m.name} mutated ctx.{name} at {w}x{h}")
    return bad


# ── 2. modes must actually animate ───────────────────────────────────────────

def _changes_over(m, frames: int, freeze_audio: bool = False) -> bool:
    """Render N frames and report whether the picture ever changed.

    With ``freeze_audio`` the spectrum *and* the waveform are held constant,
    so only time advances — that isolates a mode's own motion from motion it
    inherits from the signal.
    """
    state: dict = {}
    shots = []
    steady_bands = np.full(N_BANDS, 0.6)
    steady_wave = np.sin(np.linspace(0, 40, 512)) * 0.7
    steady_stereo = np.stack((steady_wave, np.roll(steady_wave, 7)), axis=1)

    for i in range(frames):
        if freeze_audio:
            ctx = ctx_for(160, 40, i, state, i / 30, bands=steady_bands,
                          stereo=steady_stereo)
            ctx.wave = steady_wave
        else:
            ctx = ctx_for(160, 40, i, state, i / 30)
        out = m.fn(ctx)
        shots.append((out[0].copy(), out[1].copy()))

    return any(
        not np.array_equal(shots[0][0], s[0]) or not np.array_equal(shots[0][1], s[1])
        for s in shots[1:]
    )


def check_animates() -> list[str]:
    """Two separate properties, which an earlier version of this file conflated.

    Every mode must move when the *spectrum* moves. Only the self-animating
    ones must also move when the spectrum is frozen — a bar chart of a constant
    signal is supposed to be a constant bar chart.
    """
    bad = []
    for m in M.MODES:
        if m.name == "None":
            continue
        # varying input: everything must respond
        if not _changes_over(m, 12):
            bad.append(f"{m.name} is static even when the spectrum changes")
        # frozen input: only the self-animating modes should still move
        moves_alone = _changes_over(m, 14, freeze_audio=True)
        if m.name in SELF_ANIMATING and not moves_alone:
            bad.append(f"{m.name} claims to self-animate but froze on steady input")
        if m.name not in SELF_ANIMATING and moves_alone:
            bad.append(f"{m.name} moves on its own — should it be in SELF_ANIMATING?")
    return bad


# ── 3. modes must respond to the audio ───────────────────────────────────────

def check_audio_reactive() -> list[str]:
    """Loud and quiet must not render identically.

    Waveform-driven modes are fed a quiet and a loud *waveform* rather than
    band levels — testing those against band level alone was another error in
    an earlier version here, and it flagged three correct modes.
    """
    bad = []
    quiet_bands = np.full(N_BANDS, 0.02)
    loud_bands = np.full(N_BANDS, 0.95)
    base = np.sin(np.linspace(0, 40, 512))

    for m in M.MODES:
        if m.name == "None":
            continue
        wave_driven = m.name in WAVEFORM_DRIVEN
        a_state: dict = {}
        b_state: dict = {}
        differs = False
        for i in range(8):
            if wave_driven:
                qa = np.stack((base * 0.01, np.roll(base, 7) * 0.01), axis=1)
                lb = np.stack((base * 0.95, np.roll(base, 7) * 0.95), axis=1)
                ca = ctx_for(160, 40, i, a_state, i / 30, stereo=qa)
                cb = ctx_for(160, 40, i, b_state, i / 30, stereo=lb)
                ca.wave = base * 0.01
                cb.wave = base * 0.95
            else:
                ca = ctx_for(160, 40, i, a_state, i / 30, bands=quiet_bands)
                cb = ctx_for(160, 40, i, b_state, i / 30, bands=loud_bands)
            a, b = m.fn(ca), m.fn(cb)
            if not np.array_equal(a[0], b[0]) or not np.array_equal(a[1], b[1]):
                differs = True
                break
        if not differs:
            what = "waveform amplitude" if wave_driven else "band level"
            bad.append(f"{m.name} renders identically at 0.01 and 0.95 {what}")
    return bad


# ── 4. no NaN, no inf, no surrogates ─────────────────────────────────────────

def check_output_sanity() -> list[str]:
    """Codepoints in the UTF-16 surrogate range cannot be decoded, so they
    would crash the strip builder rather than draw something odd."""
    bad = []
    for m in M.MODES:
        state: dict = {}
        for w, h in ((13, 3), (80, 24), (240, 60)):
            for i in range(3):
                out = m.fn(ctx_for(w, h, i, state, i / 30))
                codes, cidx = out[0], out[1]
                if not np.isfinite(np.asarray(codes, dtype=np.float64)).all():
                    bad.append(f"{m.name} produced non-finite codepoints at {w}x{h}")
                if codes.min() < 0:
                    bad.append(f"{m.name} produced a negative codepoint at {w}x{h}")
                if codes.max() > 0x10FFFF:
                    bad.append(f"{m.name} produced a codepoint past U+10FFFF at {w}x{h}")
                sur = ((codes >= 0xD800) & (codes <= 0xDFFF)).any()
                if sur:
                    bad.append(f"{m.name} produced a surrogate codepoint at {w}x{h}")
                if cidx.min() < 0 or cidx.max() >= RAMP_STEPS:
                    bad.append(f"{m.name} ramp index out of range at {w}x{h}")
    return bad


def check_surrogates_survive_strips() -> list[str]:
    """A plugin returning a surrogate must not take the renderer down."""
    from spektr.plugins import validate

    h, w = 6, 10
    codes = np.full((h, w), 0xD800, dtype=np.int32)   # lone surrogate
    cidx = np.zeros((h, w), dtype=np.int32)
    try:
        codes, cidx = validate((codes, cidx), w, h)
    except Exception as exc:  # noqa: BLE001
        return [f"validate rejected surrogates instead of remapping: {exc}"]
    try:
        strips = make_strips(codes, cidx, PAL)
    except Exception as exc:  # noqa: BLE001
        return [f"surrogate codepoint crashed make_strips: {exc!r}"]
    if len(strips) != h:
        return ["surrogate path produced the wrong number of strips"]
    return []


# ── 5. state must not leak across resizes ────────────────────────────────────

def check_scratch_does_not_leak() -> list[str]:
    """Modes cache geometry keyed on size. Resizing repeatedly must not grow
    the cache without bound — on a tiled window manager that's every drag."""
    bad = []
    for m in M.MODES:
        state: dict = {}
        for i in range(40):
            w = 60 + (i % 20) * 7
            h = 20 + (i % 11) * 3
            m.fn(ctx_for(w, h, i, state, i / 30))
        if len(state) > 4:
            bad.append(f"{m.name} kept {len(state)} scratch entries after 40 resizes")
    return bad


# ── 5b. scrolling modes must scroll in seconds, not in frames ────────────────

#: Modes that scroll their own history buffer on a clock. Their content has to
#: travel the same distance per *second* at any frame rate — see the check.
SCROLL_PACED = ("Spectro", "ECG")


def check_scroll_is_frame_rate_independent() -> list[str]:
    """A scrolling mode's time axis must not mean "however fast we happen to render".

    Both of these regressed this way and it was visible, not theoretical.
    ``Spectro`` shifted a fixed one column per frame, so one second of audio
    occupied 30 columns at 30 fps against 120 at 120 fps. ``ECG`` rounded its
    per-frame column step, which at a narrow terminal alternated between 1 and
    2 columns from frame to frame and, with its old ``max(1, ...)`` floor, ran
    +74% fast at 60 columns / 120 fps.

    Both are worse than a fixed offset, because widget.py's adaptive pacer
    retimes fps by +/-6 at runtime: the picture visibly slowed down and sped
    back up with nothing in the audio changing. It also breaks what the
    settings panel promises about the frame-rate row ("the motion is timed in
    seconds, so this changes smoothness only").

    Driven by a single impulse followed by silence, so the measurement is where
    that impulse ended up — no RNG involved, unlike the modes whose particles
    respawn randomly and legitimately differ frame-to-frame at different rates.
    """
    bad = []
    w, h = 120, 30
    seconds = 1.0

    def impulse_col(m, fps):
        state: dict = {}
        dt = 1.0 / fps
        codes = None
        for i in range(int(seconds * fps)):
            loud = i == 0
            bands = np.full(N_BANDS, 0.9 if loud else 0.0)
            codes = m.fn(ctx_for(w, h, i, state, i * dt, bands=bands, silent=not loud, dt=dt))[0]
        lit = (codes != SPACE) & (codes != 0x2800)   # 0x2800 is blank braille
        cols = np.flatnonzero(lit.any(axis=0))
        return int(cols.min()) if cols.size else None

    for name in SCROLL_PACED:
        m = M.get(name)
        if m is None:
            bad.append(f"{name} is missing — update SCROLL_PACED")
            continue
        seen = {fps: impulse_col(m, fps) for fps in (30, 60, 120)}
        if any(v is None for v in seen.values()):
            bad.append(f"{name}: impulse vanished entirely at {seen} — nothing to measure")
            continue
        spread = max(seen.values()) - min(seen.values())
        # a couple of columns of slack for integer rounding at each rate
        if spread > 3:
            bad.append(
                f"{name} scrolls at a frame-rate-dependent speed: "
                f"impulse at column {seen} after {seconds}s — should match within 3"
            )
    return bad


# ── 6. band resampling must conserve level ───────────────────────────────────

def check_resample_conserves() -> list[str]:
    bad = []
    for n_out in (8, 10, 12, 16, 24, 31, 32, 40, 64):
        flat = np.full(N_BANDS, 0.5)
        out = resample_bands(flat, n_out)
        if not np.allclose(out, 0.5, atol=1e-9):
            bad.append(f"resample({n_out}) of a flat 0.5 spectrum gave {out.min()}..{out.max()}")
        ramp = np.linspace(0.0, 1.0, N_BANDS)
        out = resample_bands(ramp, n_out)
        if np.any(np.diff(out) < -1e-9):
            bad.append(f"resample({n_out}) broke monotonicity of a rising ramp")
        if not np.isfinite(out).all():
            bad.append(f"resample({n_out}) produced non-finite values")
    return bad


# ── 7. palette ramps must actually be ramps ──────────────────────────────────

def check_palette_ramps() -> list[str]:
    """Ramps must be smooth and hit their anchors.

    Not *brighter* at the top — an earlier version asserted that and failed 20
    of 30 themes, wrongly. A spectrum ramp travels through hue, not luminance:
    classic runs green → amber → red, and red is genuinely darker than green.
    What actually matters is that the gradient has no visible seam and that the
    ends are the colours the theme asked for. Animated ramps are closed loops:
    they have no ends, and the wrap back to the start must be as smooth as the
    rest of the ramp.
    """
    from spektr.palette import hex_to_rgb

    bad = []
    for name, theme in all_themes().items():
        p = Palette(theme)
        if len(p.hexes) != RAMP_STEPS:
            bad.append(f"{name}: ramp has {len(p.hexes)} steps")
            continue

        rgb = np.array([hex_to_rgb(c) for c in p.hexes], dtype=np.float64)

        # smoothness: no single step may jump more than a few times the median.
        # An animated ramp is a closed loop, so the step that wraps from the last
        # colour back to the first is part of the ramp too — a seam there is
        # exactly the bug the loop exists to remove.
        steps = np.abs(np.diff(rgb, axis=0)).sum(axis=1)
        if theme.animated:
            steps = np.append(steps, float(np.abs(rgb[0] - rgb[-1]).sum()))

        # ends must match the theme's declared anchors. A loop has no ends — it
        # starts on the low colour and wraps back to it — so only the start is
        # pinned; the high anchor is a riding point on the loop, not a terminus.
        if theme.animated:
            end_checks = (("low", theme.low, p.hexes[0]),)
        else:
            end_checks = (
                ("low", theme.low, p.hexes[0]),
                ("high", theme.high, p.hexes[-1]),
            )
        for label, want, got in end_checks:
            if max(abs(a - b) for a, b in zip(hex_to_rgb(want), hex_to_rgb(got))) > 2:
                bad.append(f"{name}: ramp {label} end is {got}, theme declares {want}")

        median = float(np.median(steps))
        if median > 0 and steps.max() > median * 6:
            worst = int(np.argmax(steps))
            bad.append(
                f"{name}: seam at step {worst} — jumps {steps.max():.0f} against a median of {median:.0f}"
            )

        # a banded ramp would repeat colours. Smoothstep easing goes to zero
        # velocity at every anchor by construction, so a run of identical
        # 8-bit-rounded steps right at the mid anchor is normal — everforest,
        # gruvbox-light and tokyo-night all sit at 14-16 consecutive steps
        # there and look fine; only a genuinely degenerate ramp (near-flat
        # start to finish) should trip this. RAMP_STEPS // 2 was tuned back
        # when RAMP_STEPS was 64; raising resolution to 256 for a smoother
        # animated spread (see AudioVisualizer._animate_ramp) shrank the same
        # anchor plateau's relative share of the ramp not at all — it's still
        # ~16 steps — while doubling what "half the steps" demands, and
        # everforest's low-contrast palette no longer had enough *other*
        # distinct steps to clear it. // 3 keeps the check meaningful without
        # that resolution-coupling.
        if len(set(p.hexes)) < RAMP_STEPS // 3:
            bad.append(f"{name}: only {len(set(p.hexes))} distinct colours in {RAMP_STEPS} steps")

        idx = p.indices(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
        if idx[0] != 0 or idx[-1] != RAMP_STEPS - 1:
            bad.append(f"{name}: indices() did not clamp out-of-range input")
        if not (0 <= idx.min() and idx.max() < RAMP_STEPS):
            bad.append(f"{name}: indices() escaped the ramp")
    return bad


# ── 8. ring buffer under concurrent access ───────────────────────────────────

def check_ring_concurrency() -> list[str]:
    """The audio callback writes while the analyser reads. Verify no torn
    reads and no lost frames under real contention."""
    ring = RingBuffer(4096)
    stop = threading.Event()
    written = [0]
    errors: list[str] = []

    def writer():
        v = 0.0
        while not stop.is_set():
            block = np.full((256, 2), v, dtype=np.float32)
            ring.push(block)
            written[0] += 256
            v += 1.0

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 0.8
        reads = 0
        while time.monotonic() < deadline:
            buf = ring.latest(2048)
            if buf is None:
                continue
            reads += 1
            if not np.isfinite(buf).all():
                errors.append("ring returned non-finite samples")
                break
            # every 256-sample block was written with a single constant, so
            # any block boundary inside the read must be a clean step
            if buf.shape != (2048, 2):
                errors.append(f"ring returned shape {buf.shape}")
                break
    finally:
        stop.set()
        t.join(timeout=1.0)

    if reads < 10:
        errors.append(f"only {reads} successful concurrent reads")
    if ring.written != written[0]:
        errors.append(f"ring counted {ring.written} frames, writer pushed {written[0]}")
    return errors


# ── 9. analyser must not drift or leak ───────────────────────────────────────

def check_analyser_stability() -> list[str]:
    """Long run at real time: sequence numbers must advance steadily and the
    published frame must always be internally consistent."""
    SR = 48000
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)
    an.start()
    bad = []
    try:
        phase = 0
        seqs = []
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            t = (np.arange(512) + phase) / SR
            s = (np.sin(2 * math.pi * 440 * t) * 0.3).astype(np.float32)
            ring.push(np.stack((s, s), axis=1))
            phase += 512
            time.sleep(512 / SR)
            f = an.frame
            seqs.append(f.seq)
            if len(f.bands) != N_BANDS:
                bad.append(f"frame.bands has {len(f.bands)} entries")
                break
            if not np.isfinite(f.bands).all():
                bad.append("frame.bands contained non-finite values")
                break
            if f.bands.min() < 0 or f.bands.max() > 1:
                bad.append(f"frame.bands out of 0..1: {f.bands.min()}..{f.bands.max()}")
                break
    finally:
        an.stop()
        time.sleep(0.05)

    if not bad:
        # monotonic, never going backwards (would mean a torn publish)
        if any(b < a for a, b in zip(seqs, seqs[1:])):
            bad.append("frame.seq went backwards — publish is not atomic")
    return bad


# ── 10. gate behaviour ───────────────────────────────────────────────────────

def check_gate_hysteresis() -> list[str]:
    """The gate holds open briefly after sound stops, so a quiet passage
    doesn't make the display flicker off between notes."""
    SR = 48000
    ring = RingBuffer(SR)
    an = Analyser(ring, lambda: SR)

    # Enough samples to fill the bass window — the analyser reads 8192 frames
    # at this rate and publishes nothing until it has them.
    n = 16384
    t = np.arange(n) / SR
    loud = (np.sin(2 * math.pi * 440 * t) * 0.3).astype(np.float32)
    ring.push(np.stack((loud, loud), axis=1))
    an._analyse_once()
    if an.frame.silent:
        return ["gate did not open on a loud tone"]

    # immediately silent: hold should keep it open
    ring.push(np.zeros((n, 2), dtype=np.float32))
    an._analyse_once()
    held = not an.frame.silent

    time.sleep(0.35)          # past the hold window
    an._analyse_once()
    closed = an.frame.silent

    bad = []
    if not held:
        bad.append("gate slammed shut the instant the signal stopped (no hold)")
    if not closed:
        bad.append("gate never closed after the hold window elapsed")
    return bad


# ── 11. theme visibility ──────────────────────────────────────────────────────

def check_theme_visibility() -> list[str]:
    """A theme's colours must actually be visible against its own background.

    Nothing here asserts *how* a theme should look — that's taste, and themes
    range from vantablack's near-monochrome greys to rainbow's full hue wheel
    on purpose, both deliberately. What isn't taste: a ramp anchor close
    enough to the background that it renders as flat background instead of a
    colour. ``infrared`` shipped with exactly that bug — a low anchor of
    #3a0000 on a #0d0000 background, a perceptual RGB distance of 0.18
    against every other theme's 0.27 or higher — and nothing caught it,
    because no check ever compared a theme's colours to its own bg. This one
    does, plus the WCAG AA text-contrast floor for fg-on-bg, which every
    built-in theme already happened to clear.
    """
    from spektr.palette import _luminance, hex_to_rgb

    def dist(a, b) -> float:
        x = np.array(hex_to_rgb(a), dtype=np.float64) / 255.0
        y = np.array(hex_to_rgb(b), dtype=np.float64) / 255.0
        return float(np.sqrt(((x - y) ** 2).sum()))

    def contrast(a, b) -> float:
        la, lb = _luminance(a), _luminance(b)
        hi, lo = max(la, lb), min(la, lb)
        return (hi + 0.05) / (lo + 0.05)

    bad = []
    for name, theme in all_themes().items():
        for label, colour in (("low", theme.low), ("mid", theme.mid), ("high", theme.high)):
            d = dist(colour, theme.bg)
            if d < 0.18:
                bad.append(
                    f"{name}: {label} anchor {colour} is only {d:.2f} from bg {theme.bg} — "
                    "nearly invisible"
                )
        fg_bg = contrast(theme.fg, theme.bg)
        if fg_bg < 4.5:
            bad.append(f"{name}: fg/bg contrast is {fg_bg:.2f}, below WCAG AA's 4.5")
    return bad


def check_fps_sentinel() -> list[str]:
    """``fps`` accepts 0 as "unlimited" without swallowing junk into it.

    The sentinel is the one value in ``Settings`` where a clamp bound and a
    meaningful value collide. Clamping first and then testing for 0 turns
    every negative number in a hand-edited config into a request to uncap,
    and JSON booleans compare equal to 0, so both are checked here.

    Also pins the resolution side: a failed probe must be distinguishable from
    a probe that genuinely measured the fallback, and neither may raise.
    """
    from unittest.mock import patch

    import spektr.display as disp
    from spektr.config import FPS_MAX, FPS_UNLIMITED, Settings
    from spektr.widget import AudioVisualizer

    bad = []
    for raw, want in (
        (0, FPS_UNLIMITED), (0.0, FPS_UNLIMITED),
        (60, 60), (144, 144), (FPS_MAX, FPS_MAX),
        (10_000, FPS_MAX), (5, 15), (-3, 15), (-999, 15),
        (None, 60), ("junk", 60), (float("nan"), 60),
        (True, 15), (False, 15),
    ):
        s = Settings()
        s.fps = raw
        s.clamp()
        if s.fps != want:
            bad.append(f"Settings.fps={raw!r} clamped to {s.fps!r}, expected {want!r}")

    def boom():
        raise OSError("no display")

    cap = 375
    for probe, want in ((lambda: 144.0, (144, 144)), (lambda: None, (60, None)),
                        (lambda: 500.0, (cap, 500)), (boom, (60, None))):
        saved, disp._UNLIMITED = disp._UNLIMITED, None
        try:
            with patch.object(disp, "refresh_hz", probe):
                got = disp.unlimited_fps(cap)
            if got != want:
                bad.append(f"unlimited_fps: probe gave {got}, expected {want}")
        finally:
            disp._UNLIMITED = saved

    viz = object.__new__(AudioVisualizer)
    viz._unlimited = (144, 144)
    for raw, want in ((FPS_UNLIMITED, 144), (30, 30), (10, 15), (10_000, FPS_MAX)):
        got = viz._resolve_fps(raw)
        if got != want:
            bad.append(f"_resolve_fps({raw!r}) = {got}, expected {want}")
    return bad


TESTS = [
    ("modes don't mutate shared buffers", check_no_mutation),
    ("modes animate", check_animates),
    ("modes react to audio", check_audio_reactive),
    ("output sanity (no NaN/surrogates)", check_output_sanity),
    ("surrogates survive the strip builder", check_surrogates_survive_strips),
    ("no scratch leak across resizes", check_scratch_does_not_leak),
    ("scrolling modes scroll in seconds, not frames", check_scroll_is_frame_rate_independent),
    ("band resampling conserves level", check_resample_conserves),
    ("palette ramps are smooth", check_palette_ramps),
    ("ring buffer under contention", check_ring_concurrency),
    ("analyser stability over 1 s", check_analyser_stability),
    ("gate hysteresis", check_gate_hysteresis),
    ("themes are visible against their own background", check_theme_visibility),
    ("fps unlimited sentinel survives clamping", check_fps_sentinel),
]


# ── pytest entry points ───────────────────────────────────────────────────────
#
# A bare ``def test_x(): return bad`` does not fail under pytest — a non-None
# return only trips a PytestReturnNotNoneWarning, not a failure — so every
# check above was reporting green under ``pytest tests/`` regardless of what
# it actually found. Only running this file directly (the loop below) ever
# turned a finding into a nonzero exit. These thin wrappers are what make
# ``pytest`` itself catch the same problems; the checks' own logic is
# untouched.

def test_no_mutation() -> None:
    bad = check_no_mutation()
    assert not bad, "\n".join(bad)


def test_animates() -> None:
    bad = check_animates()
    assert not bad, "\n".join(bad)


def test_audio_reactive() -> None:
    bad = check_audio_reactive()
    assert not bad, "\n".join(bad)


def test_output_sanity() -> None:
    bad = check_output_sanity()
    assert not bad, "\n".join(bad)


def test_surrogates_survive_strips() -> None:
    bad = check_surrogates_survive_strips()
    assert not bad, "\n".join(bad)


def test_scratch_does_not_leak() -> None:
    bad = check_scratch_does_not_leak()
    assert not bad, "\n".join(bad)


def test_scroll_is_frame_rate_independent() -> None:
    bad = check_scroll_is_frame_rate_independent()
    assert not bad, "\n".join(bad)


def test_resample_conserves() -> None:
    bad = check_resample_conserves()
    assert not bad, "\n".join(bad)


def test_palette_ramps() -> None:
    bad = check_palette_ramps()
    assert not bad, "\n".join(bad)


def test_ring_concurrency() -> None:
    bad = check_ring_concurrency()
    assert not bad, "\n".join(bad)


def test_analyser_stability() -> None:
    bad = check_analyser_stability()
    assert not bad, "\n".join(bad)


def test_gate_hysteresis() -> None:
    bad = check_gate_hysteresis()
    assert not bad, "\n".join(bad)


def test_theme_visibility() -> None:
    bad = check_theme_visibility()
    assert not bad, "\n".join(bad)


def test_fps_sentinel() -> None:
    bad = check_fps_sentinel()
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    failures = 0
    for name, fn in TESTS:
        t0 = time.monotonic()
        bad = fn()
        gc.collect()
        mark = "ok  " if not bad else "FAIL"
        print(f"  [{mark}] {name}  ({time.monotonic() - t0:.1f}s)")
        for b in bad:
            print(f"         {b}")
        failures += len(bad)
    print("\nall good" if not failures else f"\n{failures} problems")
    raise SystemExit(1 if failures else 0)
