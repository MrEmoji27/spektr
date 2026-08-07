"""ASCII animation loading tests — spektr/asciiart.py, no real config dir touched.

Redirects palette.config_dir the same way test_app.py does, for the same
reason: asciiart.py derives its directory from that one function, and a
careless test here writing into the developer's real %APPDATA%/spektr/ascii
is exactly the bug that leaked a preset into the real config dir earlier in
this project's history.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spektr.palette as _palette  # noqa: E402

_scratch_dir = Path(tempfile.mkdtemp(prefix="spektr-ascii-test-"))
_palette.config_dir = lambda: _scratch_dir  # type: ignore[attr-defined]

import spektr.asciiart as asciiart  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _reset() -> Path:
    """A clean, empty ascii/ dir for one check, with the module cache dropped."""
    d = asciiart.ascii_dir()
    if d.is_dir():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    asciiart.reload()
    asciiart.restore("", "warp")
    return d


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ── parsing ──────────────────────────────────────────────────────────────────

def check_ragged_lines_padded() -> list[str]:
    d = _reset()
    folder = d / "anim"
    folder.mkdir()
    _write(folder / "001.txt", "ab\nc")
    _write(folder / "002.txt", "wxyz")
    asciiart.reload()

    reel = next(r for r in asciiart.reels() if r.name == "anim")
    codes, _density = reel.frames()
    bad = []
    if codes.shape != (2, 2, 4):
        return [f"expected (2, 2, 4), got {codes.shape}"]
    row0 = "".join(chr(c) for c in codes[0, 0])
    row1 = "".join(chr(c) for c in codes[0, 1])
    if row0 != "ab  ":
        bad.append(f"frame 0 row 0: {row0!r}")
    if row1 != "c   ":
        bad.append(f"frame 0 row 1: {row1!r}")
    return bad


def check_tabs_expanded() -> list[str]:
    d = _reset()
    _write(d / "solo.txt", "a\tb")
    asciiart.reload()

    reel = next(r for r in asciiart.reels() if r.name == "solo")
    codes, _ = reel.frames()
    text = "".join(chr(c) for c in codes[0, 0])
    # expandtabs(8): "a" then padding to column 8, then "b"
    return [] if text == "a" + " " * 7 + "b" else [f"tab not expanded to a stop of 8: {text!r}"]


def check_ansi_stripped() -> list[str]:
    d = _reset()
    _write(d / "solo.txt", "\x1b[31mred\x1b[0m")
    asciiart.reload()

    reel = next(r for r in asciiart.reels() if r.name == "solo")
    codes, _ = reel.frames()
    text = "".join(chr(c) for c in codes[0, 0]).rstrip()
    return [] if text == "red" else [f"ANSI codes leaked into the glyphs: {text!r}"]


def check_crlf_normalised() -> list[str]:
    d = _reset()
    # write_bytes, not write_text: text-mode writes translate "\n" to the
    # platform line separator, which on Windows would silently turn an
    # already-CRLF source string into CRCRLF and defeat the point of the test.
    (d / "solo.txt").write_bytes(b"one\r\ntwo\r\n")
    asciiart.reload()

    reel = next(r for r in asciiart.reels() if r.name == "solo")
    codes, _ = reel.frames()
    if codes.shape[1] != 2:
        return [f"expected 2 rows from a 2-line file, got {codes.shape[1]}"]
    if "\r" in "".join(chr(c) for row in codes[0] for c in row):
        return ["a stray \\r ended up as a rendered glyph"]
    return []


# ── discovery ────────────────────────────────────────────────────────────────

def check_natural_frame_order() -> list[str]:
    """frame2 before frame10 — a plain string sort would get this backwards."""
    d = _reset()
    folder = d / "anim"
    folder.mkdir()
    for name, marker in (("frame1.txt", "A"), ("frame2.txt", "B"), ("frame10.txt", "C")):
        _write(folder / name, marker)
    asciiart.reload()

    reel = next(r for r in asciiart.reels() if r.name == "anim")
    codes, _ = reel.frames()
    order = "".join(chr(codes[i, 0, 0]) for i in range(3))
    return [] if order == "ABC" else [f"expected frame order A,B,C — got {order!r}"]


def check_bare_txt_is_a_single_frame_reel() -> list[str]:
    d = _reset()
    _write(d / "logo.txt", "hi")
    asciiart.reload()

    reel = next((r for r in asciiart.reels() if r.name == "logo"), None)
    if reel is None:
        return ["a bare .txt at the top level did not become a reel"]
    return [] if reel.n_frames == 1 else [f"expected 1 frame, got {reel.n_frames}"]


def check_non_txt_files_are_ignored() -> list[str]:
    """A stray non-animation file must be skipped, not crash discovery."""
    d = _reset()
    _write(d / "README.md", "not an animation")
    asciiart.reload()

    names = [r.name for r in asciiart.reels()]
    return [] if "README" not in names else ["a non-.txt file was picked up as a reel"]


def check_missing_frame_file_is_skipped_not_fatal() -> list[str]:
    """A frame deleted between discovery and decode must not raise."""
    d = _reset()
    folder = d / "anim"
    folder.mkdir()
    _write(folder / "001.txt", "one")
    _write(folder / "002.txt", "two")
    asciiart.reload()

    reel = next(r for r in asciiart.reels() if r.name == "anim")
    (folder / "002.txt").unlink()   # discovery already recorded this path

    try:
        codes, _ = reel.frames()
    except OSError as exc:
        return [f"a missing frame raised instead of being skipped: {exc}"]
    return [] if codes.shape[0] == 1 else [f"expected the 1 surviving frame, got {codes.shape[0]}"]


def check_builtins_always_present() -> list[str]:
    """No reels on disk at all must still leave something to play."""
    _reset()
    names = [r.name for r in asciiart.reels()]
    return [] if names else ["reels() was empty with nothing on disk and no built-ins"]


# ── caps ─────────────────────────────────────────────────────────────────────

def check_frame_cap_enforced() -> list[str]:
    d = _reset()
    folder = d / "anim"
    folder.mkdir()
    real_cap = asciiart._MAX_FRAMES
    asciiart._MAX_FRAMES = 3
    try:
        for i in range(5):
            _write(folder / f"{i:03d}.txt", str(i))
        asciiart.reload()
        reel = next(r for r in asciiart.reels() if r.name == "anim")
        n = reel.n_frames
        codes, _ = reel.frames()
    finally:
        asciiart._MAX_FRAMES = real_cap
    bad = []
    if n != 3:
        bad.append(f"discovery reported {n} frames, expected the cap of 3")
    if codes.shape[0] != 3:
        bad.append(f"decoded {codes.shape[0]} frames, expected the cap of 3")
    return bad


def check_dimension_cap_enforced() -> list[str]:
    d = _reset()
    real_w, real_h = asciiart._MAX_W, asciiart._MAX_H
    asciiart._MAX_W, asciiart._MAX_H = 5, 4
    try:
        _write(d / "solo.txt", "\n".join("x" * 20 for _ in range(20)))
        asciiart.reload()
        reel = next(r for r in asciiart.reels() if r.name == "solo")
        codes, density = reel.frames()
    finally:
        asciiart._MAX_W, asciiart._MAX_H = real_w, real_h
    bad = []
    if codes.shape != (1, 4, 5):
        bad.append(f"expected capped shape (1, 4, 5), got {codes.shape}")
    if density.shape != codes.shape:
        bad.append(f"density shape {density.shape} does not match codes {codes.shape}")
    return bad


# ── laziness ─────────────────────────────────────────────────────────────────

def check_discovery_does_not_decode() -> list[str]:
    d = _reset()
    folder = d / "anim"
    folder.mkdir()
    _write(folder / "001.txt", "x")
    asciiart.reload()

    reel = next(r for r in asciiart.reels() if r.name == "anim")
    bad = []
    if reel._cache is not None:
        bad.append("a reel was already decoded right after discovery")
    reel.frames()
    if reel._cache is None:
        bad.append("frames() did not populate the cache")
    return bad


# ── selection ────────────────────────────────────────────────────────────────

def check_reel_and_fx_stepping_wraps() -> list[str]:
    _reset()
    names = [r.name for r in asciiart.reels()]
    if len(names) < 2:
        return [f"need at least 2 reels (builtins) to test wrap, got {names}"]

    first = asciiart.step_reel(0)
    back_one = None
    for _ in range(len(names)):
        back_one = asciiart.step_reel(1)
    bad = []
    if back_one is None or back_one.name != first.name:
        bad.append("stepping reels all the way around did not return to the start")

    fx_seen = {asciiart.step_fx(1) for _ in range(3)}
    if fx_seen != {"warp", "dissolve", "lit"}:
        bad.append(f"fx stepping did not cycle all three: {fx_seen}")
    return bad


def check_restore_touches_no_disk() -> list[str]:
    """restore() must only record names — never trigger discovery itself."""
    _reset()
    asciiart._reels_cache = None
    asciiart.restore("some-reel-name", "dissolve")
    bad = []
    if asciiart._reels_cache is not None:
        bad.append("restore() triggered discovery, it should only store names")
    if asciiart.current_fx() != "dissolve":
        bad.append("restore() did not record the fx")
    return bad


TESTS = [
    ("ragged lines padded to a rectangle", check_ragged_lines_padded),
    ("tabs expand to a stop of 8", check_tabs_expanded),
    ("ANSI escapes are stripped", check_ansi_stripped),
    ("CRLF is normalised", check_crlf_normalised),
    ("frame2 sorts before frame10", check_natural_frame_order),
    ("a bare .txt is a 1-frame reel", check_bare_txt_is_a_single_frame_reel),
    ("non-.txt files are ignored", check_non_txt_files_are_ignored),
    ("a frame deleted before decode is skipped", check_missing_frame_file_is_skipped_not_fatal),
    ("built-in reels always exist", check_builtins_always_present),
    ("frame count is capped", check_frame_cap_enforced),
    ("frame dimensions are capped", check_dimension_cap_enforced),
    ("discovery does not decode frames", check_discovery_does_not_decode),
    ("reel/fx stepping wraps around", check_reel_and_fx_stepping_wraps),
    ("restore() never touches disk", check_restore_touches_no_disk),
]


# ── pytest entry points ───────────────────────────────────────────────────────

def test_ragged_lines_padded() -> None:
    bad = check_ragged_lines_padded()
    assert not bad, "\n".join(bad)


def test_tabs_expanded() -> None:
    bad = check_tabs_expanded()
    assert not bad, "\n".join(bad)


def test_ansi_stripped() -> None:
    bad = check_ansi_stripped()
    assert not bad, "\n".join(bad)


def test_crlf_normalised() -> None:
    bad = check_crlf_normalised()
    assert not bad, "\n".join(bad)


def test_natural_frame_order() -> None:
    bad = check_natural_frame_order()
    assert not bad, "\n".join(bad)


def test_bare_txt_is_a_single_frame_reel() -> None:
    bad = check_bare_txt_is_a_single_frame_reel()
    assert not bad, "\n".join(bad)


def test_non_txt_files_are_ignored() -> None:
    bad = check_non_txt_files_are_ignored()
    assert not bad, "\n".join(bad)


def test_missing_frame_file_is_skipped_not_fatal() -> None:
    bad = check_missing_frame_file_is_skipped_not_fatal()
    assert not bad, "\n".join(bad)


def test_builtins_always_present() -> None:
    bad = check_builtins_always_present()
    assert not bad, "\n".join(bad)


def test_frame_cap_enforced() -> None:
    bad = check_frame_cap_enforced()
    assert not bad, "\n".join(bad)


def test_dimension_cap_enforced() -> None:
    bad = check_dimension_cap_enforced()
    assert not bad, "\n".join(bad)


def test_discovery_does_not_decode() -> None:
    bad = check_discovery_does_not_decode()
    assert not bad, "\n".join(bad)


def test_reel_and_fx_stepping_wraps() -> None:
    bad = check_reel_and_fx_stepping_wraps()
    assert not bad, "\n".join(bad)


def test_restore_touches_no_disk() -> None:
    bad = check_restore_touches_no_disk()
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    failures = 0
    for name, fn in TESTS:
        bad = fn()
        mark = "ok  " if not bad else "FAIL"
        print(f"  [{mark}] {name}")
        for b in bad:
            print(f"         {b}")
        failures += len(bad)
    print("\nall good" if not failures else f"\n{failures} problems")
    raise SystemExit(1 if failures else 0)
