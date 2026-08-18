"""The port's version, which is not the desktop app's.

spektr 0.4.0 is the release; the APK inside it says 0.2.0, because that is how
many versions of the Android build there have been. Two numbers that mean
different things and look the same is exactly the kind of thing that drifts,
so the one on the APK and the one in the release notes are checked against
each other here rather than kept in step by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROPS = ROOT / "android" / "gradle.properties"
CHANGELOG = ROOT / "CHANGELOG.md"

#: `## Android v0.2.0 — ships in spektr 0.4.0`
_HEADING = re.compile(r"^## Android v(\d+)\.(\d+)\.(\d+)\b(.*)$", re.M)


def _declared() -> str:
    for line in PROPS.read_text(encoding="utf-8").splitlines():
        if line.startswith("spektrAndroidVersion="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"spektrAndroidVersion is missing from {PROPS}")


def _version_code(major: int, minor: int, patch: int) -> int:
    """The same arithmetic build.gradle.kts does."""
    return major * 10000 + minor * 100 + patch


def test_the_apk_version_is_the_one_the_changelog_announces():
    releases = _HEADING.findall(CHANGELOG.read_text(encoding="utf-8"))
    assert releases, "no `## Android vX.Y.Z` heading in CHANGELOG.md"
    newest = ".".join(releases[0][:3])
    assert _declared() == newest, (
        f"the APK would report {_declared()} and the changelog announces {newest}"
    )


def test_the_newest_android_release_says_which_spektr_release_it_ships_in():
    """The whole reason the two numbers differ is worth stating on the page."""
    releases = _HEADING.findall(CHANGELOG.read_text(encoding="utf-8"))
    tail = releases[0][3]
    assert re.search(r"spektr \d+\.\d+\.\d+", tail), (
        f"the newest Android heading does not name a spektr release: {tail!r}"
    )


def test_the_version_code_goes_up():
    """Android refuses an update whose code is not higher than the installed one.

    Derived rather than hand-bumped for that reason, but derivation only helps
    if the versions themselves increase — 0.10.0 after 0.9.0 is fine, 0.2.0
    after 0.10.0 would install nowhere and there is no way to take it back.
    """
    releases = _HEADING.findall(CHANGELOG.read_text(encoding="utf-8"))
    codes = [_version_code(int(a), int(b), int(c)) for a, b, c, _ in releases]
    assert codes == sorted(codes, reverse=True), (
        "the Android changelog is not in descending version order, so a release "
        f"would ship a versionCode that does not increase: {codes}"
    )
    assert len(set(codes)) == len(codes), f"two Android releases share a version: {codes}"


def test_the_declared_version_parses_the_way_gradle_parses_it():
    """build.gradle.kts splits on `-` then on `.`; anything else fails the build."""
    core = _declared().split("-")[0].split(".")
    assert len(core) == 3, f"{_declared()!r} is not major.minor.patch"
    assert all(p.isdigit() for p in core), f"{_declared()!r} has a non-numeric part"
    assert _version_code(*(int(p) for p in core)) > 0


# ── the port's README ────────────────────────────────────────────────────────

def test_the_android_readme_counts_the_modes_the_picker_offers():
    """It went stale within minutes of being written, which is the argument.

    `android/README.md` is the public face of the port and it states how many
    modes the picker offers. That number changes every time a mode is added,
    and nothing about adding a mode reminds you to open that file — the root
    README has a test for exactly this reason, and this is the same guarantee
    for the other one.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    import spektr.modes as M

    text = (ROOT / "android" / "README.md").read_text(encoding="utf-8")
    offered, total = len(M.listed()), len(M.MODES)
    assert f"offers {offered} of the engine's {total} modes" in text, (
        f"the port's README does not say it offers {offered} of {total} modes"
    )
