"""The desktop app's version, which is declared in two files and a tag.

``spektr.__version__`` is what ``spektr --version`` prints and what the
Windows version resource is built from; ``pyproject.toml`` is what pip
installs and what the wheel is named after. They are the same number written
twice, and a release cuts from a tag that is a third copy of it.

Nothing about editing one reminds you to edit the others, and the failure is
quiet in the worst way: the build succeeds, the artefacts are named correctly
because the workflow reads the tag, and only the version the app reports about
itself is wrong. That is a bad thing to discover from a bug report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spektr  # noqa: E402

_PYPROJECT = re.compile(r'^version = "([^"]+)"', re.M)
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _pyproject_version() -> str:
    found = _PYPROJECT.findall((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert found, "no top-level `version = \"...\"` in pyproject.toml"
    return found[0]


def test_the_package_and_the_project_agree_on_the_version():
    assert spektr.__version__ == _pyproject_version(), (
        f"spektr.__version__ is {spektr.__version__} and pyproject.toml says "
        f"{_pyproject_version()} — `spektr --version` and `pip show spektr` "
        f"would disagree"
    )


def test_the_version_is_a_plain_three_part_number():
    """``packaging/spektr.spec`` does ``int(p) for p in version.split(".")[:3]``.

    A suffix like ``0.4.0rc1`` gets that far and then fails on the int, which
    is a Windows-only build failure discovered on a tag rather than here.
    """
    assert _SEMVER.match(spektr.__version__), (
        f"{spektr.__version__!r} is not major.minor.patch, which the Windows "
        f"version resource in packaging/spektr.spec cannot build from"
    )


# There was a test here that tried to police the release tag, twice, and both
# versions were wrong in the same way: a tag's relationship to HEAD is not a
# property of the code.
#
# The first asserted the tag did not exist yet. That is true before cutting a
# release and false during one — the build workflows are triggered by the tag
# and check it out — so it failed all three jobs on the first tag it ever saw.
#
# The second asserted the tag did not point somewhere other than HEAD. That
# holds for exactly one commit. The moment anything lands after the release,
# which is immediately and forever, the tag correctly stays behind and the test
# is red on a clean tree until the next version bump.
#
# Nothing replaced it, because git already refuses to move a tag that exists
# without --force, which is the actual protection. What is worth testing is
# below: the two declarations agree, the version is shaped the way the Windows
# build needs, and the changelog has a section for it.


def test_the_changelog_has_a_section_for_this_version():
    """The build workflows only upload files; they write none of the release.

    So the notes have to exist before the tag is pushed, and the changelog is
    where the full story of a version lives.
    """
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"^## spektr {re.escape(spektr.__version__)}\b", re.M)
    assert heading.search(text), (
        f"CHANGELOG.md has no `## spektr {spektr.__version__}` section, so a "
        f"release cut now would ship without its story"
    )


# ── the release body, and who writes it ──────────────────────────────────────

_WORKFLOWS = ["build-windows.yml", "build-linux.yml", "build-android.yml"]


def _android_version() -> str:
    for line in (ROOT / "android" / "gradle.properties").read_text(encoding="utf-8").splitlines():
        if line.startswith("spektrAndroidVersion="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("spektrAndroidVersion is missing from android/gradle.properties")


def test_the_release_notes_name_every_file_the_builds_upload():
    """The notes file is the whole release body, download guide included.

    The build jobs used to append a download section each, and three jobs
    editing one body at once raced: 0.5.0 lost its Linux section. Now they only
    upload, so a notes file that does not say which file to download ships a
    release that does not say it either.
    """
    v = spektr.__version__
    notes = ROOT / "docs" / f"release-notes-{v}.md"
    assert notes.exists(), f"no {notes.relative_to(ROOT)} for the release this version would cut"
    text = notes.read_text(encoding="utf-8")
    for name in ("spektr.exe", f"spektr-{v}.0-setup.exe", "`spektr`",
                 f"spektr-android-{_android_version()}-arm64-v8a.apk"):
        assert name in text, f"the {v} release notes never mention {name}"
    assert "Which file" in text, f"the {v} release notes have no download guide"


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_no_build_job_edits_the_release_body(workflow):
    """Only uploads: a body edit from a build job is the race coming back."""
    text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    assert "append_body" not in text, f"{workflow} appends to the release body"
    assert "action-gh-release" not in text, f"{workflow} uses action-gh-release, which rewrites the body and publishes"
    assert "gh release upload" in text, f"{workflow} no longer attaches its files"
