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
import subprocess
import sys
from pathlib import Path

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


def test_the_tag_for_this_version_is_not_pointing_somewhere_else():
    """A tag is immutable in practice once pushed, and the workflows key on it.

    The thing to catch is a version that has already shipped: bump it, or the
    next tag push builds over a release that is already out.

    "Already taken" is the wrong test for that, and it failed the first
    release it ran on. The three build workflows are triggered *by* the tag
    and check out the tag, so the tag they are building always exists — a test
    that objects to it existing objects to every release build there will ever
    be. What matters is whether it points at this commit or at a different
    one.
    """
    wanted = f"v{spektr.__version__}"
    at = subprocess.run(
        ["git", "-C", str(ROOT), "rev-list", "-n", "1", wanted],
        capture_output=True, text=True,
    )
    if at.returncode:
        return                                    # no such tag yet: nothing to say
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert at.stdout.strip() == head, (
        f"{wanted} already exists and points at {at.stdout.strip()[:12]}, not at "
        f"HEAD {head[:12]} — bump the version, or a tag push here would build "
        f"over a release that is already out"
    )


def test_the_changelog_has_a_section_for_this_version():
    """The three build workflows append to the release body; they do not write it.

    So the notes have to exist before the tag is pushed, and this is where
    they live.
    """
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"^## spektr {re.escape(spektr.__version__)}\b", re.M)
    assert heading.search(text), (
        f"CHANGELOG.md has no `## spektr {spektr.__version__}` section, so a "
        f"release cut now would ship with only the build jobs' download notes"
    )
