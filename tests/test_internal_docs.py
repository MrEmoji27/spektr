"""What is ours stays ours.

The repository is public, and two kinds of document live in it. One explains
how the thing works — architecture, the plugin contract, the port's design —
and that is worth publishing. The other is what we intend to build next:
scope, build order, risks, open questions, an exploration of a renderer that
does not exist yet. That is ours, and it went public once already, which is
why there is a test rather than a habit.

`.git/info/exclude` keeps `docs/internal/` untracked, but an exclude file is
local to one clone and silent when it fails. This is the part that travels.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


def test_nothing_under_docs_internal_is_tracked():
    leaked = [p for p in _tracked() if p.startswith("docs/internal/")]
    assert not leaked, (
        "internal planning documents are staged for the public repo: "
        + ", ".join(leaked)
    )


def test_the_working_notes_are_not_tracked():
    """The specific files that have been kept out by hand, named.

    A directory rule only catches what someone remembered to put in the
    directory. These are the ones that already exist at the top of `docs/`,
    and one of them — the comment audit — reached the public history before
    anyone noticed.
    """
    ours = {
        "docs/comment-audit.md",
        "docs/next-session.md",
        "docs/mode-ideas.md",
        "docs/development.md",
        "docs/how-it-works.md",
        "docs/audio-capture.md",
    }
    tracked = set(_tracked())
    leaked = sorted(ours & tracked)
    assert not leaked, "working notes are tracked: " + ", ".join(leaked)
    assert not [p for p in tracked if p.startswith("docs/debug/")]
    assert not [p for p in tracked if p.startswith("docs/spektr")]


def test_the_public_docs_are_still_public():
    """The split has to cut both ways, or it is just deletion.

    These explain how the thing works and are linked from the README; losing
    them to an over-broad rule would be the opposite failure.
    """
    tracked = set(_tracked())
    for keep in (
        "docs/architecture.md",
        "docs/plugins.md",
        "docs/android-port.md",
        "docs/gallery.md",
        "CHANGELOG.md",
    ):
        assert keep in tracked, f"{keep} should be published and is not tracked"


def test_no_public_doc_links_to_an_internal_one():
    """A dead link is how a reader learns the internal file exists."""
    bad = []
    for path in ROOT.glob("**/*.md"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("docs/internal/") or ".git" in rel:
            continue
        if rel not in _tracked():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "docs/internal/" in text or "android-3d.md" in text:
            bad.append(rel)
    assert not bad, "public documents pointing at internal ones: " + ", ".join(bad)
