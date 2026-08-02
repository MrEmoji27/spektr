"""Plugin discovery, trust, and loading.

A plugin is a Python file (or a directory with ``__init__.py``) in
``~/.config/spektr/plugins/`` that calls :func:`spektr.api.mode`. That is the
whole contract — the same decorator the built-in modes use.

**On sandboxing, plainly.** These are Python files and they run with your
privileges. There is no VM to isolate them the way cliamp isolates Lua, and
pretending otherwise in Python is security theatre: ``import`` is ``import``.
So instead of a fake sandbox, spektr makes the trust decision explicit and
visible. A plugin does not run until you have approved its exact contents, the
approval records a SHA-256, and any edit invalidates it. You are trusting the
author, and spektr makes sure you know that you are.

What *is* contained is failure. A plugin that raises gets quarantined rather
than taking the app down, and one that renders too slowly has its previous
frame reused rather than stuttering the whole UI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import modes as registry
from . import palette

#: How many consecutive exceptions before a mode is disabled for the session.
FAILURE_LIMIT = 3

TRUST_FILE = ".trust.json"


def plugins_dir() -> Path:
    # resolved through the module rather than a bound name, so the config root
    # can be redirected (tests, and eventually a --config-dir flag)
    return palette.config_dir() / "plugins"


@dataclass
class Plugin:
    name: str
    path: Path
    digest: str
    lines: int = 0
    trusted: bool = False
    loaded: bool = False
    error: str | None = None
    modes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if not self.trusted:
            return "untrusted"
        return "ok" if self.loaded else "pending"


# ── hashing ──────────────────────────────────────────────────────────────────

def _digest(path: Path) -> tuple[str, int]:
    """SHA-256 over the plugin's source, plus a line count for the prompt.

    A directory hashes every ``.py`` it contains, sorted, with the relative
    path mixed in — so adding, renaming or removing a file changes the digest
    just as editing one does.
    """
    h = hashlib.sha256()
    lines = 0
    if path.is_dir():
        for f in sorted(path.rglob("*.py")):
            rel = f.relative_to(path).as_posix().encode()
            data = f.read_bytes()
            h.update(len(rel).to_bytes(4, "big"))
            h.update(rel)
            h.update(len(data).to_bytes(8, "big"))
            h.update(data)
            lines += data.count(b"\n") + 1
    else:
        data = path.read_bytes()
        h.update(data)
        lines = data.count(b"\n") + 1
    return h.hexdigest(), lines


# ── trust store ──────────────────────────────────────────────────────────────

def _trust_path() -> Path:
    return plugins_dir() / TRUST_FILE


def read_trust() -> dict[str, str]:
    try:
        data = json.loads(_trust_path().read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}


def write_trust(store: dict[str, str]) -> None:
    path = _trust_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass  # windows, or a filesystem without unix modes


def trust(name: str) -> tuple[bool, str]:
    """Approve a plugin's current contents. Returns ``(ok, message)``."""
    found = {p.name: p for p in discover()}
    p = found.get(name)
    if p is None:
        return False, f"no plugin named {name!r} in {plugins_dir()}"
    store = read_trust()
    store[name] = p.digest
    write_trust(store)
    return True, f"trusted {name} ({p.digest[:12]}…)"


def untrust(name: str) -> tuple[bool, str]:
    store = read_trust()
    if store.pop(name, None) is None:
        return False, f"{name} was not trusted"
    write_trust(store)
    return True, f"revoked trust for {name}"


def remove(name: str) -> tuple[bool, str]:
    """Delete a plugin from disk and forget its approval."""
    found = {p.name: p for p in discover()}
    p = found.get(name)
    if p is None:
        return False, f"no plugin named {name!r}"
    try:
        if p.path.is_dir():
            for f in sorted(p.path.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            p.path.rmdir()
        else:
            p.path.unlink()
    except OSError as exc:
        return False, f"could not remove {name}: {exc}"
    untrust(name)
    return True, f"removed {name}"


# ── discovery ────────────────────────────────────────────────────────────────

def discover() -> list[Plugin]:
    """Everything that looks like a plugin, whether or not it is trusted."""
    try:
        folder = plugins_dir()
        if not folder.is_dir():
            return []
        entries = sorted(folder.iterdir())
    except OSError:
        return []

    store = read_trust()
    out: list[Plugin] = []
    seen: set[str] = set()

    for entry in entries:
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_dir():
            if not (entry / "__init__.py").is_file():
                continue
            name = entry.name
        elif entry.suffix == ".py":
            name = entry.stem
        else:
            continue
        if name in seen:
            continue
        seen.add(name)

        try:
            digest, lines = _digest(entry)
        except OSError as exc:
            out.append(Plugin(name=name, path=entry, digest="", error=f"unreadable: {exc}"))
            continue

        out.append(
            Plugin(
                name=name,
                path=entry,
                digest=digest,
                lines=lines,
                trusted=store.get(name) == digest,
            )
        )
    return out


# ── loading ──────────────────────────────────────────────────────────────────

def _load_one(p: Plugin) -> None:
    """Import a trusted plugin and record which modes it registered."""
    target = p.path / "__init__.py" if p.path.is_dir() else p.path
    module_name = f"spektr_plugin_{p.name}"

    spec = importlib.util.spec_from_file_location(module_name, target)
    if spec is None or spec.loader is None:
        p.error = "could not build an import spec"
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    registry._LOADING = p.name
    try:
        spec.loader.exec_module(module)
    except registry.ModeNameTaken as exc:
        p.error = str(exc)
    except Exception:
        p.error = traceback.format_exc(limit=6).strip()
    finally:
        registry._LOADING = None
        sys.modules.pop(module_name, None)

    declared = getattr(module, "SPEKTR_API", None)
    if declared is not None:
        from .api import API_VERSION

        if int(declared) != API_VERSION:
            p.error = (
                f"needs spektr API v{declared}, this build provides v{API_VERSION}"
            )

    p.modes = [m.name for m in registry.MODES if m.plugin == p.name]

    if p.error:
        # never leave a half-registered plugin in the mode list
        registry.unregister_plugin(p.name)
        p.modes = []
        p.loaded = False
        return

    if not p.modes:
        p.error = "loaded but registered no modes — did you call @mode(...)?"
        return

    p.loaded = True


def load_all() -> list[Plugin]:
    """Load every trusted plugin. Safe to call once at startup."""
    found = discover()
    for p in found:
        if p.error or not p.trusted:
            continue
        _load_one(p)
    return found


def reload_all() -> list[Plugin]:
    """Drop every plugin mode and load again — for a `plugins reload` action."""
    for p in discover():
        registry.unregister_plugin(p.name)
    return load_all()


# ── quarantine ───────────────────────────────────────────────────────────────

class Quarantine:
    """Tracks misbehaving modes so one bad plugin can't ruin the session.

    A mode that raises is retried a couple of times — a transient failure at an
    awkward terminal size shouldn't be a death sentence — and then disabled,
    with the traceback kept so ``spektr plugins doctor`` can show it.
    """

    def __init__(self, limit: int = FAILURE_LIMIT):
        self._limit = limit
        self._fails: dict[str, int] = {}
        self.errors: dict[str, str] = {}
        self.disabled: set[str] = set()

    def record_failure(self, name: str, detail: str) -> bool:
        """Returns True if this failure disabled the mode."""
        self.errors[name] = detail
        self._fails[name] = self._fails.get(name, 0) + 1
        if self._fails[name] >= self._limit and name not in self.disabled:
            self.disabled.add(name)
            return True
        return False

    def record_success(self, name: str) -> None:
        if self._fails.pop(name, None) is not None:
            self.errors.pop(name, None)

    def is_disabled(self, name: str) -> bool:
        return name in self.disabled

    def clear(self, name: str | None = None) -> None:
        if name is None:
            self._fails.clear()
            self.errors.clear()
            self.disabled.clear()
        else:
            self._fails.pop(name, None)
            self.errors.pop(name, None)
            self.disabled.discard(name)


# ── output validation ────────────────────────────────────────────────────────

class BadModeOutput(Exception):
    pass


def validate(out, w: int, h: int):
    """Check and coerce what a plugin returned.

    Built-in modes skip this — they're covered by the test suite. Plugins get
    it because the alternative is a shape mismatch surfacing as a crash deep
    inside the strip builder, where the traceback says nothing useful about
    which plugin caused it.
    """
    import numpy as np

    if not isinstance(out, tuple) or len(out) not in (2, 3):
        raise BadModeOutput(
            f"render must return (codes, cidx) or (codes, cidx, bgidx), got {type(out).__name__}"
        )

    from .palette import RAMP_STEPS

    fixed = []
    for i, arr in enumerate(out):
        a = np.asarray(arr)
        if a.shape != (h, w):
            what = ("codes", "cidx", "bgidx")[i]
            raise BadModeOutput(f"{what} has shape {a.shape}, expected {(h, w)}")
        a = a.astype(np.int32, copy=False)
        if i == 0:
            # Codepoints in the UTF-16 surrogate range are not decodable, and
            # neither is anything past U+10FFFF. Substitute rather than clamp:
            # clamping a surrogate to 0xD7FF would silently draw an unrelated
            # Hangul glyph, where U+FFFD visibly says "this cell is wrong".
            a = np.where(
                ((a >= 0xD800) & (a <= 0xDFFF)) | (a < 0) | (a > 0x10FFFF),
                0xFFFD,
                a,
            ).astype(np.int32)
        else:
            a = np.clip(a, 0, RAMP_STEPS - 1)
        fixed.append(a)

    return tuple(fixed)
