"""Minimal `.env` loader.

`.env.example` tells people to copy the file to `.env`, so the app has to
actually read it.  This is a deliberately small parser rather than a dependency:
it handles `KEY=value`, comments, blank lines, `export ` prefixes and quoted
values, and nothing more.

Values already present in the real environment always win, so a shell variable
is never silently overridden by a stale file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .logging_setup import register_secret

log = logging.getLogger(__name__)

SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


def parse_env(text: str) -> dict[str, str]:
    """Parse `.env` content into a mapping.  Malformed lines are skipped."""
    values: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Load a `.env` file into `os.environ`.

    Returns the names of the variables that were set.  Values are never logged,
    and anything that looks like a credential is registered with the redactor.
    """
    target = Path(path) if path is not None else _default_path()
    if target is None or not target.is_file():
        return []

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Could not read %s: %s", target.name, exc)
        return []

    applied: list[str] = []
    for key, value in parse_env(text).items():
        if not value:
            continue
        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
        if key.upper().endswith(SECRET_SUFFIXES):
            register_secret(value)

    if applied:
        log.info("Loaded %d variable(s) from %s: %s", len(applied), target.name, ", ".join(applied))
    return applied


def _default_path() -> Path | None:
    """`.env` beside the project root, or next to a frozen executable."""
    candidates = [Path.cwd() / ".env"]
    here = Path(__file__).resolve()
    candidates.append(here.parent.parent.parent / ".env")   # src/copilot -> repo root
    import sys

    if getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable).parent / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
