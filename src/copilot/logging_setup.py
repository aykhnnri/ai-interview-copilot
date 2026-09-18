"""Logging with hard guarantees that API keys never reach a log sink.

Two layers of defence:

1. `SecretRedactor` scrubs anything that *looks* like a credential (provider key
   prefixes, long opaque tokens, `xi-api-key: ...` headers).
2. `register_secret()` registers the exact live key strings so they are masked
   even if a provider ever changes its key format.
"""
from __future__ import annotations

import logging
import logging.handlers
import re
import sys
import threading
from typing import Any

from .paths import logs_dir

_MASK = "***REDACTED***"

# Known credential shapes.  Order matters: longest / most specific first.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{8,}", re.I),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}", re.I),
    re.compile(r"\bsk_[A-Za-z0-9]{24,}", re.I),          # ElevenLabs
    re.compile(r"\bxi-api-key\b\s*[:=]\s*\S+", re.I),
    re.compile(r"\bauthorization\b\s*[:=]\s*\S+", re.I),
    re.compile(r"\b(?:api[_-]?key|token)\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{12,}", re.I),
)

_exact_secrets: set[str] = set()
_lock = threading.Lock()


def register_secret(value: str | None) -> None:
    """Remember a literal secret so it is masked wherever it appears."""
    if value and len(value) >= 8:
        with _lock:
            _exact_secrets.add(value)


def forget_secret(value: str | None) -> None:
    if value:
        with _lock:
            _exact_secrets.discard(value)


def redact(text: str) -> str:
    """Return `text` with every known or credential-shaped substring masked."""
    if not text:
        return text
    with _lock:
        secrets = tuple(_exact_secrets)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, _MASK)
    for pattern in _PATTERNS:
        text = pattern.sub(_MASK, text)
    return text


class SecretRedactor(logging.Filter):
    """Applies `redact()` to the formatted message and every argument."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        try:
            record.msg = redact(str(record.msg))
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: _scrub(v) for k, v in record.args.items()}
                else:
                    record.args = tuple(_scrub(a) for a in record.args)
        except Exception:  # logging must never raise
            record.msg = "<log record suppressed by redactor>"
            record.args = ()
        return True


def _scrub(value: Any) -> Any:
    return redact(value) if isinstance(value, str) else value


_configured = False


def configure_logging(level: int = logging.INFO, to_file: bool = True) -> None:
    """Install console (+ rotating file) handlers, both redacted."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)
    redactor = SecretRedactor()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-28s %(message)s")

    # The Windows console defaults to cp1252, which turns every Azerbaijani
    # character in a log line into a \uXXXX escape.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - redirected stream
                pass

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(redactor)
    root.addHandler(console)

    if to_file:
        try:
            handler = logging.handlers.RotatingFileHandler(
                logs_dir() / "copilot.log", maxBytes=2_000_000, backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(fmt)
            handler.addFilter(redactor)
            root.addHandler(handler)
        except OSError:
            pass  # a read-only profile must not stop the app from starting

    # Third-party libraries are chatty at DEBUG and can echo request headers.
    for noisy in ("websockets", "openai", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
