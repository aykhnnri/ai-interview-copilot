"""Filesystem locations used by the application."""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "AzInterviewCopilot"


def app_data_dir() -> Path:
    """Per-user writable directory for settings, logs and extracted documents."""
    override = os.environ.get("AZCOPILOT_DATA_DIR")
    if override:
        base = Path(override)
    else:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
        base = base / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def documents_dir() -> Path:
    d = app_data_dir() / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = app_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def benchmarks_dir() -> Path:
    d = app_data_dir() / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    return d
