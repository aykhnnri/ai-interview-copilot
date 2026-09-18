"""Launcher: `python run.py` from a source checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from copilot.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
