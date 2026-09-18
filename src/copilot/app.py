"""Application entry point."""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="az-interview-copilot",
        description=(
            "Azerbaijani interview copilot: ElevenLabs Scribe v2 Realtime for "
            "speech recognition, OpenAI for answer generation."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    parser.add_argument("--no-log-file", action="store_true", help="console logging only")
    parser.add_argument(
        "--check", action="store_true",
        help="report environment readiness and exit without opening a window",
    )
    args = parser.parse_args(argv)

    from .logging_setup import configure_logging

    configure_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        to_file=not args.no_log_file,
    )

    # A .env file is a documented fallback for the two API keys; real
    # environment variables always take priority over it.
    from .env_file import load_env_file

    load_env_file()

    if args.check:
        return _environment_check()

    # Qt picks up the display scale factor from Windows by default; make text
    # crisp on high-DPI laptops without forcing integer scaling.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    from .ui.main_window import run

    return run()


def _environment_check() -> int:
    """Print what is configured and what is missing, then exit."""
    from .audio.capture import AudioCaptureError, list_loopback_devices
    from .config import Settings
    from .credentials import PROVIDERS, CredentialStore, mask

    settings = Settings.load()
    store = CredentialStore()

    print("AZ Interview Copilot - environment check")
    print(f"  Python            : {sys.version.split()[0]} ({sys.platform})")
    print(f"  Credential backend: {store.backend_name}")

    ok = True
    for provider, spec in PROVIDERS.items():
        key = store.get(provider)  # type: ignore[arg-type]
        source = store.source_of(provider) or "-"  # type: ignore[arg-type]
        status = f"{mask(key)} via {source}" if key else "NOT CONFIGURED"
        print(f"  {spec.label:<18}: {status}")
        ok = ok and bool(key)

    try:
        devices = list_loopback_devices()
        if devices:
            for device in devices:
                print(f"  Loopback device   : {device.label}")
        else:
            print("  Loopback device   : none found")
            ok = False
    except AudioCaptureError as exc:
        print(f"  Loopback device   : unavailable ({exc})")
        ok = False

    print(f"  STT model         : {settings.stt.model_id} ({settings.stt.language_code})")
    print(f"  Answer model      : {settings.llm.model}")
    print("Ready." if ok else "Not ready - see the entries above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
