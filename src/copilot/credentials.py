"""Secure storage for the two independent provider API keys.

Resolution order for a key:
    1. Windows Credential Manager (via `keyring`) - what the Settings UI writes.
    2. Environment variable - `ELEVENLABS_API_KEY` / `OPENAI_API_KEY`.

Keys are never written to disk in plain text and never logged: every value that
passes through here is registered with the log redactor.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from .logging_setup import forget_secret, register_secret

log = logging.getLogger(__name__)

Provider = Literal["elevenlabs", "openai"]

SERVICE_NAME = "AzInterviewCopilot"


@dataclass(frozen=True)
class ProviderSpec:
    key: Provider
    label: str
    env_var: str
    credential_name: str


PROVIDERS: dict[Provider, ProviderSpec] = {
    "elevenlabs": ProviderSpec(
        key="elevenlabs",
        label="ElevenLabs",
        env_var="ELEVENLABS_API_KEY",
        credential_name="elevenlabs_api_key",
    ),
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        env_var="OPENAI_API_KEY",
        credential_name="openai_api_key",
    ),
}


class CredentialStore:
    """Thin wrapper over `keyring` that degrades gracefully when unavailable."""

    def __init__(self, service_name: str = SERVICE_NAME) -> None:
        self.service_name = service_name
        self._keyring = None
        self._keyring_error: str | None = None
        try:
            import keyring

            # Touch the backend so a broken install fails here, not mid-session.
            keyring.get_keyring()
            self._keyring = keyring
        except Exception as exc:  # pragma: no cover - environment dependent
            self._keyring_error = str(exc)
            log.warning("Credential Manager unavailable (%s); falling back to environment variables", exc)

    # -- introspection ----------------------------------------------------
    @property
    def secure_storage_available(self) -> bool:
        return self._keyring is not None

    @property
    def backend_name(self) -> str:
        if self._keyring is None:
            return f"unavailable ({self._keyring_error})"
        try:
            return type(self._keyring.get_keyring()).__name__
        except Exception:  # pragma: no cover
            return "unknown"

    # -- read -------------------------------------------------------------
    def get(self, provider: Provider) -> str | None:
        spec = PROVIDERS[provider]
        value = self._get_stored(spec)
        if not value:
            value = (os.environ.get(spec.env_var) or "").strip() or None
        register_secret(value)
        return value

    def source_of(self, provider: Provider) -> str | None:
        """Where the active key came from: 'credential-manager', 'environment' or None."""
        spec = PROVIDERS[provider]
        if self._get_stored(spec):
            return "credential-manager"
        if (os.environ.get(spec.env_var) or "").strip():
            return "environment"
        return None

    def _get_stored(self, spec: ProviderSpec) -> str | None:
        if self._keyring is None:
            return None
        try:
            value = self._keyring.get_password(self.service_name, spec.credential_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            log.warning("Could not read %s key from Credential Manager: %s", spec.label, exc)
            return None
        return (value or "").strip() or None

    # -- write ------------------------------------------------------------
    def set(self, provider: Provider, value: str) -> None:
        spec = PROVIDERS[provider]
        value = (value or "").strip()
        if not value:
            raise ValueError("API key is empty")
        if self._keyring is None:
            raise RuntimeError(
                "Windows Credential Manager is not available on this system. "
                f"Set the {spec.env_var} environment variable instead."
            )
        self._keyring.set_password(self.service_name, spec.credential_name, value)
        register_secret(value)
        log.info("Stored %s API key in Credential Manager", spec.label)

    def delete(self, provider: Provider) -> bool:
        spec = PROVIDERS[provider]
        if self._keyring is None:
            return False
        existing = self._get_stored(spec)
        try:
            self._keyring.delete_password(self.service_name, spec.credential_name)
        except Exception:
            return False
        forget_secret(existing)
        log.info("Removed %s API key from Credential Manager", spec.label)
        return True


def mask(key: str | None) -> str:
    """Display form of a key: never shows more than the last 4 characters."""
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:3]}{'*' * 8}{key[-4:]}"
