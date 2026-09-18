"""Credential storage, provider separation, and the no-secrets-in-logs guarantee."""
from __future__ import annotations

import logging

import pytest

from copilot.credentials import PROVIDERS, CredentialStore, mask
from copilot.logging_setup import SecretRedactor, forget_secret, redact, register_secret

ELEVENLABS_KEY = "sk_1234567890abcdef1234567890abcdef"
OPENAI_KEY = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX"


class FakeKeyring:
    """In-memory stand-in for Windows Credential Manager."""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.fail = fail

    def get_keyring(self):
        if self.fail:
            raise RuntimeError("no backend")
        return self

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.store:
            raise KeyError(name)
        del self.store[(service, name)]


@pytest.fixture
def store(monkeypatch):
    fake = FakeKeyring()
    credentials = CredentialStore("TestService")
    credentials._keyring = fake
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return credentials


# =========================================================================
# Two independent providers
# =========================================================================
def test_both_providers_are_registered():
    assert set(PROVIDERS) == {"elevenlabs", "openai"}
    assert PROVIDERS["elevenlabs"].env_var == "ELEVENLABS_API_KEY"
    assert PROVIDERS["openai"].env_var == "OPENAI_API_KEY"


def test_providers_use_separate_credential_entries():
    assert (
        PROVIDERS["elevenlabs"].credential_name != PROVIDERS["openai"].credential_name
    )


def test_keys_are_stored_and_read_back_independently(store):
    store.set("elevenlabs", ELEVENLABS_KEY)
    store.set("openai", OPENAI_KEY)

    assert store.get("elevenlabs") == ELEVENLABS_KEY
    assert store.get("openai") == OPENAI_KEY
    # The decisive property: one provider never sees the other's key.
    assert store.get("elevenlabs") != store.get("openai")


def test_removing_one_key_leaves_the_other(store):
    store.set("elevenlabs", ELEVENLABS_KEY)
    store.set("openai", OPENAI_KEY)

    assert store.delete("elevenlabs")
    assert store.get("elevenlabs") is None
    assert store.get("openai") == OPENAI_KEY


def test_environment_variable_is_used_when_nothing_is_stored(store, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    assert store.get("openai") == OPENAI_KEY
    assert store.source_of("openai") == "environment"


def test_stored_key_takes_priority_over_the_environment(store, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-value-0000")
    store.set("openai", OPENAI_KEY)
    assert store.get("openai") == OPENAI_KEY
    assert store.source_of("openai") == "credential-manager"


def test_missing_key_reports_no_source(store):
    assert store.get("openai") is None
    assert store.source_of("openai") is None


def test_empty_key_is_rejected(store):
    with pytest.raises(ValueError):
        store.set("openai", "   ")


def test_whitespace_is_trimmed(store):
    store.set("openai", f"  {OPENAI_KEY}  ")
    assert store.get("openai") == OPENAI_KEY


def test_store_degrades_without_a_backend(monkeypatch):
    credentials = CredentialStore("TestService")
    credentials._keyring = None
    credentials._keyring_error = "no backend"

    assert not credentials.secure_storage_available
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    assert credentials.get("openai") == OPENAI_KEY     # env var still works
    with pytest.raises(RuntimeError, match="Credential Manager"):
        credentials.set("openai", OPENAI_KEY)


def test_delete_without_a_backend_returns_false():
    credentials = CredentialStore("TestService")
    credentials._keyring = None
    assert credentials.delete("openai") is False


# =========================================================================
# Masking and redaction
# =========================================================================
def test_mask_never_reveals_more_than_the_tail():
    masked = mask(OPENAI_KEY)
    assert OPENAI_KEY not in masked
    assert masked.endswith(OPENAI_KEY[-4:])
    assert mask(None) == "(not set)"
    assert mask("short") == "*****"


@pytest.mark.parametrize(
    "text",
    [
        "key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX here",
        "elevenlabs sk_1234567890abcdef1234567890abcdef",
        "header xi-api-key: sk_abcdefabcdefabcdefabcdef",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        'api_key="ABCDEFGHIJKLMNOP1234"',
    ],
)
def test_credential_shaped_strings_are_redacted(text):
    assert "***REDACTED***" in redact(text)


def test_registered_secrets_are_redacted_whatever_their_shape():
    odd_key = "totally-unusual-format-9876"
    assert odd_key in redact(odd_key)     # not yet registered
    register_secret(odd_key)
    try:
        assert odd_key not in redact(f"using {odd_key} now")
    finally:
        forget_secret(odd_key)


def test_storing_a_key_registers_it_for_redaction(store):
    store.set("elevenlabs", ELEVENLABS_KEY)
    assert ELEVENLABS_KEY not in redact(f"connecting with {ELEVENLABS_KEY}")


def test_log_records_are_scrubbed_before_formatting():
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="connecting with key %s", args=(OPENAI_KEY,), exc_info=None,
    )
    SecretRedactor().filter(record)
    assert OPENAI_KEY not in record.getMessage()


def test_redactor_never_raises_on_odd_records():
    class Exploding:
        def __str__(self):
            raise ValueError("nope")

    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg=Exploding(), args=(), exc_info=None,
    )
    assert SecretRedactor().filter(record) is True


def test_redact_handles_empty_input():
    assert redact("") == ""
