"""Typed view over the ElevenLabs realtime speech-to-text websocket protocol.

Parsing lives here, apart from the transport, so the protocol can be tested
against recorded frames without opening a socket.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .. import messages

# Server -> client message types, per the realtime STT reference.
SESSION_STARTED = "session_started"
PARTIAL_TRANSCRIPT = "partial_transcript"
COMMITTED_TRANSCRIPT = "committed_transcript"
COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS = "committed_transcript_with_timestamps"
COMMITTED_TRANSCRIPT_ENTITIES = "committed_transcript_entities"
WARNING = "warning"

#: Error frames.  Each maps to a category the UI reacts to differently.
AUTH_ERRORS = frozenset({"auth_error", "unaccepted_terms"})
BILLING_ERRORS = frozenset({"quota_exceeded", "resource_exhausted"})
RATE_LIMIT_ERRORS = frozenset({"rate_limited", "commit_throttled", "queue_overflow"})
FATAL_ERRORS = frozenset({"session_time_limit_exceeded", "transcriber_error", "error"})
REQUEST_ERRORS = frozenset(
    {"input_error", "invalid_request", "chunk_size_exceeded", "insufficient_audio_activity"}
)
ERROR_TYPES = AUTH_ERRORS | BILLING_ERRORS | RATE_LIMIT_ERRORS | FATAL_ERRORS | REQUEST_ERRORS


@dataclass(frozen=True)
class SttEvent:
    """One decoded frame from the transcription websocket."""

    type: str
    text: str = ""
    session_id: str = ""
    language_code: str = ""
    message: str = ""
    words: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    # -- classification ---------------------------------------------------
    @property
    def is_partial(self) -> bool:
        return self.type == PARTIAL_TRANSCRIPT

    @property
    def is_committed(self) -> bool:
        return self.type in (
            COMMITTED_TRANSCRIPT,
            COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS,
        )

    @property
    def is_error(self) -> bool:
        return self.type in ERROR_TYPES

    @property
    def is_auth_error(self) -> bool:
        return self.type in AUTH_ERRORS

    @property
    def is_billing_error(self) -> bool:
        return self.type in BILLING_ERRORS

    @property
    def is_rate_limited(self) -> bool:
        return self.type in RATE_LIMIT_ERRORS

    @property
    def is_recoverable(self) -> bool:
        """Whether reconnecting could plausibly help."""
        if self.is_auth_error or self.is_billing_error:
            return False
        return True


def parse_event(payload: str | bytes | dict[str, Any]) -> SttEvent:
    """Decode one websocket frame.  Unknown types are preserved, never dropped."""
    if isinstance(payload, (str, bytes)):
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            return SttEvent(type="error", message="Malformed frame from ElevenLabs", raw={})
    else:
        data = payload
    if not isinstance(data, dict):
        return SttEvent(type="error", message="Unexpected frame from ElevenLabs", raw={})

    msg_type = str(data.get("message_type") or data.get("type") or "unknown")
    text = data.get("text") or ""

    message = ""
    for key in ("error", "warning", "message", "detail"):
        value = data.get(key)
        if isinstance(value, str) and value:
            message = value
            break
        if isinstance(value, dict):
            nested = value.get("message") or value.get("detail")
            if isinstance(nested, str) and nested:
                message = nested
                break
    if not message and msg_type in ERROR_TYPES:
        message = msg_type.replace("_", " ")

    words = data.get("words") if isinstance(data.get("words"), list) else []
    entities = data.get("entities") if isinstance(data.get("entities"), list) else []

    return SttEvent(
        type=msg_type,
        text=str(text),
        session_id=str(data.get("session_id") or ""),
        language_code=str(data.get("language_code") or ""),
        message=message,
        words=words,
        entities=entities,
        raw=data,
    )


def describe_error(event: SttEvent) -> str:
    """Azerbaijani-facing explanation of an error frame."""
    if event.is_auth_error:
        return messages.STT_AUTH_ERROR
    if event.is_billing_error:
        return messages.STT_BILLING_ERROR
    if event.is_rate_limited:
        return messages.STT_RATE_LIMITED
    if event.message:
        return f"{messages.STT_GENERIC_ERROR} {event.message}"
    return messages.STT_GENERIC_ERROR
