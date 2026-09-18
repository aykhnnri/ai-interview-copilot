"""ElevenLabs realtime speech-to-text: protocol, connection handling, errors."""
from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse

import pytest

from copilot import messages as msg
from copilot.config import SttSettings
from copilot.stt.elevenlabs_client import (
    MAX_CHUNK_BYTES,
    ElevenLabsRealtimeClient,
    SttAuthError,
    SttBillingError,
    SttCallbacks,
    build_realtime_url,
    encode_audio_chunk,
)
from copilot.stt.events import SttEvent, describe_error, parse_event

from .conftest import frame

AZ_QUESTION = "Production mühitində RAG pipeline-ın retrieval quality-sini necə evaluate edərdiniz?"


# =========================================================================
# URL construction
# =========================================================================
def test_url_uses_the_documented_endpoint_and_model():
    url = build_realtime_url(SttSettings(), 16000)
    parsed = urlparse(url)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.elevenlabs.io"
    assert parsed.path == "/v1/speech-to-text/realtime"

    params = parse_qs(parsed.query)
    assert params["model_id"] == ["scribe_v2_realtime"]
    assert params["audio_format"] == ["pcm_16000"]


def test_url_requests_azerbaijani_as_the_primary_language():
    params = parse_qs(urlparse(build_realtime_url(SttSettings(), 16000)).query)
    assert params["language_code"] == ["aze"]


def test_url_carries_vad_configuration():
    settings = SttSettings(
        commit_strategy="vad", vad_threshold=0.65,
        vad_silence_threshold_secs=1.2, min_speech_duration_ms=200,
        min_silence_duration_ms=600,
    )
    params = parse_qs(urlparse(build_realtime_url(settings, 16000)).query)
    assert params["commit_strategy"] == ["vad"]
    assert params["vad_threshold"] == ["0.65"]
    assert params["vad_silence_threshold_secs"] == ["1.2"]
    assert params["min_speech_duration_ms"] == ["200"]
    assert params["min_silence_duration_ms"] == ["600"]


def test_manual_commit_strategy_omits_vad_parameters():
    params = parse_qs(urlparse(build_realtime_url(SttSettings(commit_strategy="manual"), 16000)).query)
    assert params["commit_strategy"] == ["manual"]
    assert "vad_threshold" not in params


def test_url_includes_technical_keyterms_when_configured():
    settings = SttSettings(keyterms=["RAG", "LLM", "FastAPI", "LangGraph", "Embeddings"])
    params = parse_qs(urlparse(build_realtime_url(settings, 16000)).query)
    for expected in ("RAG", "LLM", "FastAPI", "LangGraph", "Embeddings"):
        assert expected in params["keyterms"]


def test_keyterms_respect_the_documented_limits():
    settings = SttSettings(keyterms=["ok", "x" * 21, "RAG", "rag"] + [f"term{i}" for i in range(60)])
    effective = settings.effective_keyterms()
    assert len(effective) <= 50
    assert all(len(term) <= 20 for term in effective)
    assert "x" * 21 not in effective
    assert effective.count("RAG") + effective.count("rag") == 1  # deduplicated


def test_url_reflects_a_different_sample_rate():
    params = parse_qs(urlparse(build_realtime_url(SttSettings(), 48000)).query)
    assert params["audio_format"] == ["pcm_48000"]


# =========================================================================
# Audio chunk preparation
# =========================================================================
def test_audio_chunk_is_base64_encoded_pcm():
    pcm = b"\x01\x02\x03\x04" * 100
    message = json.loads(encode_audio_chunk(pcm, 16000))
    assert message["message_type"] == "input_audio_chunk"
    assert base64.b64decode(message["audio_base_64"]) == pcm
    assert message["sample_rate"] == 16000
    assert "commit" not in message


def test_audio_chunk_can_request_a_commit():
    message = json.loads(encode_audio_chunk(b"", 16000, commit=True))
    assert message["commit"] is True


def test_audio_chunk_rejects_oversized_frames():
    with pytest.raises(ValueError, match="exceeds"):
        encode_audio_chunk(b"\x00" * (MAX_CHUNK_BYTES + 1), 16000)


def test_audio_chunk_can_omit_sample_rate():
    assert "sample_rate" not in json.loads(encode_audio_chunk(b"\x00\x01", None))


# =========================================================================
# Event parsing
# =========================================================================
def test_partial_transcript_event():
    event = parse_event(frame("partial_transcript", text="RAG nə"))
    assert event.is_partial and not event.is_committed
    assert event.text == "RAG nə"


def test_committed_transcript_event():
    event = parse_event(frame("committed_transcript", text="RAG nədir?"))
    assert event.is_committed and not event.is_partial
    assert event.text == "RAG nədir?"


def test_committed_transcript_with_timestamps_counts_as_committed():
    event = parse_event(
        frame("committed_transcript_with_timestamps", text="RAG nədir?",
              language_code="aze", words=[{"text": "RAG", "start": 0.0, "end": 0.4}])
    )
    assert event.is_committed
    assert event.language_code == "aze"
    assert event.words[0]["text"] == "RAG"


def test_session_started_event_carries_the_session_id():
    event = parse_event(frame("session_started", session_id="abc123"))
    assert event.type == "session_started"
    assert event.session_id == "abc123"


def test_azerbaijani_characters_survive_parsing():
    """The transcript must come through byte-for-byte, not transliterated."""
    event = parse_event(frame("committed_transcript", text=AZ_QUESTION))
    assert event.text == AZ_QUESTION
    # The specific characters this sentence contains, none of them normalised away.
    for char in "üəı":
        assert char in event.text

    # A sentence exercising every Azerbaijani-specific letter.
    alphabet = "Çətin şəraitdə də işlədiyim üçün öyrəndiklərim çoxdur: ğ, ı, İstanbul."
    round_tripped = parse_event(frame("committed_transcript", text=alphabet))
    assert round_tripped.text == alphabet
    for char in "əığşçöü":
        assert char in round_tripped.text.lower()


def test_malformed_frame_becomes_an_error_not_a_crash():
    event = parse_event("{not json")
    assert event.is_error
    assert "Malformed" in event.message


def test_unknown_event_type_is_preserved():
    event = parse_event(frame("something_new", text="x"))
    assert event.type == "something_new"
    assert not event.is_error


@pytest.mark.parametrize(
    "event_type,auth,billing,rate,recoverable",
    [
        ("auth_error", True, False, False, False),
        ("unaccepted_terms", True, False, False, False),
        ("quota_exceeded", False, True, False, False),
        ("resource_exhausted", False, True, False, False),
        ("rate_limited", False, False, True, True),
        ("commit_throttled", False, False, True, True),
        ("transcriber_error", False, False, False, True),
        ("invalid_request", False, False, False, True),
    ],
)
def test_error_frames_are_classified(event_type, auth, billing, rate, recoverable):
    event = parse_event(frame(event_type, error="boom"))
    assert event.is_error
    assert event.is_auth_error is auth
    assert event.is_billing_error is billing
    assert event.is_rate_limited is rate
    assert event.is_recoverable is recoverable


def test_error_descriptions_are_azerbaijani():
    assert describe_error(parse_event(frame("auth_error"))) == msg.STT_AUTH_ERROR
    assert describe_error(parse_event(frame("quota_exceeded"))) == msg.STT_BILLING_ERROR
    assert describe_error(parse_event(frame("rate_limited"))) == msg.STT_RATE_LIMITED


def test_nested_error_message_is_extracted():
    event = parse_event(json.dumps({"message_type": "error", "error": {"message": "detail here"}}))
    assert event.message == "detail here"


# =========================================================================
# Client behaviour
# =========================================================================
def _client(connect, **kwargs) -> ElevenLabsRealtimeClient:
    return ElevenLabsRealtimeClient(
        api_key="sk_test_key", settings=SttSettings(), sample_rate=16000,
        connect_factory=connect, **kwargs,
    )


async def _settle(times: int = 12) -> None:
    for _ in range(times):
        await asyncio.sleep(0.01)


async def test_client_requires_an_api_key():
    with pytest.raises(SttAuthError):
        ElevenLabsRealtimeClient(api_key="", settings=SttSettings(), sample_rate=16000)


async def test_client_sends_the_api_key_as_a_header(fake_ws_factory):
    captured: dict = {}

    async def connect(url, **kwargs):
        captured.update(kwargs)
        captured["url"] = url
        from .conftest import FakeWebSocket

        return FakeWebSocket([])

    client = _client(connect)
    await client.start()
    await _settle()
    assert captured["additional_headers"]["xi-api-key"] == "sk_test_key"
    await client.stop()


async def test_partial_and_committed_events_reach_the_callbacks(fake_ws_factory):
    partials: list[str] = []
    commits: list[str] = []
    connect = fake_ws_factory([
        frame("session_started", session_id="s1"),
        frame("partial_transcript", text="Production mühitində"),
        frame("partial_transcript", text="Production mühitində RAG"),
        frame("committed_transcript", text=AZ_QUESTION),
    ])
    client = _client(connect)
    client._cb = SttCallbacks(on_partial=partials.append, on_committed=commits.append)

    await client.start()
    await _settle()

    assert partials == ["Production mühitində", "Production mühitində RAG"]
    assert commits == [AZ_QUESTION]
    assert client.session_id == "s1"
    assert client.stats.partials == 2 and client.stats.commits == 1
    await client.stop()


async def test_audio_is_streamed_as_input_audio_chunks(fake_ws_factory):
    connect = fake_ws_factory([])
    client = _client(connect)
    await client.start()
    await _settle(4)

    for _ in range(3):
        assert client.push_audio(b"\x00\x01" * 800)
    await _settle()

    ws = connect.created[0]
    messages = ws.sent_messages()
    assert len(messages) == 3
    assert all(m["message_type"] == "input_audio_chunk" for m in messages)
    assert client.stats.chunks_sent == 3
    assert client.stats.bytes_sent == 3 * 1600
    await client.stop()


async def test_full_send_queue_drops_rather_than_lagging(fake_ws_factory):
    """Falling behind must not build an ever-growing delay."""
    client = _client(fake_ws_factory([]), queue_maxsize=2)
    # No start(), so nothing drains the queue.
    assert client.push_audio(b"a")
    assert client.push_audio(b"b")
    assert not client.push_audio(b"c")
    assert client.dropped_chunks == 1


async def test_auth_error_frame_stops_the_session_without_retrying(fake_ws_factory):
    errors: list[tuple[str, bool]] = []
    connect = fake_ws_factory([frame("auth_error", error="invalid key")])
    client = _client(connect)
    client._cb = SttCallbacks(on_error=lambda m, r: errors.append((m, r)))

    await client.start()
    await _settle(20)

    assert errors and errors[-1] == (msg.STT_AUTH_ERROR, False)
    assert client.stats.reconnects == 0   # never retried
    assert len(connect.created) == 1
    await client.stop()


async def test_billing_error_frame_stops_the_session(fake_ws_factory):
    errors: list[tuple[str, bool]] = []
    connect = fake_ws_factory([frame("quota_exceeded")])
    client = _client(connect)
    client._cb = SttCallbacks(on_error=lambda m, r: errors.append((m, r)))

    await client.start()
    await _settle(20)
    assert errors[-1] == (msg.STT_BILLING_ERROR, False)
    assert len(connect.created) == 1
    await client.stop()


async def test_rate_limit_frame_is_reported_but_recoverable(fake_ws_factory):
    errors: list[tuple[str, bool]] = []
    connect = fake_ws_factory([frame("rate_limited", error="slow down")])
    client = _client(connect)
    client._cb = SttCallbacks(on_error=lambda m, r: errors.append((m, r)))

    await client.start()
    await _settle()
    assert (msg.STT_RATE_LIMITED, True) in errors
    await client.stop()


async def test_interrupted_connection_reconnects(monkeypatch):
    """A dropped socket must be retried, with the disconnect surfaced first."""
    from .conftest import FakeWebSocket

    attempts: list[FakeWebSocket] = []

    async def connect(url, **kwargs):
        ws = FakeWebSocket([frame("partial_transcript", text=f"try{len(attempts)}")])
        attempts.append(ws)
        if len(attempts) == 1:
            # First socket dies as soon as its frames run out.
            ws.release()
        return ws

    states: list[str] = []
    errors: list[tuple[str, bool]] = []
    client = _client(connect)
    client._cb = SttCallbacks(on_state=states.append, on_error=lambda m, r: errors.append((m, r)))
    monkeypatch.setattr("random.random", lambda: 0.0)

    await client.start()
    await asyncio.sleep(0.6)

    assert len(attempts) >= 2, "client did not reconnect"
    assert client.stats.reconnects >= 1
    assert any(message == msg.STT_RECONNECTING for message, _ in errors)
    assert "connected" in states
    await client.stop()


async def test_reconnection_gives_up_after_the_limit(monkeypatch):
    async def connect(url, **kwargs):
        raise OSError("network unreachable")

    errors: list[tuple[str, bool]] = []
    client = _client(connect, max_reconnects=2)
    client._cb = SttCallbacks(on_error=lambda m, r: errors.append((m, r)))
    monkeypatch.setattr("random.random", lambda: 0.0)

    await client.start()
    await asyncio.sleep(1.5)

    assert errors[-1] == (msg.STT_DISCONNECTED, False)
    await client.stop()


async def test_http_401_during_handshake_is_an_auth_error(monkeypatch):
    from websockets.exceptions import InvalidStatus

    class _Response:
        status_code = 401

    async def connect(url, **kwargs):
        raise InvalidStatus(_Response())

    errors: list[tuple[str, bool]] = []
    client = _client(connect)
    client._cb = SttCallbacks(on_error=lambda m, r: errors.append((m, r)))
    await client.start()
    await _settle(20)

    assert errors[-1] == (msg.STT_AUTH_ERROR, False)
    await client.stop()


async def test_stop_closes_the_socket_and_releases_state(fake_ws_factory):
    connect = fake_ws_factory([])
    client = _client(connect)
    await client.start()
    await _settle(4)
    ws = connect.created[0]

    await client.stop()
    assert ws.closed
    assert client.state == "disconnected"
    assert not client.push_audio(b"x")   # no audio accepted after stop


# =========================================================================
# Defaults chosen from live measurements (see docs/stt-tuning.md)
# =========================================================================
def test_secondary_languages_is_empty_by_default():
    """Measured: `secondary_languages=eng` made Scribe translate the
    Azerbaijani sentence into English, which this app must never do."""
    assert SttSettings().secondary_languages == []
    params = parse_qs(urlparse(build_realtime_url(SttSettings(), 16000)).query)
    assert "secondary_languages" not in params


def test_keyterms_are_empty_by_default():
    """Measured: any keyterm list turned a spoken sentence into a
    comma-separated list of the keyterms and mangled Azerbaijani suffixes."""
    assert SttSettings().keyterms == []
    params = parse_qs(urlparse(build_realtime_url(SttSettings(), 16000)).query)
    assert "keyterms" not in params


def test_no_verbatim_is_off_by_default():
    """Measured: it rewrote the question as a nominalisation, removing the
    grammatical cues question detection depends on."""
    assert SttSettings().no_verbatim is False
    params = parse_qs(urlparse(build_realtime_url(SttSettings(), 16000)).query)
    assert "no_verbatim" not in params


def test_secondary_languages_still_work_when_explicitly_enabled():
    settings = SttSettings(secondary_languages=["eng", "rus"])
    params = parse_qs(urlparse(build_realtime_url(settings, 16000)).query)
    assert params["secondary_languages"] == ["eng", "rus"]


def test_keyterms_still_work_when_explicitly_enabled():
    settings = SttSettings(keyterms=["RAG", "LangGraph"])
    params = parse_qs(urlparse(build_realtime_url(settings, 16000)).query)
    assert params["keyterms"] == ["RAG", "LangGraph"]


# =========================================================================
# VAD values the server will actually accept
# =========================================================================
def test_vad_silence_is_clamped_to_the_accepted_range():
    """Out of range is not a warning from this API - it closes the socket with
    `invalid_request`, losing the whole session mid-interview."""
    params = parse_qs(urlparse(build_realtime_url(SttSettings(vad_silence_threshold_secs=0.05), 16000)).query)
    assert params["vad_silence_threshold_secs"] == ["0.3"]

    params = parse_qs(urlparse(build_realtime_url(SttSettings(vad_silence_threshold_secs=9.0), 16000)).query)
    assert params["vad_silence_threshold_secs"] == ["3"]


def test_vad_threshold_is_clamped():
    params = parse_qs(urlparse(build_realtime_url(SttSettings(vad_threshold=5.0), 16000)).query)
    assert params["vad_threshold"] == ["1"]


def test_a_valid_vad_silence_passes_through_unchanged():
    params = parse_qs(urlparse(build_realtime_url(SttSettings(vad_silence_threshold_secs=0.8), 16000)).query)
    assert params["vad_silence_threshold_secs"] == ["0.8"]


def test_default_vad_silence_is_inside_the_accepted_range():
    from copilot.config import VAD_SILENCE_RANGE

    low, high = VAD_SILENCE_RANGE
    assert low <= SttSettings().vad_silence_threshold_secs <= high
