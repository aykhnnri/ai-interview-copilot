"""ElevenLabs Scribe v2 Realtime speech-to-text over websockets.

One long-lived connection is held for the whole listening session: audio chunks
go up as base64 `input_audio_chunk` messages, `partial_transcript` and
`committed_transcript` frames come back down.

Reconnection is automatic and bounded for transport faults, but *not* for
authentication or billing failures - those are surfaced to the user instead of
being retried, and we never fall back to a different provider.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Sequence
from urllib.parse import urlencode

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from ..config import SttSettings
from .events import SttEvent, describe_error, parse_event

log = logging.getLogger(__name__)

REALTIME_PATH = "/v1/speech-to-text/realtime"

#: ElevenLabs rejects oversized frames; keep well under the limit.
MAX_CHUNK_BYTES = 96_000


class SttAuthError(RuntimeError):
    """API key rejected - retrying will not help."""


class SttBillingError(RuntimeError):
    """Account out of credit - retrying will not help."""


class SttConnectionError(RuntimeError):
    """Transport-level failure; a reconnect may succeed."""


@dataclass
class SttCallbacks:
    """Hooks invoked from the client task.  All are optional."""

    on_partial: Callable[[str], None] | None = None
    on_committed: Callable[[str], None] | None = None
    on_event: Callable[[SttEvent], None] | None = None
    on_state: Callable[[str], None] | None = None        # connecting/connected/disconnected
    on_error: Callable[[str, bool], None] | None = None  # (message, recoverable)


@dataclass
class SttStats:
    chunks_sent: int = 0
    bytes_sent: int = 0
    partials: int = 0
    commits: int = 0
    reconnects: int = 0
    connected_at: float | None = None
    last_audio_sent_at: float | None = None
    last_partial_at: float | None = None
    last_commit_at: float | None = None
    audio_to_partial_ms: list[float] = field(default_factory=list)

    def note_partial(self) -> None:
        now = time.perf_counter()
        self.partials += 1
        self.last_partial_at = now
        if self.last_audio_sent_at is not None:
            self.audio_to_partial_ms.append((now - self.last_audio_sent_at) * 1000.0)
            del self.audio_to_partial_ms[:-200]

    @property
    def median_audio_to_partial_ms(self) -> float | None:
        if not self.audio_to_partial_ms:
            return None
        ordered = sorted(self.audio_to_partial_ms)
        return ordered[len(ordered) // 2]


def build_realtime_url(settings: SttSettings, sample_rate: int) -> str:
    """Compose the realtime websocket URL from the documented query parameters."""
    from ..audio.processing import audio_format_for

    params: list[tuple[str, str]] = [
        ("model_id", settings.model_id),
        ("audio_format", audio_format_for(sample_rate)),
    ]
    if settings.language_code:
        params.append(("language_code", settings.language_code))
    for secondary in settings.secondary_languages or []:
        if secondary and secondary != settings.language_code:
            params.append(("secondary_languages", secondary))
    if settings.commit_strategy:
        params.append(("commit_strategy", settings.commit_strategy))
    if settings.commit_strategy == "vad":
        params.append(("vad_threshold", _fmt(settings.effective_vad_threshold())))
        params.append(
            ("vad_silence_threshold_secs", _fmt(settings.effective_vad_silence()))
        )
        params.append(("min_speech_duration_ms", str(int(settings.min_speech_duration_ms))))
        params.append(("min_silence_duration_ms", str(int(settings.min_silence_duration_ms))))
    if settings.include_timestamps:
        params.append(("include_timestamps", "true"))
    if settings.include_language_detection:
        params.append(("include_language_detection", "true"))
    if settings.no_verbatim:
        params.append(("no_verbatim", "true"))
    if settings.filter_background_audio:
        params.append(("filter_background_audio", "true"))
    for term in settings.effective_keyterms():
        params.append(("keyterms", term))

    base = (settings.base_url or "wss://api.elevenlabs.io").rstrip("/")
    return f"{base}{REALTIME_PATH}?{urlencode(params)}"


def _fmt(value: float) -> str:
    """Compact float formatting, so 0.5 does not become '0.5000000001'."""
    return f"{float(value):g}"


def encode_audio_chunk(
    pcm: bytes, sample_rate: int | None = None, commit: bool = False
) -> str:
    """Build one `input_audio_chunk` message as a JSON string."""
    if len(pcm) > MAX_CHUNK_BYTES:
        raise ValueError(
            f"audio chunk of {len(pcm)} bytes exceeds the {MAX_CHUNK_BYTES} byte limit"
        )
    payload: dict[str, object] = {
        "message_type": "input_audio_chunk",
        "audio_base_64": base64.b64encode(pcm).decode("ascii"),
    }
    if commit:
        payload["commit"] = True
    if sample_rate is not None:
        payload["sample_rate"] = int(sample_rate)
    return json.dumps(payload)


class ElevenLabsRealtimeClient:
    """Persistent realtime transcription session.

    Usage::

        client = ElevenLabsRealtimeClient(api_key, settings, sample_rate, callbacks)
        await client.start()
        client.push_audio(pcm_bytes)      # thread-safe-ish: call from the loop
        await client.stop()
    """

    def __init__(
        self,
        api_key: str,
        settings: SttSettings,
        sample_rate: int,
        callbacks: SttCallbacks | None = None,
        connect_factory: Callable[..., Awaitable[object]] | None = None,
        max_reconnects: int = 8,
        queue_maxsize: int = 400,
    ) -> None:
        if not api_key:
            raise SttAuthError("ElevenLabs API key is not configured")
        self._api_key = api_key
        self._settings = settings
        self._sample_rate = sample_rate
        self._cb = callbacks or SttCallbacks()
        self._connect_factory = connect_factory or websockets.connect
        self._max_reconnects = max_reconnects

        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_maxsize)
        self._task: asyncio.Task | None = None
        self._ws = None
        self._state = "disconnected"
        self._stopping = False
        self._session_id = ""
        self.stats = SttStats()
        self._dropped_chunks = 0

    # -- public API -------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == "connected"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def dropped_chunks(self) -> int:
        return self._dropped_chunks

    @property
    def url(self) -> str:
        return build_realtime_url(self._settings, self._sample_rate)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="elevenlabs-stt")

    def push_audio(self, pcm: bytes) -> bool:
        """Enqueue a chunk for transmission.  Returns False if the queue is full.

        Dropping the newest chunk under pressure is deliberate: it keeps the
        stream close to real time instead of accumulating an ever-growing lag.
        """
        if self._stopping or not pcm:
            return False
        try:
            self._queue.put_nowait(pcm)
            return True
        except asyncio.QueueFull:
            self._dropped_chunks += 1
            if self._dropped_chunks % 25 == 1:
                log.warning("Audio send queue is full; dropped %d chunks", self._dropped_chunks)
            return False

    async def commit(self) -> None:
        """Force the current audio segment to be finalised (manual strategy)."""
        ws = self._ws
        if ws is None:
            return
        with contextlib.suppress(Exception):
            await ws.send(encode_audio_chunk(b"", self._effective_sample_rate(), commit=True))

    async def stop(self) -> None:
        """Stop sending, close the socket and release the task."""
        self._stopping = True
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._close_socket()
        self._set_state("disconnected")

    # -- internals --------------------------------------------------------
    def _effective_sample_rate(self) -> int | None:
        return self._sample_rate if self._settings.send_sample_rate_field else None

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        if self._cb.on_state is not None:
            self._cb.on_state(state)

    def _report(self, message: str, recoverable: bool) -> None:
        if self._cb.on_error is not None:
            self._cb.on_error(message, recoverable)

    async def _run(self) -> None:
        attempt = 0
        while not self._stopping:
            session_started = time.perf_counter()
            try:
                await self._session_once()
            except (SttAuthError, SttBillingError) as exc:
                # A rejected key or an empty balance will not fix itself.
                self._set_state("error")
                self._report(str(exc), False)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason: BaseException = exc
            else:
                if self._stopping:
                    return
                # The socket closed without us asking: still a disconnection.
                reason = ConnectionResetError("connection closed by ElevenLabs")

            if self._stopping:
                return

            # A session that ran for a while was healthy; start its backoff fresh.
            if time.perf_counter() - session_started > 30.0:
                attempt = 0
            attempt += 1
            self.stats.reconnects += 1
            self._set_state("disconnected")

            if attempt > self._max_reconnects:
                log.error("Giving up on ElevenLabs after %d attempts: %s", attempt, reason)
                self._report(_msg().STT_DISCONNECTED, False)
                return

            delay = min(30.0, 0.5 * (2 ** (attempt - 1))) * (0.8 + 0.4 * random.random())
            log.warning(
                "ElevenLabs connection lost (%s); reconnecting in %.1fs (attempt %d/%d)",
                reason, delay, attempt, self._max_reconnects,
            )
            self._report(_msg().STT_RECONNECTING, True)
            await asyncio.sleep(delay)

    async def _session_once(self) -> None:
        self._set_state("connecting")
        url = self.url
        log.info("Connecting to ElevenLabs realtime STT (model=%s, language=%s)",
                 self._settings.model_id, self._settings.language_code)
        try:
            ws = await self._connect_factory(
                url,
                additional_headers={"xi-api-key": self._api_key},
                max_size=2**22,
                ping_interval=20,
                ping_timeout=20,
                open_timeout=15,
            )
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise SttAuthError(_msg().STT_AUTH_ERROR) from exc
            if status == 402:
                raise SttBillingError(_msg().STT_BILLING_ERROR) from exc
            if status == 429:
                raise SttConnectionError(_msg().STT_RATE_LIMITED) from exc
            raise SttConnectionError(f"{_msg().STT_GENERIC_ERROR} (HTTP {status})") from exc

        self._ws = ws
        self._set_state("connected")
        self.stats.connected_at = time.perf_counter()
        sender = asyncio.create_task(self._send_loop(ws), name="stt-send")
        try:
            await self._receive_loop(ws)
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
            await self._close_socket()

    async def _send_loop(self, ws) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            try:
                await ws.send(encode_audio_chunk(chunk, self._effective_sample_rate()))
            except ConnectionClosed:
                return
            except Exception:
                log.exception("Failed to send audio chunk")
                return
            self.stats.chunks_sent += 1
            self.stats.bytes_sent += len(chunk)
            self.stats.last_audio_sent_at = time.perf_counter()

    async def _receive_loop(self, ws) -> None:
        async for frame in ws:
            event = parse_event(frame)
            self._dispatch(event)
            if event.is_auth_error:
                raise SttAuthError(describe_error(event))
            if event.is_billing_error:
                raise SttBillingError(describe_error(event))

    def _dispatch(self, event: SttEvent) -> None:
        if self._cb.on_event is not None:
            self._cb.on_event(event)

        if event.type == "session_started":
            self._session_id = event.session_id
            log.info("ElevenLabs session started (%s)", event.session_id or "no id")
            return
        if event.is_partial:
            self.stats.note_partial()
            if event.text and self._cb.on_partial is not None:
                self._cb.on_partial(event.text)
            return
        if event.is_committed:
            self.stats.commits += 1
            self.stats.last_commit_at = time.perf_counter()
            if event.text and self._cb.on_committed is not None:
                self._cb.on_committed(event.text)
            return
        if event.type == "warning":
            log.warning("ElevenLabs warning: %s", event.message)
            return
        if event.is_error:
            log.error("ElevenLabs error frame: %s", event.type)
            self._report(describe_error(event), event.is_recoverable)

    async def _close_socket(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        with contextlib.suppress(Exception):
            await ws.close()


def _msg():
    from .. import messages

    return messages


async def test_api_key(api_key: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Authenticated, minimal-cost check of an ElevenLabs key.

    Calls `GET /v1/user/subscription`, which bills nothing but does require a
    valid key, and reports the tier so the user sees a real response.
    """
    import urllib.error
    import urllib.request

    if not api_key or not api_key.strip():
        return False, _msg().KEY_MISSING_ELEVENLABS

    def _call() -> tuple[bool, str]:
        request = urllib.request.Request(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": api_key.strip(), "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, _msg().STT_AUTH_ERROR
            if exc.code == 429:
                return False, _msg().STT_RATE_LIMITED
            return False, f"{_msg().STT_GENERIC_ERROR} (HTTP {exc.code})"
        except urllib.error.URLError as exc:
            return False, f"{_msg().STT_GENERIC_ERROR} ({exc.reason})"
        except Exception as exc:
            return False, f"{_msg().STT_GENERIC_ERROR} ({exc})"

        tier = data.get("tier") or "unknown"
        used = data.get("character_count")
        limit = data.get("character_limit")
        detail = f"tier: {tier}"
        if isinstance(used, int) and isinstance(limit, int):
            detail += f", {used:,}/{limit:,} characters used"
        return True, detail

    return await asyncio.to_thread(_call)
