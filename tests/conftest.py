"""Shared fixtures and fake providers.

Nothing in the default test run touches a real API or a real audio device; the
fakes here reproduce the documented protocols closely enough to exercise the
real client code paths.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from copilot.config import Settings
from copilot.documents.extract import extract_text
from copilot.documents.profile import CandidateProfile

FIXTURES = Path(__file__).parent / "fixtures"


def _ensure_writable_temp_root() -> None:
    """Point pytest's `tmp_path` somewhere writable.

    Some hardened Windows profiles and CI images deny writes to the shared
    `%TEMP%`, which otherwise fails every test that uses `tmp_path` before it
    has run a line of our code.
    """
    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return

    # pytest puts its per-run directories under <temp>/pytest-of-<user>, so that
    # exact path is what has to be writable - a fresh sibling directory can
    # succeed while the real one is denied.
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    root = Path(tempfile.gettempdir()) / f"pytest-of-{user}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".probe-{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError:
        fallback = Path(__file__).resolve().parent.parent / ".pytest-tmp"
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["PYTEST_DEBUG_TEMPROOT"] = str(fallback)


_ensure_writable_temp_root()

# Make a local .env available to the opt-in live tests.  They stay gated behind
# AZCOPILOT_LIVE_TESTS=1, so this cannot cause an accidental billable run.
from copilot.env_file import load_env_file  # noqa: E402

load_env_file()


# =========================================================================
# Documents
# =========================================================================
@pytest.fixture
def cv_document():
    return extract_text(FIXTURES / "sample_cv.txt", "cv")


@pytest.fixture
def jd_document():
    return extract_text(FIXTURES / "sample_job.txt", "job_description")


@pytest.fixture
def profile(cv_document, jd_document) -> CandidateProfile:
    return CandidateProfile.build(cv_document, jd_document)


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("AZCOPILOT_DATA_DIR", str(tmp_path / "appdata"))
    return Settings()


# =========================================================================
# Fake ElevenLabs websocket
# =========================================================================
class FakeWebSocket:
    """Stands in for a `websockets` client connection.

    Frames queued in `incoming` are yielded by iteration; everything the client
    sends is recorded in `sent`.
    """

    def __init__(self, incoming: list[str] | None = None, fail_after: int | None = None) -> None:
        self.incoming = list(incoming or [])
        self.sent: list[str] = []
        self.closed = False
        self.fail_after = fail_after
        self._released = asyncio.Event()

    async def send(self, payload: str) -> None:
        if self.closed:
            from websockets.exceptions import ConnectionClosedOK

            raise ConnectionClosedOK(None, None)
        self.sent.append(payload)
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            self.closed = True
            from websockets.exceptions import ConnectionClosedError

            raise ConnectionClosedError(None, None)

    async def close(self) -> None:
        self.closed = True
        self._released.set()

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for frame in self.incoming:
            await asyncio.sleep(0)
            yield frame
        # Stay open until closed, mirroring a live connection that simply has
        # nothing more to say.
        await self._released.wait()

    # -- helpers for assertions ------------------------------------------
    def sent_messages(self) -> list[dict]:
        return [json.loads(payload) for payload in self.sent]

    def release(self) -> None:
        self._released.set()


def frame(message_type: str, **fields) -> str:
    return json.dumps({"message_type": message_type, **fields})


@pytest.fixture
def fake_ws_factory():
    """Builds a `connect_factory` that hands back a prepared FakeWebSocket."""

    created: list[FakeWebSocket] = []

    def make(incoming: list[str] | None = None, **kwargs):
        async def connect(url, **_ignored):
            ws = FakeWebSocket(incoming, **kwargs)
            ws.url = url
            created.append(ws)
            return ws

        connect.created = created
        return connect

    make.created = created
    return make


# =========================================================================
# Fake OpenAI Responses stream
# =========================================================================
class FakeEvent:
    def __init__(self, type: str, **fields) -> None:
        self.type = type
        for key, value in fields.items():
            setattr(self, key, value)


class FakeUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class FakeResponse:
    def __init__(self, usage=None, incomplete_details=None, error=None) -> None:
        self.usage = usage
        self.incomplete_details = incomplete_details
        self.error = error


class FakeStream:
    def __init__(self, events: list[FakeEvent], delay: float = 0.0) -> None:
        self._events = events
        self._delay = delay

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self._events:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield event


class FakeResponses:
    """Records requests and replays a scripted stream (or raises)."""

    def __init__(self, deltas: list[str] | None = None, error: Exception | None = None) -> None:
        self.deltas = deltas if deltas is not None else ["Salam", ", ", "dünya."]
        self.error = error
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        if not kwargs.get("stream"):
            return FakeResponse(usage=FakeUsage())
        events = [FakeEvent("response.created")]
        events += [FakeEvent("response.output_text.delta", delta=d) for d in self.deltas]
        events.append(
            FakeEvent("response.completed", response=FakeResponse(usage=FakeUsage()))
        )
        return FakeStream(events)


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponses | None = None) -> None:
        self.responses = responses or FakeResponses()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_openai(monkeypatch):
    """Patches `AnswerGenerator` to use a scripted client instead of the SDK."""

    holder: dict[str, FakeOpenAIClient] = {}

    def install(deltas: list[str] | None = None, error: Exception | None = None):
        client = FakeOpenAIClient(FakeResponses(deltas, error))
        holder["client"] = client

        from copilot.llm import openai_client as module

        original_init = module.AnswerGenerator.__init__

        def patched_init(self, api_key, settings=None):
            if not api_key:
                raise module.LlmAuthError("no key", recoverable=False)
            from copilot.config import LlmSettings

            self.settings = settings or LlmSettings()
            self._client = client
            self._history = []
            self.service_tier_rejected = None
            self.stream_retries = 0

        monkeypatch.setattr(module.AnswerGenerator, "__init__", patched_init)
        return client

    install.holder = holder
    return install
