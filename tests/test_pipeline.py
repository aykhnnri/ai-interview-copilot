"""End-to-end pipeline behaviour with both providers faked.

These wire the real `CopilotPipeline` to a fake websocket and a fake OpenAI
client, so the orchestration - audio in, transcript out, question detected,
answer streamed - is exercised for real without touching a network or a device.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from copilot import messages as msg
from copilot.config import Settings
from copilot.core.metrics import PipelineMetrics
from copilot.core.pipeline import IDLE, LISTENING, PAUSED, CopilotPipeline, PipelineEvents
from copilot.credentials import CredentialStore

from .conftest import FakeWebSocket, frame

AZ_QUESTION = "RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?"


class StubCredentials(CredentialStore):
    """Hands out fixed keys without touching Credential Manager."""

    def __init__(self, elevenlabs="sk_eleven_test", openai="sk-openai-test") -> None:
        self._values = {"elevenlabs": elevenlabs, "openai": openai}
        self._keyring = None
        self.service_name = "test"
        self._keyring_error = None

    def get(self, provider):
        return self._values.get(provider)

    def source_of(self, provider):
        return "credential-manager" if self._values.get(provider) else None


class Recorder:
    """Collects everything the pipeline emits."""

    def __init__(self) -> None:
        self.states: list[str] = []
        self.providers: list[tuple[str, str]] = []
        self.partials: list[str] = []
        self.commits: list[str] = []
        self.questions: list[str] = []
        self.answer_starts: list[str] = []
        self.deltas: list[str] = []
        self.answers: list[object] = []
        self.errors: list[tuple[str, str]] = []
        self.logs: list[str] = []

    def events(self) -> PipelineEvents:
        return PipelineEvents(
            on_state=self.states.append,
            on_provider_state=lambda p, s: self.providers.append((p, s)),
            on_partial_transcript=self.partials.append,
            on_committed_transcript=self.commits.append,
            on_question=self.questions.append,
            on_answer_start=self.answer_starts.append,
            on_answer_delta=self.deltas.append,
            on_answer_done=self.answers.append,
            on_error=lambda p, m: self.errors.append((p, m)),
            on_log=self.logs.append,
        )


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("AZCOPILOT_DATA_DIR", str(tmp_path))
    settings = Settings()
    settings.question.merge_window_ms = 100
    recorder = Recorder()
    pipe = CopilotPipeline(settings, recorder.events(), StubCredentials())
    pipe.start()
    pipe.recorder = recorder  # type: ignore[attr-defined]
    yield pipe
    pipe.shutdown()


def _wait_for(predicate, timeout: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# =========================================================================
# Lifecycle
# =========================================================================
def test_pipeline_starts_and_stops_its_loop(pipeline):
    assert pipeline.loop is not None
    assert pipeline.session.state == IDLE


def test_missing_elevenlabs_key_is_reported_and_nothing_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("AZCOPILOT_DATA_DIR", str(tmp_path))
    recorder = Recorder()
    pipe = CopilotPipeline(Settings(), recorder.events(), StubCredentials(elevenlabs=None))
    pipe.start()
    try:
        pipe.start_listening().result(timeout=5)
        assert ("elevenlabs", msg.KEY_MISSING_ELEVENLABS) in recorder.errors
        assert pipe.session.state == IDLE
    finally:
        pipe.shutdown()


def test_audio_device_failure_stops_cleanly(pipeline, monkeypatch):
    """A missing loopback device must not leave the STT socket open."""
    from copilot.audio import capture as capture_module
    from copilot.core import pipeline as pipeline_module
    from copilot.stt import elevenlabs_client

    async def fake_connect(url, **kwargs):
        return FakeWebSocket([])

    monkeypatch.setattr(elevenlabs_client.websockets, "connect", fake_connect)

    def explode(*args, **kwargs):
        raise capture_module.AudioCaptureError(msg.AUDIO_NO_DEVICE)

    monkeypatch.setattr(pipeline_module, "LoopbackCapture", explode)

    pipeline.start_listening().result(timeout=5)
    assert any(provider == "audio" for provider, _ in pipeline.recorder.errors)
    assert pipeline.session.state == IDLE


# =========================================================================
# Transcript -> question -> answer
# =========================================================================
@pytest.fixture
def wired(pipeline, monkeypatch, fake_openai):
    """Pipeline with a scripted websocket and a scripted OpenAI client."""
    sockets: list[FakeWebSocket] = []

    async def fake_connect(url, **kwargs):
        ws = FakeWebSocket([])
        sockets.append(ws)
        return ws

    from copilot.stt import elevenlabs_client

    monkeypatch.setattr(elevenlabs_client.websockets, "connect", fake_connect)

    class FakeCapture:
        def __init__(self, on_chunk, **kwargs):
            self.on_chunk = on_chunk
            self.started = self.stopped = self.paused = False

        def start(self):
            from copilot.audio.capture import LoopbackDevice

            self.started = True
            return LoopbackDevice(1, "Fake [Loopback]", 2, 48000, True)

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

        def stop(self):
            self.stopped = True

    from copilot.core import pipeline as pipeline_module

    captures: list[FakeCapture] = []

    def make_capture(*args, **kwargs):
        capture = FakeCapture(*args, **kwargs)
        captures.append(capture)
        return capture

    monkeypatch.setattr(pipeline_module, "LoopbackCapture", make_capture)
    fake_openai(["Retrieval quality-ni ", "Recall@k və MRR ", "ilə ölçürəm."])

    pipeline.start_listening().result(timeout=5)
    assert _wait_for(lambda: pipeline.session.state == LISTENING)
    pipeline.sockets = sockets        # type: ignore[attr-defined]
    pipeline.captures = captures      # type: ignore[attr-defined]
    return pipeline


@pytest.fixture
def wired_normal(wired):
    """`wired`, but with the narrow gate - for tests about classification."""
    wired.settings.question.sensitivity = "normal"
    wired._detector.settings = wired.settings.question
    return wired


def _feed(pipeline, *frames: str) -> None:
    """Deliver websocket frames the way the STT receive loop would."""
    from copilot.stt.events import parse_event

    loop = pipeline.loop
    for raw in frames:
        loop.call_soon_threadsafe(pipeline._stt._dispatch, parse_event(raw))


def test_partial_transcripts_reach_the_ui(wired):
    _feed(wired, frame("partial_transcript", text="RAG pipeline-ın"))
    assert _wait_for(lambda: wired.recorder.partials)
    assert wired.recorder.partials == ["RAG pipeline-ın"]
    assert wired.recorder.questions == []      # a partial never fires a question


def test_committed_transcript_detects_a_question_and_answers_it(wired):
    _feed(wired, frame("committed_transcript", text=AZ_QUESTION))

    assert _wait_for(lambda: wired.recorder.questions), "no question detected"
    assert wired.recorder.questions == [AZ_QUESTION]

    assert _wait_for(lambda: wired.recorder.answers), "no answer generated"
    result = wired.recorder.answers[0]
    assert result.text == "Retrieval quality-ni Recall@k və MRR ilə ölçürəm."
    assert wired.recorder.deltas          # streamed incrementally, not in one lump
    assert len(wired.recorder.deltas) > 1


def test_azerbaijani_text_survives_the_whole_pipeline(wired):
    _feed(wired, frame("committed_transcript", text=AZ_QUESTION))
    assert _wait_for(lambda: wired.recorder.questions)
    assert wired.recorder.questions[0] == AZ_QUESTION
    assert "ı" in wired.recorder.questions[0] and "ə" in wired.recorder.questions[0]


def test_a_split_question_produces_exactly_one_answer(wired_normal):
    _feed(
        wired_normal,
        frame("committed_transcript", text="Production mühitində RAG pipeline-ın"),
        frame("committed_transcript", text="retrieval quality-sini necə ölçərdiniz?"),
    )
    assert _wait_for(lambda: wired_normal.recorder.questions)
    time.sleep(0.4)
    assert len(wired_normal.recorder.questions) == 1
    assert wired_normal.recorder.questions[0].startswith("Production mühitində")


def test_a_continuation_supersedes_the_answer_in_flight(wired):
    """At the default gate the first half is answered straight away, and the
    rest of the turn replaces that answer rather than queueing a second one."""
    _feed(wired, frame("committed_transcript", text="Sizin CV-nizdə LangGraph var."))
    assert _wait_for(lambda: wired.recorder.questions)

    _feed(wired, frame("committed_transcript", text="Bu barədə eşitmək istərdim."))
    assert _wait_for(lambda: len(wired.recorder.questions) >= 2)
    assert _wait_for(lambda: wired.recorder.answers)

    # The last thing generated covers the whole turn, not just the tail.
    final = wired.recorder.answer_starts[-1]
    assert "LangGraph" in final and "eşitmək" in final


def test_a_repeated_question_is_not_answered_twice(wired):
    _feed(wired, frame("committed_transcript", text=AZ_QUESTION))
    assert _wait_for(lambda: wired.recorder.answers)
    _feed(wired, frame("committed_transcript", text=AZ_QUESTION))
    time.sleep(0.4)

    assert len(wired.recorder.questions) == 1
    assert wired.metrics.duplicates_skipped >= 1


def test_statements_do_not_trigger_generation_at_normal_sensitivity(wired_normal):
    _feed(wired_normal,
          frame("committed_transcript", text="Yaxşı, indi növbəti mövzuya keçək."))
    time.sleep(0.4)
    assert wired_normal.recorder.questions == []
    assert wired_normal.recorder.answers == []


def test_acknowledgements_never_trigger_generation_even_at_the_default(wired):
    for noise in ("Bəli.", "Yaxşı, aydındır.", "Təşəkkür edirəm."):
        _feed(wired, frame("committed_transcript", text=noise))
    time.sleep(0.5)
    assert wired.recorder.questions == []
    assert wired.recorder.answers == []


def test_every_interviewer_turn_is_answered_at_the_default(wired):
    """The reported problem: only question-shaped sentences got an answer."""
    _feed(wired, frame("committed_transcript",
                       text="Bizim komandada model deployment prosesi belədir."))
    assert _wait_for(lambda: wired.recorder.questions), "a plain statement was ignored"
    assert _wait_for(lambda: wired.recorder.answers)


def test_auto_generate_can_be_turned_off(wired):
    wired.settings.llm.auto_generate = False
    _feed(wired, frame("committed_transcript", text="Vector database necə işləyir?"))
    assert _wait_for(lambda: wired.recorder.questions)
    time.sleep(0.3)
    assert wired.recorder.answers == []

    wired.generate_answer()
    assert _wait_for(lambda: wired.recorder.answers)


def test_manual_regenerate_and_expand(wired):
    _feed(wired, frame("committed_transcript", text=AZ_QUESTION))
    assert _wait_for(lambda: wired.recorder.answers)

    wired.regenerate()
    assert _wait_for(lambda: len(wired.recorder.answers) >= 2)
    wired.expand()
    assert _wait_for(lambda: len(wired.recorder.answers) >= 3)


def test_audio_chunks_are_forwarded_to_the_transcriber(wired):
    capture = wired.captures[0]
    for _ in range(3):
        capture.on_chunk(b"\x00\x01" * 800, 0.2)
    assert _wait_for(lambda: wired._stt.stats.chunks_sent >= 3)


def test_pause_and_resume(wired):
    wired.pause_listening().result(timeout=5)
    assert _wait_for(lambda: wired.session.state == PAUSED)
    assert wired.captures[0].paused

    wired.resume_listening().result(timeout=5)
    assert _wait_for(lambda: wired.session.state == LISTENING)
    assert not wired.captures[0].paused


def test_stop_releases_audio_and_the_connection(wired):
    socket = wired.sockets[0]
    wired.stop_listening().result(timeout=5)

    assert _wait_for(lambda: wired.session.state == IDLE)
    assert wired.captures[0].stopped                 # 1. audio capture stopped
    assert _wait_for(lambda: socket.closed)          # 3. connection closed
    assert wired._stt is None                        # 4. resources released


def test_openai_failure_is_reported_without_losing_the_question(
    pipeline, monkeypatch, fake_openai
):
    error = type("RateLimitError", (Exception,), {})("slow down")
    error.status_code = 429
    fake_openai(error=error)

    pipeline.generate_answer(AZ_QUESTION)
    assert _wait_for(lambda: pipeline.recorder.errors)
    provider, message = pipeline.recorder.errors[-1]
    assert provider == "openai"
    assert message == msg.LLM_RATE_LIMITED
    # The question is preserved so the user can retry it.
    assert pipeline._last_question == AZ_QUESTION


def test_stt_failure_never_falls_back_to_another_provider(wired):
    _feed(wired, frame("rate_limited", error="too fast"))
    assert _wait_for(lambda: wired.recorder.errors)
    assert all(provider == "elevenlabs" for provider, _ in wired.recorder.errors)
    assert wired.recorder.answers == []     # no invented transcript, no answer


# =========================================================================
# Metrics
# =========================================================================
def test_metrics_record_the_four_stage_latencies():
    metrics = PipelineMetrics()
    metrics.note_audio_sent()
    metrics.note_partial()
    metrics.note_commit()
    metrics.note_question("k", detected_at=time.monotonic())
    metrics.note_first_token("k")
    metrics.note_answer_done("k", total_ms=1234.0)

    assert metrics.audio_to_partial.count == 1
    assert metrics.audio_to_commit.count == 1
    assert metrics.question_to_first_token.count == 1
    assert metrics.answer_total.last == 1234.0
    assert metrics.questions_detected == 1 and metrics.answers_generated == 1
    assert "Sual" in metrics.report()


def test_metrics_reset():
    metrics = PipelineMetrics()
    metrics.note_audio_sent()
    metrics.note_partial()
    metrics.questions_detected = 5
    metrics.reset()
    assert metrics.audio_to_partial.count == 0
    assert metrics.questions_detected == 0


def test_samples_summarise_percentiles():
    metrics = PipelineMetrics()
    for value in range(1, 101):
        metrics.answer_total.add(float(value))
    assert metrics.answer_total.median == pytest.approx(50.5)
    # Nearest-rank on a 0-based index: round(0.95 * 99) == 94 -> the 95th value.
    assert metrics.answer_total.p95 == pytest.approx(95.0)


# =========================================================================
# Answering things the classifier did not flag
# =========================================================================
def test_requests_without_a_question_mark_are_answered(wired):
    """'Tell me about your multi-agent work' is a prompt, not a question."""
    _feed(wired, frame("committed_transcript",
                       text="Mənə multi-agent layihənizdən bəhs edin."))
    assert _wait_for(lambda: wired.recorder.questions), "request was not detected"
    assert _wait_for(lambda: wired.recorder.answers)


def test_generate_answer_falls_back_to_the_last_thing_said(wired_normal):
    wired = wired_normal
    """Detection is a heuristic, so the button must never do nothing."""
    remark = "Sizin CV-nizdə maraqlı bir layihə görürəm."
    _feed(wired, frame("committed_transcript", text=remark))
    assert _wait_for(lambda: remark in wired.session.committed_lines)
    time.sleep(0.3)
    assert wired.recorder.questions == []      # correctly not auto-detected

    wired.generate_answer()                    # user presses Generate Answer
    assert _wait_for(lambda: wired.recorder.answers), "manual generation did nothing"
    assert wired.recorder.answer_starts[-1] == remark


def test_the_fallback_joins_recent_segments(wired_normal):
    wired = wired_normal
    _feed(
        wired,
        frame("committed_transcript", text="Bizim komandada"),
        frame("committed_transcript", text="model deployment prosesi belədir"),
    )
    assert _wait_for(lambda: len(wired.session.committed_lines) >= 2)
    assert "Bizim komandada" in wired.last_committed_text()
    assert "deployment" in wired.last_committed_text()


def test_the_fallback_is_empty_before_anything_is_said(pipeline):
    assert pipeline.last_committed_text() == ""
    pipeline.generate_answer()                 # must not raise or hang
