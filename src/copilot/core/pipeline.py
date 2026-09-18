"""The orchestrator: system audio -> ElevenLabs -> question -> OpenAI -> UI.

Everything asynchronous lives on one asyncio loop running in a background
thread.  The UI never blocks on the network, and the PortAudio callback thread
only hands buffers across with `call_soon_threadsafe`.

Provider separation is strict: ElevenLabs does speech recognition, OpenAI does
answer generation, and a failure in one is never covered up by the other.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .. import messages as msg
from ..audio.capture import AudioCaptureError, LoopbackCapture
from ..config import Settings
from ..credentials import CredentialStore
from ..documents.profile import CandidateProfile, normalized_question_key
from ..llm.openai_client import AnswerGenerator, AnswerResult, LlmError
from ..nlp.question_detector import DetectedQuestion, QuestionDetector
from ..stt.elevenlabs_client import ElevenLabsRealtimeClient, SttCallbacks
from .metrics import PipelineMetrics

log = logging.getLogger(__name__)

IDLE = "idle"
LISTENING = "listening"
PAUSED = "paused"
STOPPING = "stopping"


@dataclass
class PipelineEvents:
    """UI hooks.  Every one is called on the pipeline loop thread."""

    on_state: Callable[[str], None] | None = None
    on_provider_state: Callable[[str, str], None] | None = None   # (provider, state)
    on_partial_transcript: Callable[[str], None] | None = None
    on_committed_transcript: Callable[[str], None] | None = None
    on_question: Callable[[str], None] | None = None
    on_answer_start: Callable[[str], None] | None = None
    on_answer_delta: Callable[[str], None] | None = None
    on_answer_done: Callable[[AnswerResult], None] | None = None
    on_error: Callable[[str, str], None] | None = None            # (provider, message)
    on_audio_level: Callable[[float], None] | None = None
    on_log: Callable[[str], None] | None = None


@dataclass
class SessionState:
    state: str = IDLE
    stt_state: str = "disconnected"
    llm_state: str = "disconnected"
    current_question: str = ""
    last_answer: str = ""
    committed_lines: list[str] = field(default_factory=list)


class CopilotPipeline:
    """Owns the background event loop and the full listen/answer cycle."""

    def __init__(
        self,
        settings: Settings,
        events: PipelineEvents | None = None,
        credentials: CredentialStore | None = None,
    ) -> None:
        self.settings = settings
        self.events = events or PipelineEvents()
        self.credentials = credentials or CredentialStore()
        self.metrics = PipelineMetrics()
        self.session = SessionState()
        self.profile = CandidateProfile.build()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

        self._stt: ElevenLabsRealtimeClient | None = None
        self._capture: LoopbackCapture | None = None
        self._generator: AnswerGenerator | None = None
        self._generator_key: str | None = None

        self._detector = QuestionDetector(settings.question)
        self._flush_task: asyncio.Task | None = None
        self._answer_task: asyncio.Task | None = None
        self._answered: set[str] = set()
        self._last_question: str = ""

    # =====================================================================
    # Loop lifecycle
    # =====================================================================
    def start(self) -> None:
        """Spin up the background event loop (safe to call more than once)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, name="copilot-loop", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Pipeline event loop failed to start")

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The background event loop, once `start()` has run."""
        return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def shutdown(self, timeout: float = 8.0) -> None:
        """Stop listening, close providers and stop the loop."""
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(Exception):
            future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), loop)
            future.result(timeout=timeout)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._loop = None
        self._thread = None

    async def _shutdown_async(self) -> None:
        await self._stop_listening_async()
        # Stopping listening deliberately lets a streaming answer finish, but on
        # shutdown the client is about to close underneath it, so end it first.
        task, self._answer_task = self._answer_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._generator is not None:
            await self._generator.aclose()
            self._generator = None
            self._generator_key = None

    def _submit(self, coro) -> "asyncio.Future":
        loop = self._loop
        if loop is None:
            raise RuntimeError("Pipeline is not running")
        return asyncio.run_coroutine_threadsafe(coro, loop)

    # =====================================================================
    # Emitting to the UI
    # =====================================================================
    def _emit(self, hook: Callable | None, *args) -> None:
        if hook is None:
            return
        try:
            hook(*args)
        except Exception:  # a UI slot must never break the pipeline
            log.exception("UI callback failed")

    def _set_state(self, state: str) -> None:
        if self.session.state != state:
            self.session.state = state
            self._emit(self.events.on_state, state)

    def _set_provider_state(self, provider: str, state: str) -> None:
        if provider == "elevenlabs":
            if self.session.stt_state == state:
                return
            self.session.stt_state = state
        else:
            if self.session.llm_state == state:
                return
            self.session.llm_state = state
        self._emit(self.events.on_provider_state, provider, state)

    def _error(self, provider: str, message: str) -> None:
        log.error("[%s] %s", provider, message)
        self._emit(self.events.on_error, provider, message)

    def _log(self, message: str) -> None:
        log.info(message)
        self._emit(self.events.on_log, message)

    # =====================================================================
    # Documents
    # =====================================================================
    def set_profile(self, profile: CandidateProfile) -> None:
        self.profile = profile
        self._log(f"Sənədlər yeniləndi: {profile.describe()}")

    # =====================================================================
    # Listening controls
    # =====================================================================
    def start_listening(self) -> "asyncio.Future":
        return self._submit(self._start_listening_async())

    def pause_listening(self) -> "asyncio.Future":
        return self._submit(self._pause_listening_async())

    def resume_listening(self) -> "asyncio.Future":
        return self._submit(self._resume_listening_async())

    def stop_listening(self) -> "asyncio.Future":
        return self._submit(self._stop_listening_async())

    async def _start_listening_async(self) -> None:
        if self.session.state == LISTENING:
            return
        if self.session.state == PAUSED:
            await self._resume_listening_async()
            return

        api_key = self.credentials.get("elevenlabs")
        if not api_key:
            self._error("elevenlabs", msg.KEY_MISSING_ELEVENLABS)
            return

        # Warm the OpenAI client now so no client setup sits between the
        # finalised question and the first answer token.
        self._ensure_generator(report=True)

        self._detector.reset()
        self._detector.clear_history()
        self._answered.clear()
        self.session.committed_lines.clear()

        sample_rate = self.settings.audio.target_sample_rate
        self._stt = ElevenLabsRealtimeClient(
            api_key=api_key,
            settings=self.settings.stt,
            sample_rate=sample_rate,
            callbacks=SttCallbacks(
                on_partial=self._on_partial,
                on_committed=self._on_committed,
                on_state=lambda state: self._set_provider_state("elevenlabs", state),
                on_error=lambda message, recoverable: self._error("elevenlabs", message),
            ),
        )
        await self._stt.start()

        try:
            self._capture = LoopbackCapture(
                on_chunk=self._on_audio_chunk,
                device_index=self.settings.audio.device_index,
                target_sample_rate=sample_rate,
                chunk_ms=self.settings.audio.chunk_ms,
                gain=self.settings.audio.input_gain,
                on_error=self._on_capture_error,
            )
            device = await asyncio.to_thread(self._capture.start)
        except AudioCaptureError as exc:
            self._capture = None
            await self._stop_stt()
            self._error("audio", str(exc))
            self._set_state(IDLE)
            return

        self._log(f"Səs mənbəyi: {device.name} ({device.sample_rate} Hz)")
        self._flush_task = asyncio.create_task(self._flush_loop(), name="question-flush")
        self._set_state(LISTENING)

    async def _pause_listening_async(self) -> None:
        if self.session.state != LISTENING:
            return
        if self._capture is not None:
            self._capture.pause()
        self._set_state(PAUSED)
        self._log("Dinləmə dayandırıldı")

    async def _resume_listening_async(self) -> None:
        if self.session.state != PAUSED:
            return
        if self._capture is not None:
            self._capture.resume()
        self._set_state(LISTENING)
        self._log("Dinləmə davam edir")

    async def _stop_listening_async(self) -> None:
        if self.session.state in (IDLE, STOPPING):
            if self._capture is None and self._stt is None:
                return
        self._set_state(STOPPING)

        # 1. stop Windows audio capture
        capture, self._capture = self._capture, None
        if capture is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(capture.stop)

        # 2 + 3. stop transmitting and close the ElevenLabs connection
        await self._stop_stt()

        # 4. release the question timer and any in-flight generation
        task, self._flush_task = self._flush_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        for question in self._detector.force_flush():
            self._handle_question(question)

        self._set_provider_state("elevenlabs", "disconnected")
        self._set_state(IDLE)
        self._log("Dinləmə tamamlandı")

    async def _stop_stt(self) -> None:
        stt, self._stt = self._stt, None
        if stt is not None:
            with contextlib.suppress(Exception):
                await stt.stop()

    def _on_capture_error(self, exc: Exception) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._error, "audio", f"{msg.AUDIO_DEVICE_LOST} ({exc})")

    # =====================================================================
    # Audio -> ElevenLabs
    # =====================================================================
    def _on_audio_chunk(self, pcm: bytes, level: float) -> None:
        """Called on the PortAudio thread; hop to the loop, do nothing heavy."""
        loop = self._loop
        if loop is None:
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._forward_audio, pcm, level)

    def _forward_audio(self, pcm: bytes, level: float) -> None:
        stt = self._stt
        if stt is not None:
            stt.push_audio(pcm)
            self.metrics.note_audio_sent()
        self._emit(self.events.on_audio_level, level)

    # =====================================================================
    # Transcription -> questions
    # =====================================================================
    def _on_partial(self, text: str) -> None:
        self.metrics.note_partial()
        self._emit(self.events.on_partial_transcript, text)

    def _on_committed(self, text: str) -> None:
        self.metrics.note_commit()
        self.session.committed_lines.append(text)
        del self.session.committed_lines[:-200]
        self._emit(self.events.on_committed_transcript, text)

        before = self._detector.skipped_duplicates
        for question in self._detector.add_committed(text):
            self._handle_question(question)
        self.metrics.duplicates_skipped += self._detector.skipped_duplicates - before

    async def _flush_loop(self) -> None:
        """Fires questions whose coalescing window has elapsed."""
        try:
            while True:
                await asyncio.sleep(0.1)
                for question in self._detector.flush_due():
                    self._handle_question(question)
        except asyncio.CancelledError:
            raise

    def _handle_question(self, question: DetectedQuestion) -> None:
        key = normalized_question_key(question.text)
        if key in self._answered:
            self.metrics.duplicates_skipped += 1
            return
        self._answered.add(key)
        self.session.current_question = question.text
        self._last_question = question.text
        self.metrics.note_question(key, question.detected_at)
        self._log(f"Sual aşkarlandı ({question.reason}): {question.text}")
        self._emit(self.events.on_question, question.text)

        if self.settings.llm.auto_generate:
            self.generate_answer(question.text)

    # =====================================================================
    # Answer generation
    # =====================================================================
    def _ensure_generator(self, report: bool = False) -> AnswerGenerator | None:
        api_key = self.credentials.get("openai")
        if not api_key:
            if report:
                self._error("openai", msg.KEY_MISSING_OPENAI)
            self._set_provider_state("openai", "disconnected")
            return None
        if self._generator is None or self._generator_key != api_key:
            try:
                self._generator = AnswerGenerator(api_key, self.settings.llm)
                self._generator_key = api_key
            except LlmError as exc:
                self._error("openai", exc.user_message)
                self._set_provider_state("openai", "error")
                return None
        self._generator.settings = self.settings.llm
        self._set_provider_state("openai", "ready")
        return self._generator

    def generate_answer(self, question: str | None = None, mode: str = "answer") -> None:
        """Start (or restart) answer generation.

        With no explicit question this answers the last *detected* one, and
        failing that whatever the interviewer last said.  Detection can only
        ever be a heuristic, so pressing the button must always do something
        rather than silently ignoring a prompt the classifier did not like.
        """
        target = (question or self._last_question or self.last_committed_text()).strip()
        if not target:
            return
        self._submit(self._generate_async(target, mode))

    def last_committed_text(self, max_segments: int = 3) -> str:
        """The most recent committed speech, whether or not it was detected.

        Several trailing segments are joined because the recogniser commits on
        silence, so the prompt the user wants answered may have been split.
        """
        lines = [line.strip() for line in self.session.committed_lines if line.strip()]
        if not lines:
            return ""
        recent = lines[-max_segments:]
        # Keep only the tail that is not already an answered question.
        while len(recent) > 1 and normalized_question_key(" ".join(recent)) in self._answered:
            recent = recent[1:]
        return " ".join(recent).strip()

    def regenerate(self) -> None:
        self.generate_answer(self._last_question, mode="regenerate")

    def expand(self) -> None:
        self.generate_answer(self._last_question, mode="expand")

    def cancel_generation(self) -> None:
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._cancel_answer_task)

    def _cancel_answer_task(self) -> None:
        task, self._answer_task = self._answer_task, None
        if task is not None and not task.done():
            task.cancel()

    async def _generate_async(self, question: str, mode: str) -> None:
        # The interviewer has moved on: drop any answer still streaming.
        self._cancel_answer_task()
        self._answer_task = asyncio.create_task(
            self._stream_answer(question, mode), name="answer"
        )
        with contextlib.suppress(asyncio.CancelledError):
            await self._answer_task

    async def _stream_answer(self, question: str, mode: str) -> None:
        generator = self._ensure_generator(report=True)
        if generator is None:
            return

        key = normalized_question_key(question)
        self._last_question = question
        self._emit(self.events.on_answer_start, question)
        self._set_provider_state("openai", "generating")
        first = True
        started = time.perf_counter()

        try:
            async for item in generator.stream_answer(
                question, profile=self.profile, mode=mode
            ):
                if isinstance(item, AnswerResult):
                    generator.remember(question, item.text)
                    self.session.last_answer = item.text
                    self.metrics.note_answer_done(key, item.total_ms)
                    self._set_provider_state("openai", "ready")
                    self._emit(self.events.on_answer_done, item)
                    return
                if first:
                    first = False
                    self.metrics.note_first_token(key)
                self._emit(self.events.on_answer_delta, item)
        except asyncio.CancelledError:
            self._set_provider_state("openai", "ready")
            raise
        except LlmError as exc:
            self.metrics.note_answer_failed(key)
            self._set_provider_state("openai", "error" if not exc.recoverable else "ready")
            log.warning("Answer generation failed after %.0f ms: %s",
                        (time.perf_counter() - started) * 1000.0, exc.detail)
            self._error("openai", exc.user_message)
        except Exception as exc:  # pragma: no cover - defensive
            self.metrics.note_answer_failed(key)
            self._set_provider_state("openai", "error")
            log.exception("Unexpected answer generation failure")
            self._error("openai", f"{msg.LLM_FAILED} ({exc})")

    # =====================================================================
    # Introspection for the developer panel
    # =====================================================================
    def stt_debug(self) -> dict[str, object]:
        stt = self._stt
        if stt is None:
            return {"state": "disconnected"}
        return {
            "state": stt.state,
            "session_id": stt.session_id,
            "chunks_sent": stt.stats.chunks_sent,
            "bytes_sent": stt.stats.bytes_sent,
            "partials": stt.stats.partials,
            "commits": stt.stats.commits,
            "reconnects": stt.stats.reconnects,
            "dropped_chunks": stt.dropped_chunks,
        }
