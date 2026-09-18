"""Adapter between the pipeline (asyncio thread) and Qt (GUI thread).

Pipeline callbacks fire on the background loop.  Touching widgets from there
would be a crash, so every callback becomes a Qt signal emission; Qt queues it
onto the GUI thread automatically because the receiver lives there.
"""
from __future__ import annotations

import asyncio
from typing import Any, Coroutine

from PySide6.QtCore import QObject, Signal

from ..core.pipeline import CopilotPipeline, PipelineEvents
from ..llm.openai_client import AnswerResult


class PipelineBridge(QObject):
    """Qt-side view of one `CopilotPipeline`."""

    state_changed = Signal(str)
    provider_state_changed = Signal(str, str)
    partial_transcript = Signal(str)
    committed_transcript = Signal(str)
    question_detected = Signal(str)
    answer_started = Signal(str)
    answer_delta = Signal(str)
    answer_finished = Signal(object)     # AnswerResult
    error_raised = Signal(str, str)      # provider, message
    audio_level = Signal(float)
    log_line = Signal(str)

    #: Generic completion channel for one-off coroutines (connection tests,
    #: benchmark runs).  Carries (tag, ok, payload).
    task_finished = Signal(str, bool, object)

    def __init__(self, pipeline: CopilotPipeline, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        pipeline.events = PipelineEvents(
            on_state=self.state_changed.emit,
            on_provider_state=self.provider_state_changed.emit,
            on_partial_transcript=self.partial_transcript.emit,
            on_committed_transcript=self.committed_transcript.emit,
            on_question=self.question_detected.emit,
            on_answer_start=self.answer_started.emit,
            on_answer_delta=self.answer_delta.emit,
            on_answer_done=self._emit_answer,
            on_error=self.error_raised.emit,
            on_audio_level=self.audio_level.emit,
            on_log=self.log_line.emit,
        )

    def _emit_answer(self, result: AnswerResult) -> None:
        self.answer_finished.emit(result)

    # -- running coroutines from the UI -----------------------------------
    def run_task(self, tag: str, coro: Coroutine[Any, Any, Any]) -> None:
        """Run `coro` on the pipeline loop; the result arrives as `task_finished`.

        Results are delivered only through the signal, so handlers always run on
        the GUI thread.  `tag` lets several tasks share the one channel.
        """
        loop = self.pipeline.loop
        if loop is None:
            self.task_finished.emit(tag, False, RuntimeError("Pipeline is not running"))
            return

        future = asyncio.run_coroutine_threadsafe(coro, loop)

        def _done(fut) -> None:
            try:
                value, ok = fut.result(), True
            except Exception as exc:  # surfaced to the UI, never swallowed
                value, ok = exc, False
            self.task_finished.emit(tag, ok, value)

        future.add_done_callback(_done)
