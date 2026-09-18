"""Latency bookkeeping for the capture -> transcript -> question -> answer path.

Every stage records a monotonic timestamp so the developer panel can show where
the time actually goes, rather than a single end-to-end number.
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Samples:
    """A bounded window of millisecond samples."""

    name: str
    values: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def add(self, ms: float) -> None:
        if ms >= 0:
            self.values.append(float(ms))

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def last(self) -> float | None:
        return self.values[-1] if self.values else None

    @property
    def median(self) -> float | None:
        return statistics.median(self.values) if self.values else None

    @property
    def p95(self) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[index]

    def summary(self) -> str:
        if not self.values:
            return f"{self.name}: -"
        return (
            f"{self.name}: son {self.last:.0f} ms | median {self.median:.0f} ms "
            f"| p95 {self.p95:.0f} ms | n={self.count}"
        )


@dataclass
class PipelineMetrics:
    """The four latencies the spec asks to be measured, plus counters."""

    audio_to_partial: Samples = field(default_factory=lambda: Samples("Səs → ilkin transkript"))
    audio_to_commit: Samples = field(default_factory=lambda: Samples("Səs → yekun transkript"))
    question_finalize: Samples = field(default_factory=lambda: Samples("Sual yekunlaşma"))
    question_to_first_token: Samples = field(default_factory=lambda: Samples("Sual → ilk token"))
    answer_total: Samples = field(default_factory=lambda: Samples("Tam cavab müddəti"))

    questions_detected: int = 0
    answers_generated: int = 0
    answers_failed: int = 0
    duplicates_skipped: int = 0

    _last_audio_at: float | None = None
    _question_started: dict[str, float] = field(default_factory=dict)

    # -- capture ----------------------------------------------------------
    def note_audio_sent(self) -> None:
        self._last_audio_at = time.perf_counter()

    def note_partial(self) -> None:
        if self._last_audio_at is not None:
            self.audio_to_partial.add((time.perf_counter() - self._last_audio_at) * 1000.0)

    def note_commit(self) -> None:
        if self._last_audio_at is not None:
            self.audio_to_commit.add((time.perf_counter() - self._last_audio_at) * 1000.0)

    # -- question ---------------------------------------------------------
    def note_question(self, key: str, detected_at: float | None = None) -> None:
        self.questions_detected += 1
        now = time.perf_counter()
        self._question_started[key] = now
        if detected_at is not None:
            # `detected_at` is monotonic-clock based (time.monotonic) from the
            # detector; convert the gap rather than the absolute value.
            gap = (time.monotonic() - detected_at) * 1000.0
            if 0 <= gap < 60_000:
                self.question_finalize.add(gap)

    def note_first_token(self, key: str) -> None:
        started = self._question_started.get(key)
        if started is not None:
            self.question_to_first_token.add((time.perf_counter() - started) * 1000.0)

    def note_answer_done(self, key: str, total_ms: float | None = None) -> None:
        self.answers_generated += 1
        started = self._question_started.pop(key, None)
        if total_ms is not None:
            self.answer_total.add(total_ms)
        elif started is not None:
            self.answer_total.add((time.perf_counter() - started) * 1000.0)

    def note_answer_failed(self, key: str) -> None:
        self.answers_failed += 1
        self._question_started.pop(key, None)

    # -- reporting --------------------------------------------------------
    def reset(self) -> None:
        for samples in self.all_samples():
            samples.values.clear()
        self.questions_detected = self.answers_generated = 0
        self.answers_failed = self.duplicates_skipped = 0
        self._question_started.clear()
        self._last_audio_at = None

    def all_samples(self) -> list[Samples]:
        return [
            self.audio_to_partial,
            self.audio_to_commit,
            self.question_finalize,
            self.question_to_first_token,
            self.answer_total,
        ]

    def report(self) -> str:
        lines = [samples.summary() for samples in self.all_samples()]
        lines.append(
            f"Suallar: {self.questions_detected} | Cavablar: {self.answers_generated} "
            f"| Xətalar: {self.answers_failed} | Təkrarlar: {self.duplicates_skipped}"
        )
        return "\n".join(lines)
