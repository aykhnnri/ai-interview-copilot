"""Model benchmark: measure, do not guess, which model is fastest here.

Every model sees the identical question bank with the identical output-token
budget, so response lengths stay comparable.  Measured per question:

* time to first output token
* time to complete response
* total response length (characters and output tokens)
* Azerbaijani language quality (and Turkish drift)
* technical answer correctness, as expected-concept coverage
* approximate API cost

Benchmarks call the live OpenAI API and therefore cost money; nothing here runs
unless the user explicitly starts it.  The result never changes the active model
on its own - the UI only ever recommends.
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ..config import LlmSettings
from ..documents.profile import CandidateProfile
from ..paths import benchmarks_dir
from .benchmark_questions import QUESTIONS, BenchmarkQuestion
from .models import get_spec
from .openai_client import AnswerGenerator, LlmError
from .quality import score_answer

log = logging.getLogger(__name__)

ProgressHook = Callable[[str, int, int], None]
"""progress(model_id, completed, total)"""


@dataclass
class QuestionResult:
    question_id: str
    question: str
    ok: bool
    first_token_ms: float | None = None
    total_ms: float | None = None
    char_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    azerbaijani: float = 0.0
    english_preservation: float = 0.0
    coverage: float = 0.0
    overall: float = 0.0
    turkish_hits: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    error: str = ""
    answer_preview: str = ""
    retries: int = 0


@dataclass
class ModelBenchmark:
    model: str
    label: str
    results: list[QuestionResult] = field(default_factory=list)

    # -- aggregates -------------------------------------------------------
    @property
    def successes(self) -> list[QuestionResult]:
        return [r for r in self.results if r.ok]

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def failure_rate(self) -> float:
        return self.failure_count / len(self.results) if self.results else 0.0

    @property
    def transient_retries(self) -> int:
        """Transport hiccups that were retried; a health signal, not a score."""
        return sum(r.retries for r in self.results)

    def _median(self, attribute: str) -> float | None:
        values = [
            getattr(r, attribute) for r in self.successes if getattr(r, attribute) is not None
        ]
        return round(statistics.median(values), 1) if values else None

    def _mean(self, attribute: str) -> float | None:
        values = [
            getattr(r, attribute) for r in self.successes if getattr(r, attribute) is not None
        ]
        return round(statistics.fmean(values), 3) if values else None

    @property
    def median_first_token_ms(self) -> float | None:
        return self._median("first_token_ms")

    @property
    def p95_first_token_ms(self) -> float | None:
        values = sorted(r.first_token_ms for r in self.successes if r.first_token_ms is not None)
        if not values:
            return None
        index = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
        return round(values[index], 1)

    @property
    def median_total_ms(self) -> float | None:
        return self._median("total_ms")

    @property
    def mean_chars(self) -> float | None:
        return self._mean("char_count")

    @property
    def mean_azerbaijani(self) -> float | None:
        return self._mean("azerbaijani")

    @property
    def mean_coverage(self) -> float | None:
        return self._mean("coverage")

    @property
    def mean_english(self) -> float | None:
        return self._mean("english_preservation")

    @property
    def mean_overall(self) -> float | None:
        return self._mean("overall")

    @property
    def total_cost_usd(self) -> float | None:
        costs = [r.cost_usd for r in self.successes if r.cost_usd is not None]
        return round(sum(costs), 6) if costs else None

    @property
    def cost_per_answer_usd(self) -> float | None:
        total = self.total_cost_usd
        return round(total / len(self.successes), 6) if total is not None and self.successes else None

    def summary_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "label": self.label,
            "answers": len(self.successes),
            "failures": self.failure_count,
            "retries": self.transient_retries,
            "ttft_median_ms": self.median_first_token_ms,
            "ttft_p95_ms": self.p95_first_token_ms,
            "total_median_ms": self.median_total_ms,
            "mean_chars": self.mean_chars,
            "azerbaijani": self.mean_azerbaijani,
            "english_terms": self.mean_english,
            "coverage": self.mean_coverage,
            "overall_quality": self.mean_overall,
            "cost_per_answer_usd": self.cost_per_answer_usd,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class BenchmarkReport:
    started_at: str
    finished_at: str
    question_count: int
    max_output_tokens: int
    models: list[ModelBenchmark] = field(default_factory=list)
    used_profile: bool = False

    # -- recommendation ---------------------------------------------------
    def recommended(
        self, min_quality: float = 0.55, max_failure_rate: float = 0.1
    ) -> ModelBenchmark | None:
        """Fastest model whose answer quality clears the bar.

        Speed alone is not enough - a model that answers in Turkish or misses
        the concept is not a usable interview assistant.  A small share of
        failures is tolerated because a dropped connection mid-run says nothing
        about the model; a model failing often is excluded.
        """
        eligible = [
            benchmark for benchmark in self.models
            if benchmark.successes
            and benchmark.failure_rate <= max_failure_rate
            and (benchmark.mean_overall or 0) >= min_quality
            and benchmark.median_first_token_ms is not None
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda b: b.median_first_token_ms)  # type: ignore[arg-type]

    def best_quality(self) -> ModelBenchmark | None:
        """Highest-scoring model, regardless of speed.

        Reported next to the speed recommendation because the two often
        disagree, and which one matters is the user's call, not the tool's.
        """
        scored = [b for b in self.models if b.successes and b.mean_overall is not None]
        if not scored:
            return None
        return max(scored, key=lambda b: b.mean_overall)  # type: ignore[arg-type]

    def most_consistent(self) -> ModelBenchmark | None:
        """Lowest p95 time-to-first-token - the least likely to visibly stall."""
        scored = [b for b in self.models if b.p95_first_token_ms is not None]
        if not scored:
            return None
        return min(scored, key=lambda b: b.p95_first_token_ms)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "question_count": self.question_count,
            "max_output_tokens": self.max_output_tokens,
            "used_profile": self.used_profile,
            "summary": [benchmark.summary_row() for benchmark in self.models],
            "recommended": (self.recommended().model if self.recommended() else None),
            "best_quality": (self.best_quality().model if self.best_quality() else None),
            "most_consistent": (self.most_consistent().model if self.most_consistent() else None),
            "details": {
                benchmark.model: [asdict(result) for result in benchmark.results]
                for benchmark in self.models
            },
        }

    def save(self, directory: Path | None = None) -> Path:
        target_dir = directory or benchmarks_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = target_dir / f"benchmark-{stamp}.json"
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def to_text(self) -> str:
        lines = [
            f"Benchmark: {self.question_count} sual, max_output_tokens={self.max_output_tokens}",
            f"Başlama: {self.started_at}  Bitmə: {self.finished_at}",
            "",
        ]
        header = (
            f"{'Model':<16}{'Cavab':>6}{'Xəta':>6}{'TTFT med':>10}{'TTFT p95':>10}"
            f"{'Tam med':>9}{'Simvol':>8}{'AZ':>6}{'EN':>6}{'Əhatə':>7}{'Ümumi':>7}{'Xərc/cavab':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for benchmark in self.models:
            row = benchmark.summary_row()
            lines.append(
                f"{row['label']:<16}{row['answers']:>6}{row['failures']:>6}"
                f"{_ms(row['ttft_median_ms']):>10}{_ms(row['ttft_p95_ms']):>10}"
                f"{_ms(row['total_median_ms']):>9}{_num(row['mean_chars'], 0):>8}"
                f"{_num(row['azerbaijani'], 2):>6}{_num(row['english_terms'], 2):>6}"
                f"{_num(row['coverage'], 2):>7}{_num(row['overall_quality'], 2):>7}"
                f"{_usd(row['cost_per_answer_usd']):>12}"
            )
        lines.append("")
        best = self.recommended()
        if best is not None:
            lines.append(f"Ən sürətli (median TTFT): {best.label}")
        else:
            lines.append("Sürət tövsiyəsi yoxdur: heç bir model keyfiyyət həddini keçmədi.")

        quality = self.best_quality()
        if quality is not None:
            lines.append(
                f"Ən yüksək keyfiyyət:      {quality.label} "
                f"(əhatə {_num(quality.mean_coverage, 2)})"
            )
        steady = self.most_consistent()
        if steady is not None:
            lines.append(
                f"Ən sabit (p95 TTFT):      {steady.label} "
                f"({_ms(steady.p95_first_token_ms)})"
            )

        if best is not None and quality is not None and best.model != quality.model:
            lines.append("")
            lines.append(
                "Sürət və keyfiyyət fərqli modellərə işarə edir - seçim sizindir."
            )
        lines.append("")
        lines.append("Model avtomatik dəyişmir - Settings → Cavab bölməsindən seçin.")
        lines.append(
            "Qeyd: AZ/Əhatə balları heuristikdir və insan yoxlamasını əvəz etmir."
        )
        return "\n".join(lines)


def _ms(value: object) -> str:
    return "-" if value is None else f"{float(value):.0f}ms"


def _num(value: object, digits: int) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _usd(value: object) -> str:
    return "-" if value is None else f"${float(value):.5f}"


async def run_benchmark(
    api_key: str,
    models: Sequence[str],
    questions: Sequence[BenchmarkQuestion] | None = None,
    settings: LlmSettings | None = None,
    profile: CandidateProfile | None = None,
    max_output_tokens: int = 500,
    progress: ProgressHook | None = None,
    delay_between_calls: float = 0.15,
) -> BenchmarkReport:
    """Run every model over the identical question bank."""
    bank = list(questions if questions is not None else QUESTIONS)
    if not bank:
        raise ValueError("benchmark question bank is empty")

    base = settings or LlmSettings()
    # Identical budget and no conversation history, so runs are comparable.
    bench_settings = LlmSettings(
        model=base.model,
        base_url=base.base_url,
        max_output_tokens=max_output_tokens,
        expand_max_output_tokens=max_output_tokens,
        temperature=base.temperature,
        request_timeout_s=max(base.request_timeout_s, 60.0),
        history_turns=0,
        service_tier=base.service_tier,
    )

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    generator = AnswerGenerator(api_key, bench_settings)
    report_models: list[ModelBenchmark] = []

    try:
        for model_id in models:
            spec = get_spec(model_id)
            benchmark = ModelBenchmark(model=spec.id, label=spec.label)
            log.info("Benchmarking %s over %d questions", spec.id, len(bank))

            for index, question in enumerate(bank, start=1):
                generator.clear_history()
                benchmark.results.append(
                    await _run_one(generator, question, spec.id, profile)
                )
                if progress is not None:
                    progress(spec.id, index, len(bank))
                if delay_between_calls:
                    await asyncio.sleep(delay_between_calls)

            report_models.append(benchmark)
    finally:
        await generator.aclose()

    return BenchmarkReport(
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        question_count=len(bank),
        max_output_tokens=max_output_tokens,
        models=report_models,
        used_profile=profile is not None and profile.has_documents,
    )


async def _run_one(
    generator: AnswerGenerator,
    question: BenchmarkQuestion,
    model_id: str,
    profile: CandidateProfile | None,
) -> QuestionResult:
    """Ask one question and score the answer.

    Transport drops are already retried inside `AnswerGenerator` (only before
    any token is emitted), so there is no second retry layer here - the
    benchmark just records how many reopens it took, because a model served
    over a flaky path is worth seeing.  Rate limits and model errors are *not*
    retried anywhere: those are real results.
    """
    retries_before = generator.stream_retries
    started = time.perf_counter()
    try:
        result = await generator.generate(
            question.text, profile=profile, model=model_id, use_history=False
        )
    except LlmError as exc:
        return QuestionResult(
            question_id=question.id, question=question.text, ok=False,
            total_ms=(time.perf_counter() - started) * 1000.0,
            error=f"{exc.user_message} ({exc.detail})".strip(),
            retries=generator.stream_retries - retries_before,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return QuestionResult(
            question_id=question.id, question=question.text, ok=False,
            total_ms=(time.perf_counter() - started) * 1000.0, error=str(exc),
            retries=generator.stream_retries - retries_before,
        )

    scores = score_answer(result.text, question.expected_terms, question.english_terms)
    return QuestionResult(
        question_id=question.id,
        question=question.text,
        ok=True,
        first_token_ms=result.first_token_ms,
        total_ms=result.total_ms,
        char_count=result.char_count,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        azerbaijani=scores.azerbaijani,
        english_preservation=scores.english_preservation,
        coverage=scores.coverage,
        overall=scores.overall,
        turkish_hits=list(scores.turkish_hits),
        missing_terms=list(scores.missing_terms),
        answer_preview=result.text[:240],
        retries=generator.stream_retries - retries_before,
    )


def load_latest_report(directory: Path | None = None) -> dict[str, object] | None:
    """Most recent saved benchmark, for showing in the developer panel."""
    target = directory or benchmarks_dir()
    files = sorted(target.glob("benchmark-*.json"), reverse=True)
    for path in files:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None
