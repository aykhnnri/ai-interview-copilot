"""The model benchmark: comparability, scoring, reporting, recommendation."""
from __future__ import annotations

import json

import pytest

from copilot.config import LlmSettings
from copilot.llm.benchmark import (
    BenchmarkReport,
    ModelBenchmark,
    QuestionResult,
    load_latest_report,
    run_benchmark,
)
from copilot.llm.benchmark_questions import QUESTIONS, categories, question_by_id

SPEC_QUESTIONS = [
    "RAG nədir?",
    "Fine-tuning və RAG arasında əsas fərq nədir?",
    "Multi-agent sistemlərdə agentlər arasında kommunikasiya necə qurulur?",
    "Production mühitində LLM hallucination problemini necə azaldarsınız?",
    "Python-da async və await nədir?",
    "Vector database necə işləyir?",
    "LLM tətbiqlərində latency-ni necə optimizasiya edərdiniz?",
    "Modelin performansını hansı metriklərlə qiymətləndirərdiniz?",
    "Sisteminizdə məlumatların təhlükəsizliyini necə təmin edərdiniz?",
    "RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?",
]


# =========================================================================
# Question bank
# =========================================================================
def test_bank_has_at_least_twenty_questions():
    assert len(QUESTIONS) >= 20


def test_bank_contains_every_specified_question():
    texts = [q.text for q in QUESTIONS]
    for question in SPEC_QUESTIONS:
        assert question in texts, f"missing from the bank: {question}"


def test_question_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_questions_carry_expected_concepts():
    assert all(q.expected_terms for q in QUESTIONS)


def test_bank_covers_several_categories():
    assert len(categories()) >= 5


def test_question_lookup():
    assert question_by_id("q01").text == "RAG nədir?"
    assert question_by_id("nope") is None


# =========================================================================
# Running
# =========================================================================
async def test_benchmark_asks_every_model_the_same_questions(fake_openai):
    client = fake_openai(["RAG ", "retrieval ", "və generasiya deməkdir."])
    report = await run_benchmark(
        api_key="sk-test",
        models=["gpt-4.1-nano", "gpt-5.6-luna"],
        questions=QUESTIONS[:3],
        max_output_tokens=300,
        delay_between_calls=0,
    )

    assert len(report.models) == 2
    assert all(len(benchmark.results) == 3 for benchmark in report.models)

    # Same questions, same output budget - otherwise the comparison is meaningless.
    by_model: dict[str, list[str]] = {}
    for request in client.responses.requests:
        by_model.setdefault(request["model"], []).append(str(request["input"][-1]["content"]))
    assert list(by_model["gpt-4.1-nano"]) == list(by_model["gpt-5.6-luna"])
    assert {request["max_output_tokens"] for request in client.responses.requests} == {300}


async def test_benchmark_measures_the_required_dimensions(fake_openai):
    fake_openai(["Retrieval ", "quality-ni Recall@k ", "ilə ölçürəm."])
    report = await run_benchmark(
        api_key="sk-test", models=["gpt-4.1-nano"],
        questions=QUESTIONS[:2], delay_between_calls=0,
    )
    result = report.models[0].results[0]

    assert result.ok
    assert result.first_token_ms is not None       # time to first token
    assert result.total_ms is not None             # time to complete
    assert result.char_count > 0                   # response length
    assert 0.0 <= result.azerbaijani <= 1.0        # language quality
    assert 0.0 <= result.coverage <= 1.0           # technical correctness
    assert result.cost_usd is not None             # approximate cost


async def test_benchmark_carries_no_conversation_history_between_questions(fake_openai):
    client = fake_openai(["cavab"])
    await run_benchmark(
        api_key="sk-test", models=["gpt-4.1-nano"],
        questions=QUESTIONS[:3], delay_between_calls=0,
    )
    # Each request holds exactly one user message: the question itself.
    for request in client.responses.requests:
        assert len(request["input"]) == 1


async def test_a_failing_question_is_recorded_not_swallowed(fake_openai):
    error = type("RateLimitError", (Exception,), {})("429")
    error.status_code = 429
    fake_openai(error=error)

    report = await run_benchmark(
        api_key="sk-test", models=["gpt-4.1-nano"],
        questions=QUESTIONS[:2], delay_between_calls=0,
    )
    benchmark = report.models[0]
    assert benchmark.failure_count == 2
    assert benchmark.successes == []
    assert all(result.error for result in benchmark.results)


async def test_benchmark_rejects_an_empty_bank():
    with pytest.raises(ValueError):
        await run_benchmark(api_key="sk-test", models=["gpt-4.1-nano"], questions=[])


# =========================================================================
# Aggregation and recommendation
# =========================================================================
def _benchmark(model: str, ttft: float, quality: float, failures: int = 0) -> ModelBenchmark:
    benchmark = ModelBenchmark(model=model, label=model)
    for index in range(5):
        benchmark.results.append(
            QuestionResult(
                question_id=f"q{index}", question="x", ok=True,
                first_token_ms=ttft, total_ms=ttft * 3, char_count=400,
                input_tokens=100, output_tokens=120, cost_usd=0.0001,
                azerbaijani=quality, english_preservation=quality,
                coverage=quality, overall=quality,
            )
        )
    for index in range(failures):
        benchmark.results.append(
            QuestionResult(question_id=f"f{index}", question="x", ok=False, error="boom")
        )
    return benchmark


def _report(*benchmarks: ModelBenchmark) -> BenchmarkReport:
    return BenchmarkReport(
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:05:00+00:00",
        question_count=5, max_output_tokens=500, models=list(benchmarks),
    )


def test_aggregates_are_computed():
    benchmark = _benchmark("m", ttft=300.0, quality=0.8)
    assert benchmark.median_first_token_ms == 300.0
    assert benchmark.median_total_ms == 900.0
    assert benchmark.mean_chars == 400.0
    assert benchmark.total_cost_usd == pytest.approx(0.0005)
    assert benchmark.cost_per_answer_usd == pytest.approx(0.0001)


def test_recommendation_picks_the_fastest_model_that_is_good_enough():
    fast_but_bad = _benchmark("fast", ttft=100.0, quality=0.2)
    slower_and_good = _benchmark("good", ttft=400.0, quality=0.9)
    report = _report(fast_but_bad, slower_and_good)
    assert report.recommended().model == "good"


def test_recommendation_prefers_speed_when_quality_is_comparable():
    report = _report(_benchmark("a", 500.0, 0.85), _benchmark("b", 200.0, 0.84))
    assert report.recommended().model == "b"


def test_a_model_with_failures_is_not_recommended():
    report = _report(_benchmark("flaky", 100.0, 0.9, failures=1), _benchmark("solid", 800.0, 0.9))
    assert report.recommended().model == "solid"


def test_no_recommendation_when_nothing_clears_the_bar():
    assert _report(_benchmark("a", 100.0, 0.1)).recommended() is None


def test_report_text_lists_every_model_and_states_the_caveat():
    text = _report(_benchmark("nano", 200.0, 0.9), _benchmark("luna", 300.0, 0.8)).to_text()
    assert "nano" in text and "luna" in text
    assert "Ən sürətli" in text
    assert "heuristik" in text          # the scores do not replace human review


def test_report_serialises_with_azerbaijani_intact(tmp_path):
    report = _report(_benchmark("nano", 200.0, 0.9))
    report.models[0].results[0].question = "RAG nədir?"
    path = report.save(tmp_path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["recommended"] == "nano"
    assert data["details"]["nano"][0]["question"] == "RAG nədir?"
    assert data["summary"][0]["ttft_median_ms"] == 200.0


def test_latest_report_is_loaded_back(tmp_path):
    assert load_latest_report(tmp_path) is None
    _report(_benchmark("nano", 200.0, 0.9)).save(tmp_path)
    assert load_latest_report(tmp_path)["recommended"] == "nano"


def test_unknown_price_reports_no_cost_rather_than_a_guess():
    benchmark = ModelBenchmark(model="mystery", label="Mystery")
    benchmark.results.append(
        QuestionResult(question_id="q", question="x", ok=True, first_token_ms=10.0,
                       total_ms=20.0, char_count=10, cost_usd=None)
    )
    assert benchmark.total_cost_usd is None
    assert "-" in benchmark.summary_row().__str__() or benchmark.cost_per_answer_usd is None


# =========================================================================
# Transient failures must not be blamed on the model
# =========================================================================
class _FlakyResponses:
    """Fails with a transport error `fail_times` times, then succeeds."""

    def __init__(self, fail_times: int, error_name: str = "APIConnectionError"):
        self.remaining = fail_times
        self.error_name = error_name
        self.requests: list[dict] = []
        self.attempts = 0

    async def create(self, **kwargs):
        self.attempts += 1
        self.requests.append(kwargs)
        if self.remaining > 0:
            self.remaining -= 1
            raise type(self.error_name, (Exception,), {})("transport broke")
        from .conftest import FakeEvent, FakeResponse, FakeStream, FakeUsage

        events = [FakeEvent("response.output_text.delta", delta="Azərbaycan cavabı.")]
        events.append(FakeEvent("response.completed", response=FakeResponse(usage=FakeUsage())))
        return FakeStream(events)


def _install_flaky(monkeypatch, responses):
    from copilot.llm import openai_client as module

    class Client:
        def __init__(self):
            self.responses = responses

        async def close(self):
            pass

    def patched_init(self, api_key, settings=None):
        from copilot.config import LlmSettings

        self.settings = settings or LlmSettings()
        self._client = Client()
        self._history = []
        self.service_tier_rejected = None
        self.stream_retries = 0

    monkeypatch.setattr(module.AnswerGenerator, "__init__", patched_init)


async def test_a_dropped_connection_is_retried_not_recorded_as_a_model_failure(monkeypatch):
    """The retry lives in AnswerGenerator, so the app benefits too; the
    benchmark only records how many reopens the answer needed."""
    responses = _FlakyResponses(fail_times=2)
    _install_flaky(monkeypatch, responses)

    report = await run_benchmark(
        api_key="sk-test", models=["gpt-4.1-nano"],
        questions=QUESTIONS[:1], delay_between_calls=0,
    )
    result = report.models[0].results[0]
    assert result.ok, result.error
    assert responses.attempts == 3      # two drops, then a clean stream
    assert result.retries == 2          # and the benchmark says so


async def test_a_persistent_transport_failure_is_still_recorded(monkeypatch):
    responses = _FlakyResponses(fail_times=99)
    _install_flaky(monkeypatch, responses)

    report = await run_benchmark(
        api_key="sk-test", models=["gpt-4.1-nano"],
        questions=QUESTIONS[:1], delay_between_calls=0,
    )
    result = report.models[0].results[0]
    assert not result.ok
    assert responses.attempts == 3      # tried the configured number of times
    assert "AI cavabı yaradıla bilmədi" in result.error


async def test_rate_limits_are_not_retried(monkeypatch):
    """A 429 is a real result about capacity, not a transport hiccup."""
    responses = _FlakyResponses(fail_times=99, error_name="RateLimitError")
    _install_flaky(monkeypatch, responses)

    report = await run_benchmark(
        api_key="sk-test", models=["gpt-4.1-nano"],
        questions=QUESTIONS[:1], delay_between_calls=0,
    )
    assert responses.attempts == 1
    assert report.models[0].results[0].retries == 0


def test_a_single_transient_failure_does_not_disqualify_a_model():
    good = _benchmark("solid", ttft=500.0, quality=0.9)          # 5 answers, 0 failures
    blipped = _benchmark("fast", ttft=200.0, quality=0.9)
    # One failure in 21 attempts is under the tolerated rate.
    for _ in range(16):
        blipped.results.append(
            QuestionResult(question_id="x", question="x", ok=True, first_token_ms=200.0,
                           total_ms=600.0, char_count=400, overall=0.9,
                           azerbaijani=0.9, english_preservation=0.9, coverage=0.9)
        )
    blipped.results.append(QuestionResult(question_id="f", question="x", ok=False, error="conn"))
    report = _report(good, blipped)
    assert report.recommended().model == "fast"


def test_a_model_that_fails_often_is_still_excluded():
    flaky = _benchmark("flaky", ttft=100.0, quality=0.9)
    for index in range(5):   # 5 failures in 10 -> 50%
        flaky.results.append(QuestionResult(question_id=f"f{index}", question="x", ok=False, error="boom"))
    report = _report(flaky, _benchmark("solid", 800.0, 0.9))
    assert report.recommended().model == "solid"


def test_report_names_the_quality_leader_as_well_as_the_speed_leader():
    fast = _benchmark("fast", ttft=200.0, quality=0.70)
    thorough = _benchmark("thorough", ttft=900.0, quality=0.95)
    report = _report(fast, thorough)

    assert report.recommended().model == "fast"
    assert report.best_quality().model == "thorough"
    assert report.most_consistent().model == "fast"

    text = report.to_text()
    assert "Ən sürətli" in text and "Ən yüksək keyfiyyət" in text
    # When the two disagree the report says so rather than picking silently.
    assert "seçim sizindir" in text
    assert "avtomatik dəyişmir" in text


def test_report_does_not_claim_disagreement_when_one_model_wins_both():
    report = _report(_benchmark("a", 200.0, 0.95), _benchmark("b", 900.0, 0.70))
    assert report.recommended().model == report.best_quality().model == "a"
    assert "seçim sizindir" not in report.to_text()


def test_serialised_report_records_all_three_leaders(tmp_path):
    report = _report(_benchmark("fast", 200.0, 0.70), _benchmark("thorough", 900.0, 0.95))
    data = json.loads(report.save(tmp_path).read_text(encoding="utf-8"))
    assert data["recommended"] == "fast"
    assert data["best_quality"] == "thorough"
    assert data["most_consistent"] == "fast"
