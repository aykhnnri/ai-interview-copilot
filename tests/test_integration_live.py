"""Live integration tests against the real ElevenLabs and OpenAI APIs.

These cost money and need valid credentials, so they are skipped unless you opt
in explicitly:

    set AZCOPILOT_LIVE_TESTS=1
    set ELEVENLABS_API_KEY=...
    set OPENAI_API_KEY=...
    pytest -m live

Without `AZCOPILOT_LIVE_TESTS=1` every test here is skipped even when keys are
present, so a stray key in the environment can never bill you by accident.
"""
from __future__ import annotations

import asyncio
import math
import os
import struct

import pytest

from copilot.config import LlmSettings, SttSettings
from copilot.documents.profile import CandidateProfile
from copilot.llm.openai_client import AnswerGenerator
from copilot.llm.openai_client import test_api_key as openai_test_key
from copilot.llm.quality import score_answer, score_azerbaijani
from copilot.stt.elevenlabs_client import ElevenLabsRealtimeClient, SttCallbacks
from copilot.stt.elevenlabs_client import test_api_key as elevenlabs_test_key

pytestmark = pytest.mark.live

LIVE_ENABLED = os.environ.get("AZCOPILOT_LIVE_TESTS") == "1"

requires_live = pytest.mark.skipif(
    not LIVE_ENABLED,
    reason="live API tests are opt-in; set AZCOPILOT_LIVE_TESTS=1 to run them",
)


def _key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _speech_pcm(seconds: float = 2.0, rate: int = 16000) -> bytes:
    """A synthetic speech-like signal, used only to keep the socket fed.

    It is deliberately not expected to transcribe into anything meaningful -
    these tests assert on protocol behaviour, not on recognition of a tone.
    """
    samples = []
    for index in range(int(rate * seconds)):
        t = index / rate
        envelope = 0.4 * (1 + math.sin(2 * math.pi * 3 * t)) / 2
        value = envelope * (
            math.sin(2 * math.pi * 140 * t)
            + 0.5 * math.sin(2 * math.pi * 320 * t)
            + 0.3 * math.sin(2 * math.pi * 900 * t)
        )
        samples.append(int(max(-1.0, min(1.0, value / 1.8)) * 32767))
    return struct.pack(f"<{len(samples)}h", *samples)


# =========================================================================
# Connection tests (cheapest possible authenticated calls)
# =========================================================================
@requires_live
async def test_elevenlabs_key_is_accepted():
    ok, detail = await elevenlabs_test_key(_key("ELEVENLABS_API_KEY"))
    assert ok, detail
    assert detail


@requires_live
async def test_openai_key_is_accepted():
    ok, detail = await openai_test_key(_key("OPENAI_API_KEY"))
    assert ok, detail


@requires_live
async def test_a_bad_elevenlabs_key_is_rejected():
    ok, detail = await elevenlabs_test_key("sk_definitely_not_a_real_key_00000000")
    assert not ok
    assert "açar" in detail.lower() or "xəta" in detail.lower()


@requires_live
async def test_the_openai_key_is_not_accepted_by_elevenlabs():
    """Provider separation, verified against the live services."""
    ok, _ = await elevenlabs_test_key(_key("OPENAI_API_KEY"))
    assert not ok


# =========================================================================
# ElevenLabs realtime streaming
# =========================================================================
@requires_live
async def test_realtime_session_opens_and_accepts_audio():
    events: list[str] = []
    states: list[str] = []
    errors: list[tuple[str, bool]] = []

    client = ElevenLabsRealtimeClient(
        api_key=_key("ELEVENLABS_API_KEY"),
        settings=SttSettings(),
        sample_rate=16000,
        callbacks=SttCallbacks(
            on_event=lambda event: events.append(event.type),
            on_state=states.append,
            on_error=lambda message, recoverable: errors.append((message, recoverable)),
        ),
    )
    await client.start()

    # Wait for the socket to come up before streaming.
    for _ in range(100):
        if client.is_connected:
            break
        await asyncio.sleep(0.05)
    assert client.is_connected, f"never connected; errors={errors}"

    pcm = _speech_pcm(2.0)
    chunk = 3200  # 100 ms at 16 kHz mono PCM16
    for offset in range(0, len(pcm), chunk):
        client.push_audio(pcm[offset : offset + chunk])
        await asyncio.sleep(0.05)

    await asyncio.sleep(3.0)
    await client.stop()

    assert client.stats.chunks_sent >= 15
    assert "session_started" in events, f"no session_started; saw {set(events)}"
    assert not [message for message, recoverable in errors if not recoverable]


# =========================================================================
# OpenAI answer generation
# =========================================================================
@requires_live
@pytest.mark.parametrize(
    "question",
    [
        "RAG nədir?",
        "Python-da async və await nədir?",
        "RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?",
    ],
)
async def test_answers_are_generated_in_azerbaijani(question):
    generator = AnswerGenerator(_key("OPENAI_API_KEY"), LlmSettings(max_output_tokens=350))
    try:
        result = await generator.generate(question)
    finally:
        await generator.aclose()

    assert result.text.strip()
    assert result.first_token_ms is not None

    score, turkish_hits = score_azerbaijani(result.text)
    assert score > 0.4, f"answer does not read as Azerbaijani (turkish markers: {turkish_hits}):\n{result.text}"
    assert "ə" in result.text, "no Azerbaijani-specific characters in the answer"


@requires_live
async def test_answers_stream_incrementally():
    generator = AnswerGenerator(_key("OPENAI_API_KEY"), LlmSettings(max_output_tokens=300))
    deltas: list[str] = []
    try:
        async for item in generator.stream_answer("Vector database necə işləyir?"):
            if isinstance(item, str):
                deltas.append(item)
    finally:
        await generator.aclose()
    assert len(deltas) > 3, "the response did not arrive incrementally"


@requires_live
async def test_the_model_knows_the_cv_without_being_asked_to_search(cv_document, jd_document):
    """The whole profile is in the instruction layer, so specific facts come
    back without any per-question retrieval step."""
    profile = CandidateProfile.build(cv_document, jd_document)
    generator = AnswerGenerator(_key("OPENAI_API_KEY"), LlmSettings(max_output_tokens=400))
    try:
        result = await generator.generate(
            "Hansı vector database-lərlə işləmisiniz?", profile=profile
        )
    finally:
        await generator.aclose()

    lowered = result.text.lower()
    # These appear only in the CV's skills section.
    assert sum(term in lowered for term in ("pgvector", "qdrant", "faiss")) >= 2, result.text
    assert any("tam" in label for label in result.context_labels)


@requires_live
async def test_experience_the_cv_does_not_show_is_not_claimed(cv_document, jd_document):
    """Grounding has to fail closed: no Rust anywhere in the CV."""
    profile = CandidateProfile.build(cv_document, jd_document)
    assert "rust" not in (cv_document.text + jd_document.text).lower()

    generator = AnswerGenerator(_key("OPENAI_API_KEY"), LlmSettings(max_output_tokens=400))
    try:
        result = await generator.generate(
            "Rust dilində nə qədər təcrübəniz var?", profile=profile
        )
    finally:
        await generator.aclose()

    lowered = result.text.lower()
    disclaimers = ("olmayıb", "yoxdur", "təcrübəm yox", "iddia etmirəm",
                   "işləməmişəm", "olmamışdır", "yoxdu")
    assert any(d in lowered for d in disclaimers), (
        f"the model claimed Rust experience the CV does not support:\n{result.text}"
    )


@requires_live
async def test_the_fast_service_tier_is_accepted():
    """If the account cannot use it we must find out here, not mid-interview."""
    generator = AnswerGenerator(
        _key("OPENAI_API_KEY"), LlmSettings(max_output_tokens=200, service_tier="fast")
    )
    try:
        result = await generator.generate("RAG nədir?")
    finally:
        await generator.aclose()
    assert result.text.strip()
    assert generator.service_tier_rejected is None, (
        f"service_tier 'fast' was refused and downgraded to default"
    )


@requires_live
async def test_answers_use_the_uploaded_cv(cv_document, jd_document):
    profile = CandidateProfile.build(cv_document, jd_document)
    generator = AnswerGenerator(_key("OPENAI_API_KEY"), LlmSettings(max_output_tokens=400))
    try:
        result = await generator.generate(
            "Multi-agent sistemlərlə işləmisiniz?", profile=profile
        )
    finally:
        await generator.aclose()

    lowered = result.text.lower()
    assert any(term in lowered for term in ("langgraph", "multi-agent", "agent")), result.text
    assert result.context_labels, "no CV/job-description context was selected"


@requires_live
async def test_english_technical_terms_are_preserved():
    generator = AnswerGenerator(_key("OPENAI_API_KEY"), LlmSettings(max_output_tokens=350))
    try:
        result = await generator.generate(
            "Fine-tuning və RAG arasında əsas fərq nədir?"
        )
    finally:
        await generator.aclose()

    scores = score_answer(result.text, (), ("RAG", "fine-tuning"))
    assert scores.english_preservation >= 0.5, (
        f"technical terms were translated away: missing {scores.missing_english}\n{result.text}"
    )


@requires_live
async def test_an_invalid_openai_key_fails_clearly():
    ok, detail = await openai_test_key("sk-proj-invalid-key-for-testing-000000")
    assert not ok
    assert detail
