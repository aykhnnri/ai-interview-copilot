"""Answer generation through the OpenAI Responses API, streamed.

The client object is created once and reused for the whole session so no
connection setup sits in the critical path between a finalised question and the
first token on screen.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from .. import messages as msg
from ..config import LlmSettings
from ..documents.profile import CandidateProfile
from .models import DEFAULT_MODEL, ModelSpec, get_spec
from .prompts import SYSTEM_INSTRUCTION, Turn, build_input, build_instructions

log = logging.getLogger(__name__)

#: How many times to open the answer stream before giving up.  Only retried
#: while nothing has been rendered yet - see `stream_answer`.
STREAM_ATTEMPTS = 3


class LlmError(RuntimeError):
    """Answer generation failed.  `user_message` is safe to show in the UI.

    `recoverable` means "retrying later could work".  `transient` is narrower:
    the request never reached a verdict because the transport broke, so a retry
    measures the same thing again.  A rate limit is recoverable but *not*
    transient - it is a real answer about capacity, and the benchmark records it
    rather than retrying it away.
    """

    def __init__(
        self,
        user_message: str,
        detail: str = "",
        recoverable: bool = True,
        transient: bool = False,
    ) -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail
        self.recoverable = recoverable
        self.transient = transient


class LlmAuthError(LlmError):
    pass


class LlmBillingError(LlmError):
    pass


@dataclass
class AnswerResult:
    """Everything a completed generation produced, including its timings."""

    question: str
    text: str
    model: str
    first_token_ms: float | None = None
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    context_labels: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)


#: Exception names that mean the transport broke before any verdict arrived.
TRANSPORT_ERRORS = frozenset(
    {
        "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout",
        "ReadError", "ReadTimeout", "WriteError", "RemoteProtocolError",
        "ProtocolError", "ConnectionError", "ConnectionResetError",
    }
)


def classify_error(exc: Exception) -> LlmError:
    """Map an SDK exception onto a user-facing Azerbaijani message."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)

    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
        return LlmAuthError(msg.LLM_AUTH_ERROR, str(exc), recoverable=False)
    if status == 402 or "insufficient_quota" in str(exc) or name == "InsufficientQuotaError":
        return LlmBillingError(msg.LLM_BILLING_ERROR, str(exc), recoverable=False)
    if name == "RateLimitError" or status == 429:
        # Recoverable, but a real result about capacity - never retried away.
        return LlmError(msg.LLM_RATE_LIMITED, str(exc), recoverable=True, transient=False)
    if name == "APITimeoutError" or isinstance(exc, asyncio.TimeoutError):
        return LlmError(msg.LLM_TIMEOUT, str(exc), recoverable=True, transient=True)
    if name == "BadRequestError" or status == 400:
        return LlmError(msg.LLM_FAILED, str(exc), recoverable=False)
    if name in TRANSPORT_ERRORS:
        return LlmError(msg.LLM_FAILED, f"{name}: {exc}", recoverable=True, transient=True)
    return LlmError(msg.LLM_FAILED, f"{name}: {exc}", recoverable=True)


class AnswerGenerator:
    """Streams Azerbaijani answers for detected interview questions."""

    def __init__(self, api_key: str, settings: LlmSettings | None = None) -> None:
        if not api_key:
            raise LlmAuthError(msg.KEY_MISSING_OPENAI, recoverable=False)
        from openai import AsyncOpenAI

        self.settings = settings or LlmSettings()
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.settings.base_url or None,
            timeout=self.settings.request_timeout_s,
            max_retries=1,  # the UI offers an explicit retry; keep latency predictable
        )
        self._history: list[Turn] = []
        #: Set when the API refused the configured tier and we downgraded.
        self.service_tier_rejected: str | None = None
        #: Streams reopened after a transport drop, for the developer panel.
        self.stream_retries = 0

    # -- conversation memory ---------------------------------------------
    @property
    def history(self) -> list[Turn]:
        return list(self._history)

    def remember(self, question: str, answer: str) -> None:
        self._history.append(Turn(question=question, answer=answer))
        keep = max(0, self.settings.history_turns)
        if keep == 0:
            self._history.clear()
        else:
            del self._history[:-keep]

    def clear_history(self) -> None:
        self._history.clear()

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception:  # pragma: no cover
            pass

    # -- generation -------------------------------------------------------
    def _request_kwargs(
        self,
        spec: ModelSpec,
        messages: list[dict[str, object]],
        max_output_tokens: int,
        instructions: str,
        service_tier: str | None,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": spec.id,
            "instructions": instructions,
            "input": messages,
            "max_output_tokens": max_output_tokens,
            "stream": True,
            "store": False,
        }
        if spec.supports_temperature:
            kwargs["temperature"] = self.settings.temperature
        if spec.supports_reasoning and spec.default_reasoning_effort:
            kwargs["reasoning"] = {"effort": spec.default_reasoning_effort}
        if service_tier:
            kwargs["service_tier"] = service_tier
        return kwargs

    async def _create_stream(self, kwargs: dict[str, object]):
        """Open the stream, retrying once without `service_tier` if rejected.

        A tier the account cannot use must not cost the user their answer, but
        it also must not be hidden: the downgrade is logged and recorded on
        `service_tier_rejected` so the UI can say so rather than implying the
        fast tier was used.
        """
        try:
            return await self._client.responses.create(**kwargs)
        except Exception as exc:
            tier = kwargs.get("service_tier")
            if not tier or "service_tier" not in str(exc):
                raise
            log.warning(
                "service_tier=%r was rejected (%s); retrying on the default tier", tier, exc
            )
            self.service_tier_rejected = str(tier)
            downgraded = dict(kwargs)
            downgraded.pop("service_tier", None)
            return await self._client.responses.create(**downgraded)

    async def stream_answer(
        self,
        question: str,
        profile: CandidateProfile | None = None,
        mode: str = "answer",
        model: str | None = None,
        use_history: bool = True,
    ) -> AsyncIterator[str | AnswerResult]:
        """Yield text deltas as they arrive, then a final `AnswerResult`.

        The caller can render every `str` immediately; the trailing
        `AnswerResult` carries the assembled text, timings and token usage.
        """
        model_id = model or self.settings.model
        spec = get_spec(model_id)
        max_tokens = (
            self.settings.expand_max_output_tokens
            if mode == "expand"
            else self.settings.max_output_tokens
        )
        messages = build_input(
            question,
            history=self._history if use_history else [],
            mode=mode,
        )
        # The whole profile rides in the instruction layer, so there is no
        # per-question retrieval between the question and the first token.
        instructions = build_instructions(profile)
        context_labels: list[str] = []
        if profile is not None and profile.has_documents:
            if profile.cv is not None:
                context_labels.append(f"CV (tam: {profile.cv.name})")
            if profile.job_description is not None:
                context_labels.append(f"Vakansiya (tam: {profile.job_description.name})")
            if profile.truncated:
                context_labels.append("⚠ sənədlər kəsildi")

        started = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        input_tokens = output_tokens = 0
        truncated = False

        # A dropped connection is common enough that losing the answer to one
        # would be felt mid-interview.  Retrying is only safe while nothing has
        # been shown yet - once a delta has been yielded, restarting would
        # duplicate text on screen, so from that point the error is reported
        # and the user retries deliberately.
        for attempt in range(1, STREAM_ATTEMPTS + 1):
            started = time.perf_counter()
            first_token_at = None
            parts = []
            input_tokens = output_tokens = 0
            truncated = False
            try:
                stream = await self._create_stream(
                    self._request_kwargs(
                        spec, messages, max_tokens, instructions,
                        self.settings.service_tier,
                    )
                )
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "") or ""
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            parts.append(delta)
                            yield delta
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        usage = getattr(response, "usage", None)
                        if usage is not None:
                            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                        if getattr(response, "incomplete_details", None) is not None:
                            truncated = True
                    elif event_type == "response.incomplete":
                        truncated = True
                    elif event_type == "response.failed":
                        response = getattr(event, "response", None)
                        error = getattr(response, "error", None)
                        detail = getattr(error, "message", "") or "response failed"
                        raise LlmError(msg.LLM_FAILED, detail, recoverable=True)
                    elif event_type == "error":
                        detail = getattr(event, "message", "") or "stream error"
                        raise LlmError(msg.LLM_FAILED, detail, recoverable=True)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = exc if isinstance(exc, LlmError) else classify_error(exc)
                retryable = (
                    error.transient
                    and first_token_at is None
                    and attempt < STREAM_ATTEMPTS
                )
                if not retryable:
                    raise error from exc
                log.warning(
                    "Answer stream dropped before any token (%s); retrying %d/%d",
                    error.detail or error.user_message, attempt + 1, STREAM_ATTEMPTS,
                )
                self.stream_retries += 1
                await asyncio.sleep(0.4 * attempt)

        text = "".join(parts)
        total_ms = (time.perf_counter() - started) * 1000.0
        if not text.strip():
            raise LlmError(msg.LLM_FAILED, "model returned no text", recoverable=True)

        yield AnswerResult(
            question=question,
            text=text,
            model=spec.id,
            first_token_ms=None if first_token_at is None else (first_token_at - started) * 1000.0,
            total_ms=total_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=spec.cost_usd(input_tokens, output_tokens),
            context_labels=context_labels,
            truncated=truncated,
        )

    async def generate(self, question: str, **kwargs) -> AnswerResult:
        """Non-streaming convenience wrapper used by the benchmark and tests."""
        result: AnswerResult | None = None
        async for item in self.stream_answer(question, **kwargs):
            if isinstance(item, AnswerResult):
                result = item
        if result is None:  # pragma: no cover - stream_answer always ends with one
            raise LlmError(msg.LLM_FAILED, "stream ended without a result")
        return result


async def test_api_key(
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 20.0,
    service_tier: str | None = None,
) -> tuple[bool, str]:
    """Minimal authenticated request that proves the key works.

    Sends a 1-token generation, which is the cheapest call that still exercises
    authentication, model access and billing.
    """
    if not api_key or not api_key.strip():
        return False, msg.KEY_MISSING_OPENAI
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key.strip(), timeout=timeout, max_retries=0)
    try:
        spec = get_spec(model)
        kwargs: dict[str, object] = {
            "model": spec.id,
            "input": "ping",
            "max_output_tokens": 16,
            "store": False,
        }
        if spec.supports_reasoning and spec.default_reasoning_effort:
            kwargs["reasoning"] = {"effort": spec.default_reasoning_effort}
        if service_tier:
            kwargs["service_tier"] = service_tier
        try:
            response = await client.responses.create(**kwargs)
        except Exception as exc:
            if not service_tier or "service_tier" not in str(exc):
                raise
            kwargs.pop("service_tier", None)
            response = await client.responses.create(**kwargs)
            usage = getattr(response, "usage", None)
            used = getattr(usage, "total_tokens", 0) if usage is not None else 0
            return True, (
                f"model: {spec.id}, {used} token "
                f"(diqqət: '{service_tier}' service tier qəbul edilmədi, defolt istifadə olundu)"
            )
        usage = getattr(response, "usage", None)
        used = getattr(usage, "total_tokens", 0) if usage is not None else 0
        return True, f"model: {spec.id}, {used} token"
    except Exception as exc:
        error = classify_error(exc)
        return False, error.user_message
    finally:
        try:
            await client.close()
        except Exception:  # pragma: no cover
            pass
