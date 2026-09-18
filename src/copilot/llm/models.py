"""Which OpenAI models this app can drive, and what parameters each accepts.

Sending a reasoning parameter to a non-reasoning model is a 400, and so is
sending `temperature` to a model that does not take it, so capabilities are
declared per model rather than guessed at call time.

Prices are USD per 1M tokens and are only used for the benchmark cost estimate.
They change; `price_source` records where a number came from, and a model whose
price is unknown reports a cost of `None` rather than a made-up figure.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    supports_temperature: bool = True
    supports_reasoning: bool = False
    default_reasoning_effort: str | None = None
    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None
    price_source: str = ""
    notes: str = ""

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        """Approximate cost, or None when this model has no known price."""
        if self.input_price_per_1m is None or self.output_price_per_1m is None:
            return None
        return (
            input_tokens * self.input_price_per_1m
            + output_tokens * self.output_price_per_1m
        ) / 1_000_000


DEFAULT_MODEL = "gpt-5.6-luna"

MODELS: dict[str, ModelSpec] = {
    "gpt-4.1-nano": ModelSpec(
        id="gpt-4.1-nano",
        label="GPT-4.1 Nano",
        supports_temperature=True,
        supports_reasoning=False,
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        price_source="OpenAI published pricing; verify before relying on it",
        notes="Fastest and cheapest, but measured the shallowest coverage. "
              "No reasoning parameter - sending one is rejected.",
    ),
    "gpt-5.6-luna": ModelSpec(
        id="gpt-5.6-luna",
        label="GPT-5.6 Luna",
        supports_temperature=False,
        supports_reasoning=True,
        default_reasoning_effort="none",
        input_price_per_1m=None,
        output_price_per_1m=None,
        price_source="price not configured - no cost estimate is shown rather than a guess",
        notes="Default. Reasoning effort 'none' for interview latency; measured "
              "the best answer coverage and the best p95 time-to-first-token.",
    ),
    "gpt-4.1-mini": ModelSpec(
        id="gpt-4.1-mini",
        label="GPT-4.1 Mini",
        supports_temperature=True,
        supports_reasoning=False,
        input_price_per_1m=0.40,
        output_price_per_1m=1.60,
        price_source="OpenAI published pricing; verify before relying on it",
        notes="Fallback for harder questions; slower than nano.",
    ),
}


def get_spec(model_id: str) -> ModelSpec:
    """Spec for a model id; unknown ids get conservative defaults."""
    spec = MODELS.get(model_id)
    if spec is not None:
        return spec
    return ModelSpec(
        id=model_id,
        label=model_id,
        supports_temperature=True,
        supports_reasoning=False,
        notes="Unknown model - capabilities assumed, price unavailable.",
    )


def selectable_models() -> list[ModelSpec]:
    return list(MODELS.values())
