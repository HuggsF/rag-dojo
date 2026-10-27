"""Cost calculator — computes dollar cost from token counts.

Analogy for a C dev:
    sizeof(int) * count = memory cost.
    price_per_1M_tokens * token_count / 1M = dollar cost.
    Every LLM call is a malloc that costs real money.
"""

from __future__ import annotations

from dataclasses import dataclass

from spiral_05_economics.config import MODEL_PRICING
from spiral_05_economics.token_counter import TokenCount


@dataclass
class CostRecord:
    """Cost breakdown for a single LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

    def __repr__(self) -> str:
        return (
            f"CostRecord(model={self.model}, "
            f"in={self.input_tokens}, out={self.output_tokens}, "
            f"${self.total_cost:.6f})"
        )


class CostCalculator:
    """Calculates cost from token counts using pricing tables."""

    def __init__(self, pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self._pricing = pricing if pricing is not None else dict(MODEL_PRICING)

    def calculate(self, token_count: TokenCount) -> CostRecord:
        """Calculate cost for a token count.

        Raises ValueError if model is not in the pricing table.
        """
        if token_count.model not in self._pricing:
            raise ValueError(
                f"Unknown model: {token_count.model!r}. "
                f"Available: {', '.join(sorted(self._pricing.keys()))}"
            )

        input_price, output_price = self._pricing[token_count.model]
        input_cost = (token_count.input_tokens / 1_000_000) * input_price
        output_cost = (token_count.output_tokens / 1_000_000) * output_price

        return CostRecord(
            model=token_count.model,
            input_tokens=token_count.input_tokens,
            output_tokens=token_count.output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )

    def get_pricing(self, model: str) -> tuple[float, float]:
        """Get (input_price_per_1M, output_price_per_1M) for a model."""
        if model not in self._pricing:
            raise ValueError(f"Unknown model: {model!r}")
        return self._pricing[model]

    def list_models(self) -> list[str]:
        """List all known models."""
        return sorted(self._pricing.keys())
