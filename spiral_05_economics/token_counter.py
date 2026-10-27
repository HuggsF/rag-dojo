"""Token counter — counts tokens before sending to LLM.

Analogy for a C dev:
    strlen() tells you the byte count.
    TokenCounter tells you the *token* count — the unit LLMs bill on.
    Overshoot your malloc and you segfault; overshoot your token budget and you pay.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass
class TokenCount:
    """Result of counting tokens for a prompt/response pair."""

    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        return (
            f"TokenCount(in={self.input_tokens}, out={self.output_tokens}, "
            f"total={self.total}, model={self.model})"
        )


# tiktoken encoding mapping for known model families
_ENCODING_MAP: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "claude-haiku": "cl100k_base",
    "claude-sonnet": "cl100k_base",
    "claude-opus": "cl100k_base",
}

# Default encoding for models not in the map
_DEFAULT_ENCODING = "cl100k_base"


class TokenCounter:
    """Counts tokens using tiktoken, with fallback for unknown models."""

    def __init__(self) -> None:
        self._encoders: dict[str, tiktoken.Encoding] = {}

    def _get_encoder(self, model: str) -> tiktoken.Encoding | None:
        """Get or create encoder for a model."""
        if model not in self._encoders:
            encoding_name = _ENCODING_MAP.get(model, _DEFAULT_ENCODING)
            try:
                self._encoders[model] = tiktoken.get_encoding(encoding_name)
            except Exception:
                return None
        return self._encoders[model]

    def count_text(self, text: str, model: str = "cl100k_base") -> int:
        """Count tokens in a text string.

        Falls back to char/4 estimation if encoding not available.
        """
        if not text:
            return 0
        encoder = self._get_encoder(model)
        if encoder is None:
            return max(1, len(text) // 4)
        return len(encoder.encode(text))

    def count_prompt_and_response(self, prompt: str, response: str, model: str) -> TokenCount:
        """Count tokens for both prompt and response."""
        return TokenCount(
            input_tokens=self.count_text(prompt, model),
            output_tokens=self.count_text(response, model),
            model=model,
        )

    def estimate_prompt(self, prompt: str, model: str) -> TokenCount:
        """Estimate tokens for a prompt (before LLM call, no response yet)."""
        return TokenCount(
            input_tokens=self.count_text(prompt, model),
            output_tokens=0,
            model=model,
        )
