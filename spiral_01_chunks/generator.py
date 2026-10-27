"""Generator — takes context + question and produces an answer via LLM.

Analogy for a C dev:
- The generator is like printf() with a template
- The template (prompt) has placeholders for context and question
- The LLM fills in the answer based on the context provided
- Without context (RAG), the LLM would hallucinate (like reading uninitialized memory)
- With context, it's grounded (like reading from a properly initialized buffer)

This is the G in RAG — Generation.
We use Groq's free tier (llama-3.3-70b) to keep cost at $0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from groq import Groq

from spiral_01_chunks.config import (
    GROQ_API_KEY,
    GROQ_API_KEYS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

SYSTEM_PROMPT = """You are a helpful technical assistant. Answer questions based ONLY on the provided context.

Rules:
1. If the context contains the answer, provide it clearly and concisely.
2. If the context does NOT contain enough information, say "I don't have enough context to answer this."
3. Always cite which source document you're drawing from.
4. Never make up information that isn't in the context.
5. Respond in the same language as the question.
6. When the context contains code examples, include the most relevant snippet in your answer.
7. Mention specific function names, classes, or parameters from the context to give actionable answers."""

USER_TEMPLATE = """Context:
{context}

---

Question: {question}

Instructions: Answer the question using the context above. Include specific API names, \
functions, or classes mentioned in the context. If the context shows how something is used \
in practice, describe the concrete mechanism (e.g. which function to call, which parameter \
to pass)."""


@dataclass
class GenerationResult:
    """Result of LLM generation with metadata."""

    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD (Groq free tier = $0)."""
        return 0.0  # Groq free tier


class Generator:
    """Generates answers using retrieved context + LLM.

    Supports key rotation: pass api_keys (list) for round-robin across
    multiple Groq API keys. When a key hits 429, the next key is tried.
    Falls back to single api_key if api_keys is not provided.
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        api_keys: list[str] | None = None,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        keys = api_keys or GROQ_API_KEYS or ([api_key] if api_key else [])
        if not keys:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        self._clients = [Groq(api_key=k) for k in keys]
        self._current_idx = 0
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _next_client(self) -> Groq:
        """Return current client and advance round-robin index."""
        client = self._clients[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._clients)
        return client

    def _call_with_rotation(self, messages: list[dict[str, str]]) -> Any:
        """Try each key until one succeeds or all fail with 429."""
        last_err = None
        for _ in range(len(self._clients)):
            client = self._next_client()
            try:
                return client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    last_err = e
                    continue
                raise
        raise last_err  # type: ignore[misc]

    def generate(self, context: str, question: str) -> GenerationResult:
        """Generate an answer given context and a question.

        Args:
            context: Retrieved chunks formatted as text.
            question: The user's question.

        Returns:
            GenerationResult with answer and token usage.
        """
        user_message = USER_TEMPLATE.format(context=context, question=question)

        response = self._call_with_rotation(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        choice = response.choices[0]
        usage = response.usage

        return GenerationResult(
            answer=choice.message.content or "",
            model=self._model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    def generate_without_context(self, question: str) -> GenerationResult:
        """Generate without RAG context (for comparison/baseline).

        This shows what the LLM would answer WITHOUT retrieval.
        Useful for measuring RAG's value-add.
        """
        response = self._call_with_rotation(
            [
                {
                    "role": "system",
                    "content": "You are a helpful technical assistant. Answer concisely.",
                },
                {"role": "user", "content": question},
            ]
        )

        choice = response.choices[0]
        usage = response.usage

        return GenerationResult(
            answer=choice.message.content or "",
            model=self._model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
