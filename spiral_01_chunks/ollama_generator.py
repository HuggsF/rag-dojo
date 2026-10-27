"""Ollama Generator — local LLM fallback when Groq is rate-limited.

Analogy for a C dev:
    Generator uses Groq (remote, fast, rate-limited).
    OllamaGenerator uses Ollama (local, slower, no limits).
    Same interface, different backend. Like swapping libc implementations.

Satisfies the LLMBackend Protocol from spiral_05_economics.layer_pipeline.
"""

from __future__ import annotations

import httpx

from spiral_01_chunks.generator import SYSTEM_PROMPT, USER_TEMPLATE, GenerationResult


class OllamaGenerator:
    """Generates answers using Ollama (local LLM).

    Drop-in replacement for Generator when Groq is rate-limited.
    """

    def __init__(
        self,
        base_url: str = "http://host.docker.internal:11434",
        model: str = "qwen2.5:3b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, context: str, question: str) -> GenerationResult:
        """Generate an answer given context and a question."""
        user_message = USER_TEMPLATE.format(context=context, question=question)

        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            },
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            answer=data.get("message", {}).get("content", ""),
            model=self._model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )
