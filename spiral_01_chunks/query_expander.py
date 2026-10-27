"""Query expander — generates query variations for multi-query retrieval.

Analogy for a C dev:
    A single query is like a single probe into memory. Query expansion
    is like issuing multiple probes with different addresses — you cover
    more of the address space and reduce the chance of missing relevant data.

Two strategies:

1. QueryExpander (rule-based, zero LLM cost)
   Generates variations by rephrasing, adding synonyms, and decomposing
   compound questions. Purely deterministic — no API calls.

2. HyDEExpander (Hypothetical Document Embeddings, uses LLM)
   Generates a *hypothetical document* that would answer the query, then
   uses that document as the embedding input. The document embedding
   typically lands closer to real answer documents than the query embedding
   does — improving recall for factual questions.

Both are STANDALONE modules:
- They do NOT modify Retriever.retrieve() or HybridRetriever.retrieve()
- Integration is via composition: pass an expander to retrieve_expanded()
  (a new method) or call expand() manually before retrieval.
"""

from __future__ import annotations

import re
from typing import Protocol

from groq import Groq

from spiral_01_chunks.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE

# ─── Protocol ─────────────────────────────────────────────────────────────────


class Expander(Protocol):
    """Common interface for all query expanders."""

    def expand(self, query: str) -> list[str]:
        """Return a list of query variations including the original."""
        ...


# ─── Rule-based expander ───────────────────────────────────────────────────────


_STOP_WORDS = frozenset(
    ["how", "what", "where", "when", "why", "which", "who", "is", "are", "the", "a", "an"]
)


class QueryExpander:
    """Generates query variations using rule-based rewrites (no LLM).

    Strategies applied:
    - Original query (always included)
    - Keyword extraction (remove stop words, embed the key terms)
    - Question → statement rewrite (improves recall for factual docs)
    """

    def __init__(self, max_variations: int = 3) -> None:
        self._max_variations = max_variations

    def expand(self, query: str) -> list[str]:
        """Return up to max_variations query variations including the original.

        Args:
            query: The original user query.

        Returns:
            List of query strings. First element is always the original.
        """
        variations: list[str] = [query]

        # Variation 1: keyword extraction
        keywords = self._extract_keywords(query)
        if keywords and keywords != query:
            variations.append(keywords)

        # Variation 2: question → statement
        statement = self._question_to_statement(query)
        if statement and statement not in variations:
            variations.append(statement)

        return variations[: self._max_variations]

    def _extract_keywords(self, query: str) -> str:
        """Remove stop words and question marks, keep content words."""
        words = re.sub(r"[?!.,]", "", query.lower()).split()
        keywords = [w for w in words if w not in _STOP_WORDS]
        return " ".join(keywords)

    def _question_to_statement(self, query: str) -> str:
        """Turn 'How do I create a route in FastAPI?' into 'creating a route in FastAPI'."""
        q = query.strip().rstrip("?")
        # "How do I X" → "X"
        q = re.sub(r"^how\s+do\s+i\s+", "", q, flags=re.IGNORECASE)
        # "What is X" → "X explanation"
        q = re.sub(r"^what\s+is\s+", "", q, flags=re.IGNORECASE)
        # "How to X" → "X tutorial"
        q = re.sub(r"^how\s+to\s+", "", q, flags=re.IGNORECASE)
        return q.strip()


# ─── HyDE expander ────────────────────────────────────────────────────────────


_HYDE_SYSTEM = (
    "You are a technical documentation writer. "
    "Given a question, write a short passage (2-3 sentences) that would "
    "directly answer it, as if it were an excerpt from official documentation. "
    "Do NOT say you don't know. Always write a plausible passage."
)


class HyDEExpander:
    """Hypothetical Document Embeddings expander (uses LLM).

    Instead of embedding the query directly, this generates a *hypothetical*
    answer document and returns it. When you retrieve using this hypothetical
    document as the query, you get results that are closer to real answers
    because the embedding space aligns document-to-document, not question-to-document.

    Reference: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        self._client = Groq(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def expand(self, query: str) -> list[str]:
        """Return [original_query, hypothetical_document].

        Args:
            query: The original user query.

        Returns:
            List with [original_query, hyde_document]. Always 2 elements.
        """
        hyde_doc = self._generate_hypothetical_doc(query)
        return [query, hyde_doc]

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Call LLM to generate a hypothetical answer document."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=self._temperature,
            max_tokens=256,
        )
        return response.choices[0].message.content or query
