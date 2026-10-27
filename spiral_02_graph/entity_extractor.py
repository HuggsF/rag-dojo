"""Entity Extractor — uses LLM to extract entities and relations from chunks.

Analogy for a C dev:
- Chunks are raw byte buffers
- Entity extraction is like parsing structs out of raw data
- The LLM acts as a smart parser that understands context
- Entities are the structs, relations are the pointers between them

The extraction uses Groq (free tier) to identify:
- Entities: named concepts, modules, functions, classes, patterns
- Relations: how entities connect (USES, DEPENDS_ON, PART_OF, etc.)

Output is structured JSON that feeds into the graph store.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from groq import Groq

from spiral_02_graph.config import (
    ALLOWED_LABELS,
    ALLOWED_RELATIONS,
    EXTRACTION_MODEL,
    GROQ_API_KEY,
    GROQ_API_KEYS,
    LLM_TEMPERATURE,
    MAX_ENTITIES_PER_CHUNK,
    MAX_RELATIONS_PER_CHUNK,
)


@dataclass
class Entity:
    """A named entity extracted from text."""

    name: str
    label: str  # Concept, Module, Function, Class, Pattern
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunk_id: str = ""

    def __post_init__(self):
        self.name = self.name.strip()
        if self.label not in ALLOWED_LABELS:
            self.label = "Concept"  # fallback to safest label

    def __repr__(self) -> str:
        return f"Entity({self.label}:{self.name})"


@dataclass
class Relation:
    """A directed relation between two entities."""

    source: str  # entity name
    target: str  # entity name
    relation_type: str  # USES, DEPENDS_ON, PART_OF, etc.
    properties: dict[str, Any] = field(default_factory=dict)
    source_chunk_id: str = ""

    def __post_init__(self):
        self.source = self.source.strip()
        self.target = self.target.strip()
        if self.relation_type not in ALLOWED_RELATIONS:
            self.relation_type = "RELATED_TO"  # fallback

    def __repr__(self) -> str:
        return f"Relation({self.source} -[{self.relation_type}]-> {self.target})"


@dataclass
class ExtractionResult:
    """Result of entity extraction from a chunk."""

    entities: list[Entity]
    relations: list[Relation]
    chunk_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph extractor. Given a text chunk about software documentation, extract entities and their relationships.

Output ONLY valid JSON with this exact structure:
{
  "entities": [
    {"name": "entity name", "label": "Concept|Module|Function|Class|Pattern", "description": "brief description"}
  ],
  "relations": [
    {"source": "entity name", "target": "entity name", "type": "USES|DEPENDS_ON|PART_OF|RELATED_TO|IMPLEMENTS|EXTENDS"}
  ]
}

Entity labels:
- Concept: abstract ideas (e.g., "Dependency Injection", "Middleware", "CORS")
- Module: libraries or packages (e.g., "FastAPI", "Starlette", "Pydantic")
- Function: callable functions (e.g., "Depends", "add_middleware")
- Class: class definitions (e.g., "FastAPI", "APIRouter", "HTTPException")
- Pattern: design patterns (e.g., "Request-Response cycle", "Decorator pattern")

Relation types:
- USES: A uses B (FastAPI USES Starlette)
- DEPENDS_ON: A requires B (Middleware DEPENDS_ON App)
- PART_OF: A is part of B (Depends PART_OF Dependency Injection)
- RELATED_TO: A is related to B (CORS RELATED_TO Middleware)
- IMPLEMENTS: A implements B (add_middleware IMPLEMENTS Middleware)
- EXTENDS: A extends B (APIRouter EXTENDS FastAPI)

Rules:
1. Extract at most 10 entities and 15 relations per chunk.
2. Entity names should be concise (1-4 words).
3. Only extract entities that are clearly mentioned in the text.
4. Relations must connect entities that are both in the entities list.
5. Output ONLY the JSON, nothing else."""

EXTRACTION_USER_TEMPLATE = """Extract entities and relationships from this text chunk:

---
{text}
---

Output the JSON:"""


class EntityExtractor:
    """Extracts entities and relations from text chunks using an LLM.

    Supports key rotation: pass api_keys (list) for round-robin across
    multiple Groq API keys. When a key hits 429, the next key is tried.
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        api_keys: list[str] | None = None,
        model: str = EXTRACTION_MODEL,
    ):
        keys = api_keys or GROQ_API_KEYS or ([api_key] if api_key else [])
        if not keys:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        self._clients = [Groq(api_key=k) for k in keys]
        self._current_idx = 0
        self._model = model

    def _next_client(self) -> Groq:
        """Return current client and advance round-robin index."""
        client = self._clients[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._clients)
        return client

    def _call_with_rotation(self, messages: list[dict[str, str]], **kwargs) -> Any:
        """Try each key until one succeeds or all fail with 429."""
        last_err = None
        for _ in range(len(self._clients)):
            client = self._next_client()
            try:
                return client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    **kwargs,
                )
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    last_err = e
                    continue
                raise
        raise last_err  # type: ignore[misc]

    def extract_from_chunk(self, chunk_text: str, chunk_id: str = "") -> ExtractionResult:
        """Extract entities and relations from a single text chunk.

        Args:
            chunk_text: The text to extract from.
            chunk_id: Identifier for the source chunk.

        Returns:
            ExtractionResult with entities and relations.
        """
        user_message = EXTRACTION_USER_TEMPLATE.format(text=chunk_text)

        response = self._call_with_rotation(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=1024,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        result = self._parse_llm_response(content, chunk_id)
        result.model = self._model
        result.input_tokens = usage.prompt_tokens if usage else 0
        result.output_tokens = usage.completion_tokens if usage else 0

        return result

    def extract_from_chunks(
        self,
        chunks: list,
        batch_delay: float = 1.0,
    ) -> list[ExtractionResult]:
        """Extract from multiple chunks with rate limiting.

        Groq free tier allows ~30 req/min. batch_delay controls the
        pause between requests to avoid rate limiting.

        Args:
            chunks: List of Chunk objects from spiral_01.
            batch_delay: Seconds to wait between requests.

        Returns:
            List of ExtractionResult, one per chunk.
        """
        results: list[ExtractionResult] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk.source}::chunk_{chunk.chunk_index}"
            try:
                result = self.extract_from_chunk(chunk.text, chunk_id)
                results.append(result)
            except Exception as e:
                # On error, return empty result for this chunk
                results.append(
                    ExtractionResult(
                        entities=[],
                        relations=[],
                        chunk_id=chunk_id,
                        model=self._model,
                    )
                )
                print(f"Warning: extraction failed for {chunk_id}: {e}")

            # Rate limiting (skip delay after last chunk)
            if i < len(chunks) - 1 and batch_delay > 0:
                time.sleep(batch_delay)

        return results

    @staticmethod
    def _parse_llm_response(response_text: str, chunk_id: str) -> ExtractionResult:
        """Parse JSON from LLM response, handling common issues.

        Handles:
        - Clean JSON
        - JSON wrapped in markdown code fences
        - Malformed JSON (returns empty result)
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response using regex
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return ExtractionResult(
                        entities=[],
                        relations=[],
                        chunk_id=chunk_id,
                        model="",
                    )
            else:
                return ExtractionResult(
                    entities=[],
                    relations=[],
                    chunk_id=chunk_id,
                    model="",
                )

        # Parse entities
        entities: list[Entity] = []
        for e in data.get("entities", [])[:MAX_ENTITIES_PER_CHUNK]:
            if not isinstance(e, dict) or "name" not in e:
                continue
            entities.append(
                Entity(
                    name=e["name"],
                    label=e.get("label", "Concept"),
                    properties={"description": e.get("description", "")},
                    source_chunk_id=chunk_id,
                )
            )

        # Parse relations (only between known entities)
        entity_names = {e.name for e in entities}
        relations: list[Relation] = []
        for r in data.get("relations", [])[:MAX_RELATIONS_PER_CHUNK]:
            if not isinstance(r, dict):
                continue
            source = r.get("source", "").strip()
            target = r.get("target", "").strip()
            if source in entity_names and target in entity_names:
                relations.append(
                    Relation(
                        source=source,
                        target=target,
                        relation_type=r.get("type", "RELATED_TO"),
                        source_chunk_id=chunk_id,
                    )
                )

        return ExtractionResult(
            entities=entities,
            relations=relations,
            chunk_id=chunk_id,
            model="",
        )


class OllamaEntityExtractor:
    """Extracts entities and relations using a local Ollama model.

    Zero rate limits, zero cost. Uses Ollama's OpenAI-compatible API.
    Requires: ollama running locally (default http://localhost:11434).
    """

    def __init__(
        self,
        model: str = "mistral:7b",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def extract_from_chunk(self, chunk_text: str, chunk_id: str = "") -> ExtractionResult:
        """Extract entities and relations from a single text chunk via Ollama."""
        import httpx

        user_message = EXTRACTION_USER_TEMPLATE.format(text=chunk_text)

        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": LLM_TEMPERATURE},
            },
            timeout=120.0,
        )
        response.raise_for_status()

        data = response.json()
        content = data.get("message", {}).get("content", "")

        result = EntityExtractor._parse_llm_response(content, chunk_id)
        result.model = self._model
        return result

    def extract_from_chunks(
        self,
        chunks: list,
        batch_delay: float = 0.0,
    ) -> list[ExtractionResult]:
        """Extract from multiple chunks. No rate limiting needed for local models."""
        results: list[ExtractionResult] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk.source}::chunk_{chunk.chunk_index}"
            try:
                result = self.extract_from_chunk(chunk.text, chunk_id)
                results.append(result)
            except Exception as e:
                results.append(
                    ExtractionResult(
                        entities=[],
                        relations=[],
                        chunk_id=chunk_id,
                        model=self._model,
                    )
                )
                print(f"Warning: extraction failed for {chunk_id}: {e}")

            if i < len(chunks) - 1 and batch_delay > 0:
                time.sleep(batch_delay)

        return results
