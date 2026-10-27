"""Layer pipeline — L0 cache → L1 keyword → L2 RAG → L3 LLM.

Analogy for a C dev:
    Like CPU cache hierarchy: L1 (register) → L2 (cache) → L3 (RAM) → disk.
    Each level is slower and more expensive. Never skip L0 — always check cache first.
    The cheapest token is the one you never send.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from spiral_05_economics.budget_tracker import BudgetTracker
from spiral_05_economics.circuit_breaker import CircuitBreaker
from spiral_05_economics.config import CACHE_MAX_ENTRIES, KEYWORD_MIN_SCORE
from spiral_05_economics.cost_calculator import CostCalculator
from spiral_05_economics.model_router import ModelRouter
from spiral_05_economics.token_counter import TokenCount


@runtime_checkable
class CognitiveObserver(Protocol):
    """Protocol for cognitive monitoring (spiral_04 CognitiveLayer)."""

    def enrich(
        self,
        bot_name: str,
        command: str,
        raw_output: str,
        metrics: dict[str, float] | None = None,
    ) -> Any: ...


@runtime_checkable
class RAGBackend(Protocol):
    """Protocol for a RAG retrieval backend (spiral_02 HybridRetriever)."""

    def retrieve(self, query: str, top_k: int = 5) -> list[Any]: ...

    @staticmethod
    def format_context(results: list[Any], max_tokens: int = 2000) -> str: ...


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for an LLM generation backend (spiral_01 Generator)."""

    def generate(self, context: str, question: str) -> Any: ...


@runtime_checkable
class MetricsObserver(Protocol):
    """Protocol for metrics collection (spiral_04 MetricsCollector)."""

    def record_latency(self, operation: str, seconds: float) -> None: ...
    def record_tokens(self, model: str, count: int) -> None: ...
    def record_cache_hit(self, layer: str) -> None: ...
    def record_cache_miss(self, layer: str) -> None: ...


# Layer constants
LAYER_CACHE = "L0_CACHE"
LAYER_KEYWORD = "L1_KEYWORD"
LAYER_RAG = "L2_RAG"
LAYER_LLM = "L3_LLM"


@dataclass
class LayerResult:
    """Result from the layer pipeline."""

    query: str
    answer: str
    layer: str
    cost: float
    model: str | None
    cached: bool
    metadata: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"LayerResult(layer={self.layer}, cost=${self.cost:.6f}, "
            f"model={self.model}, cached={self.cached})"
        )


def _normalize_query(query: str) -> str:
    """Normalize query for cache key."""
    return query.strip().lower()


def _cache_key(query: str) -> str:
    """Generate a deterministic cache key from a normalized query."""
    normalized = _normalize_query(query)
    return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()  # nosec B324


class LayerPipeline:
    """Cost-aware retrieval pipeline: L0 cache → L1 keyword → L2 RAG → L3 LLM."""

    def __init__(
        self,
        tracker: BudgetTracker,
        calculator: CostCalculator,
        router: ModelRouter,
        breaker: CircuitBreaker,
        rag_backend: RAGBackend | None = None,
        llm_backend: LLMBackend | None = None,
        cognitive: CognitiveObserver | None = None,
        metrics: MetricsObserver | None = None,
        semantic_cache: Any | None = None,
        embedder: Any | None = None,
        cache_max_entries: int = CACHE_MAX_ENTRIES,
        keyword_min_score: float = KEYWORD_MIN_SCORE,
    ) -> None:
        self.tracker = tracker
        self._calculator = calculator
        self._router = router
        self._breaker = breaker
        self._rag = rag_backend
        self._llm = llm_backend
        self._cognitive = cognitive
        self._metrics = metrics
        self.metrics = metrics  # public access for dashboard/tests
        self._semantic_cache = semantic_cache
        self._embedder = embedder
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max = cache_max_entries
        self._keywords: dict[str, str] = {}
        self._keyword_min_score = keyword_min_score

    # --- L0: Cache ---

    def store_in_cache(self, query: str, answer: str) -> None:
        """Store a query/answer pair in the cache."""
        key = _cache_key(query)
        self._cache[key] = answer
        self._cache.move_to_end(key)

        # Evict oldest entries if over capacity
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def check_cache(self, query: str) -> str | None:
        """Check if a query is in the cache. Returns answer or None."""
        key = _cache_key(query)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def cache_size(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    # --- L1: Keyword ---

    def add_keyword_entry(self, keyword: str, answer: str) -> None:
        """Add a keyword → answer mapping for L1 lookup."""
        self._keywords[keyword.lower()] = answer

    def check_keywords(self, query: str) -> str | None:
        """Check if query matches any keyword entries."""
        query_lower = query.lower()
        for keyword, answer in self._keywords.items():
            if keyword in query_lower:
                return answer
        return None

    # --- Cognitive observation ---

    def _observe(self, result: LayerResult) -> LayerResult:
        """Pass result through cognitive layer for monitoring (if configured)."""
        if self._cognitive is None:
            return result
        metrics = {
            "cost": result.cost,
            "layer_depth": {"L0_CACHE": 0, "L1_KEYWORD": 1, "L2_RAG": 2, "L3_LLM": 3}.get(
                result.layer, -1
            ),
        }
        if "generation_latency" in result.metadata:
            metrics["latency"] = float(result.metadata["generation_latency"])  # type: ignore[arg-type]
        if "retrieval_latency" in result.metadata:
            metrics["latency"] = float(result.metadata["retrieval_latency"])  # type: ignore[arg-type]
        analysis = self._cognitive.enrich(
            bot_name="layer_pipeline",
            command="query",
            raw_output=result.answer[:500],
            metrics=metrics,
        )
        result.metadata["cognitive_analysis"] = {
            "status": analysis.status,
            "alerts": analysis.alerts,
            "is_regression": analysis.is_regression,
        }
        return result

    # --- Full pipeline ---

    def query(self, query: str) -> LayerResult:
        """Run the full layer pipeline for a query.

        L0 Cache → L1 Keyword → L2 RAG → L3 LLM
        Each layer is cheaper than the next. Stop at first hit.
        Result is observed by cognitive layer (if configured).
        """
        result = self._execute_pipeline(query)
        return self._observe(result)

    def _execute_pipeline(self, query: str) -> LayerResult:
        """Internal pipeline execution — separated for cognitive observation."""
        # L0: Cache
        cached = self.check_cache(query)
        if cached is not None:
            if self._metrics:
                self._metrics.record_cache_hit(LAYER_CACHE)
            self.tracker.record_cost(
                model="cache",
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                query=query,
            )
            return LayerResult(
                query=query,
                answer=cached,
                layer=LAYER_CACHE,
                cost=0.0,
                model=None,
                cached=True,
            )
        if self._metrics:
            self._metrics.record_cache_miss(LAYER_CACHE)

        # L0.5: Semantic cache (opt-in)
        if self._semantic_cache is not None and self._embedder is not None:
            query_emb = self._embedder.embed_text(query)
            semantic_hit = self._semantic_cache.lookup(query_emb)
            if semantic_hit is not None:
                if self._metrics:
                    self._metrics.record_cache_hit("L0_SEMANTIC")
                self.tracker.record_cost(
                    model="semantic_cache",
                    input_tokens=0,
                    output_tokens=0,
                    cost=0.0,
                    query=query,
                )
                return LayerResult(
                    query=query,
                    answer=semantic_hit,
                    layer=LAYER_CACHE,
                    cost=0.0,
                    model=None,
                    cached=True,
                    metadata={"cache_type": "semantic"},
                )
            if self._metrics:
                self._metrics.record_cache_miss("L0_SEMANTIC")

        # L1: Keyword
        keyword_hit = self.check_keywords(query)
        if keyword_hit is not None:
            if self._metrics:
                self._metrics.record_cache_hit(LAYER_KEYWORD)
            self.store_in_cache(query, keyword_hit)
            self.tracker.record_cost(
                model="keyword",
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                query=query,
            )
            return LayerResult(
                query=query,
                answer=keyword_hit,
                layer=LAYER_KEYWORD,
                cost=0.0,
                model=None,
                cached=False,
            )
        if self._metrics:
            self._metrics.record_cache_miss(LAYER_KEYWORD)

        # L2: RAG retrieval (spiral_02 HybridRetriever)
        context = ""
        if self._rag is not None:
            t0 = time.monotonic()
            results = self._rag.retrieve(query, top_k=5)
            latency = time.monotonic() - t0
            if self._metrics:
                self._metrics.record_latency("retrieval", latency)
            if results:
                context = self._rag.format_context(results)
                # RAG-only answer: return context without LLM if no LLM backend
                if self._llm is None:
                    self.store_in_cache(query, context)
                    self.tracker.record_cost(
                        model="rag_retrieval",
                        input_tokens=0,
                        output_tokens=0,
                        cost=0.0,
                        query=query,
                    )
                    return LayerResult(
                        query=query,
                        answer=context,
                        layer=LAYER_RAG,
                        cost=0.0,
                        model=None,
                        cached=False,
                        metadata={"retrieval_latency": latency, "chunks": len(results)},
                    )

        # L3: LLM generation (spiral_01 Generator)
        routing = self._router.route(query)

        if self._breaker.is_open():
            return LayerResult(
                query=query,
                answer="[circuit breaker open — LLM unavailable]",
                layer=LAYER_LLM,
                cost=0.0,
                model=routing.model,
                cached=False,
                metadata={"circuit_breaker": "open"},
            )

        if self._llm is not None:
            t0 = time.monotonic()
            gen_result = self._llm.generate(context or "", query)
            latency = time.monotonic() - t0
            answer = gen_result.answer
            actual_model = gen_result.model
            token_count = TokenCount(
                input_tokens=gen_result.input_tokens,
                output_tokens=gen_result.output_tokens,
                model=actual_model,
            )
            cost_record = self._calculator.calculate(token_count)
            cost = cost_record.total_cost
            self.store_in_cache(query, answer)
            # Also store in semantic cache for fuzzy matches
            if self._semantic_cache is not None and self._embedder is not None:
                query_emb = self._embedder.embed_text(query)
                self._semantic_cache.store(query, query_emb, answer)
            self.tracker.record_cost(
                model=actual_model,
                input_tokens=gen_result.input_tokens,
                output_tokens=gen_result.output_tokens,
                cost=cost,
                query=query,
            )
            if self._metrics:
                self._metrics.record_latency("generation", latency)
                self._metrics.record_tokens(
                    actual_model, gen_result.input_tokens + gen_result.output_tokens
                )
            return LayerResult(
                query=query,
                answer=answer,
                layer=LAYER_LLM,
                cost=cost,
                model=gen_result.model,
                cached=False,
                metadata={
                    "generation_latency": latency,
                    "input_tokens": gen_result.input_tokens,
                    "output_tokens": gen_result.output_tokens,
                    "context_provided": bool(context),
                    "routed_tier": routing.tier,
                },
            )

        # Fallback: no backends available
        self.tracker.record_cost(
            model=routing.model,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            query=query,
        )
        return LayerResult(
            query=query,
            answer="[no RAG or LLM backend configured]",
            layer=LAYER_LLM,
            cost=0.0,
            model=routing.model,
            cached=False,
            metadata={"routing": routing.__dict__},
        )
