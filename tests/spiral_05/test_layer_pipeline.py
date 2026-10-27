"""Tests for spiral_05_economics.layer_pipeline — L0-L3 cost-aware retrieval."""

from spiral_05_economics.layer_pipeline import (
    LAYER_CACHE,
    LAYER_KEYWORD,
    LAYER_LLM,
    LAYER_RAG,
    LayerPipeline,
    LayerResult,
)


class TestLayerResult:
    """Tests for LayerResult dataclass."""

    def test_has_required_fields(self):
        lr = LayerResult(
            query="test",
            answer="response",
            layer=LAYER_CACHE,
            cost=0.0,
            model=None,
            cached=True,
        )
        assert lr.layer == LAYER_CACHE
        assert lr.cached is True
        assert lr.cost == 0.0

    def test_repr(self):
        lr = LayerResult(
            query="test",
            answer="response",
            layer=LAYER_LLM,
            cost=0.01,
            model="claude-sonnet",
            cached=False,
        )
        assert LAYER_LLM in repr(lr)


class TestLayerPipelineCache:
    """Tests for L0 — Cache layer."""

    def test_cache_miss_on_first_query(self, layer_pipeline):
        result = layer_pipeline.check_cache("What is FastAPI?")
        assert result is None

    def test_cache_hit_after_store(self, layer_pipeline):
        layer_pipeline.store_in_cache("What is FastAPI?", "A web framework")
        result = layer_pipeline.check_cache("What is FastAPI?")
        assert result is not None
        assert result == "A web framework"

    def test_cache_is_case_insensitive(self, layer_pipeline):
        layer_pipeline.store_in_cache("What is FastAPI?", "A web framework")
        result = layer_pipeline.check_cache("what is fastapi?")
        assert result is not None

    def test_cache_respects_max_entries(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            cache_max_entries=3,
        )
        for i in range(5):
            pipeline.store_in_cache(f"query_{i}", f"answer_{i}")
        assert pipeline.cache_size() <= 3


class TestLayerPipelineKeyword:
    """Tests for L1 — Keyword layer."""

    def test_keyword_match_returns_answer(self, layer_pipeline):
        layer_pipeline.add_keyword_entry("FastAPI", "FastAPI is a modern Python web framework.")
        result = layer_pipeline.check_keywords("What is FastAPI?")
        assert result is not None
        assert "FastAPI" in result

    def test_keyword_miss(self, layer_pipeline):
        result = layer_pipeline.check_keywords("What is quantum computing?")
        assert result is None

    def test_keyword_partial_match(self, layer_pipeline):
        layer_pipeline.add_keyword_entry("middleware", "Middleware processes requests.")
        result = layer_pipeline.check_keywords("How does middleware work in FastAPI?")
        assert result is not None


class TestLayerPipelineQuery:
    """Tests for the full pipeline query flow."""

    def test_query_hits_cache_first(self, layer_pipeline):
        layer_pipeline.store_in_cache("What is FastAPI?", "Cached answer")
        result = layer_pipeline.query("What is FastAPI?")
        assert isinstance(result, LayerResult)
        assert result.layer == LAYER_CACHE
        assert result.cost == 0.0
        assert result.cached is True

    def test_query_hits_keyword_when_no_cache(self, layer_pipeline):
        layer_pipeline.add_keyword_entry("FastAPI", "FastAPI is a web framework.")
        result = layer_pipeline.query("What is FastAPI?")
        assert result.layer == LAYER_KEYWORD
        assert result.cost == 0.0

    def test_query_returns_layer_result(self, layer_pipeline):
        """Even without RAG/LLM backends, query should not crash."""
        result = layer_pipeline.query("What is dependency injection?")
        assert isinstance(result, LayerResult)
        assert result.layer in (LAYER_CACHE, LAYER_KEYWORD, LAYER_RAG, LAYER_LLM)

    def test_query_records_cost(self, layer_pipeline):
        layer_pipeline.store_in_cache("cached q", "cached a")
        layer_pipeline.query("cached q")
        summary = layer_pipeline.tracker.daily_summary()
        assert summary.total_requests >= 1


class TestLayerPipelineMetrics:
    """Tests for MetricsObserver integration."""

    def test_cache_hit_records_metric(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        from spiral_04_cognitive.metrics import InMemoryMetrics

        metrics = InMemoryMetrics()
        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            metrics=metrics,
        )
        pipeline.store_in_cache("test q", "test a")
        pipeline.query("test q")
        assert metrics.cache_hit_rate(LAYER_CACHE) == 1.0

    def test_cache_miss_records_metric(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        from spiral_04_cognitive.metrics import InMemoryMetrics

        metrics = InMemoryMetrics()
        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            metrics=metrics,
        )
        pipeline.query("unknown query")
        assert metrics.cache_hit_rate(LAYER_CACHE) == 0.0

    def test_no_metrics_does_not_crash(self, layer_pipeline):
        """Pipeline without metrics still works (None metrics)."""
        layer_pipeline.query("any query")  # should not raise

    def test_llm_records_latency_and_tokens(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        from unittest.mock import MagicMock

        from spiral_04_cognitive.metrics import InMemoryMetrics

        metrics = InMemoryMetrics()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            answer="test answer",
            model="llama-3.3-70b-versatile",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            llm_backend=mock_llm,
            metrics=metrics,
        )
        pipeline.query("test query")
        # Mock is near-instant, so latency may be 0.0 — check it was recorded
        assert "generation" in metrics._latencies
        assert len(metrics._latencies["generation"]) == 1
        assert metrics.total_tokens("llama-3.3-70b-versatile") == 150

    def test_rag_records_retrieval_latency(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        from unittest.mock import MagicMock

        from spiral_04_cognitive.metrics import InMemoryMetrics

        metrics = InMemoryMetrics()
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = []
        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            rag_backend=mock_rag,
            metrics=metrics,
        )
        pipeline.query("test query")
        assert metrics.avg_latency("retrieval") >= 0


class TestSemanticCacheIntegration:
    """Tests for L0.5 semantic cache path in layer_pipeline."""

    def test_semantic_cache_hit(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        """Semantic cache hit returns cached answer without LLM call."""
        from unittest.mock import MagicMock

        from spiral_04_cognitive.metrics import InMemoryMetrics
        from spiral_05_economics.semantic_cache import SemanticCache

        metrics = InMemoryMetrics()
        sem_cache = SemanticCache(similarity_threshold=0.9)
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [1.0, 0.0, 0.0]

        # Pre-populate semantic cache
        sem_cache.store("What is FastAPI?", [1.0, 0.0, 0.0], "A web framework")

        mock_llm = MagicMock()
        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            llm_backend=mock_llm,
            metrics=metrics,
            semantic_cache=sem_cache,
            embedder=mock_embedder,
        )
        result = pipeline.query("What is FastAPI?")
        assert result.answer == "A web framework"
        assert result.cached is True
        assert result.metadata.get("cache_type") == "semantic"
        assert result.cost == 0.0
        mock_llm.generate.assert_not_called()

    def test_semantic_cache_miss(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        """Semantic cache miss falls through to next layer."""
        from unittest.mock import MagicMock

        from spiral_04_cognitive.metrics import InMemoryMetrics
        from spiral_05_economics.semantic_cache import SemanticCache

        metrics = InMemoryMetrics()
        sem_cache = SemanticCache(similarity_threshold=0.99)
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.0, 1.0, 0.0]

        # Cache has a very different embedding
        sem_cache.store("unrelated", [1.0, 0.0, 0.0], "wrong answer")

        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            metrics=metrics,
            semantic_cache=sem_cache,
            embedder=mock_embedder,
        )
        result = pipeline.query("completely different query")
        assert result.cached is False  # fell through to L1/noLLM

    def test_semantic_cache_miss_records_metric(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        """Semantic cache miss records metric."""
        from unittest.mock import MagicMock

        from spiral_04_cognitive.metrics import InMemoryMetrics
        from spiral_05_economics.semantic_cache import SemanticCache

        metrics = InMemoryMetrics()
        sem_cache = SemanticCache(similarity_threshold=0.99)
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.0, 1.0, 0.0]

        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            metrics=metrics,
            semantic_cache=sem_cache,
            embedder=mock_embedder,
        )
        pipeline.query("q")
        assert metrics.cache_hit_rate("L0_SEMANTIC") == 0.0

    def test_llm_stores_in_semantic_cache(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        """After LLM generates, answer is stored in semantic cache."""
        from unittest.mock import MagicMock

        from spiral_05_economics.semantic_cache import SemanticCache

        sem_cache = SemanticCache(similarity_threshold=0.9)
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [1.0, 0.0, 0.0]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            answer="generated answer",
            model="llama-3.3-70b-versatile",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            llm_backend=mock_llm,
            semantic_cache=sem_cache,
            embedder=mock_embedder,
        )
        pipeline.query("new question")
        assert sem_cache.size() == 1


class TestCognitiveRetrievalLatency:
    """Test cognitive observation with retrieval_latency metadata."""

    def test_cognitive_sees_retrieval_latency(
        self, budget_tracker, cost_calculator, model_router, circuit_breaker
    ):
        """Cognitive layer receives retrieval_latency when RAG + LLM runs."""
        from unittest.mock import MagicMock

        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [MagicMock(text="chunk", source="doc.md")]
        mock_rag.format_context.return_value = "chunk context"
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            answer="answer",
            model="llama-3.3-70b-versatile",
            input_tokens=50,
            output_tokens=20,
            total_tokens=70,
        )
        mock_cognitive = MagicMock()
        mock_cognitive.enrich.return_value = MagicMock(status="ok", alerts=[], is_regression=False)

        pipeline = LayerPipeline(
            tracker=budget_tracker,
            calculator=cost_calculator,
            router=model_router,
            breaker=circuit_breaker,
            rag_backend=mock_rag,
            llm_backend=mock_llm,
            cognitive=mock_cognitive,
        )
        pipeline.query("test")
        mock_cognitive.enrich.assert_called_once()
        call_kwargs = mock_cognitive.enrich.call_args
        metrics_arg = call_kwargs.kwargs.get("metrics") or call_kwargs[1].get("metrics")
        assert "latency" in metrics_arg
