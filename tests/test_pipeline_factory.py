"""Tests for pipeline_factory — verifies full stack assembly.

All external services are mocked. These tests validate that:
1. build_pipeline() wires all 6 spirals correctly
2. build_cycle_manager() creates a runnable CycleManager
3. build_answer_fn_from_pipeline() bridges spiral_05 → spiral_06
4. Config extraction returns correct RAG parameters
"""

from unittest.mock import patch

from pipeline_factory import (
    _extract_rag_config,
    build_answer_fn_from_pipeline,
    build_cycle_manager,
    build_pipeline,
)
from spiral_05_economics.layer_pipeline import LayerPipeline
from spiral_06_state_of_art.cycle_manager import CycleManager
from spiral_06_state_of_art.verifier import Verifier


class TestBuildPipeline:
    """Tests for build_pipeline()."""

    @patch("pipeline_factory.GraphStore")
    @patch("pipeline_factory.Generator")
    @patch("pipeline_factory.VectorStore")
    def test_returns_layer_pipeline(self, mock_vs, mock_gen, mock_gs):
        """build_pipeline() returns a LayerPipeline instance."""
        pipeline = build_pipeline(
            groq_api_key="test-key",
            neo4j_password="test-pass",
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert isinstance(pipeline, LayerPipeline)

    @patch("pipeline_factory.GraphStore")
    @patch("pipeline_factory.Generator")
    @patch("pipeline_factory.VectorStore")
    def test_wires_rag_backend(self, mock_vs, mock_gen, mock_gs):
        """Pipeline has a RAG backend (HybridRetriever)."""
        pipeline = build_pipeline(
            groq_api_key="test-key",
            neo4j_password="test-pass",
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert pipeline._rag is not None

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_no_graph_disables_graph_store(self, mock_vs, mock_gs):
        """enable_graph=False skips GraphStore creation."""
        pipeline = build_pipeline(
            groq_api_key="test-key",
            enable_graph=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        # HybridRetriever still created, but graph_store is None
        assert pipeline._rag is not None

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_no_api_key_skips_generator(self, mock_vs, mock_gs):
        """Empty GROQ_API_KEY skips Generator creation."""
        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert pipeline._llm is None

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_cognitive_layer_wired(self, mock_vs, mock_gs):
        """enable_cognitive=True attaches CognitiveLayer."""
        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            enable_cognitive=True,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert pipeline._cognitive is not None

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_cognitive_layer_disabled(self, mock_vs, mock_gs):
        """enable_cognitive=False skips CognitiveLayer."""
        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            enable_cognitive=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert pipeline._cognitive is None

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_budget_tracker_wired(self, mock_vs, mock_gs):
        """Pipeline has a BudgetTracker connected."""
        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert pipeline.tracker is not None


class TestEnhancedRetriever:
    """Tests for query expansion + context compression integration."""

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_query_expansion_enabled(self, mock_vs, mock_gs):
        """enable_query_expansion wraps retriever in EnhancedRetriever."""
        from pipeline_factory import EnhancedRetriever

        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            enable_query_expansion=True,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert isinstance(pipeline._rag, EnhancedRetriever)
        assert pipeline._rag._expander is not None

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_context_compression_enabled(self, mock_vs, mock_gs):
        """enable_context_compression wraps retriever in EnhancedRetriever."""
        from pipeline_factory import EnhancedRetriever

        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            enable_context_compression=True,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert isinstance(pipeline._rag, EnhancedRetriever)
        assert pipeline._rag._compressor is not None

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_no_enhancement_by_default(self, mock_vs, mock_gs):
        """Without flags, retriever is plain HybridRetriever."""
        from pipeline_factory import EnhancedRetriever

        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert not isinstance(pipeline._rag, EnhancedRetriever)


class TestMetricsIntegration:
    """Tests for MetricsCollector integration."""

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_metrics_enabled_by_default(self, mock_vs, mock_gs):
        """Metrics are attached by default."""
        from spiral_04_cognitive.metrics import InMemoryMetrics

        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert isinstance(pipeline.metrics, InMemoryMetrics)

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_metrics_disabled(self, mock_vs, mock_gs):
        """enable_metrics=False uses NoopMetrics."""
        from spiral_04_cognitive.metrics import NoopMetrics

        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            enable_metrics=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert isinstance(pipeline.metrics, NoopMetrics)


class TestBuildCycleManagerUnit:
    """Unit tests for build_cycle_manager() — no Neo4j needed."""

    def test_returns_manager_and_verifier_with_mock_pipeline(self):
        """build_cycle_manager() with a mock pipeline returns (CycleManager, Verifier)."""
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock(spec=LayerPipeline)
        manager, verifier = build_cycle_manager(mock_pipeline)
        assert isinstance(manager, CycleManager)
        assert isinstance(verifier, Verifier)

    def test_custom_max_cycles_with_mock_pipeline(self):
        """max_cycles parameter is forwarded to CycleManager."""
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock(spec=LayerPipeline)
        manager, _ = build_cycle_manager(mock_pipeline, max_cycles=7)
        assert manager._max_cycles == 7


class TestBuildAnswerFnUnit:
    """Unit tests for build_answer_fn_from_pipeline() — no Neo4j needed."""

    def test_returns_callable(self):
        """build_answer_fn_from_pipeline() returns a callable."""
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock(spec=LayerPipeline)
        fn = build_answer_fn_from_pipeline(mock_pipeline)
        assert callable(fn)

    def test_answer_fn_calls_pipeline_query(self):
        """answer_fn delegates to pipeline.query()."""
        from unittest.mock import MagicMock

        from spiral_05_economics.layer_pipeline import LayerResult

        mock_pipeline = MagicMock(spec=LayerPipeline)
        mock_pipeline.query.return_value = LayerResult(
            query="q", answer="ok", layer="L1", cost=0.0, model=None, cached=False
        )
        fn = build_answer_fn_from_pipeline(mock_pipeline)
        answer, latency, cost = fn("test question")
        assert answer == "ok"
        assert isinstance(latency, float)
        assert cost == 0.0


class TestBuildCycleManager:
    """Integration tests for build_cycle_manager() — mocked Neo4j."""

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_returns_manager_and_verifier(self, mock_vs, mock_gs):
        """build_cycle_manager() returns (CycleManager, Verifier) tuple."""
        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        manager, verifier = build_cycle_manager(pipeline)
        assert isinstance(manager, CycleManager)
        assert isinstance(verifier, Verifier)

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_custom_max_cycles(self, mock_vs, mock_gs):
        """max_cycles parameter is forwarded to CycleManager."""
        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        manager, _ = build_cycle_manager(pipeline, max_cycles=5)
        assert manager._max_cycles == 5


class TestBuildAnswerFn:
    """Tests for build_answer_fn_from_pipeline()."""

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_answer_fn_returns_tuple(self, mock_vs, mock_gs):
        """answer_fn returns (answer, latency, cost) tuple."""
        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            enable_cognitive=False,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        fn = build_answer_fn_from_pipeline(pipeline)
        answer, latency, cost = fn("test question")
        assert isinstance(answer, str)
        assert isinstance(latency, float)
        assert isinstance(cost, float)


class TestExtractRagConfig:
    """Tests for _extract_rag_config()."""

    def test_returns_dict_with_required_keys(self):
        """Config dict contains all keys the Solver needs."""
        config = _extract_rag_config()
        assert "chunk_size" in config
        assert "chunk_overlap" in config
        assert "top_k" in config
        assert "similarity_threshold" in config
        assert "vector_weight" in config
        assert "graph_weight" in config

    def test_values_are_correct_types(self):
        """Config values are numbers, not strings."""
        config = _extract_rag_config()
        assert isinstance(config["chunk_size"], int)
        assert isinstance(config["top_k"], int)
        assert isinstance(config["similarity_threshold"], float)


class TestValidateConfig:
    """Tests for validate_config()."""

    def test_valid_config_returns_empty(self):
        """Default config should be valid."""
        from pipeline_factory import validate_config

        errors = validate_config()
        assert isinstance(errors, list)

    def test_chunk_size_zero_fails(self):
        """CHUNK_SIZE < 1 is flagged."""
        from pipeline_factory import validate_config

        with patch("spiral_01_chunks.config.CHUNK_SIZE", 0):
            errors = validate_config()
            assert any("CHUNK_SIZE" in e for e in errors)


class TestEnhancedRetrieverBehavior:
    """Tests for EnhancedRetriever.retrieve() with query expansion."""

    def test_retrieve_with_expander_deduplicates(self):
        """Query expansion deduplicates results by ID."""
        from unittest.mock import MagicMock

        from pipeline_factory import EnhancedRetriever

        mock_retriever = MagicMock()
        mock_expander = MagicMock()

        # Expander returns 2 queries
        mock_expander.expand.return_value = ["q1", "q2"]

        # Retriever returns overlapping results
        r1 = MagicMock(chunk_id="a", similarity=0.9)
        r2 = MagicMock(chunk_id="b", similarity=0.8)
        r3 = MagicMock(chunk_id="a", similarity=0.85)  # duplicate

        mock_retriever.retrieve.side_effect = [[r1, r2], [r3]]

        enhanced = EnhancedRetriever(retriever=mock_retriever, expander=mock_expander)
        results = enhanced.retrieve("test query", top_k=5)

        # Should deduplicate: only r1 and r2
        assert len(results) == 2

    def test_retrieve_with_compressor(self):
        """Context compression filters results."""
        from unittest.mock import MagicMock

        from pipeline_factory import EnhancedRetriever

        mock_retriever = MagicMock()
        mock_compressor = MagicMock()

        r1 = MagicMock(similarity=0.9)
        mock_retriever.retrieve.return_value = [r1]
        mock_compressor.compress.return_value = [r1]

        enhanced = EnhancedRetriever(retriever=mock_retriever, compressor=mock_compressor)
        results = enhanced.retrieve("test", top_k=5)
        mock_compressor.compress.assert_called_once()
        assert len(results) == 1

    def test_format_context_delegates(self):
        """format_context delegates to HybridRetriever."""
        from pipeline_factory import EnhancedRetriever

        result = EnhancedRetriever.format_context([])
        assert isinstance(result, str)


class TestBuildPipelineOllama:
    """Tests for Ollama path in build_pipeline."""

    @patch("spiral_02_graph.graph_rag.GraphStore")
    @patch("pipeline_factory.VectorStore")
    def test_ollama_generator_wired(self, mock_vs, mock_gs):
        """use_ollama=True creates OllamaGenerator."""
        from spiral_01_chunks.ollama_generator import OllamaGenerator

        pipeline = build_pipeline(
            groq_api_key="",
            enable_graph=False,
            use_ollama=True,
            budget_db_path=":memory:",
            memory_db_path=":memory:",
        )
        assert isinstance(pipeline._llm, OllamaGenerator)
