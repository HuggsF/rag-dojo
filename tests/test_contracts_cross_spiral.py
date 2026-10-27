"""Tests for cross-spiral contracts and config validation."""

from contracts import (
    CostTracker,
    GenerationBackend,
    RetrievalBackend,
    SelfImprover,
    verify_all_spirals,
)
from pipeline_factory import validate_config


class TestValidateConfig:
    """Tests for cross-spiral config validation."""

    def test_valid_config_returns_empty(self):
        """Default config should be valid."""
        errors = validate_config()
        assert errors == []

    def test_validates_weight_sum(self, monkeypatch):
        """Detects VECTOR_WEIGHT + GRAPH_WEIGHT != 1.0."""
        import spiral_02_graph.config as cfg

        monkeypatch.setattr(cfg, "VECTOR_WEIGHT", 0.3)
        monkeypatch.setattr(cfg, "GRAPH_WEIGHT", 0.3)
        errors = validate_config()
        assert any("VECTOR_WEIGHT" in e for e in errors)

    def test_validates_overlap_less_than_size(self, monkeypatch):
        """Detects CHUNK_OVERLAP >= CHUNK_SIZE."""
        import spiral_01_chunks.config as cfg

        monkeypatch.setattr(cfg, "CHUNK_OVERLAP", 1000)
        monkeypatch.setattr(cfg, "CHUNK_SIZE", 500)
        errors = validate_config()
        assert any("CHUNK_OVERLAP" in e for e in errors)

    def test_validates_top_k_positive(self, monkeypatch):
        """Detects TOP_K < 1."""
        import spiral_01_chunks.config as cfg

        monkeypatch.setattr(cfg, "TOP_K", 0)
        errors = validate_config()
        assert any("TOP_K" in e for e in errors)


class TestVerifyAllSpirals:
    """Tests for verify_all_spirals contract checking."""

    def test_all_spirals_satisfy_contracts(self):
        """All current implementations should pass."""
        errors = verify_all_spirals()
        assert errors == [], f"Contract violations: {errors}"


class TestProtocolSatisfaction:
    """Direct isinstance checks for cross-spiral Protocols."""

    def test_retriever_satisfies_retrieval_backend(self):
        from spiral_01_chunks.retriever import Retriever

        assert issubclass(Retriever, RetrievalBackend)

    def test_hybrid_retriever_satisfies_retrieval_backend(self):
        from spiral_02_graph.hybrid_retriever import HybridRetriever

        assert issubclass(HybridRetriever, RetrievalBackend)

    def test_generator_satisfies_generation_backend(self):
        from spiral_01_chunks.generator import Generator

        assert issubclass(Generator, GenerationBackend)

    def test_ollama_generator_satisfies_generation_backend(self):
        from spiral_01_chunks.ollama_generator import OllamaGenerator

        assert issubclass(OllamaGenerator, GenerationBackend)

    def test_budget_tracker_satisfies_cost_tracker(self):
        from spiral_05_economics.budget_tracker import BudgetTracker

        assert issubclass(BudgetTracker, CostTracker)

    def test_cycle_manager_satisfies_self_improver(self):
        from spiral_06_state_of_art.cycle_manager import CycleManager

        assert issubclass(CycleManager, SelfImprover)


class TestDashboardMetricsWiring:
    """Tests for Dashboard wiring with InMemoryMetrics."""

    def test_summary_includes_metrics_when_wired(self, tmp_path):
        from spiral_04_cognitive.metrics import InMemoryMetrics
        from spiral_05_economics.budget_tracker import BudgetTracker
        from spiral_05_economics.dashboard import Dashboard

        metrics = InMemoryMetrics()
        metrics.record_latency("test_op", 0.5)
        metrics.record_tokens("test_model", 100)

        tracker = BudgetTracker(db_path=str(tmp_path / "test.db"))
        dashboard = Dashboard(tracker=tracker, metrics=metrics)
        summary = dashboard.summary()

        assert "pipeline_metrics" in summary
        assert isinstance(summary["pipeline_metrics"], dict)
        tracker.close()

    def test_summary_without_metrics(self, tmp_path):
        from spiral_05_economics.budget_tracker import BudgetTracker
        from spiral_05_economics.dashboard import Dashboard

        tracker = BudgetTracker(db_path=str(tmp_path / "test.db"))
        dashboard = Dashboard(tracker=tracker)
        summary = dashboard.summary()

        assert "pipeline_metrics" not in summary
        tracker.close()
