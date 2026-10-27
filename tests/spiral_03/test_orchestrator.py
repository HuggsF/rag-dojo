"""Tests for spiral_03_orchestration.orchestrator — main orchestration loop."""

from unittest.mock import MagicMock, patch

import pytest

from spiral_03_orchestration.compass import Action, CompassReading
from spiral_03_orchestration.confidence import ConfidenceResult
from spiral_03_orchestration.orchestrator import OrchestrationResult, Orchestrator, main
from spiral_03_orchestration.tension import TensionMetrics
from spiral_03_orchestration.tension_profile import ProfileType, get_profile


class TestOrchestratorInit:
    """Tests for Orchestrator initialization."""

    def test_default_profile(self):
        """Default profile should be STANDARD."""
        orch = Orchestrator()
        assert orch.profile.name == ProfileType.STANDARD

    def test_custom_profile_string(self):
        """Should accept profile as string."""
        orch = Orchestrator(profile="CRITICAL")
        assert orch.profile.name == ProfileType.CRITICAL

    def test_custom_profile_enum(self):
        """Should accept profile as enum."""
        orch = Orchestrator(profile=ProfileType.EXPLORATORY)
        assert orch.profile.name == ProfileType.EXPLORATORY

    def test_set_profile(self):
        """set_profile should change the active profile."""
        orch = Orchestrator()
        orch.set_profile("DESTRUCTIVE")
        assert orch.profile.name == ProfileType.DESTRUCTIVE

    def test_empty_history(self):
        """Fresh orchestrator should have empty history."""
        orch = Orchestrator()
        assert len(orch.history.records) == 0


class TestLazyPipeline:
    """Tests for lazy pipeline loading."""

    def test_pipeline_lazy_load_via_import(self):
        """Lazy load should import and instantiate GraphRAGPipeline."""
        orch = Orchestrator()
        with patch("spiral_02_graph.pipeline.GraphRAGPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance
            result = orch.pipeline
            assert result is mock_instance
            assert orch._pipeline is mock_instance


class TestEvaluateTension:
    """Tests for Orchestrator.evaluate_tension."""

    def test_short_query_low_creativity(self):
        """Short simple query → lower creativity."""
        orch = Orchestrator()
        metrics = orch.evaluate_tension("hello", previous_delta=0.5)
        assert 0.0 <= metrics.creativity <= 1.0
        assert 0.0 <= metrics.logic <= 1.0
        assert metrics.delta_estado == pytest.approx(0.5)

    def test_long_diverse_query_higher_creativity(self):
        """Long query with diverse words → higher creativity."""
        orch = Orchestrator()
        short = orch.evaluate_tension("FastAPI", previous_delta=0.5)
        long_query = (
            "Explica a relacao entre dependency injection middleware "
            "routing authentication FastAPI Starlette Pydantic models"
        )
        long_m = orch.evaluate_tension(long_query, previous_delta=0.5)
        assert long_m.creativity >= short.creativity

    def test_delta_from_previous(self):
        """delta_estado should come from previous_delta."""
        orch = Orchestrator()
        metrics = orch.evaluate_tension("test", previous_delta=0.8)
        assert metrics.delta_estado == pytest.approx(0.8)

    def test_all_values_in_range(self):
        """All metric values should be in [0, 1]."""
        orch = Orchestrator()
        for query in ["", "a", "a b c d e f g h i j k l m n o p q r s t"]:
            m = orch.evaluate_tension(query, previous_delta=0.5)
            assert 0.0 <= m.creativity <= 1.0
            assert 0.0 <= m.logic <= 1.0
            assert 0.0 <= m.delta_estado <= 1.0


class TestMeasurePostDelta:
    """Tests for Orchestrator.measure_post_delta."""

    def test_none_result_returns_zero(self):
        """None result → delta = 0."""
        orch = Orchestrator()
        assert orch.measure_post_delta(None) == pytest.approx(0.0)

    def test_no_answer_low_delta(self):
        """Result with no useful answer → low delta."""
        orch = Orchestrator()
        result = {"answer": "No relevant context found.", "sources": []}
        delta = orch.measure_post_delta(result)
        assert delta == pytest.approx(0.0)

    def test_good_result_positive_delta(self):
        """Good result with answer and sources → positive delta."""
        orch = Orchestrator()
        result = {
            "answer": "FastAPI uses dependency injection via the Depends function. " * 5,
            "sources": [
                {"text": "source1", "score": 0.85},
                {"text": "source2", "score": 0.72},
                {"text": "source3", "score": 0.68},
            ],
        }
        delta = orch.measure_post_delta(result)
        assert delta > 0.0
        assert delta <= 1.0

    def test_more_sources_higher_delta(self):
        """More sources should contribute to higher delta."""
        orch = Orchestrator()
        few_sources = {
            "answer": "Some answer " * 20,
            "sources": [{"text": "s1", "score": 0.8}],
        }
        many_sources = {
            "answer": "Some answer " * 20,
            "sources": [{"text": f"s{i}", "score": 0.8} for i in range(5)],
        }
        delta_few = orch.measure_post_delta(few_sources)
        delta_many = orch.measure_post_delta(many_sources)
        assert delta_many >= delta_few


class TestOrchestrationResult:
    """Tests for OrchestrationResult serialization."""

    def test_to_dict(self, standard_profile, balanced_metrics):
        """to_dict should contain all expected keys."""
        from spiral_03_orchestration.confidence import ConfidenceResult

        confidence = ConfidenceResult(
            score=0.5,
            weighted_creativity=0.3,
            weighted_logic=0.4,
            delta_estado=0.7,
            threshold=0.3,
            passed=True,
            zero_delta_collapse=False,
        )
        result = OrchestrationResult(
            query="test",
            profile=standard_profile,
            pre_tension=balanced_metrics,
            confidence=confidence,
            action=Action.RETRIEVE,
            answer="test answer",
            post_tension=balanced_metrics,
            elapsed_s=1.5,
        )
        d = result.to_dict()
        assert d["query"] == "test"
        assert d["action"] == "RETRIEVE"
        assert d["answer"] == "test answer"
        assert d["elapsed_s"] == pytest.approx(1.5)


class TestOrchestratorAsk:
    """Tests for Orchestrator.ask with mocked pipeline."""

    def test_ask_records_history(self):
        """ask() should record tension in history."""
        orch = Orchestrator()
        mock_pipeline = MagicMock()
        mock_pipeline.ask_hybrid.return_value = {
            "answer": "Test answer " * 20,
            "sources": [{"text": "s1", "score": 0.8}],
        }
        orch._pipeline = mock_pipeline

        result = orch.ask("test question", show_details=False)
        assert len(orch.history.records) == 1
        assert result.answer is not None

    def test_ask_with_mock_pipeline(self):
        """Full ask flow with mocked pipeline."""
        orch = Orchestrator(profile="STANDARD")
        mock_pipeline = MagicMock()
        mock_pipeline.ask_hybrid.return_value = {
            "answer": "FastAPI is a modern framework " * 10,
            "sources": [
                {"text": "source1", "score": 0.85},
                {"text": "source2", "score": 0.72},
            ],
        }
        orch._pipeline = mock_pipeline

        result = orch.ask("Explain FastAPI", show_details=False)
        assert isinstance(result, OrchestrationResult)
        assert result.query == "Explain FastAPI"
        assert result.profile.name == ProfileType.STANDARD

    def test_ask_retrieve_path(self):
        """ask() should execute RETRIEVE when compass decides so."""
        orch = Orchestrator()
        mock_pipeline = MagicMock()
        mock_pipeline.ask_hybrid.return_value = {
            "answer": "FastAPI uses Depends for DI " * 10,
            "sources": [{"text": "s1", "score": 0.9}],
        }
        orch._pipeline = mock_pipeline

        # Mock compass to force RETRIEVE action
        profile = get_profile("STANDARD")
        mock_reading = CompassReading(
            action=Action.RETRIEVE,
            confidence=ConfidenceResult(
                score=0.5,
                weighted_creativity=0.4,
                weighted_logic=0.4,
                delta_estado=0.8,
                threshold=0.3,
                passed=True,
                zero_delta_collapse=False,
            ),
            metrics=TensionMetrics(creativity=0.8, logic=0.8, delta_estado=0.8),
            profile=profile,
            reason="Confidence passed threshold.",
        )
        orch._compass = MagicMock()
        orch._compass.evaluate.return_value = mock_reading

        result = orch.ask("test query", show_details=False)
        assert result.action == Action.RETRIEVE
        assert result.answer is not None
        mock_pipeline.ask_hybrid.assert_called_once()

    def test_ask_with_show_details(self):
        """ask() with show_details=True should call display methods."""
        orch = Orchestrator()
        mock_pipeline = MagicMock()
        mock_pipeline.ask_hybrid.return_value = {
            "answer": "Test answer " * 20,
            "sources": [{"text": "s1", "score": 0.8}],
        }
        orch._pipeline = mock_pipeline

        result = orch.ask("test question", show_details=True)
        assert result.answer is not None

    def test_ask_uses_previous_delta_from_history(self):
        """Second ask should use delta from first ask's result."""
        orch = Orchestrator()
        mock_pipeline = MagicMock()
        mock_pipeline.ask_hybrid.return_value = {
            "answer": "Good answer " * 20,
            "sources": [{"text": "s1", "score": 0.9}],
        }
        orch._pipeline = mock_pipeline

        orch.ask("first question", show_details=False)
        assert len(orch.history.records) == 1

        orch.ask("second question", show_details=False)
        assert len(orch.history.records) == 2

    def test_ask_fallback_when_no_good_answer(self):
        """Fallback cascade should run when retrieval returns no useful answer."""
        orch = Orchestrator()
        mock_pipeline = MagicMock()
        mock_pipeline.ask_hybrid.return_value = {
            "answer": "No relevant context found.",
            "sources": [],
        }
        orch._pipeline = mock_pipeline

        # Force zero delta to trigger fallback
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))

        result = orch.ask("failing query", show_details=False)
        assert len(result.fallback_attempts) > 0

    def test_ask_fallback_with_show_details(self):
        """Fallback with show_details should display messages."""
        orch = Orchestrator()
        mock_pipeline = MagicMock()
        mock_pipeline.ask_hybrid.return_value = {
            "answer": "No relevant context found.",
            "sources": [],
        }
        orch._pipeline = mock_pipeline

        # Force inertia zero law
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))

        result = orch.ask("failing query", show_details=True)
        assert len(result.fallback_attempts) > 0

    def test_ask_fallback_breaks_on_good_answer(self):
        """Fallback cascade should stop when a good answer is found."""
        orch = Orchestrator()
        mock_pipeline = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return {
                    "answer": "Found a good answer " * 10,
                    "sources": [{"text": "s1", "score": 0.85}],
                }
            return {"answer": "No relevant context found.", "sources": []}

        mock_pipeline.ask_hybrid.side_effect = side_effect
        orch._pipeline = mock_pipeline

        # Force fallback
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))
        orch._history.record(TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.01))

        result = orch.ask("query", show_details=False)
        assert result.answer is not None
        assert "Found a good answer" in result.answer


class TestDisplayMethods:
    """Tests for display helper methods."""

    def test_display_post_action_with_none_post_tension(self):
        """_display_post_action should handle None post_tension."""
        orch = Orchestrator()
        from spiral_03_orchestration.confidence import ConfidenceResult

        result = OrchestrationResult(
            query="test",
            profile=orch.profile,
            pre_tension=TensionMetrics(),
            confidence=ConfidenceResult(
                score=0.0,
                weighted_creativity=0.0,
                weighted_logic=0.0,
                delta_estado=0.0,
                threshold=0.3,
                passed=False,
                zero_delta_collapse=True,
            ),
            action=Action.FALLBACK,
            answer=None,
            post_tension=None,
        )
        # Should not raise
        orch._display_post_action(result)

    def test_display_post_action_with_negative_delta(self):
        """_display_post_action should handle negative delta change."""
        orch = Orchestrator()
        from spiral_03_orchestration.confidence import ConfidenceResult

        result = OrchestrationResult(
            query="test",
            profile=orch.profile,
            pre_tension=TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.8),
            confidence=ConfidenceResult(
                score=0.1,
                weighted_creativity=0.25,
                weighted_logic=0.25,
                delta_estado=0.8,
                threshold=0.3,
                passed=False,
                zero_delta_collapse=False,
            ),
            action=Action.RETRIEVE,
            answer="some answer",
            post_tension=TensionMetrics(creativity=0.5, logic=0.5, delta_estado=0.3),
        )
        # Should not raise — delta_change is negative
        orch._display_post_action(result)


class TestMain:
    """Tests for CLI main() function."""

    def test_main_no_args(self):
        """main() with no args should exit with code 1."""
        with patch("sys.argv", ["orchestrator"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_no_query(self):
        """main() with only --profile should exit with code 1."""
        with patch("sys.argv", ["orchestrator", "--profile", "STANDARD"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_with_query(self):
        """main() with valid args should call orchestrator.ask."""
        with patch("sys.argv", ["orchestrator", "--profile", "STANDARD", "test query"]):
            with patch.object(Orchestrator, "ask") as mock_ask:
                mock_ask.return_value = MagicMock()
                main()
                mock_ask.assert_called_once_with("test query")

    def test_main_default_profile(self):
        """main() without --profile should default to STANDARD."""
        with patch("sys.argv", ["orchestrator", "test query"]):
            with patch.object(Orchestrator, "ask") as mock_ask:
                mock_ask.return_value = MagicMock()
                main()
                mock_ask.assert_called_once_with("test query")
