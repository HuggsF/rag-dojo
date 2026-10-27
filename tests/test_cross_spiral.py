"""Cross-spiral integration tests.

Validates that spirals 01-06 connect correctly through their interfaces.
All external services (ChromaDB, Neo4j, Groq) are mocked — these tests
verify the WIRING, not the external calls.
"""

from dataclasses import dataclass

import pytest

from spiral_04_cognitive.cognitive_layer import CognitiveLayer
from spiral_05_economics.budget_tracker import BudgetTracker
from spiral_05_economics.circuit_breaker import CircuitBreaker
from spiral_05_economics.cost_calculator import CostCalculator
from spiral_05_economics.layer_pipeline import LayerPipeline
from spiral_05_economics.model_router import ModelRouter
from spiral_06_state_of_art.benchmark_suite import BenchmarkQuestion, BenchmarkSuite
from spiral_06_state_of_art.cycle_manager import CycleManager
from spiral_06_state_of_art.guardrails import Guardrails
from spiral_06_state_of_art.integration import build_answer_fn
from spiral_06_state_of_art.reporter import Reporter
from spiral_06_state_of_art.scorer import Scorer
from spiral_06_state_of_art.verifier import Verifier

# --- Mock backends that satisfy the Protocol interfaces ---


@dataclass
class MockRetrievalResult:
    text: str
    similarity: float = 0.9
    combined_score: float = 0.85
    is_relevant: bool = True


class MockRAGBackend:
    """Mock HybridRetriever satisfying RAGBackend protocol."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self.call_count = 0

    def retrieve(self, query: str, top_k: int = 5) -> list[MockRetrievalResult]:
        self.call_count += 1
        text = self._responses.get(query, f"Context for: {query}")
        return [MockRetrievalResult(text=text)]

    @staticmethod
    def format_context(results: list[MockRetrievalResult], max_tokens: int = 2000) -> str:
        return "\n".join(r.text for r in results)


@dataclass
class MockGenerationResult:
    answer: str
    model: str = "llama-3.3-70b-versatile"
    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150


class MockLLMBackend:
    """Mock Generator satisfying LLMBackend protocol."""

    def __init__(self, answers: dict[str, str] | None = None):
        self._answers = answers or {}
        self.call_count = 0

    def generate(self, context: str, question: str) -> MockGenerationResult:
        self.call_count += 1
        answer = self._answers.get(question, f"Answer based on context: {context[:50]}")
        return MockGenerationResult(answer=answer)


# --- Fixtures ---


@pytest.fixture
def mock_rag():
    return MockRAGBackend(
        responses={
            "What is FastAPI?": "FastAPI is a modern Python web framework for building APIs.",
            "How does middleware work?": "Middleware processes requests before path operations.",
        }
    )


@pytest.fixture
def mock_llm():
    return MockLLMBackend(
        answers={
            "What is FastAPI?": "FastAPI is a fast Python web framework with automatic docs.",
            "How does middleware work?": "Middleware intercepts HTTP requests for preprocessing.",
        }
    )


@pytest.fixture
def cognitive():
    """Real CognitiveLayer with in-memory store."""
    from spiral_04_cognitive.memory_store import MemoryStore
    from spiral_04_cognitive.pattern_detector import PatternDetector
    from spiral_04_cognitive.threshold_checker import ThresholdChecker

    return CognitiveLayer(
        memory=MemoryStore(":memory:"),
        detector=PatternDetector(),
        checker=ThresholdChecker(),
    )


@pytest.fixture
def integrated_pipeline(mock_rag, mock_llm, cognitive, tmp_path):
    """Fully integrated LayerPipeline with all backends."""
    db_path = str(tmp_path / "budget.db")
    tracker = BudgetTracker(db_path=db_path)
    calculator = CostCalculator()
    router = ModelRouter()
    breaker = CircuitBreaker(tracker=tracker)

    return LayerPipeline(
        tracker=tracker,
        calculator=calculator,
        router=router,
        breaker=breaker,
        rag_backend=mock_rag,
        llm_backend=mock_llm,
        cognitive=cognitive,
    )


# --- Tests: Spiral 05 ↔ Spiral 01/02 integration ---


class TestL2RAGIntegration:
    """Verify L2 (RAG) layer calls the retriever backend."""

    def test_rag_backend_called_on_cache_miss(self, integrated_pipeline, mock_rag):
        """Query that misses cache should call RAG backend."""
        result = integrated_pipeline.query("What is FastAPI?")
        assert mock_rag.call_count == 1
        assert result.layer == "L3_LLM"  # RAG feeds into LLM

    def test_rag_only_mode(self, mock_rag, tmp_path):
        """Pipeline with RAG but no LLM returns at L2."""
        db_path = str(tmp_path / "budget.db")
        tracker = BudgetTracker(db_path=db_path)
        pipeline = LayerPipeline(
            tracker=tracker,
            calculator=CostCalculator(),
            router=ModelRouter(),
            breaker=CircuitBreaker(tracker=tracker),
            rag_backend=mock_rag,
            llm_backend=None,
        )
        result = pipeline.query("What is FastAPI?")
        assert result.layer == "L2_RAG"
        assert "FastAPI" in result.answer


class TestL3LLMIntegration:
    """Verify L3 (LLM) layer calls the generator backend."""

    def test_llm_backend_called(self, integrated_pipeline, mock_llm):
        """LLM backend should be called with RAG context."""
        result = integrated_pipeline.query("What is FastAPI?")
        assert mock_llm.call_count == 1
        assert "FastAPI" in result.answer
        assert result.cost >= 0

    def test_llm_tracks_real_cost(self, integrated_pipeline):
        """Cost should reflect actual token usage."""
        result = integrated_pipeline.query("What is FastAPI?")
        # MockLLMBackend returns 100 input + 50 output tokens
        assert result.metadata.get("input_tokens") == 100
        assert result.metadata.get("output_tokens") == 50

    def test_circuit_breaker_blocks_llm(self, mock_rag, mock_llm, tmp_path):
        """Open circuit breaker should prevent LLM calls."""
        db_path = str(tmp_path / "budget.db")
        tracker = BudgetTracker(db_path=db_path, daily_budget=0.0001)
        # Record cost to exceed the tiny budget
        tracker.record_cost(
            model="test", input_tokens=1000, output_tokens=1000, cost=1.0, query="test"
        )
        breaker = CircuitBreaker(tracker=tracker)

        pipeline = LayerPipeline(
            tracker=tracker,
            calculator=CostCalculator(),
            router=ModelRouter(),
            breaker=breaker,
            rag_backend=mock_rag,
            llm_backend=mock_llm,
        )
        result = pipeline.query("What is FastAPI?")
        assert mock_llm.call_count == 0
        assert "circuit breaker" in result.answer


# --- Tests: Spiral 05 ↔ Spiral 04 integration ---


class TestCognitiveIntegration:
    """Verify cognitive layer monitors pipeline results."""

    def test_cognitive_enriches_result(self, integrated_pipeline):
        """Pipeline results should include cognitive analysis."""
        result = integrated_pipeline.query("What is FastAPI?")
        assert "cognitive_analysis" in result.metadata
        analysis = result.metadata["cognitive_analysis"]
        assert "status" in analysis
        assert "alerts" in analysis

    def test_cognitive_detects_regression(self, mock_rag, mock_llm, tmp_path):
        """Cognitive layer detects status changes across queries."""
        from spiral_04_cognitive.memory_store import MemoryStore
        from spiral_04_cognitive.pattern_detector import PatternDetector
        from spiral_04_cognitive.threshold_checker import ThresholdChecker

        cognitive = CognitiveLayer(
            memory=MemoryStore(":memory:"),
            detector=PatternDetector(),
            checker=ThresholdChecker(),
        )
        db_path = str(tmp_path / "budget.db")
        tracker = BudgetTracker(db_path=db_path)
        pipeline = LayerPipeline(
            tracker=tracker,
            calculator=CostCalculator(),
            router=ModelRouter(),
            breaker=CircuitBreaker(tracker=tracker),
            rag_backend=mock_rag,
            llm_backend=mock_llm,
            cognitive=cognitive,
        )

        # First query: establishes baseline
        r1 = pipeline.query("What is FastAPI?")
        assert r1.metadata["cognitive_analysis"]["status"] == "ok"

        # Second query: should not show regression (both OK)
        r2 = pipeline.query("How does middleware work?")
        assert r2.metadata["cognitive_analysis"]["is_regression"] is False


# --- Tests: Spiral 06 ↔ Spiral 05 integration ---


class TestSpiralSixIntegration:
    """Verify cycle_manager can use pipeline via answer_fn bridge."""

    def test_build_answer_fn_returns_tuple(self, integrated_pipeline):
        """answer_fn should return (answer, latency, cost) tuple."""
        fn = build_answer_fn(integrated_pipeline)
        answer, latency, cost = fn("What is FastAPI?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert latency >= 0
        assert cost >= 0

    def test_cycle_manager_with_real_pipeline(self, integrated_pipeline):
        """CycleManager should run cycles using the integrated pipeline."""
        fn = build_answer_fn(integrated_pipeline)

        suite = BenchmarkSuite(
            questions=[
                BenchmarkQuestion(
                    question="What is FastAPI?",
                    ground_truth="FastAPI is a modern Python web framework.",
                    keywords=["fast", "python", "web framework"],
                    category="basics",
                    difficulty="easy",
                ),
            ]
        )
        scorer = Scorer()
        verifier = Verifier(suite=suite, scorer=scorer)
        reporter = Reporter()
        guardrails = Guardrails()

        # Get initial scorecard
        _, initial_scorecard = verifier.run_benchmark(fn)

        manager = CycleManager(
            config={"chunk_size": 512, "top_k": 5},
            scorer=scorer,
            guardrails=guardrails,
            verifier=verifier,
            reporter=reporter,
            max_cycles=1,
        )

        result = manager.run(initial_scorecard, fn)
        assert result.cycles_run >= 1
        assert result.report is not None

    def test_no_backends_fallback(self, tmp_path):
        """Pipeline without backends returns placeholder, not crash."""
        db_path = str(tmp_path / "budget.db")
        tracker = BudgetTracker(db_path=db_path)
        pipeline = LayerPipeline(
            tracker=tracker,
            calculator=CostCalculator(),
            router=ModelRouter(),
            breaker=CircuitBreaker(tracker=tracker),
        )
        result = pipeline.query("test question")
        assert result.layer == "L3_LLM"
        assert "no RAG or LLM backend" in result.answer
