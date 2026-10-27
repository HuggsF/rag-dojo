"""Tests for spiral_05_economics.model_router — complexity-based model triage."""

from spiral_05_economics.config import TIER_COMPLEX, TIER_FAST, TIER_STANDARD
from spiral_05_economics.model_router import RoutingDecision


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_has_required_fields(self):
        rd = RoutingDecision(
            query="test",
            tier=TIER_FAST,
            model="llama-3.1-8b-instant",
            complexity_score=0.1,
            reason="simple keyword query",
        )
        assert rd.tier == TIER_FAST
        assert rd.model == "llama-3.1-8b-instant"

    def test_repr(self):
        rd = RoutingDecision(
            query="test",
            tier=TIER_FAST,
            model="llama-3.1-8b-instant",
            complexity_score=0.1,
            reason="simple",
        )
        assert TIER_FAST in repr(rd)


class TestModelRouter:
    """Tests for ModelRouter."""

    def test_simple_query_routes_fast(self, model_router):
        result = model_router.route("What is FastAPI?")
        assert isinstance(result, RoutingDecision)
        assert result.tier == TIER_FAST
        assert result.complexity_score <= 0.3

    def test_medium_query_routes_standard(self, model_router):
        result = model_router.route(
            "Compare middleware patterns between FastAPI and Django, "
            "explaining the trade-offs of each approach in production."
        )
        assert result.tier in (TIER_STANDARD, TIER_COMPLEX)

    def test_complex_query_routes_complex(self, model_router):
        result = model_router.route(
            "Design an event-driven architecture with FastAPI that handles "
            "real-time websocket connections, implements CQRS pattern, "
            "integrates with message queues for async processing, "
            "and includes circuit breakers for external service calls. "
            "Consider scalability, fault tolerance, and observability."
        )
        assert result.tier == TIER_COMPLEX
        assert result.complexity_score > 0.7

    def test_empty_query_routes_fast(self, model_router):
        result = model_router.route("")
        assert result.tier == TIER_FAST

    def test_score_is_between_0_and_1(self, model_router):
        result = model_router.route("Explain dependency injection in FastAPI")
        assert 0.0 <= result.complexity_score <= 1.0

    def test_route_returns_valid_model(self, model_router):
        result = model_router.route("What is a REST API?")
        assert result.model is not None
        assert len(result.model) > 0

    def test_forced_tier(self, model_router):
        result = model_router.route("simple question", force_tier=TIER_COMPLEX)
        assert result.tier == TIER_COMPLEX

    def test_complexity_factors(self, model_router):
        """Longer queries with technical terms should score higher."""
        simple_score = model_router.score_complexity("What is FastAPI?")
        complex_score = model_router.score_complexity(
            "Design an event-driven microservices architecture with "
            "CQRS, event sourcing, saga pattern, and circuit breakers "
            "using FastAPI, considering horizontal scalability and "
            "fault tolerance across multiple availability zones."
        )
        assert complex_score > simple_score
