"""Shared fixtures for Spiral 06 tests."""

import pytest

from spiral_06_state_of_art.benchmark_suite import (
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkSuite,
)
from spiral_06_state_of_art.guardrails import Guardrails
from spiral_06_state_of_art.proposer import Proposer
from spiral_06_state_of_art.reporter import Reporter
from spiral_06_state_of_art.scorer import ScoreCard, Scorer
from spiral_06_state_of_art.solver import Solver
from spiral_06_state_of_art.verifier import Verifier


@pytest.fixture
def sample_questions() -> list[BenchmarkQuestion]:
    """Small set of benchmark questions for fast tests."""
    return [
        BenchmarkQuestion(
            question="What is FastAPI?",
            ground_truth="FastAPI is a modern Python web framework.",
            keywords=["fast", "python", "web framework"],
            category="basics",
            difficulty="easy",
        ),
        BenchmarkQuestion(
            question="How does middleware work?",
            ground_truth="Middleware processes requests before path operations.",
            keywords=["middleware", "request", "process"],
            category="middleware",
            difficulty="medium",
        ),
        BenchmarkQuestion(
            question="What is dependency injection?",
            ground_truth="Depends() resolves dependencies at request time.",
            keywords=["depends", "injection", "request"],
            category="di",
            difficulty="medium",
        ),
    ]


@pytest.fixture
def benchmark_suite(sample_questions):
    """BenchmarkSuite with 3 test questions."""
    return BenchmarkSuite(questions=sample_questions)


@pytest.fixture
def scorer():
    """Default Scorer."""
    return Scorer()


@pytest.fixture
def good_report(sample_questions, benchmark_suite) -> BenchmarkReport:
    """BenchmarkReport with good results."""
    results = []
    for q in sample_questions:
        results.append(
            benchmark_suite.evaluate_answer(
                q,
                q.ground_truth,
                latency_s=0.3,
                cost=0.001,
            )
        )
    return benchmark_suite.compile_report(results)


@pytest.fixture
def bad_report(sample_questions, benchmark_suite) -> BenchmarkReport:
    """BenchmarkReport with poor results."""
    results = []
    for q in sample_questions:
        results.append(
            benchmark_suite.evaluate_answer(
                q,
                "I don't know.",
                latency_s=5.0,
                cost=0.05,
            )
        )
    return benchmark_suite.compile_report(results)


@pytest.fixture
def good_scorecard(scorer, good_report) -> ScoreCard:
    """ScoreCard from good benchmark."""
    return scorer.score_benchmark(good_report)


@pytest.fixture
def bad_scorecard(scorer, bad_report) -> ScoreCard:
    """ScoreCard from bad benchmark."""
    return scorer.score_benchmark(bad_report)


@pytest.fixture
def sample_config() -> dict[str, object]:
    """Sample RAG config for solver tests."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5,
        "similarity_threshold": 0.3,
    }


@pytest.fixture
def solver(sample_config):
    """Solver with sample config."""
    return Solver(sample_config)


@pytest.fixture
def proposer(sample_config):
    """Proposer with sample config."""
    return Proposer(current_config=sample_config)


@pytest.fixture
def guardrails():
    """Default Guardrails."""
    return Guardrails()


@pytest.fixture
def verifier(benchmark_suite, scorer):
    """Verifier with test suite."""
    return Verifier(suite=benchmark_suite, scorer=scorer)


@pytest.fixture
def reporter():
    """Default Reporter."""
    return Reporter()
