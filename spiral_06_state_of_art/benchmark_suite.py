"""Benchmark suite — 20 questions with ground truth for measuring RAG quality.

Analogy for a C dev:
    Like a test harness with known inputs and expected outputs.
    You run your program, compare output vs expected, compute pass rate.
    Here: question → RAG → answer, compare vs ground_truth → precision/recall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with expected answer and keywords."""

    question: str
    ground_truth: str
    keywords: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard

    def keyword_recall(self, answer: str) -> float:
        """Fraction of expected keywords found in the answer."""
        if not self.keywords:
            return 1.0
        answer_lower = answer.lower()
        hits = sum(1 for kw in self.keywords if kw.lower() in answer_lower)
        return hits / len(self.keywords)


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark question."""

    question: str
    expected: str
    actual: str
    keyword_recall: float
    latency_s: float
    cost: float
    hit: bool  # True if keyword_recall >= threshold

    def __repr__(self) -> str:
        status = "HIT" if self.hit else "MISS"
        return (
            f"BenchmarkResult({status}, recall={self.keyword_recall:.2f}, "
            f"latency={self.latency_s:.3f}s, cost=${self.cost:.6f})"
        )


@dataclass
class BenchmarkReport:
    """Aggregated results from running the full benchmark suite."""

    results: list[BenchmarkResult]
    total_questions: int
    precision_at_k: float  # fraction of hits
    mean_keyword_recall: float
    mean_latency_s: float
    total_cost: float
    pass_rate: float  # fraction of questions with recall >= threshold

    def __repr__(self) -> str:
        return (
            f"BenchmarkReport(precision={self.precision_at_k:.2f}, "
            f"recall={self.mean_keyword_recall:.2f}, "
            f"latency={self.mean_latency_s:.3f}s, "
            f"cost=${self.total_cost:.4f}, "
            f"pass_rate={self.pass_rate:.0%})"
        )


# Default benchmark questions — FastAPI documentation (public data only)
DEFAULT_QUESTIONS: list[dict[str, object]] = [
    {
        "question": "What is FastAPI?",
        "ground_truth": "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.",
        "keywords": ["fast", "web framework", "python", "type hints"],
        "category": "basics",
        "difficulty": "easy",
    },
    {
        "question": "How do you create a path parameter in FastAPI?",
        "ground_truth": "Path parameters are declared in the path string using curly braces and as function parameters with type annotations.",
        "keywords": ["path", "curly braces", "function parameter", "type"],
        "category": "routing",
        "difficulty": "easy",
    },
    {
        "question": "What is dependency injection in FastAPI?",
        "ground_truth": "FastAPI has a dependency injection system using Depends() that allows declaring dependencies that are resolved at request time.",
        "keywords": ["depends", "injection", "request", "resolved"],
        "category": "di",
        "difficulty": "medium",
    },
    {
        "question": "How does middleware work in FastAPI?",
        "ground_truth": "Middleware in FastAPI processes every request before it reaches the path operation and every response before returning it.",
        "keywords": ["middleware", "request", "response", "process"],
        "category": "middleware",
        "difficulty": "medium",
    },
    {
        "question": "What is the purpose of Pydantic models in FastAPI?",
        "ground_truth": "Pydantic models are used for request body validation, serialization, and documentation generation in FastAPI.",
        "keywords": ["pydantic", "validation", "serialization", "request body"],
        "category": "validation",
        "difficulty": "easy",
    },
    {
        "question": "How do you handle authentication in FastAPI?",
        "ground_truth": "FastAPI supports OAuth2 with Password flow, JWT tokens, and API keys through its security utilities.",
        "keywords": ["oauth2", "jwt", "security", "token"],
        "category": "security",
        "difficulty": "medium",
    },
    {
        "question": "What is the difference between Query and Path parameters?",
        "ground_truth": "Path parameters are part of the URL path, while Query parameters are appended after ? in the URL.",
        "keywords": ["path", "query", "url", "parameter"],
        "category": "routing",
        "difficulty": "easy",
    },
    {
        "question": "How do you configure CORS in FastAPI?",
        "ground_truth": "CORS is configured using CORSMiddleware, specifying allowed origins, methods, and headers.",
        "keywords": ["cors", "middleware", "origins", "headers"],
        "category": "middleware",
        "difficulty": "medium",
    },
    {
        "question": "What are background tasks in FastAPI?",
        "ground_truth": "Background tasks run after the response is sent, useful for operations like sending emails or processing data.",
        "keywords": ["background", "after response", "task"],
        "category": "async",
        "difficulty": "medium",
    },
    {
        "question": "How does FastAPI generate automatic API documentation?",
        "ground_truth": "FastAPI automatically generates OpenAPI schema and provides Swagger UI and ReDoc documentation interfaces.",
        "keywords": ["openapi", "swagger", "documentation", "automatic"],
        "category": "docs",
        "difficulty": "easy",
    },
    {
        "question": "What is the lifespan event in FastAPI?",
        "ground_truth": "Lifespan events handle startup and shutdown logic for the application, replacing the deprecated on_event decorators.",
        "keywords": ["lifespan", "startup", "shutdown"],
        "category": "lifecycle",
        "difficulty": "medium",
    },
    {
        "question": "How do you handle file uploads in FastAPI?",
        "ground_truth": "File uploads use UploadFile parameter type which provides async file handling with metadata like filename and content type.",
        "keywords": ["uploadfile", "file", "upload", "async"],
        "category": "request",
        "difficulty": "medium",
    },
    {
        "question": "What is APIRouter in FastAPI?",
        "ground_truth": "APIRouter allows organizing path operations into groups, similar to Flask blueprints, for modular application structure.",
        "keywords": ["apirouter", "organize", "group", "modular"],
        "category": "structure",
        "difficulty": "easy",
    },
    {
        "question": "How do you implement WebSockets in FastAPI?",
        "ground_truth": "FastAPI supports WebSockets through the WebSocket class with accept, send, and receive methods for real-time communication.",
        "keywords": ["websocket", "real-time", "accept", "send"],
        "category": "websocket",
        "difficulty": "hard",
    },
    {
        "question": "What is the role of response_model in FastAPI?",
        "ground_truth": "response_model defines the output schema, filters the response data, and generates documentation for the endpoint.",
        "keywords": ["response_model", "schema", "filter", "output"],
        "category": "validation",
        "difficulty": "medium",
    },
    {
        "question": "How does FastAPI handle async and sync functions?",
        "ground_truth": "FastAPI runs async def endpoints in the event loop and regular def endpoints in a thread pool for compatibility.",
        "keywords": ["async", "event loop", "thread pool", "def"],
        "category": "async",
        "difficulty": "hard",
    },
    {
        "question": "What are HTTPException and custom exception handlers?",
        "ground_truth": "HTTPException raises HTTP errors with status codes, and custom exception handlers process specific exception types globally.",
        "keywords": ["httpexception", "status code", "handler", "error"],
        "category": "errors",
        "difficulty": "medium",
    },
    {
        "question": "How do you test FastAPI applications?",
        "ground_truth": "FastAPI applications are tested using TestClient from Starlette which allows sending requests without running a server.",
        "keywords": ["testclient", "test", "starlette", "request"],
        "category": "testing",
        "difficulty": "medium",
    },
    {
        "question": "What is the purpose of tags in FastAPI path operations?",
        "ground_truth": "Tags organize endpoints into groups in the API documentation, making it easier to navigate large APIs.",
        "keywords": ["tags", "organize", "documentation", "group"],
        "category": "docs",
        "difficulty": "easy",
    },
    {
        "question": "How do you implement rate limiting in FastAPI?",
        "ground_truth": "Rate limiting in FastAPI is typically implemented using middleware or third-party libraries like slowapi with configurable limits per endpoint.",
        "keywords": ["rate limit", "middleware", "slowapi", "endpoint"],
        "category": "security",
        "difficulty": "hard",
    },
]


class BenchmarkSuite:
    """Benchmark suite for evaluating RAG quality."""

    def __init__(
        self,
        questions: list[BenchmarkQuestion] | None = None,
        recall_threshold: float = 0.5,
    ) -> None:
        if questions is not None:
            self._questions = questions
        else:
            self._questions = [
                BenchmarkQuestion(**q)  # type: ignore[arg-type]
                for q in DEFAULT_QUESTIONS
            ]
        self._recall_threshold = recall_threshold

    @property
    def questions(self) -> list[BenchmarkQuestion]:
        """Return the list of benchmark questions."""
        return list(self._questions)

    @property
    def size(self) -> int:
        """Number of questions in the suite."""
        return len(self._questions)

    def evaluate_answer(
        self,
        question: BenchmarkQuestion,
        answer: str,
        latency_s: float = 0.0,
        cost: float = 0.0,
    ) -> BenchmarkResult:
        """Evaluate a single answer against its ground truth."""
        recall = question.keyword_recall(answer)
        return BenchmarkResult(
            question=question.question,
            expected=question.ground_truth,
            actual=answer,
            keyword_recall=recall,
            latency_s=latency_s,
            cost=cost,
            hit=recall >= self._recall_threshold,
        )

    def compile_report(self, results: list[BenchmarkResult]) -> BenchmarkReport:
        """Compile individual results into an aggregated report."""
        if not results:
            return BenchmarkReport(
                results=[],
                total_questions=0,
                precision_at_k=0.0,
                mean_keyword_recall=0.0,
                mean_latency_s=0.0,
                total_cost=0.0,
                pass_rate=0.0,
            )

        hits = sum(1 for r in results if r.hit)
        total = len(results)

        return BenchmarkReport(
            results=results,
            total_questions=total,
            precision_at_k=hits / total,
            mean_keyword_recall=sum(r.keyword_recall for r in results) / total,
            mean_latency_s=sum(r.latency_s for r in results) / total,
            total_cost=sum(r.cost for r in results),
            pass_rate=hits / total,
        )

    def load_from_file(self, path: str) -> None:
        """Load benchmark questions from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self._questions = [BenchmarkQuestion(**q) for q in data.get("questions", data)]

    def save_to_file(self, path: str) -> None:
        """Save benchmark questions to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "question": q.question,
                "ground_truth": q.ground_truth,
                "keywords": q.keywords,
                "category": q.category,
                "difficulty": q.difficulty,
            }
            for q in self._questions
        ]
        with open(path, "w") as f:
            json.dump({"questions": data}, f, indent=2)
