"""Pipeline factory — assembles the full 6-spiral RAG stack.

Analogy for a C dev:
    Each spiral is a .o file. This module is the linker.
    It resolves all symbols, wires dependencies, and produces the executable.

Usage:
    from pipeline_factory import build_pipeline, build_answer_fn_from_pipeline

    pipeline = build_pipeline()                      # full stack
    result = pipeline.query("What is FastAPI?")      # L0→L1→L2→L3 + cognitive

    answer_fn = build_answer_fn_from_pipeline(pipeline)  # bridge to spiral_06
    manager = build_cycle_manager(pipeline)
    run_result = manager.run(scorecard, answer_fn)       # self-improvement loop
"""

from __future__ import annotations

from spiral_01_chunks.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, GROQ_API_KEY
from spiral_01_chunks.context_compressor import ContextCompressor
from spiral_01_chunks.generator import Generator
from spiral_01_chunks.ollama_generator import OllamaGenerator
from spiral_01_chunks.query_expander import QueryExpander
from spiral_01_chunks.shutdown import register_for_cleanup
from spiral_01_chunks.store import VectorStore
from spiral_02_graph.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from spiral_02_graph.graph_store import GraphStore
from spiral_02_graph.hybrid_retriever import HybridRetriever
from spiral_04_cognitive.cognitive_layer import CognitiveLayer
from spiral_04_cognitive.memory_store import MemoryStore
from spiral_04_cognitive.metrics import InMemoryMetrics, MetricsCollector, NoopMetrics
from spiral_04_cognitive.pattern_detector import PatternDetector
from spiral_04_cognitive.threshold_checker import ThresholdChecker
from spiral_05_economics.budget_tracker import BudgetTracker
from spiral_05_economics.circuit_breaker import CircuitBreaker
from spiral_05_economics.config import BUDGET_DB_PATH
from spiral_05_economics.cost_calculator import CostCalculator
from spiral_05_economics.layer_pipeline import LayerPipeline
from spiral_05_economics.model_router import ModelRouter
from spiral_06_state_of_art.benchmark_suite import BenchmarkSuite
from spiral_06_state_of_art.cycle_manager import CycleManager
from spiral_06_state_of_art.guardrails import Guardrails
from spiral_06_state_of_art.integration import build_answer_fn
from spiral_06_state_of_art.reporter import Reporter
from spiral_06_state_of_art.scorer import Scorer
from spiral_06_state_of_art.verifier import Verifier


class EnhancedRetriever:
    """Wraps HybridRetriever with optional query expansion and context compression.

    Satisfies RAGBackend Protocol — drop-in replacement for HybridRetriever.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        expander: QueryExpander | None = None,
        compressor: ContextCompressor | None = None,
    ) -> None:
        self._retriever = retriever
        self._expander = expander
        self._compressor = compressor

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve with optional query expansion."""
        if self._expander:
            queries = self._expander.expand(query)
            seen_ids: set[str] = set()
            all_results: list = []
            for q in queries[:3]:  # cap at 3 expansions
                for r in self._retriever.retrieve(q, top_k=top_k):
                    rid = getattr(r, "chunk_id", None) or getattr(r, "source", id(r))
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_results.append(r)
            # Sort by similarity descending, take top_k
            all_results.sort(key=lambda r: getattr(r, "similarity", 0), reverse=True)
            results = all_results[:top_k]
        else:
            results = self._retriever.retrieve(query, top_k=top_k)

        if self._compressor and results:
            results = self._compressor.compress(results, query)

        return results

    @staticmethod
    def format_context(results: list, max_tokens: int = 2000) -> str:
        """Delegate to HybridRetriever.format_context."""
        return HybridRetriever.format_context(results, max_tokens)


def build_pipeline(
    *,
    chroma_dir: str = CHROMA_PERSIST_DIR,
    chroma_collection: str = CHROMA_COLLECTION_NAME,
    neo4j_uri: str = NEO4J_URI,
    neo4j_user: str = NEO4J_USER,
    neo4j_password: str = NEO4J_PASSWORD,
    groq_api_key: str = GROQ_API_KEY,
    budget_db_path: str = BUDGET_DB_PATH,
    memory_db_path: str | None = None,
    enable_cognitive: bool = True,
    enable_graph: bool = True,
    use_ollama: bool = False,
    ollama_url: str = "http://host.docker.internal:11434",
    ollama_model: str = "qwen2.5:3b",
    enable_query_expansion: bool = False,
    enable_context_compression: bool = False,
    enable_metrics: bool = True,
) -> LayerPipeline:
    """Assemble a fully-integrated LayerPipeline with all 6 spirals wired.

    Args:
        chroma_dir: ChromaDB persistence directory.
        chroma_collection: ChromaDB collection name.
        neo4j_uri: Neo4j bolt URI.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        groq_api_key: Groq API key for LLM generation.
        budget_db_path: Path to SQLite budget database.
        memory_db_path: Path to cognitive memory SQLite (None = default).
        enable_cognitive: Whether to attach cognitive monitoring.
        enable_graph: Whether to use graph retrieval (requires Neo4j).

    Returns:
        Fully-wired LayerPipeline ready for .query() calls.
    """
    # --- Spiral 01: Vector store ---
    vector_store = VectorStore(
        persist_dir=chroma_dir,
        collection_name=chroma_collection,
    )

    # --- Spiral 04: Metrics (created early so other components can use it) ---
    metrics: MetricsCollector = InMemoryMetrics() if enable_metrics else NoopMetrics()

    # --- Spiral 02: Graph store + Hybrid retriever ---
    graph_store = None
    if enable_graph and neo4j_password:
        graph_store = GraphStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
        )
        register_for_cleanup(graph_store)

    retriever = HybridRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        metrics=metrics if enable_metrics else None,
    )

    # --- Spiral 01: LLM generator ---
    generator = None
    if use_ollama:
        generator = OllamaGenerator(base_url=ollama_url, model=ollama_model)
    elif groq_api_key:
        generator = Generator(api_key=groq_api_key)

    # --- Spiral 04: Cognitive layer ---
    cognitive = None
    if enable_cognitive:
        memory_store = MemoryStore(db_path=memory_db_path)
        register_for_cleanup(memory_store)
        cognitive = CognitiveLayer(
            memory=memory_store,
            detector=PatternDetector(),
            checker=ThresholdChecker(),
        )

    # --- Spiral 05: Economics ---
    tracker = BudgetTracker(db_path=budget_db_path)
    register_for_cleanup(tracker)
    calculator = CostCalculator()
    router = ModelRouter(calculator=calculator)
    breaker = CircuitBreaker(tracker=tracker)

    # --- Spiral 01: Query expansion + Context compression (opt-in) ---
    rag_backend = retriever
    if enable_query_expansion or enable_context_compression:
        expander = QueryExpander() if enable_query_expansion else None
        compressor = ContextCompressor() if enable_context_compression else None
        rag_backend = EnhancedRetriever(
            retriever=retriever,
            expander=expander,
            compressor=compressor,
        )

    # --- Assembly ---
    pipeline = LayerPipeline(
        tracker=tracker,
        calculator=calculator,
        router=router,
        breaker=breaker,
        rag_backend=rag_backend,
        llm_backend=generator,
        cognitive=cognitive,
        metrics=metrics,
    )
    return pipeline


def build_cycle_manager(
    pipeline: LayerPipeline,
    *,
    max_cycles: int = 3,
    benchmark_suite: BenchmarkSuite | None = None,
) -> tuple[CycleManager, Verifier]:
    """Build a CycleManager wired to a LayerPipeline.

    Args:
        pipeline: The assembled LayerPipeline.
        max_cycles: Maximum improvement cycles.
        benchmark_suite: Custom benchmark questions (None = defaults).

    Returns:
        Tuple of (CycleManager, Verifier) — verifier needed for initial benchmark.
    """
    scorer = Scorer()
    suite = benchmark_suite or BenchmarkSuite()
    verifier = Verifier(suite=suite, scorer=scorer)
    guardrails = Guardrails()
    reporter = Reporter()

    # Extract current RAG config for the solver
    config = _extract_rag_config()

    manager = CycleManager(
        config=config,
        scorer=scorer,
        guardrails=guardrails,
        verifier=verifier,
        reporter=reporter,
        max_cycles=max_cycles,
    )

    return manager, verifier


def build_answer_fn_from_pipeline(pipeline: LayerPipeline):
    """Create an AnswerFn (spiral_06) from a LayerPipeline (spiral_05).

    Convenience wrapper around spiral_06_state_of_art.integration.build_answer_fn.
    """
    return build_answer_fn(pipeline)


def validate_config() -> list[str]:
    """Validate cross-spiral config consistency.

    Returns a list of error messages. Empty list = all good.
    Checks invariants that span multiple spirals' configs.
    """
    from spiral_01_chunks.config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
    from spiral_02_graph.config import GRAPH_WEIGHT, VECTOR_WEIGHT

    errors: list[str] = []

    # Vector + graph weights must sum to 1.0
    weight_sum = VECTOR_WEIGHT + GRAPH_WEIGHT
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(
            f"VECTOR_WEIGHT ({VECTOR_WEIGHT}) + GRAPH_WEIGHT ({GRAPH_WEIGHT}) "
            f"= {weight_sum}, expected 1.0"
        )

    # Chunk overlap must be less than chunk size
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) >= CHUNK_SIZE ({CHUNK_SIZE}), "
            f"overlap must be smaller than chunk size"
        )

    # TOP_K must be positive
    if TOP_K < 1:
        errors.append(f"TOP_K ({TOP_K}) must be >= 1")

    # Chunk size must be positive
    if CHUNK_SIZE < 1:
        errors.append(f"CHUNK_SIZE ({CHUNK_SIZE}) must be >= 1")

    return errors


def _extract_rag_config() -> dict[str, object]:
    """Extract current RAG config parameters for the Solver.

    The Solver operates on these parameters — proposing and applying changes.
    """
    from spiral_01_chunks.config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
    from spiral_02_graph.config import GRAPH_WEIGHT, VECTOR_WEIGHT

    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "similarity_threshold": 0.3,
        "vector_weight": VECTOR_WEIGHT,
        "graph_weight": GRAPH_WEIGHT,
    }
