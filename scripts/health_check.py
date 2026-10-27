"""Semantic health check for K8s probes.

Used as readinessProbe in pod-rag-test.yaml. Checks that the core
RAG components are functional, not just importable.

Exit codes:
    0 = HEALTHY (all checks pass)
    1 = UNHEALTHY (one or more checks failed)
"""

from __future__ import annotations

import sys


def _check_chromadb() -> str | None:
    """Check ChromaDB is up and has data. Returns error string or None."""
    from spiral_01_chunks.store import VectorStore

    store = VectorStore()
    stats = store.stats()
    if stats.get("total_chunks", 0) == 0:
        return "ChromaDB: empty collection"
    return None


def _check_embedder() -> str | None:
    """Check embedding model produces correct dimensions. Returns error string or None."""
    from spiral_01_chunks.embedder import Embedder

    emb = Embedder()
    result = emb.embed_text("health check")
    if len(result) != 384:
        return f"Embedder: wrong dims {len(result)}"
    return None


def _check_neo4j() -> str | None:
    """Check Neo4j connectivity. Returns error string or None."""
    from spiral_02_graph.graph_store import GraphStore

    gs = GraphStore()
    gs.close()
    return None


def run_checks(
    check_chromadb: bool = True,
    check_embedder: bool = True,
    check_neo4j: bool = True,
) -> list[str]:
    """Run health checks and return list of error strings.

    Empty list means healthy. Neo4j failures are non-fatal (graceful degradation).
    """
    errors: list[str] = []

    if check_chromadb:
        try:
            err = _check_chromadb()
            if err:
                errors.append(err)
        except Exception as e:
            errors.append(f"ChromaDB: {e}")

    if check_embedder:
        try:
            err = _check_embedder()
            if err:
                errors.append(err)
        except Exception as e:
            errors.append(f"Embedder: {e}")

    if check_neo4j:
        try:
            _check_neo4j()
        except Exception:
            pass  # Neo4j offline is acceptable (graceful degradation)

    return errors


def check_health() -> None:
    """Run semantic health checks and exit with appropriate code."""
    errors = run_checks()

    if errors:
        print("UNHEALTHY:", "; ".join(errors))
        sys.exit(1)
    print("HEALTHY")
    sys.exit(0)


if __name__ == "__main__":
    check_health()
