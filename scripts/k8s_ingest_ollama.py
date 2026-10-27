"""Ingest chunks into Neo4j knowledge graph using Ollama (local LLM).

Use this script when Groq rate limit is hit. Runs inside the rag-test pod
and connects to Ollama on the Windows host via host.docker.internal.

Usage (from Windows):
    kubectl exec rag-test -n default -- python scripts/k8s_ingest_ollama.py

    # Custom Ollama URL or model:
    kubectl exec rag-test -n default -- python scripts/k8s_ingest_ollama.py \
        --ollama-url http://host.docker.internal:11434 \
        --model mistral:7b

Prerequisites:
    - Ollama running on Windows host with model pulled (ollama pull mistral:7b)
    - ChromaDB populated (run spiral_01 ingest first)
    - Neo4j K8s pod running (bolt://neo4j:7687)
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from spiral_01_chunks.store import VectorStore  # noqa: E402
from spiral_02_graph.entity_extractor import OllamaEntityExtractor  # noqa: E402
from spiral_02_graph.graph_store import GraphStore  # noqa: E402


def ingest_via_ollama(
    ollama_url: str = "http://host.docker.internal:11434",
    model: str = "mistral:7b",
    batch_size: int = 25,
) -> dict:
    """Extract entities from all ChromaDB chunks and persist to Neo4j.

    Returns summary dict with counts.
    """
    store = VectorStore()
    extractor = OllamaEntityExtractor(model=model, base_url=ollama_url)
    graph = GraphStore()

    nodes_before = graph.node_count
    edges_before = graph.edge_count
    print(f"Neo4j before: {nodes_before} nodes, {edges_before} edges")

    results = store._collection.get(include=["documents", "metadatas"])
    total = len(results["ids"])
    print(f"Processing {total} chunks via Ollama ({model})...")

    success = 0
    fail = 0
    total_entities = 0
    total_relations = 0
    start = time.time()

    for i in range(total):
        chunk_id = results["ids"][i]
        text = results["documents"][i]
        try:
            result = extractor.extract_from_chunk(text, chunk_id)
            if result.entities or result.relations:
                graph.persist_extraction(result)
                total_entities += len(result.entities)
                total_relations += len(result.relations)
            success += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  FAIL [{chunk_id}]: {str(e)[:120]}")

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{total}] ok={success} fail={fail} "
                f"entities={total_entities} rels={total_relations} "
                f"({rate:.1f} chunks/s, ETA {eta / 60:.0f}min)",
                flush=True,
            )

    elapsed = time.time() - start
    nodes_after = graph.node_count
    edges_after = graph.edge_count
    graph.close()

    summary = {
        "total_chunks": total,
        "success": success,
        "fail": fail,
        "entities_extracted": total_entities,
        "relations_extracted": total_relations,
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "edges_before": edges_before,
        "edges_after": edges_after,
        "elapsed_seconds": round(elapsed, 1),
        "model": model,
    }

    print(f"\n{'=' * 60}")
    print(f"  Ollama Ingest Complete ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"  Chunks: {success}/{total} ok, {fail} failed")
    print(f"  Extracted: {total_entities} entities, {total_relations} relations")
    print(f"  Neo4j: {nodes_before} → {nodes_after} nodes, {edges_before} → {edges_after} edges")
    print(f"  Model: {model}")
    print(f"{'=' * 60}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest chunks into Neo4j via Ollama")
    parser.add_argument(
        "--ollama-url",
        default="http://host.docker.internal:11434",
        help="Ollama API URL (default: http://host.docker.internal:11434)",
    )
    parser.add_argument(
        "--model",
        default="mistral:7b",
        help="Ollama model to use (default: mistral:7b)",
    )
    args = parser.parse_args()

    summary = ingest_via_ollama(ollama_url=args.ollama_url, model=args.model)
    sys.exit(0 if summary["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
