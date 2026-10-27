"""Ingest chunks into Neo4j with exponential backoff and resume support.

Analogy for a C dev:
    Like a reliable write loop with fsync — if the disk is busy (429),
    back off exponentially instead of hammering. If the process crashes,
    resume from the last successful write using a progress journal.

Usage (inside rag-test pod):
    python scripts/ingest_with_retry.py
    python scripts/ingest_with_retry.py --limit 50
    python scripts/ingest_with_retry.py --no-resume
    python scripts/ingest_with_retry.py --ollama
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Defaults
DEFAULT_PROGRESS_PATH = "./data/ingest_progress.json"
INITIAL_BACKOFF = 5.0
MAX_BACKOFF = 60.0
PROGRESS_REPORT_INTERVAL = 25


def load_progress(path: str) -> set[str]:
    """Load set of already-ingested chunk IDs from the progress file.

    Returns an empty set if the file doesn't exist or is invalid.
    """
    progress_path = Path(path)
    if not progress_path.exists():
        return set()
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        return set(data.get("ingested_ids", []))
    except (json.JSONDecodeError, KeyError, TypeError):
        return set()


def save_progress(path: str, ingested: set[str]) -> None:
    """Save the set of ingested chunk IDs to disk.

    Creates parent directories if needed. Writes atomically-ish
    by writing to the path directly (good enough for single-writer).
    """
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"ingested_ids": sorted(ingested), "count": len(ingested)}
    progress_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_chunk_id(chunk: Any) -> str:
    """Build a stable chunk ID from a Chunk object."""
    return f"{chunk.source}::chunk_{chunk.chunk_index}"


def ingest_all(
    chunks: list[Any] | None = None,
    extractor: Any | None = None,
    graph_store: Any | None = None,
    limit: int = 0,
    resume: bool = True,
    progress_path: str = DEFAULT_PROGRESS_PATH,
    use_ollama: bool = False,
) -> dict[str, Any]:
    """Ingest all chunks into Neo4j, skipping already-done ones if resume=True.

    Args:
        chunks: List of Chunk objects. If None, loads from VectorStore.
        extractor: EntityExtractor instance. If None, creates one.
        graph_store: GraphStore instance. If None, creates one.
        limit: Max chunks to process (0 = all).
        resume: If True, skip chunks already in progress file.
        progress_path: Path to the JSON progress file.
        use_ollama: If True, use OllamaEntityExtractor instead of Groq.

    Returns:
        Dict with ingested_count, skipped_count, failed_count, errors.
    """
    # Lazy imports — only needed when actually running
    if extractor is None:
        if use_ollama:
            from spiral_02_graph.entity_extractor import OllamaEntityExtractor

            extractor = OllamaEntityExtractor()
        else:
            from spiral_02_graph.entity_extractor import EntityExtractor

            extractor = EntityExtractor()

    if graph_store is None:
        from spiral_02_graph.graph_store import GraphStore

        graph_store = GraphStore()

    if chunks is None:
        from spiral_01_chunks.store import VectorStore

        store = VectorStore()
        all_data = store._collection.get(include=["documents", "metadatas"])
        # Build lightweight chunk-like objects
        from dataclasses import dataclass

        @dataclass
        class _MiniChunk:
            text: str
            source: str
            chunk_index: int

        chunks = []
        for i, doc_id in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
            chunks.append(
                _MiniChunk(
                    text=all_data["documents"][i] if all_data["documents"] else "",
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", i),
                )
            )

    # Load progress for resume
    ingested: set[str] = load_progress(progress_path) if resume else set()
    initial_count = len(ingested)

    # Apply limit
    if limit > 0:
        chunks = chunks[:limit]

    stats: dict[str, Any] = {
        "ingested_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "errors": [],
    }

    backoff = INITIAL_BACKOFF

    for i, chunk in enumerate(chunks):
        chunk_id = _make_chunk_id(chunk)

        # Skip already-ingested
        if chunk_id in ingested:
            stats["skipped_count"] += 1
            continue

        # Extract and persist with retry
        success = False
        while not success:
            try:
                result = extractor.extract_from_chunk(chunk.text, chunk_id)
                graph_store.persist_extraction(result)
                ingested.add(chunk_id)
                save_progress(progress_path, ingested)
                stats["ingested_count"] += 1
                backoff = INITIAL_BACKOFF  # reset backoff on success
                success = True
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    print(f"  Rate limited on {chunk_id}, backing off {backoff:.0f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                else:
                    stats["failed_count"] += 1
                    stats["errors"].append({"chunk_id": chunk_id, "error": error_str})
                    print(f"  FAILED {chunk_id}: {error_str}")
                    success = True  # move on, don't retry non-rate-limit errors

        # Progress report
        processed = stats["ingested_count"] + stats["skipped_count"] + stats["failed_count"]
        if processed > 0 and processed % PROGRESS_REPORT_INTERVAL == 0:
            print(
                f"  Progress: {processed}/{len(chunks)} "
                f"(ingested={stats['ingested_count']}, "
                f"skipped={stats['skipped_count']}, "
                f"failed={stats['failed_count']})"
            )

    # Final summary
    stats["total_in_progress_file"] = len(ingested)
    stats["resumed_from"] = initial_count
    return stats


def main() -> None:
    """CLI entry point for ingest_with_retry."""
    parser = argparse.ArgumentParser(description="Ingest chunks into Neo4j with retry/resume")
    parser.add_argument("--limit", type=int, default=0, help="Max chunks to ingest (0=all)")
    parser.add_argument(
        "--no-resume", action="store_true", help="Start fresh (ignore progress file)"
    )
    parser.add_argument(
        "--progress-path",
        default=DEFAULT_PROGRESS_PATH,
        help="Path to progress JSON file",
    )
    parser.add_argument("--ollama", action="store_true", help="Use Ollama instead of Groq")
    args = parser.parse_args()

    print("Starting ingest with retry/resume...")
    stats = ingest_all(
        limit=args.limit,
        resume=not args.no_resume,
        progress_path=args.progress_path,
        use_ollama=args.ollama,
    )
    print(f"\nDone! {stats}")
    sys.exit(1 if stats["failed_count"] > 0 else 0)


if __name__ == "__main__":
    main()
