"""Store — ChromaDB vector store wrapper.

Analogy for a C dev:
- ChromaDB is a heap allocator for vectors
- add_documents() = malloc + write (allocate space, store data)
- query() = search through heap for matching blocks
- stats() = memory usage report
- The "collection" is like a memory pool — isolated namespace

Under the hood, ChromaDB:
1. Stores your text + metadata
2. Generates embeddings (or accepts pre-computed ones)
3. Builds an index for fast similarity search (HNSW algorithm)
4. Persists to disk (sqlite + parquet files)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import chromadb
from chromadb.config import Settings

from spiral_01_chunks.chunker import Chunk
from spiral_01_chunks.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
)
from spiral_01_chunks.embedder import get_embedder


class QueryResult(TypedDict):
    """Typed contract for store.query() return values.

    Every consumer of VectorStore.query() depends on this shape.
    If you rename a key here, update retriever.py (and future consumers).
    """

    text: str
    metadata: dict[str, Any]
    similarity: float
    id: str


class VectorStore:
    """ChromaDB wrapper for storing and querying document chunks."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )
        self._embedder = get_embedder()

    @property
    def count(self) -> int:
        """Number of documents in the store."""
        return self._collection.count()

    def add_documents(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_batch(texts)

        ids = [f"{c.source}::chunk_{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "strategy": c.strategy,
                "token_estimate": c.token_estimate,
                **c.metadata,
            }
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """Query the vector store for similar chunks.

        Returns list of results with text, metadata, and similarity score.
        """
        query_embedding = self._embedder.embed_text(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=min(top_k, self.count) if self.count > 0 else 1,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        output: list[QueryResult] = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            similarity = 1 - distance  # ChromaDB returns distance, we want similarity

            output.append(
                QueryResult(
                    text=doc,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},  # type: ignore[typeddict-item]
                    similarity=round(similarity, 4),
                    id=results["ids"][0][i] if results["ids"] else "",
                )
            )

        return output

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        count = self.count
        return {
            "collection": self._collection.name,
            "total_chunks": count,
            "embedding_model": EMBEDDING_MODEL,
            "similarity_metric": "cosine",
        }

    def clear(self) -> None:
        """Delete all documents from the collection."""
        if self.count > 0:
            all_ids = self._collection.get()["ids"]
            if all_ids:
                self._collection.delete(ids=all_ids)
