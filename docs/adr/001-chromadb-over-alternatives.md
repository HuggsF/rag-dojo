# ADR 001 — ChromaDB over alternatives (FAISS, Pinecone, Weaviate)

**Status:** Accepted
**Date:** 2026-03-21
**Deciders:** Hugo (@HuggsF)

## Context

RAG Dojo needs a vector store to persist embeddings between sessions. The candidates
evaluated were FAISS, Pinecone, Weaviate, and ChromaDB. The project runs on:

- Local Windows dev machine (unit tests with mocks)
- Kubernetes cluster (canonical runtime, Docker Desktop)
- Zero cloud budget for infrastructure — only LLM API costs are acceptable

Key constraints:

1. **Must run embedded** (in-process) without a separate server for CI
2. **Must persist to disk** so re-ingest is not required on pod restart
3. **Free and open-source** — no SaaS subscriptions in training environment
4. **Python-native API** — no HTTP overhead for the core retrieval path
5. **Simple to operate** — spiral training, not infra training

## Decision

Use **ChromaDB** (`chromadb>=0.4.22`) as the vector store.

Configuration (`spiral_01_chunks/config.py`):

```python
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_dojo")
```

The ChromaDB data lives in a Kubernetes PersistentVolumeClaim (`rag-data`) mounted
at `/app/data` in the `rag-test` pod — surviving pod restarts without re-ingestion.

## Alternatives Rejected

### FAISS (Facebook AI Similarity Search)

- **Pro:** Fastest ANN search, GPU support, battle-tested at scale
- **Con:** In-memory only — persistence requires manual serialization to disk (`faiss.write_index`)
- **Con:** No metadata filtering — would need a separate SQLite for source/chunk_index lookups
- **Verdict:** Too much glue code for a training project. ChromaDB bundles persistence + metadata.

### Pinecone

- **Pro:** Managed, scalable, production-grade
- **Con:** SaaS — requires API key, has free tier limits (1 index, 100k vectors)
- **Con:** Network latency on every query — breaks the "zero external deps in CI" rule
- **Con:** Data leaves the machine — conflicts with Rule 1 (only public open-source data)
- **Verdict:** Wrong tool for a local dev/K8s training setup.

### Weaviate

- **Pro:** Full GraphQL API, rich filtering, hybrid search built-in
- **Con:** Requires a separate Docker container (or K8s deployment) — complexity overhead
- **Con:** Heavy (~500MB image) for a training project
- **Con:** Python client v4 had breaking changes mid-2024 — maintenance burden
- **Verdict:** Overkill. Graph retrieval in Spiral 2 uses Neo4j; two graph DBs is too much.

## Consequences

**Positive:**
- Zero infra setup — `VectorStore()` works immediately in CI (temp dirs) and in K8s (PVC)
- Metadata stored alongside vectors — source, chunk_index, strategy all queryable
- Similarity search + metadata filter in one call
- `chroma_db/` directory is portable — can `kubectl cp` out for inspection

**Negative:**
- Not suitable for >10M vectors (ChromaDB performance degrades at scale)
- No built-in ANN index tuning (HNSW params are fixed)
- If the project grows past Spiral 6, migration to Weaviate/Qdrant would be needed

**Mitigation:** The `VectorStore` class (`spiral_01_chunks/store.py`) wraps ChromaDB behind
a stable interface — swapping the backend requires changing only that file.
