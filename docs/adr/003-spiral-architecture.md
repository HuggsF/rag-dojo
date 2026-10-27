# ADR 003 — Spiral architecture for progressive RAG mastery

**Status:** Accepted
**Date:** 2026-03-21
**Deciders:** Hugo (@HuggsF)

## Context

RAG Dojo is a training project, not a production system. The goal is to teach three
engineers with different backgrounds (Go/platform engineering, data science, backend)
the full RAG stack — from basic chunking to self-improving AI systems.

Key pedagogical constraints:

1. **Each concept must build on the previous** — no jumping to reranking before retrieval works
2. **Each spiral must be independently testable** — a failing Spiral 4 must not break Spiral 1
3. **Complexity must be visible** — the architecture should make the learning progression explicit
4. **Production patterns from day one** — no "we'll add tests later", no mocks for production code

## Decision

Structure the codebase as **6 progressive spirals**, each in its own Python package:

```
spiral_01_chunks/       # Chunking + embeddings + vector retrieval
spiral_02_graph/        # Entity extraction + Neo4j graph + hybrid retrieval
spiral_03_orchestration/ # Tension field + confidence scoring + fallback cascade
spiral_04_cognitive/    # Memory store + pattern detection + observability
spiral_05_economics/    # Token counting + cost tracking + budget governance
spiral_06_state_of_art/ # Self-improving PROPOSE→SOLVE→VERIFY loop
```

Each spiral follows a fixed pattern:
- **Doc (B):** Architecture document explaining the concept
- **Dojo (A):** Kata exercises with tests (TDD)
- **Missão (C):** Integration with previous spirals

### Dependency order

```
Spiral 1 ← no deps
Spiral 2 ← Spiral 1 (uses VectorStore, Embedder)
Spiral 3 ← Spiral 2 (uses HybridRetriever)
Spiral 4 ← Spirals 1-3 (observes the full pipeline)
Spiral 5 ← Spirals 1-4 (tracks costs across all layers)
Spiral 6 ← All spirals (runs the self-improvement loop on the full system)
```

Dependencies flow forward only — no circular imports between spirals.

### Interface contracts

Each spiral exposes a stable public API via `__all__` in `__init__.py`.
Cross-spiral dependencies use only exported symbols — never internal modules.

This is enforced by the CI pipeline: if a spiral's `__init__.py` doesn't export
a symbol, downstream spirals can't import it.

## Alternatives Rejected

### Monolithic package

- **Pro:** Simpler imports, no cross-package boundary friction
- **Con:** No isolation — a bug in economics code breaks retrieval tests
- **Con:** Makes it impossible to say "this engineer has completed Spiral 2"
- **Verdict:** Destroys the pedagogical structure.

### Feature branches per spiral

- **Pro:** Clean git history per concept
- **Con:** Merges become complex as spirals build on each other
- **Con:** Hard to run cross-spiral integration tests on a single branch
- **Verdict:** The spiral packages on a single branch give the same isolation
  without merge complexity.

### Microservices (each spiral as an HTTP service)

- **Pro:** True service isolation, language-agnostic
- **Con:** Network latency on every inter-spiral call kills the tight feedback loop
- **Con:** Docker Compose / K8s service mesh overhead is infra training, not RAG training
- **Verdict:** Overkill. Python packages with typed interfaces provide sufficient contracts.

### Jupyter notebooks

- **Pro:** Interactive, great for exploration
- **Con:** Not testable (pytest doesn't run notebooks by default)
- **Con:** No coverage enforcement — a spiral "completed" in a notebook has no quality bar
- **Con:** Can't be imported by downstream spirals
- **Verdict:** Used only for analysis/visualization, never for production code.

## Consequences

**Positive:**
- Coverage per spiral is independently verifiable (`check_coverage.py --min-percent 90`)
- A new engineer can onboard to Spiral 1 without understanding Spirals 2–6
- The architecture itself is the learning material — reading the spiral dependency
  graph teaches RAG system design
- CI enforces the architecture: failing Spiral 3 tests don't gate Spiral 1 releases

**Negative:**
- Cross-spiral refactors are expensive — changing a Spiral 1 interface may break all 5 downstream spirals
- Package boundaries create import verbosity (`from spiral_02_graph.hybrid_retriever import HybridRetriever`)
- New engineers must understand the spiral ordering before making changes

**Mitigation:** Interface stability is enforced by the `REGRA CRÍTICA` section in task documents
(see `CLAUDE.md`). Breaking changes to public interfaces require explicit review by the maintainer.
The `__all__` exports in each `__init__.py` make the public API surface explicit and auditable.
