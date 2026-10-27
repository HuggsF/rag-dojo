# ADR 002 — Groq free tier + Ollama fallback strategy

**Status:** Accepted
**Date:** 2026-03-21
**Deciders:** Hugo (@HuggsF)

## Context

RAG Dojo needs an LLM for two distinct tasks:

1. **Entity extraction** (Spiral 2): parse text and identify named entities + relations
2. **Answer generation** (Spirals 1–6): synthesize retrieved context into a natural-language answer

Constraints:

- **Zero LLM spend** during spiral training — the budget is for learning, not API fees
- **High quality** — entity extraction requires a capable model (7B+ parameters)
- **No rate-limit failures in CI** — tests mock all LLM calls, but manual pipeline runs must work
- **Local fallback** — when the cloud provider rate-limits, training must not stop

## Decision

**Primary:** Groq API with `llama-3.3-70b-versatile` (free tier)
**Fallback:** Ollama with `mistral:7b` running on Windows host

### Groq (primary)

```python
# spiral_02_graph/config.py
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
```

Groq free tier provides 100k tokens per day at zero cost. The `llama-3.3-70b` model
delivers near-GPT-4 quality for structured extraction tasks at this tier.

### Ollama fallback

```python
# spiral_02_graph/entity_extractor.py
class OllamaEntityExtractor:
    def __init__(self, model: str = "mistral:7b",
                 ollama_url: str = "http://localhost:11434"): ...
```

When Groq returns 429 (rate limited), the `k8s_ingest_ollama.py` script switches to
`OllamaEntityExtractor`, which calls the Ollama server running on the Windows host.
The K8s pod reaches the host via the `host.docker.internal` gateway.

### Budget governance (Spiral 5)

Even though the training budget is $0 (free tiers), the `BudgetTracker` + `CircuitBreaker`
in Spiral 5 enforce hard stops at configurable limits:

```bash
DAILY_BUDGET_USD=1.00   # daily hard stop
MONTHLY_BUDGET_USD=10.00  # monthly hard stop
```

This was a deliberate decision to build the economics layer even on free tiers — the
patterns learned transfer directly to paid deployments.

## Alternatives Rejected

### OpenAI GPT-4o

- **Pro:** Best-in-class quality, stable API
- **Con:** No meaningful free tier — even GPT-4o-mini has per-token cost
- **Con:** Billing setup required — adds friction to onboarding new team members
- **Verdict:** Violates the zero-spend constraint for training.

### Anthropic Claude (API)

- **Pro:** Excellent reasoning, long context
- **Con:** No free tier for API access
- **Verdict:** Same as OpenAI — cost barrier for training.

### Google Gemini free tier

- **Pro:** Generous free tier (60 req/min for Gemini 1.5 Flash)
- **Con:** Groq is 5–10x faster for inference (purpose-built inference hardware)
- **Con:** Adding a second cloud provider increases .env complexity
- **Verdict:** Groq already covers the need at higher speed.

### Self-hosted vLLM on K8s

- **Pro:** No rate limits, full control
- **Con:** Requires GPU node — Docker Desktop K8s has no GPU passthrough
- **Con:** ~6GB model weights in the container image (already avoided via .dockerignore)
- **Verdict:** Infrastructure cost > training benefit at this stage.

## Consequences

**Positive:**
- Zero API cost during all 6 spirals of training
- 70B-parameter quality for entity extraction (better graphs)
- Fallback path tested and operational (`k8s_ingest_ollama.py` + 179 tests)
- Budget governance patterns established before production costs become real

**Negative:**
- 100k token/day limit means large ingest batches must be spread across days
- Groq model availability changes — `llama-3.3-70b-versatile` could be deprecated
- Ollama requires local install on each dev machine (`winget install Ollama.Ollama`)

**Mitigation:** The `LLM_PROVIDER` env var allows switching to any provider without
code changes. The `OllamaEntityExtractor` demonstrates the pattern for adding new
providers — implement the same interface, register in the factory.
