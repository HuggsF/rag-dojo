# Spiral 05: Economics — Every Token Has a Price
# Unlocked after completing Spiral 04.
#
# Modules:
#   config.py          — Pricing tables, budget limits, tier definitions
#   token_counter.py   — Count tokens (tiktoken) before sending to LLM
#   cost_calculator.py — Pricing table: model → cost per 1M input/output tokens
#   budget_tracker.py  — SQLite: daily/monthly tracking + alerts
#   model_router.py    — Triage: simple query → cheap model, complex → expensive
#   layer_pipeline.py  — L0 cache → L1 keyword → L2 RAG → L3 LLM
#   circuit_breaker.py — Hard stop when budget exceeds limits
#   dashboard.py       — CLI: cost today, month, top model, savings

__all__ = [
    "BudgetExceededError",
    "BudgetTracker",
    "CircuitBreaker",
    "CognitiveObserver",
    "CostCalculator",
    "CostRecord",
    "LayerPipeline",
    "LayerResult",
    "LLMBackend",
    "ModelRouter",
    "RAGBackend",
    "RoutingDecision",
    "TokenCount",
]
