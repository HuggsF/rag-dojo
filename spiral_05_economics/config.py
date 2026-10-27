"""Configuration for Spiral 05 — Economics settings.

Analogy for a C dev:
- PRICING_TABLE = sizeof() for each type — every model has a known cost per token
- DAILY_BUDGET = ulimit for daily spend — hard cap, no exceptions
- MONTHLY_BUDGET = ulimit for monthly spend — the real constraint
- TIER_THRESHOLDS = complexity brackets — like optimization levels (-O0, -O1, -O2)
- CACHE_TTL = cache expiration — like DNS TTL, stale cache is cheaper than fresh LLM call
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base directory for this spiral
_SPIRAL_DIR = Path(__file__).parent

# SQLite budget database
BUDGET_DB_PATH: str = os.environ.get("ECONOMICS_BUDGET_DB", str(_SPIRAL_DIR / "data" / "budget.db"))

# Budget limits (USD)
DAILY_BUDGET: float = float(os.environ.get("ECONOMICS_DAILY_BUDGET", "5.00"))
MONTHLY_BUDGET: float = float(os.environ.get("ECONOMICS_MONTHLY_BUDGET", "150.00"))

# Alert thresholds (percentage of budget)
BUDGET_ALERT_PCT: float = float(os.environ.get("ECONOMICS_ALERT_PCT", "80"))
BUDGET_CRITICAL_PCT: float = float(os.environ.get("ECONOMICS_CRITICAL_PCT", "95"))

# Tier definitions: complexity → model tier
TIER_FAST: str = "fast"
TIER_STANDARD: str = "standard"
TIER_COMPLEX: str = "complex"

# Model routing — complexity score thresholds
COMPLEXITY_FAST_MAX: float = float(os.environ.get("ECONOMICS_FAST_MAX", "0.3"))
COMPLEXITY_STANDARD_MAX: float = float(os.environ.get("ECONOMICS_STANDARD_MAX", "0.7"))
# Above STANDARD_MAX → COMPLEX tier

# Layer pipeline cache
CACHE_TTL_SECONDS: int = int(os.environ.get("ECONOMICS_CACHE_TTL", "3600"))
CACHE_MAX_ENTRIES: int = int(os.environ.get("ECONOMICS_CACHE_MAX", "1000"))

# Keyword search minimum score for L1 hit
KEYWORD_MIN_SCORE: float = float(os.environ.get("ECONOMICS_KEYWORD_MIN", "0.6"))

# Pricing table: model → (cost_per_1M_input, cost_per_1M_output) in USD
# Public pricing data — no proprietary info
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Groq free tier (our primary)
    "llama-3.3-70b-versatile": (0.00, 0.00),
    "llama-3.1-8b-instant": (0.00, 0.00),
    "gemma2-9b-it": (0.00, 0.00),
    # Anthropic tiers (reference pricing)
    "claude-haiku": (0.25, 1.25),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (15.00, 75.00),
    # OpenAI tiers (reference pricing)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Ollama local (free)
    "mistral:7b": (0.00, 0.00),
    "qwen2.5:3b": (0.00, 0.00),
}

# Model → tier mapping
MODEL_TO_TIER: dict[str, str] = {
    "llama-3.1-8b-instant": TIER_FAST,
    "gemma2-9b-it": TIER_FAST,
    "gpt-4o-mini": TIER_FAST,
    "claude-haiku": TIER_FAST,
    "mistral:7b": TIER_FAST,
    "qwen2.5:3b": TIER_FAST,
    "llama-3.3-70b-versatile": TIER_STANDARD,
    "gpt-4o": TIER_STANDARD,
    "claude-sonnet": TIER_STANDARD,
    "claude-opus": TIER_COMPLEX,
}

# Default model per tier
TIER_DEFAULT_MODEL: dict[str, str] = {
    TIER_FAST: "llama-3.1-8b-instant",
    TIER_STANDARD: "llama-3.3-70b-versatile",
    TIER_COMPLEX: "claude-sonnet",
}
