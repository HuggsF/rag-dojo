"""Configuration for Spiral 03 — Orchestration settings.

Analogy for a C dev:
- TENSION_PROFILES = predefined struct initializers for different operating modes
- ZERO_DELTA_THRESHOLD = epsilon for float comparison — below this, delta is "zero"
- FALLBACK_MAX_RETRIES = retry limit before giving up (like MAX_RETRANSMISSIONS)
- ZERO_DELTA_STREAK_LIMIT = watchdog timer — after N consecutive zeros, auto-swap strategy
"""

import os

from dotenv import load_dotenv

from spiral_02_graph.config import (  # noqa: F401
    GRAPH_WEIGHT,
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_WEIGHT,
)

load_dotenv()

# Tension evaluation
ZERO_DELTA_THRESHOLD: float = float(os.environ.get("ZERO_DELTA_THRESHOLD", "0.05"))
ZERO_DELTA_STREAK_LIMIT: int = int(os.environ.get("ZERO_DELTA_STREAK_LIMIT", "3"))

# Default tension profile
DEFAULT_PROFILE: str = os.environ.get("TENSION_PROFILE", "STANDARD")

# Confidence thresholds per profile
CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))

# Fallback cascade
FALLBACK_MAX_RETRIES: int = int(os.environ.get("FALLBACK_MAX_RETRIES", "7"))

# Orchestrator
MAX_ORCHESTRATION_ROUNDS: int = int(os.environ.get("MAX_ORCHESTRATION_ROUNDS", "3"))
