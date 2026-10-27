"""Configuration for Spiral 06 — State of the Art settings.

Analogy for a C dev:
- MAX_CYCLES = recursion depth limit — you don't recurse forever
- MAX_FILES_PER_CYCLE = blast radius — like limiting the scope of a patch
- MAX_PRS_PER_CYCLE = rate limit on external side-effects
- IMMUTABLE_FILES = read-only segments — never overwrite critical config
- SCORE_REGRESSION_THRESHOLD = watchdog — if perf drops > N, auto-revert
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base directory for this spiral
_SPIRAL_DIR = Path(__file__).parent

# Cycle limits (Lei 1: batch de 3 ciclos max)
MAX_CYCLES: int = int(os.environ.get("SOTA_MAX_CYCLES", "3"))

# Blast radius (Lei 3)
MAX_FILES_PER_CYCLE: int = int(os.environ.get("SOTA_MAX_FILES", "5"))
MAX_PRS_PER_CYCLE: int = int(os.environ.get("SOTA_MAX_PRS", "3"))
MAX_ECOSYSTEMS: int = int(os.environ.get("SOTA_MAX_ECOSYSTEMS", "1"))

# Score thresholds
SCORE_REGRESSION_THRESHOLD: float = float(os.environ.get("SOTA_REGRESSION_THRESHOLD", "5.0"))
MIN_IMPROVEMENT_THRESHOLD: float = float(os.environ.get("SOTA_MIN_IMPROVEMENT", "0.01"))

# Benchmark
BENCHMARK_PATH: str = os.environ.get(
    "SOTA_BENCHMARK_PATH", str(_SPIRAL_DIR / "data" / "benchmark.json")
)

# Immutable files — never modified by solver
IMMUTABLE_FILES: set[str] = {
    "CLAUDE.md",
    ".env",
    ".env.example",
    ".github/workflows/ci.yml",
    "requirements.txt",
}

# Circuit breaker triggers (6 from design doc)
CIRCUIT_BREAKER_TRIGGERS: dict[str, float | int] = {
    "budget_exceeded": 1,  # any budget breach
    "consecutive_failures": 3,  # 3 failed cycles in a row
    "score_regression": 5.0,  # points of regression
    "constraint_violation": 1,  # any guardrail violation
    "max_depth": MAX_CYCLES,  # cycle depth limit
    "cost_spike_pct": 200,  # cost > 200% of baseline
}

# Solver allowed config modifications
TUNABLE_PARAMS: dict[str, dict[str, object]] = {
    "chunk_size": {"min": 128, "max": 1024, "step": 64, "env": "CHUNK_SIZE"},
    "chunk_overlap": {"min": 0, "max": 200, "step": 10, "env": "CHUNK_OVERLAP"},
    "top_k": {"min": 1, "max": 20, "step": 1, "env": "TOP_K"},
    "similarity_threshold": {"min": 0.1, "max": 0.9, "step": 0.05, "env": "SIMILARITY_THRESHOLD"},
}

# Reporter output path
REPORT_PATH: str = os.environ.get("SOTA_REPORT_PATH", str(_SPIRAL_DIR / "data" / "reports"))
