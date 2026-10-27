"""Configuration for Spiral 04 — Cognitive Layer settings.

Analogy for a C dev:
- MEMORY_DB_PATH = path to the journal file (like /var/log/syslog but structured)
- PATTERNS_YAML = config file for known error signatures (like /etc/fail2ban/filter.d/)
- THRESHOLDS_YAML = ulimits for your bot metrics (breach = alert)
- FAIL_STREAK_ALERT = watchdog counter — after N consecutive fails, trigger alert
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base directory for this spiral
_SPIRAL_DIR = Path(__file__).parent

# SQLite memory store
MEMORY_DB_PATH: str = os.environ.get("COGNITIVE_MEMORY_DB", str(_SPIRAL_DIR / "data" / "memory.db"))

# YAML config files
PATTERNS_YAML_PATH: str = os.environ.get(
    "COGNITIVE_PATTERNS_YAML", str(_SPIRAL_DIR / "patterns.yaml")
)
THRESHOLDS_YAML_PATH: str = os.environ.get(
    "COGNITIVE_THRESHOLDS_YAML", str(_SPIRAL_DIR / "thresholds.yaml")
)

# L0 Memory settings
MAX_OUTPUT_LENGTH: int = int(os.environ.get("COGNITIVE_MAX_OUTPUT_LEN", "2000"))
HISTORY_QUERY_LIMIT: int = int(os.environ.get("COGNITIVE_HISTORY_LIMIT", "100"))

# L1 Pattern detection
FAIL_STREAK_ALERT: int = int(os.environ.get("COGNITIVE_FAIL_STREAK", "3"))

# Status classification
STATUS_OK: str = "ok"
STATUS_FAIL: str = "fail"
STATUS_PARTIAL: str = "partial"
