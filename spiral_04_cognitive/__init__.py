# Spiral 04: Cognitive — Observability & Self-Monitoring
# Unlocked after completing Spiral 03.
#
# 3 Cognitive Layers:
#   L0 — Memory (memory_store.py): SQLite temporal comparison, regression/recovery detection
#   L1 — Patterns (pattern_detector.py): regex-based known issue detection (YAML-driven)
#   L2 — Thresholds (threshold_checker.py): configurable metric limits (YAML-driven)

__all__ = [
    "BotBase",
    "BotOutput",
    "CognitiveLayer",
    "EnrichedAnalysis",
    "MemoryStore",
    "PatternDetector",
    "PatternMatch",
    "ThresholdChecker",
    "ThresholdResult",
]
