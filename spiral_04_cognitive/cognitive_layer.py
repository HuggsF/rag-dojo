"""Cognitive Layer — L0 + L1 + L2 unified enrichment engine.

Analogy for a C dev:
- CognitiveLayer = the main() that chains all analysis passes
- enrich() = preprocessor pipeline: raw output → tokenize → analyze → annotated result
- L0 (Memory) = compare with historical data — regression detection
- L1 (Patterns) = regex scan — known issue classification
- L2 (Thresholds) = bounds check — metric validation
- The output is an EnrichedAnalysis struct with all findings aggregated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spiral_04_cognitive.config import FAIL_STREAK_ALERT, STATUS_FAIL, STATUS_OK, STATUS_PARTIAL
from spiral_04_cognitive.memory_store import MemoryStore
from spiral_04_cognitive.pattern_detector import PatternDetector, PatternMatch
from spiral_04_cognitive.threshold_checker import ThresholdChecker, ThresholdResult


@dataclass
class EnrichedAnalysis:
    """Result of cognitive enrichment — all 3 layers combined.

    Like a structured log entry with annotations from every analysis pass.
    """

    raw_output: str
    status: str
    pattern_matches: list[PatternMatch] = field(default_factory=list)
    threshold_violations: list[ThresholdResult] = field(default_factory=list)
    is_regression: bool = False
    is_recovery: bool = False
    fail_streak: int = 0
    recommendation: str | None = None
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging/display."""
        return {
            "status": self.status,
            "pattern_matches": [m.to_dict() for m in self.pattern_matches],
            "threshold_violations": [v.to_dict() for v in self.threshold_violations],
            "is_regression": self.is_regression,
            "is_recovery": self.is_recovery,
            "fail_streak": self.fail_streak,
            "recommendation": self.recommendation,
            "alerts": self.alerts,
        }


class CognitiveLayer:
    """Unified cognitive engine combining L0 (Memory) + L1 (Patterns) + L2 (Thresholds).

    Usage:
        cognitive = CognitiveLayer()
        analysis = cognitive.enrich("docs_bot", "check", raw_output, metrics)
    """

    def __init__(
        self,
        memory: MemoryStore | None = None,
        detector: PatternDetector | None = None,
        checker: ThresholdChecker | None = None,
    ):
        self._memory = memory
        self._detector = detector
        self._checker = checker

    @property
    def memory(self) -> MemoryStore:
        """Lazy-load memory store."""
        if self._memory is None:
            self._memory = MemoryStore()
        return self._memory

    @property
    def detector(self) -> PatternDetector:
        """Lazy-load pattern detector."""
        if self._detector is None:
            self._detector = PatternDetector()
        return self._detector

    @property
    def checker(self) -> ThresholdChecker:
        """Lazy-load threshold checker."""
        if self._checker is None:
            self._checker = ThresholdChecker()
        return self._checker

    def enrich(
        self,
        bot_name: str,
        command: str,
        raw_output: str,
        metrics: dict[str, float] | None = None,
    ) -> EnrichedAnalysis:
        """Run all 3 cognitive layers and produce enriched analysis.

        Pipeline: L1 (patterns) → L2 (thresholds) → determine status →
                  save to L0 (memory) → L0 temporal analysis → build alerts.
        """
        # L1: Pattern detection
        pattern_matches = self.detector.detect(raw_output)
        pattern_status = self.detector.classify(raw_output)

        # L2: Threshold checking
        threshold_violations: list[ThresholdResult] = []
        if metrics:
            threshold_violations = self.checker.get_violations(metrics)

        # Determine final status (worst wins)
        status = self._determine_status(pattern_status, threshold_violations)

        # L0: Save to memory
        self.memory.save_action(
            bot_name=bot_name,
            command=command,
            status=status,
            output=raw_output,
        )

        # L0: Temporal analysis
        is_regression = self.memory.detect_regression(bot_name)
        is_recovery = self.memory.detect_recovery(bot_name)
        fail_streak = self.memory.detect_streak(bot_name, STATUS_FAIL)

        # Build alerts and recommendation
        alerts: list[str] = []
        recommendation: str | None = None

        if is_regression:
            alerts.append(f"REGRESSION: {bot_name} went from ok → fail")

        if is_recovery:
            alerts.append(f"RECOVERY: {bot_name} recovered from fail → ok")

        if fail_streak >= FAIL_STREAK_ALERT:
            alerts.append(f"STREAK ALERT: {bot_name} has failed {fail_streak} times consecutively")
            recommendation = self._build_recommendation(bot_name, fail_streak, pattern_matches)

        for violation in threshold_violations:
            alerts.append(f"THRESHOLD {violation.level.upper()}: {violation.message}")

        return EnrichedAnalysis(
            raw_output=raw_output,
            status=status,
            pattern_matches=pattern_matches,
            threshold_violations=threshold_violations,
            is_regression=is_regression,
            is_recovery=is_recovery,
            fail_streak=fail_streak,
            recommendation=recommendation,
            alerts=alerts,
        )

    def _determine_status(
        self,
        pattern_status: str,
        threshold_violations: list[ThresholdResult],
    ) -> str:
        """Determine final status — worst status wins.

        Priority: fail > partial > ok.
        """
        statuses = [pattern_status]
        for v in threshold_violations:
            if v.level == "critical":
                statuses.append(STATUS_FAIL)
            elif v.level == "warning":
                statuses.append(STATUS_PARTIAL)

        if STATUS_FAIL in statuses:
            return STATUS_FAIL
        if STATUS_PARTIAL in statuses:
            return STATUS_PARTIAL
        return STATUS_OK

    def _build_recommendation(
        self,
        bot_name: str,
        fail_streak: int,
        pattern_matches: list[PatternMatch],
    ) -> str:
        """Build actionable recommendation based on failure context."""
        parts = [f"fail_streak={fail_streak}"]

        if pattern_matches:
            pattern_names = [m.pattern.name for m in pattern_matches]
            parts.append(f"pattern={'|'.join(pattern_names)}")

        # Suggest action based on detected patterns
        pattern_names_set = {m.pattern.name for m in pattern_matches}
        if "stale_docs" in pattern_names_set:
            parts.append("recommendation=re-index")
        elif "rate_limited" in pattern_names_set:
            parts.append("recommendation=switch-to-fallback-llm")
        elif "connection_error" in pattern_names_set:
            parts.append("recommendation=check-service-health")
        elif "disk_full" in pattern_names_set:
            parts.append("recommendation=free-disk-space")
        elif "memory_warning" in pattern_names_set:
            parts.append("recommendation=check-memory-usage")
        else:
            parts.append("recommendation=investigate-manually")

        return ", ".join(parts)
