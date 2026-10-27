"""L2 Threshold Checker — Configurable metric limits from YAML.

Analogy for a C dev:
- ThresholdChecker = ulimit enforcement — soft limit (warning) + hard limit (critical)
- Each threshold = a named metric with min/max bounds
- check() = compare metric value vs limits — like assert(value < MAX)
- direction "max" = value must stay BELOW limit (like memory usage)
- direction "min" = value must stay ABOVE limit (like coverage %)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from spiral_04_cognitive.config import THRESHOLDS_YAML_PATH


@dataclass
class ThresholdSpec:
    """Specification for a single threshold.

    Like a ulimit entry — soft (warning) and hard (critical) limits.
    """

    name: str
    warning: float
    critical: float
    unit: str
    direction: str  # "max" or "min"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "name": self.name,
            "warning": self.warning,
            "critical": self.critical,
            "unit": self.unit,
            "direction": self.direction,
        }


@dataclass
class ThresholdResult:
    """Result of checking a metric against its threshold.

    level: "ok" | "warning" | "critical"
    """

    metric_name: str
    value: float
    level: str
    threshold: ThresholdSpec
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "metric": self.metric_name,
            "value": self.value,
            "level": self.level,
            "unit": self.threshold.unit,
            "message": self.message,
        }


class ThresholdChecker:
    """L2 cognitive layer — checks metrics against configurable limits.

    Loads thresholds from YAML. Supports both max-bounded (value must be
    below limit) and min-bounded (value must be above limit) metrics.
    """

    def __init__(self, yaml_path: str | None = None):
        self._yaml_path = yaml_path or THRESHOLDS_YAML_PATH
        self._thresholds: dict[str, ThresholdSpec] | None = None

    @property
    def thresholds(self) -> dict[str, ThresholdSpec]:
        """Lazy-load thresholds from YAML."""
        if self._thresholds is None:
            self._thresholds = self._load_thresholds()
        return self._thresholds

    def _load_thresholds(self) -> dict[str, ThresholdSpec]:
        """Parse thresholds.yaml into ThresholdSpec objects."""
        path = Path(self._yaml_path)
        if not path.exists():
            return {}

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data or "thresholds" not in data:
            return {}

        result: dict[str, ThresholdSpec] = {}
        for name, spec in data["thresholds"].items():
            result[name] = ThresholdSpec(
                name=name,
                warning=float(spec["warning"]),
                critical=float(spec["critical"]),
                unit=spec.get("unit", ""),
                direction=spec.get("direction", "max"),
            )
        return result

    def check(self, metric_name: str, value: float) -> ThresholdResult:
        """Check a metric value against its threshold.

        Returns ThresholdResult with level "ok", "warning", or "critical".
        If metric_name is not configured, returns "ok" (unknown metrics pass).
        """
        if metric_name not in self.thresholds:
            return ThresholdResult(
                metric_name=metric_name,
                value=value,
                level="ok",
                threshold=ThresholdSpec(
                    name=metric_name, warning=0, critical=0, unit="", direction="max"
                ),
                message=f"No threshold configured for '{metric_name}'",
            )

        spec = self.thresholds[metric_name]

        if spec.direction == "max":
            # Value must stay BELOW limits
            if value >= spec.critical:
                level = "critical"
                msg = f"{metric_name}={value}{spec.unit} >= critical({spec.critical}{spec.unit})"
            elif value >= spec.warning:
                level = "warning"
                msg = f"{metric_name}={value}{spec.unit} >= warning({spec.warning}{spec.unit})"
            else:
                level = "ok"
                msg = f"{metric_name}={value}{spec.unit} within limits"
        else:
            # direction == "min": value must stay ABOVE limits
            if value <= spec.critical:
                level = "critical"
                msg = f"{metric_name}={value}{spec.unit} <= critical({spec.critical}{spec.unit})"
            elif value <= spec.warning:
                level = "warning"
                msg = f"{metric_name}={value}{spec.unit} <= warning({spec.warning}{spec.unit})"
            else:
                level = "ok"
                msg = f"{metric_name}={value}{spec.unit} within limits"

        return ThresholdResult(
            metric_name=metric_name,
            value=value,
            level=level,
            threshold=spec,
            message=msg,
        )

    def check_all(self, metrics: dict[str, float]) -> list[ThresholdResult]:
        """Check multiple metrics at once. Returns all results."""
        return [self.check(name, value) for name, value in metrics.items()]

    def get_violations(self, metrics: dict[str, float]) -> list[ThresholdResult]:
        """Check metrics and return only violations (warning or critical)."""
        return [r for r in self.check_all(metrics) if r.level != "ok"]
