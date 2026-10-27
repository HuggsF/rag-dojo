"""L1 Pattern Detector — Regex-based known issue classification.

Analogy for a C dev:
- PatternDetector = a chain of if/else regex matches (like fail2ban filters)
- Each pattern = a compiled regex + severity + status mapping
- classify() = run all filters against output, return first match (priority order)
- load_patterns() = parse config file at startup (like reading /etc/*.conf)
- The YAML is the config, the code is the engine — change patterns without recompiling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spiral_04_cognitive.config import PATTERNS_YAML_PATH, STATUS_OK


@dataclass
class Pattern:
    """A single detection pattern.

    Like a filter rule in iptables — match condition + action.
    """

    name: str
    regex: str
    severity: str
    status: str
    _compiled: re.Pattern[str] | None = field(default=None, repr=False)

    @property
    def compiled(self) -> re.Pattern[str]:
        """Lazy-compile the regex pattern."""
        if self._compiled is None:
            self._compiled = re.compile(self.regex)
        return self._compiled

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "name": self.name,
            "severity": self.severity,
            "status": self.status,
        }


@dataclass
class PatternMatch:
    """Result of a pattern match against output.

    Contains the matched pattern, the matched text, and position.
    """

    pattern: Pattern
    matched_text: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "pattern": self.pattern.name,
            "severity": self.pattern.severity,
            "status": self.pattern.status,
            "matched_text": self.matched_text[:100],
            "position": self.position,
        }


class PatternDetector:
    """L1 cognitive layer — detects known issues via regex patterns.

    Loads patterns from YAML config. Patterns are evaluated in order;
    first match wins for status classification.
    """

    def __init__(self, yaml_path: str | None = None):
        self._yaml_path = yaml_path or PATTERNS_YAML_PATH
        self._patterns: list[Pattern] | None = None

    @property
    def patterns(self) -> list[Pattern]:
        """Lazy-load patterns from YAML."""
        if self._patterns is None:
            self._patterns = self._load_patterns()
        return self._patterns

    def _load_patterns(self) -> list[Pattern]:
        """Parse patterns.yaml into Pattern objects."""
        path = Path(self._yaml_path)
        if not path.exists():
            return []

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data or "patterns" not in data:
            return []

        return [
            Pattern(
                name=p["name"],
                regex=p["regex"],
                severity=p["severity"],
                status=p["status"],
            )
            for p in data["patterns"]
        ]

    def detect(self, output: str) -> list[PatternMatch]:
        """Run all patterns against output. Returns all matches.

        Like running all iptables rules — every matching rule fires.
        """
        matches: list[PatternMatch] = []
        for pattern in self.patterns:
            match = pattern.compiled.search(output)
            if match:
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        matched_text=match.group(0),
                        position=match.start(),
                    )
                )
        return matches

    def classify(self, output: str) -> str:
        """Classify output into a status based on pattern matches.

        Priority: critical > high > medium > low. If multiple severities
        match, the highest severity determines the status.
        Returns STATUS_OK if no patterns match.
        """
        matches = self.detect(output)
        if not matches:
            return STATUS_OK

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        matches.sort(key=lambda m: severity_order.get(m.pattern.severity, 99))
        return matches[0].pattern.status

    def get_highest_severity(self, matches: list[PatternMatch]) -> str | None:
        """Get the highest severity from a list of matches."""
        if not matches:
            return None
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        matches_sorted = sorted(matches, key=lambda m: severity_order.get(m.pattern.severity, 99))
        return matches_sorted[0].pattern.severity
