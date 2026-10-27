"""Per-file coverage checker.

Reads coverage.json (produced by pytest --cov-report=json) and enforces
a minimum coverage percentage on each source file individually.

Usage:
    python scripts/check_coverage.py --min-percent 90
    python scripts/check_coverage.py --min-percent 90 --exclude download_docs.py pipeline.py

Exit codes:
    0 — all files meet the threshold
    1 — one or more files below threshold
    2 — coverage.json not found or parse error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COVERAGE_FILE = Path("coverage.json")


def check_coverage(min_percent: float, exclude: list[str]) -> bool:
    """Check per-file coverage against threshold.

    Returns True if all files pass, False otherwise.
    """
    if not COVERAGE_FILE.exists():
        print(f"ERROR: {COVERAGE_FILE} not found. Run pytest with --cov-report=json first.")
        sys.exit(2)

    try:
        data = json.loads(COVERAGE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: Failed to parse {COVERAGE_FILE}: {e}")
        sys.exit(2)

    files = data.get("files", {})
    if not files:
        print("WARNING: No files found in coverage report.")
        return True

    all_passed = True
    results: list[tuple[str, float, bool]] = []

    for filepath, info in sorted(files.items()):
        filename = Path(filepath).name

        # Skip excluded files
        if filename in exclude or filepath in exclude:
            results.append((filepath, -1.0, True))
            continue

        summary = info.get("summary", {})
        percent = summary.get("percent_covered", 0.0)
        passed = percent >= min_percent
        results.append((filepath, percent, passed))

        if not passed:
            all_passed = False

    # Print results table
    print(f"\n{'=' * 70}")
    print(f"  Per-file coverage check (minimum: {min_percent}%)")
    print(f"{'=' * 70}")

    for filepath, percent, passed in results:
        if percent < 0:
            status = "SKIP"
            bar = "---"
        elif passed:
            status = "PASS"
            bar = f"{percent:5.1f}%"
        else:
            status = "FAIL"
            bar = f"{percent:5.1f}%"

        icon = {"PASS": "+", "FAIL": "!", "SKIP": "-"}[status]
        print(f"  [{icon}] {status}  {bar:>8s}  {filepath}")

    print(f"{'=' * 70}")

    if all_passed:
        print("  All files meet the coverage threshold.\n")
    else:
        failing = [f for f, p, ok in results if not ok]
        print(f"  {len(failing)} file(s) below {min_percent}% coverage.\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-file coverage checker")
    parser.add_argument(
        "--min-percent",
        type=float,
        default=90.0,
        help="Minimum coverage percentage per file (default: 90)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="File names to exclude from the check",
    )
    args = parser.parse_args()

    passed = check_coverage(args.min_percent, args.exclude)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
