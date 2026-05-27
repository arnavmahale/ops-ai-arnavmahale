"""Data quality validation for the taxi demand parquet.

Detects four classes of issues observed in week-3 corrupted data:
  1. Duplicate rows (full-row and key-based)
  2. Out-of-range `trip_count` (negatives and impossible-high values)
  3. Variance collapse on categorical features (e.g. cbd_pricing_active stuck)
  4. Distribution shifts on rate features (e.g. is_holiday rate spike)

Usage as a CLI (used by the GitHub Actions workflow):
    python -m validation.check_data_quality <path/to/parquet>

Exit code is 0 if no critical/high-severity issues are found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The cutoff is fixed for this assignment: rows before it are treated as the
# clean historical baseline, rows on/after as the candidate window to validate.
CUTOFF = pd.Timestamp("2026-01-16")

# Hard schema constraints — declared once, used both in the report and at runtime.
REQUIRED_COLUMNS = [
    "PULocationID", "time_bucket", "trip_count",
    "hour", "minute", "dayofweek", "month",
    "is_weekend", "is_holiday", "cbd_pricing_active", "is_airport_zone",
]

# Hard value ranges (impossible to be outside these).
HARD_RANGES = {
    "trip_count": (0, 500),     # negatives impossible; ceiling = ~1.6x baseline max (310)
    "hour": (0, 23),
    "minute": (0, 59),
    "dayofweek": (0, 6),
    "month": (1, 12),
    "is_weekend": (0, 1),
    "is_holiday": (0, 1),
    "cbd_pricing_active": (0, 1),
    "is_airport_zone": (0, 1),
}

# Feature that lost variance is a structural break — model can't learn from a constant.
VARIANCE_COLLAPSE_COLS = ["cbd_pricing_active", "is_weekend", "is_holiday"]

# Rate columns where a >Nx shift vs baseline is suspicious.
RATE_DRIFT_RATIO = 2.0  # 2x baseline rate triggers a flag

SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


class DataQualityValidator:
    """Validates an incoming parquet against a clean baseline."""

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None) -> None:
        self.baseline = baseline_df
        self.issues: List[Dict[str, Any]] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all checks. Returns {is_valid, num_issues, issues}."""
        self.issues = []

        # Order matters: schema first (later checks assume columns exist).
        self.check_schema(df)
        if not any(i["type"] == "missing_columns" for i in self.issues):
            self.check_null_rates(df)
            self.check_value_ranges(df)
            self.check_duplicates(df)
            self.check_variance_collapse(df)
            self.check_rate_drift(df)

        return {
            "is_valid": not self._has_blocking_issues(),
            "num_issues": len(self.issues),
            "issues": self.issues,
        }

    # ── Individual checks ──────────────────────────────────────────────────

    def check_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            self._add_issue(
                "missing_columns", "critical",
                f"Required columns missing: {missing}",
                count=len(missing), columns=missing,
            )

    def check_null_rates(self, df: pd.DataFrame, threshold: float = 0.01) -> None:
        """Flag columns whose null rate exceeds `threshold` (1% by default)."""
        nulls = df.isna().mean()
        for col, rate in nulls.items():
            if rate > threshold:
                self._add_issue(
                    "null_rate", "high",
                    f"Column '{col}' has {rate:.1%} null rate (> {threshold:.0%})",
                    count=int(df[col].isna().sum()), column=col, rate=float(rate),
                )

    def check_value_ranges(self, df: pd.DataFrame) -> None:
        """Flag rows outside hard, declared value ranges."""
        for col, (lo, hi) in HARD_RANGES.items():
            if col not in df.columns:
                continue
            below = (df[col] < lo).sum()
            above = (df[col] > hi).sum()
            if below:
                sev = "critical" if col == "trip_count" else "high"
                self._add_issue(
                    "out_of_range_low", sev,
                    f"Column '{col}' has {below:,} values below {lo} (impossible)",
                    count=int(below), column=col, threshold=lo,
                )
            if above:
                sev = "critical" if col == "trip_count" else "high"
                self._add_issue(
                    "out_of_range_high", sev,
                    f"Column '{col}' has {above:,} values above {hi} (impossible/extreme)",
                    count=int(above), column=col, threshold=hi,
                )

    def check_duplicates(self, df: pd.DataFrame) -> None:
        """Detect full-row dupes and (PULocationID, time_bucket) key dupes."""
        full_dupes = int(df.duplicated().sum())
        if full_dupes:
            self._add_issue(
                "duplicate_rows", "high",
                f"{full_dupes:,} fully duplicated rows",
                count=full_dupes,
            )
        if {"PULocationID", "time_bucket"}.issubset(df.columns):
            key_dupes = int(df.duplicated(subset=["PULocationID", "time_bucket"]).sum())
            if key_dupes > full_dupes:  # only flag if NOT explained by full dupes
                self._add_issue(
                    "duplicate_keys", "high",
                    f"{key_dupes:,} duplicate (PULocationID, time_bucket) keys",
                    count=key_dupes, key=["PULocationID", "time_bucket"],
                )

    def check_variance_collapse(self, df: pd.DataFrame) -> None:
        """Categorical features that lost all variance are dead features."""
        for col in VARIANCE_COLLAPSE_COLS:
            if col not in df.columns:
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1 and len(df) > 100:
                value = df[col].iloc[0] if len(df) else None
                self._add_issue(
                    "variance_collapse", "high",
                    f"Column '{col}' lost variance — all {len(df):,} rows = {value!r}",
                    count=int(len(df)), column=col, value=value,
                )

    def check_rate_drift(self, df: pd.DataFrame, ratio: float = RATE_DRIFT_RATIO) -> None:
        """For 0/1 rate columns, flag when current rate diverges from baseline by `ratio`x."""
        if self.baseline is None:
            return
        for col in ["is_holiday", "is_weekend", "cbd_pricing_active"]:
            if col not in df.columns or col not in self.baseline.columns:
                continue
            base_rate = float(self.baseline[col].mean())
            cur_rate = float(df[col].mean())
            if base_rate < 1e-6:
                continue  # avoid divide-by-zero
            r = cur_rate / base_rate
            if r >= ratio or (r > 0 and 1 / r >= ratio):
                self._add_issue(
                    "rate_drift", "medium",
                    f"Column '{col}' rate changed {r:.2f}x vs baseline "
                    f"({base_rate:.3f} -> {cur_rate:.3f})",
                    count=None, column=col,
                    baseline_rate=base_rate, current_rate=cur_rate, ratio=r,
                )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _add_issue(self, issue_type: str, severity: str, description: str,
                   count: Optional[int] = None, **details: Any) -> None:
        self.issues.append({
            "type": issue_type, "severity": severity,
            "description": description, "count": count, **details,
        })

    def _has_blocking_issues(self) -> bool:
        """`is_valid=False` if any critical or high-severity issue is present."""
        return any(SEVERITY_RANK[i["severity"]] >= SEVERITY_RANK["high"]
                   for i in self.issues)


# ── Convenience utilities ─────────────────────────────────────────────────────

def split_baseline_and_current(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF):
    """Split a combined parquet into (baseline, current) by `time_bucket`."""
    df = df.copy()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    return (
        df[df["time_bucket"] < cutoff].reset_index(drop=True),
        df[df["time_bucket"] >= cutoff].reset_index(drop=True),
    )


def format_report(result: Dict[str, Any]) -> str:
    """Pretty-print a validation result for logs / CI output."""
    if result["is_valid"] and result["num_issues"] == 0:
        return "Data quality OK — no issues found."
    lines = [f"Found {result['num_issues']} issue(s) "
             f"({'VALID' if result['is_valid'] else 'INVALID'}):"]
    for i in result["issues"]:
        cnt = f" ({i['count']:,} rows)" if i.get("count") else ""
        lines.append(f"  [{i['severity'].upper():<8}] {i['type']}: {i['description']}{cnt}")
    return "\n".join(lines)


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet_path", nargs="?",
        default="week3/data/demand_enriched_corrupted.parquet",
        help="Path to the parquet to validate.",
    )
    parser.add_argument(
        "--allow-warnings", action="store_true",
        help="Exit 0 even if medium-severity issues are present (still exits 1 on critical/high).",
    )
    args = parser.parse_args(argv)

    logger.info("Loading parquet: %s", args.parquet_path)
    df = pd.read_parquet(args.parquet_path)
    baseline, current = split_baseline_and_current(df)
    logger.info("Baseline rows: %d | candidate rows: %d", len(baseline), len(current))

    validator = DataQualityValidator(baseline_df=baseline)
    result = validator.validate(current)
    print(format_report(result))

    return 0 if result["is_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
