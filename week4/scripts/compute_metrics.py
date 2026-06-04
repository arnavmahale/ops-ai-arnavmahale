"""CLI: compute monitoring metrics on the upstream parquet and write JSON.

Usage:
    python -m week4.scripts.compute_metrics [parquet_path] [--out PATH]

Exit code:
    0 — all metrics within thresholds
    1 — at least one critical-severity threshold breach
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd

from week4.scripts.metrics import MetricComputer

logger = logging.getLogger(__name__)

BASELINE_START = pd.Timestamp("2026-01-01")
BASELINE_END = pd.Timestamp("2026-01-16")          # exclusive
CURRENT_START = pd.Timestamp("2026-02-02")
CURRENT_END = pd.Timestamp("2026-03-01")           # exclusive


def split_baseline_and_current(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split combined parquet into (baseline Jan 1-15, current Feb 2-28) windows."""
    df = df.copy()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    baseline = df[(df["time_bucket"] >= BASELINE_START) & (df["time_bucket"] < BASELINE_END)]
    current = df[(df["time_bucket"] >= CURRENT_START) & (df["time_bucket"] < CURRENT_END)]
    return baseline.reset_index(drop=True), current.reset_index(drop=True)


def format_summary(report: dict) -> str:
    lines = [
        f"Overall breach: {report['overall_breach'].upper()}",
        f"Critical: {report['n_critical']}  Warning: {report['n_warning']}",
        "",
    ]
    for m in report["metrics"]:
        marker = {"ok": "✓", "warn": "!", "crit": "X"}[m["breach"]]
        lines.append(f"  [{marker}] {m['name']:18s}  value={m['value']}")
        for k, v in m["detail"].items():
            lines.append(f"        {k}: {v}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet_path", nargs="?",
        default="week4/data/demand_enriched_week4.parquet",
        help="Path to combined parquet containing both baseline and current windows.",
    )
    parser.add_argument("--out", default="week4/metrics-latest.json",
                        help="Where to write the JSON report.")
    args = parser.parse_args(argv)

    logger.info("Loading parquet: %s", args.parquet_path)
    df = pd.read_parquet(args.parquet_path)
    baseline, current = split_baseline_and_current(df)
    logger.info("Baseline rows: %d (%s -> %s)",
                len(baseline), BASELINE_START.date(), (BASELINE_END - pd.Timedelta(days=1)).date())
    logger.info("Current  rows: %d (%s -> %s)",
                len(current),  CURRENT_START.date(),  (CURRENT_END - pd.Timedelta(days=1)).date())

    computer = MetricComputer(baseline)
    report = computer.compute_all(current)
    report["computed_at"] = datetime.now(timezone.utc).isoformat()
    report["baseline_window"] = {"start": str(BASELINE_START.date()),
                                  "end": str((BASELINE_END - pd.Timedelta(days=1)).date()),
                                  "rows": int(len(baseline))}
    report["current_window"] = {"start": str(CURRENT_START.date()),
                                 "end": str((CURRENT_END - pd.Timedelta(days=1)).date()),
                                 "rows": int(len(current))}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Wrote %s", out_path)

    print(format_summary(report))
    return 0 if report["overall_breach"] != "crit" else 1


if __name__ == "__main__":
    sys.exit(main())
