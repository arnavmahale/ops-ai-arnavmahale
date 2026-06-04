"""Drift detection: surfaces the 4+ distinct drift patterns in week-4 data.

Runs feature-level KS+PSI on key columns and segment-level comparisons
(per-zone, per-hour, per-day-of-week). Prints a human-readable report and
writes findings to JSON.

Usage:
    python -m week4.scripts.detect_drift [parquet_path] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from week4.scripts.compute_metrics import split_baseline_and_current
from week4.scripts.metrics import _psi, PSI_CRIT, PSI_WARN

logger = logging.getLogger(__name__)

# Per BASELINE_METRICS.md, p_value < 0.05 = significant; we add 0.01 = critical.
KS_P_CRIT = 0.01

# Features to test for distribution drift.
DRIFT_FEATURES = [
    "trip_count", "roll_mean_1day", "lag_1day", "lag_1week",
    "is_holiday", "dayofweek",
]


def detect_feature_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame,
                          feature: str) -> Dict:
    """KS + PSI test on a single feature."""
    if feature not in baseline_df.columns or feature not in new_df.columns:
        return {"feature": feature, "error": "missing column"}
    stat, p = ks_2samp(baseline_df[feature], new_df[feature])
    psi = _psi(baseline_df[feature].to_numpy(), new_df[feature].to_numpy())
    b_mean = float(baseline_df[feature].mean())
    c_mean = float(new_df[feature].mean())
    rel_shift = (c_mean - b_mean) / b_mean if b_mean else 0.0
    return {
        "feature": feature,
        "ks_stat": round(float(stat), 4),
        "p_value": float(p),
        "psi": round(psi, 4),
        "baseline_mean": round(b_mean, 3),
        "current_mean": round(c_mean, 3),
        "rel_shift_pct": round(rel_shift * 100, 1),
        "ks_significant": bool(p < KS_P_CRIT),
        "psi_significant": bool(psi > PSI_CRIT),
    }


def detect_concept_drift_by_segment(baseline_df: pd.DataFrame, new_df: pd.DataFrame,
                                     group_col: str, target: str = "trip_count",
                                     min_change_pct: float = 20.0) -> Dict:
    """Compare per-segment mean of `target` between baseline and new windows.

    Returns the top-5 segments by absolute relative shift, plus a count of
    segments whose mean shifted by more than `min_change_pct`.
    """
    if group_col not in baseline_df.columns or group_col not in new_df.columns:
        return {"group": group_col, "error": "missing column"}
    b = baseline_df.groupby(group_col)[target].mean().rename("baseline")
    c = new_df.groupby(group_col)[target].mean().rename("current")
    merged = pd.concat([b, c], axis=1).dropna()
    merged["abs_delta"] = merged["current"] - merged["baseline"]
    merged["pct_change"] = (merged["current"] / merged["baseline"] - 1) * 100
    top_drops = merged.nsmallest(5, "pct_change").round(3).reset_index().to_dict(orient="records")
    top_rises = merged.nlargest(5, "pct_change").round(3).reset_index().to_dict(orient="records")
    n_shifted = int((merged["pct_change"].abs() > min_change_pct).sum())
    return {
        "group": group_col,
        "n_segments": int(len(merged)),
        "n_shifted_more_than_pct": min_change_pct,
        "n_shifted": n_shifted,
        "top_drops": top_drops,
        "top_rises": top_rises,
    }


def detect_lag_collapse(baseline_df: pd.DataFrame, new_df: pd.DataFrame) -> Dict:
    """Specific check for the per-zone collapse of `roll_mean_1day`."""
    if "roll_mean_1day" not in baseline_df.columns:
        return {"error": "roll_mean_1day missing"}
    b = baseline_df.groupby("PULocationID")["roll_mean_1day"].mean()
    c = new_df.groupby("PULocationID")["roll_mean_1day"].mean()
    aligned = pd.concat([b.rename("baseline"), c.rename("current")], axis=1).dropna()
    aligned["pct_change"] = (aligned["current"] / aligned["baseline"] - 1) * 100
    return {
        "feature": "roll_mean_1day",
        "n_zones": int(len(aligned)),
        "n_zones_dropped_more_than_30pct": int((aligned["pct_change"] < -30).sum()),
        "median_pct_change": round(float(aligned["pct_change"].median()), 2),
        "min_pct_change": round(float(aligned["pct_change"].min()), 2),
        "max_pct_change": round(float(aligned["pct_change"].max()), 2),
    }


def format_findings(findings: Dict) -> str:
    out = ["=" * 72, "DRIFT DETECTION", "=" * 72]
    out.append("\n-- Feature-level drift (KS + PSI) --")
    for f in findings["feature_drift"]:
        flag = "DRIFT" if f.get("ks_significant") or f.get("psi_significant") else "ok"
        out.append(f"  {f['feature']:18s}  KS={f['ks_stat']:.4f} p={f['p_value']:.2e}  "
                   f"PSI={f['psi']:.4f}  rel_shift={f['rel_shift_pct']:+.1f}%  [{flag}]")
    out.append("\n-- Segment-level drift (per-group mean shifts) --")
    for s in findings["segment_drift"]:
        if "error" in s:
            continue
        out.append(f"  {s['group']:14s}  {s['n_shifted']}/{s['n_segments']} segments "
                   f"shifted > {s['n_shifted_more_than_pct']}%")
        for t in s["top_drops"][:3]:
            out.append(f"      DROP: {s['group']}={t[s['group']]}  "
                       f"{t['baseline']} -> {t['current']}  ({t['pct_change']}%)")
    if "lag_collapse" in findings:
        lc = findings["lag_collapse"]
        out.append(f"\n-- Lag feature collapse (roll_mean_1day) --")
        out.append(f"  {lc['n_zones_dropped_more_than_30pct']}/{lc['n_zones']} zones "
                   f"dropped >30%, median pct change = {lc['median_pct_change']}%")
    out.append("")
    out.append(f"Drift patterns detected: {findings['n_patterns']}")
    out.append(f"Verdict: {findings['verdict']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path", nargs="?",
                        default="week4/data/demand_enriched_week4.parquet")
    parser.add_argument("--out", default="week4/drift-findings.json")
    args = parser.parse_args(argv)

    logger.info("Loading %s", args.parquet_path)
    df = pd.read_parquet(args.parquet_path)
    baseline, current = split_baseline_and_current(df)
    logger.info("Baseline=%d rows, Current=%d rows", len(baseline), len(current))

    feat_findings = [detect_feature_drift(baseline, current, f) for f in DRIFT_FEATURES]
    seg_findings = [
        detect_concept_drift_by_segment(baseline, current, "PULocationID"),
        detect_concept_drift_by_segment(baseline, current, "hour"),
        detect_concept_drift_by_segment(baseline, current, "dayofweek"),
    ]
    lag_findings = detect_lag_collapse(baseline, current)

    n_patterns = sum(1 for f in feat_findings
                     if f.get("ks_significant") or f.get("psi_significant"))
    n_patterns += sum(1 for s in seg_findings if s.get("n_shifted", 0) > 0)
    if lag_findings.get("n_zones_dropped_more_than_30pct", 0) > 0:
        n_patterns += 1

    findings = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_patterns": n_patterns,
        "verdict": "DRIFT_DETECTED" if n_patterns >= 4 else "STABLE",
        "feature_drift": feat_findings,
        "segment_drift": seg_findings,
        "lag_collapse": lag_findings,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, default=str))
    logger.info("Wrote %s", out)

    print(format_findings(findings))
    return 0 if findings["verdict"] == "STABLE" else 1


if __name__ == "__main__":
    sys.exit(main())
