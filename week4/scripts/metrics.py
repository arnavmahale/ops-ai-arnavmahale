"""Monitoring metrics for the taxi demand pipeline.

Implements 6 of the 8 metric stubs from the week-4 template. Each metric
returns a dict with the raw value, a `breach` flag, and the threshold that
was checked, so the calling code can write structured JSON for alerts.

Baselines and thresholds are sourced from week4/BASELINE_METRICS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# ── Thresholds (from BASELINE_METRICS.md) ─────────────────────────────────────

NULL_RATE_WARN = 0.005   # 0.5%
NULL_RATE_CRIT = 0.01    # 1%
LAG_NULL_CRIT = 0.02     # lag features tolerate up to 2%
DUPLICATE_WARN = 0.0     # baseline = 0
DUPLICATE_CRIT = 0.005   # 0.5%
KS_P_WARN = 0.05
KS_P_CRIT = 0.01
PSI_WARN = 0.10
PSI_CRIT = 0.25
TRIP_MEAN_SHIFT_WARN = 0.10   # 10% relative shift
TRIP_MEAN_SHIFT_CRIT = 0.20   # 20% relative shift
ACCURACY_CRIT = 0.80          # per-zone accuracy floor
FRESHNESS_WARN_HOURS = 2
FRESHNESS_CRIT_HOURS = 24

CRITICAL_COLUMNS = ["trip_count", "PULocationID", "time_bucket"]
LAG_COLUMNS = ["lag_15min", "lag_1h", "lag_2h", "lag_1day", "lag_1week"]


@dataclass
class MetricResult:
    name: str
    value: Any
    breach: str = "ok"           # one of: 'ok', 'warn', 'crit'
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value,
                "breach": self.breach, "detail": self.detail}


def _psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using equal-frequency bin edges from baseline."""
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Constant baseline — PSI is undefined; treat as no shift.
        return 0.0
    pa, _ = np.histogram(baseline, bins=edges)
    pb, _ = np.histogram(current, bins=edges)
    pa = pa / pa.sum() if pa.sum() else pa
    pb = pb / pb.sum() if pb.sum() else pb
    eps = 1e-6
    return float(np.sum((pa - pb) * np.log((pa + eps) / (pb + eps))))


class MetricComputer:
    """Compute monitoring metrics for drift detection."""

    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline = baseline_df

    # ── Metric 1 — Accuracy proxy via baseline profile predictions ─────────

    def metric_1_accuracy_proxy(self, new_df: pd.DataFrame) -> MetricResult:
        """Use the baseline per-(zone, hour, dow) mean as a prediction.

        Reports global MAPE-like accuracy (% within ±50% of actual) and
        the worst per-zone accuracy. Production would use real model
        predictions; this proxy is what we have without a live LightGBM.
        """
        profile = (
            self.baseline.groupby(["PULocationID", "hour", "dayofweek"])
            ["trip_count"].mean().round(3).rename("predicted").reset_index()
        )
        merged = new_df.merge(profile, on=["PULocationID", "hour", "dayofweek"], how="left")
        merged["predicted"] = merged["predicted"].fillna(merged["trip_count"].mean())
        # Tolerance: prediction within ±50% of actual (or ±2 when actual is small)
        tol = np.maximum(0.5 * merged["trip_count"], 2.0)
        merged["correct"] = (merged["predicted"] - merged["trip_count"]).abs() <= tol
        overall = float(merged["correct"].mean())
        by_zone = merged.groupby("PULocationID")["correct"].mean()
        worst_zone = int(by_zone.idxmin())
        worst_acc = float(by_zone.min())
        breach = "crit" if worst_acc < ACCURACY_CRIT else "ok"
        return MetricResult(
            "accuracy_proxy",
            value=round(overall, 4),
            breach=breach,
            detail={
                "overall": round(overall, 4),
                "worst_zone": worst_zone,
                "worst_zone_accuracy": round(worst_acc, 4),
                "threshold": ACCURACY_CRIT,
                "n_zones_below_threshold": int((by_zone < ACCURACY_CRIT).sum()),
            },
        )

    # ── Metric 2 — Accuracy by zone (per-zone breakdown) ────────────────────

    def metric_2_accuracy_by_zone(self, new_df: pd.DataFrame) -> MetricResult:
        profile = (
            self.baseline.groupby(["PULocationID", "hour", "dayofweek"])
            ["trip_count"].mean().rename("predicted").reset_index()
        )
        merged = new_df.merge(profile, on=["PULocationID", "hour", "dayofweek"], how="left")
        merged["predicted"] = merged["predicted"].fillna(merged["trip_count"].mean())
        tol = np.maximum(0.5 * merged["trip_count"], 2.0)
        merged["correct"] = (merged["predicted"] - merged["trip_count"]).abs() <= tol
        by_zone = merged.groupby("PULocationID")["correct"].mean()
        breach = "crit" if (by_zone < ACCURACY_CRIT).any() else "ok"
        return MetricResult(
            "accuracy_by_zone",
            value=round(float(by_zone.mean()), 4),
            breach=breach,
            detail={
                "n_zones_below_threshold": int((by_zone < ACCURACY_CRIT).sum()),
                "min": round(float(by_zone.min()), 4),
                "max": round(float(by_zone.max()), 4),
                "median": round(float(by_zone.median()), 4),
            },
        )

    # ── Metric 3 — Null rates (data quality) ────────────────────────────────

    def metric_3_null_rates(self, new_df: pd.DataFrame) -> MetricResult:
        nulls = {c: float(new_df[c].isna().mean()) for c in new_df.columns if c in
                 (CRITICAL_COLUMNS + LAG_COLUMNS)}
        critical_breach = any(
            (r > NULL_RATE_CRIT and col not in LAG_COLUMNS) or
            (r > LAG_NULL_CRIT and col in LAG_COLUMNS)
            for col, r in nulls.items()
        )
        warn_breach = any(r > NULL_RATE_WARN for r in nulls.values())
        breach = "crit" if critical_breach else ("warn" if warn_breach else "ok")
        return MetricResult(
            "null_rates",
            value={k: round(v, 4) for k, v in nulls.items()},
            breach=breach,
            detail={"threshold_critical_critical_cols": NULL_RATE_CRIT,
                    "threshold_critical_lag_cols": LAG_NULL_CRIT},
        )

    # ── Metric 4 — KS test for trip_count distribution shift ────────────────

    def metric_4_ks_test(self, new_df: pd.DataFrame,
                          features: Iterable[str] = ("trip_count",)) -> MetricResult:
        results = {}
        any_crit = False
        any_warn = False
        for f in features:
            if f not in new_df.columns or f not in self.baseline.columns:
                continue
            stat, p = ks_2samp(self.baseline[f], new_df[f])
            results[f] = {"ks_stat": round(float(stat), 4), "p_value": float(p)}
            if p < KS_P_CRIT:
                any_crit = True
            elif p < KS_P_WARN:
                any_warn = True
        breach = "crit" if any_crit else ("warn" if any_warn else "ok")
        return MetricResult("ks_test", value=results, breach=breach,
                            detail={"p_warn": KS_P_WARN, "p_crit": KS_P_CRIT})

    # ── Metric 5 — PSI on trip_count + selected features ────────────────────

    def metric_5_psi(self, new_df: pd.DataFrame, bins: int = 10,
                     features: Iterable[str] = ("trip_count", "roll_mean_1day")) -> MetricResult:
        results = {}
        any_crit = False
        any_warn = False
        for f in features:
            if f not in new_df.columns or f not in self.baseline.columns:
                continue
            psi = _psi(self.baseline[f].to_numpy(), new_df[f].to_numpy(), bins=bins)
            results[f] = round(psi, 4)
            if psi > PSI_CRIT:
                any_crit = True
            elif psi > PSI_WARN:
                any_warn = True
        breach = "crit" if any_crit else ("warn" if any_warn else "ok")
        return MetricResult("psi", value=results, breach=breach,
                            detail={"warn": PSI_WARN, "crit": PSI_CRIT, "bins": bins})

    # ── Metric 6 — Trip count mean shift (the bottom-line signal) ───────────

    def metric_6_mean_shift(self, new_df: pd.DataFrame) -> MetricResult:
        if "trip_count" not in new_df.columns:
            return MetricResult("mean_shift", value=None, breach="warn",
                                detail={"reason": "trip_count missing"})
        b = float(self.baseline["trip_count"].mean())
        c = float(new_df["trip_count"].mean())
        rel = (c - b) / b if b else 0.0
        breach = "crit" if abs(rel) > TRIP_MEAN_SHIFT_CRIT else (
                 "warn" if abs(rel) > TRIP_MEAN_SHIFT_WARN else "ok")
        return MetricResult("mean_shift", value=round(rel, 4), breach=breach,
                            detail={"baseline_mean": round(b, 3),
                                    "current_mean": round(c, 3),
                                    "warn": TRIP_MEAN_SHIFT_WARN,
                                    "crit": TRIP_MEAN_SHIFT_CRIT})

    # ── Metric 7 — Duplicate rate (data quality) ────────────────────────────

    def metric_7_duplicate_rate(self, new_df: pd.DataFrame) -> MetricResult:
        if not {"PULocationID", "time_bucket"}.issubset(new_df.columns):
            return MetricResult("duplicate_rate", value=None, breach="warn",
                                detail={"reason": "key columns missing"})
        n = len(new_df)
        if n == 0:
            return MetricResult("duplicate_rate", value=0.0, breach="ok",
                                detail={"count": 0})
        full = int(new_df.duplicated().sum())
        key_d = int(new_df.duplicated(subset=["PULocationID", "time_bucket"]).sum())
        rate = key_d / n
        breach = "crit" if rate > DUPLICATE_CRIT else ("warn" if key_d > 0 else "ok")
        return MetricResult("duplicate_rate", value=round(rate, 6), breach=breach,
                            detail={"full_dupes": full, "key_dupes": key_d,
                                    "count_rows": n, "crit_threshold": DUPLICATE_CRIT})

    # ── Metric 8 — Data freshness (lateness) ────────────────────────────────

    def metric_8_data_freshness(self, new_df: pd.DataFrame,
                                 now: Optional[pd.Timestamp] = None) -> MetricResult:
        if "time_bucket" not in new_df.columns or new_df.empty:
            return MetricResult("freshness", value=None, breach="crit",
                                detail={"reason": "no data"})
        latest = pd.to_datetime(new_df["time_bucket"]).max()
        # Compare against the latest baseline timestamp to make this stable
        # across runs of historical data (production would use datetime.now()).
        reference = now or pd.to_datetime(self.baseline["time_bucket"]).max()
        age_hours = (reference - latest).total_seconds() / 3600
        # Negative age means current is fresher than reference — that's healthy.
        if age_hours < 0:
            return MetricResult("freshness", value=round(age_hours, 2), breach="ok",
                                detail={"latest_record": str(latest),
                                        "reference": str(reference)})
        breach = "crit" if age_hours > FRESHNESS_CRIT_HOURS else (
                 "warn" if age_hours > FRESHNESS_WARN_HOURS else "ok")
        return MetricResult("freshness", value=round(age_hours, 2), breach=breach,
                            detail={"latest_record": str(latest),
                                    "reference": str(reference),
                                    "warn_hours": FRESHNESS_WARN_HOURS,
                                    "crit_hours": FRESHNESS_CRIT_HOURS})

    # ── Roll-up ─────────────────────────────────────────────────────────────

    def compute_all(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        """Run every metric and return a dict suitable for JSON serialization."""
        results = [
            self.metric_1_accuracy_proxy(new_df),
            self.metric_2_accuracy_by_zone(new_df),
            self.metric_3_null_rates(new_df),
            self.metric_4_ks_test(new_df, features=["trip_count", "roll_mean_1day", "lag_1day"]),
            self.metric_5_psi(new_df, features=["trip_count", "roll_mean_1day", "lag_1day"]),
            self.metric_6_mean_shift(new_df),
            self.metric_7_duplicate_rate(new_df),
            self.metric_8_data_freshness(new_df),
        ]
        any_crit = any(r.breach == "crit" for r in results)
        any_warn = any(r.breach == "warn" for r in results)
        overall = "crit" if any_crit else ("warn" if any_warn else "ok")
        return {
            "overall_breach": overall,
            "n_critical": sum(r.breach == "crit" for r in results),
            "n_warning":  sum(r.breach == "warn" for r in results),
            "metrics": [r.to_dict() for r in results],
        }
