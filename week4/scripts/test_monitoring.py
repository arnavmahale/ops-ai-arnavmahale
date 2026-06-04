"""Tests for the week-4 monitoring framework.

Uses synthetic in-memory data so the suite runs in CI without the 75 MB
parquet on the runner. One opt-in smoke test exercises the real parquet
if it's present on disk.

From the repo root:
    pytest week4/scripts/test_monitoring.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from week4.scripts.metrics import (
    MetricComputer, ACCURACY_CRIT, PSI_CRIT, _psi,
)
from week4.scripts.detect_drift import (
    detect_feature_drift, detect_concept_drift_by_segment, detect_lag_collapse,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "PULocationID", "time_bucket", "trip_count",
    "hour", "dayofweek", "is_holiday",
]


def _make_df(start: str, days: int = 7, n_zones: int = 5, seed: int = 1,
             mean_trip: float = 14.0, sigma: float = 5.0,
             roll_mean_factor: float = 1.0) -> pd.DataFrame:
    """Generate synthetic enriched-demand rows: full zone × time-bucket grid."""
    rng = np.random.default_rng(seed)
    rows = []
    zones = [10 + 11 * i for i in range(n_zones)]
    start_ts = pd.Timestamp(start)
    for z in zones:
        for d in range(days):
            for slot in range(96):
                ts = start_ts + pd.Timedelta(days=d, minutes=15 * slot)
                tc = int(max(0, rng.normal(mean_trip, sigma)))
                rows.append({
                    "PULocationID": z,
                    "time_bucket": ts,
                    "trip_count": tc,
                    "hour": ts.hour,
                    "dayofweek": ts.dayofweek,
                    "is_holiday": 0,
                    "roll_mean_1day": float(tc) * roll_mean_factor,
                    "lag_1day": float(tc) * roll_mean_factor,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def baseline_df():
    return _make_df("2026-01-01", days=15, mean_trip=14.0, seed=1)


@pytest.fixture
def computer(baseline_df):
    return MetricComputer(baseline_df)


@pytest.fixture
def stable_current(baseline_df):
    """Distribution close to baseline — should NOT raise critical alerts."""
    return _make_df("2026-02-02", days=27, mean_trip=14.0, seed=2)


@pytest.fixture
def drifted_current(baseline_df):
    """Distribution clearly shifted — should raise CRIT on KS / PSI / mean shift."""
    return _make_df("2026-02-02", days=27, mean_trip=8.0,
                    roll_mean_factor=0.5, seed=3)


# ── Individual metrics ───────────────────────────────────────────────────────

class TestMetrics:
    def test_null_rates_clean(self, computer, stable_current):
        r = computer.metric_3_null_rates(stable_current)
        assert r.breach == "ok"

    def test_null_rates_dirty(self, computer, stable_current):
        df = stable_current.copy()
        idx = df.sample(frac=0.05, random_state=0).index
        df.loc[idx, "trip_count"] = None
        r = computer.metric_3_null_rates(df)
        assert r.breach in {"warn", "crit"}

    def test_ks_test_detects_shift(self, computer, drifted_current):
        r = computer.metric_4_ks_test(drifted_current,
                                      features=["trip_count", "roll_mean_1day"])
        assert r.breach == "crit"
        assert r.value["trip_count"]["p_value"] < 0.01

    def test_psi_detects_shift(self, computer, drifted_current):
        r = computer.metric_5_psi(drifted_current,
                                  features=["trip_count", "roll_mean_1day"])
        # roll_mean_1day was halved — PSI must exceed crit threshold
        assert r.value["roll_mean_1day"] > PSI_CRIT
        assert r.breach == "crit"

    def test_psi_function_handles_constant_baseline(self):
        # Edge case: PSI must not crash on a constant input.
        baseline = np.zeros(100)
        current = np.zeros(100)
        assert _psi(baseline, current) == 0.0

    def test_mean_shift_detects_drop(self, computer, drifted_current):
        r = computer.metric_6_mean_shift(drifted_current)
        assert r.breach == "crit"
        assert r.value < -0.2  # > 20% drop

    def test_duplicate_rate_clean(self, computer, stable_current):
        r = computer.metric_7_duplicate_rate(stable_current)
        assert r.breach == "ok"

    def test_duplicate_rate_with_dupes(self, computer, stable_current):
        df = pd.concat([stable_current, stable_current.head(100)], ignore_index=True)
        r = computer.metric_7_duplicate_rate(df)
        assert r.detail["key_dupes"] >= 100

    def test_freshness_ok(self, computer, stable_current):
        r = computer.metric_8_data_freshness(stable_current)
        assert r.breach == "ok"

    def test_accuracy_proxy_runs(self, computer, stable_current):
        # The proxy should at least not crash and produce a value in [0, 1].
        r = computer.metric_1_accuracy_proxy(stable_current)
        assert 0.0 <= r.value <= 1.0

    def test_compute_all_returns_structure(self, computer, drifted_current):
        report = computer.compute_all(drifted_current)
        assert report["overall_breach"] == "crit"
        assert report["n_critical"] >= 1
        assert len(report["metrics"]) == 8


# ── Drift detection helpers ──────────────────────────────────────────────────

class TestDriftDetection:
    def test_feature_drift_significant_when_shifted(self, baseline_df, drifted_current):
        r = detect_feature_drift(baseline_df, drifted_current, "trip_count")
        assert r["ks_significant"] is True

    def test_feature_drift_not_significant_when_stable(self, baseline_df, stable_current):
        r = detect_feature_drift(baseline_df, stable_current, "trip_count")
        # Should NOT trigger PSI > critical on stable data.
        assert r["psi"] < PSI_CRIT

    def test_segment_drift_finds_shifted_zones(self, baseline_df):
        # Concoct a current df where zone 10 has half the demand.
        current = _make_df("2026-02-02", days=27, mean_trip=14.0, seed=4)
        current.loc[current["PULocationID"] == 10, "trip_count"] //= 4
        s = detect_concept_drift_by_segment(baseline_df, current, "PULocationID")
        assert s["n_shifted"] >= 1
        # Top drop should include zone 10.
        top_zone_ids = [t["PULocationID"] for t in s["top_drops"]]
        assert 10 in top_zone_ids

    def test_lag_collapse_detects_across_zones(self, baseline_df, drifted_current):
        r = detect_lag_collapse(baseline_df, drifted_current)
        # The drifted fixture halved roll_mean_1day so every zone should drop.
        assert r["n_zones_dropped_more_than_30pct"] == r["n_zones"]


# ── Real-data smoke test (skipped if parquet absent) ─────────────────────────

REAL_PARQUET = Path(__file__).resolve().parents[1] / "data" / "demand_enriched_week4.parquet"


@pytest.mark.skipif(not REAL_PARQUET.exists(),
                    reason="Real week-4 parquet not present — skipping smoke test.")
def test_real_parquet_flags_four_or_more_drift_patterns():
    from week4.scripts.compute_metrics import split_baseline_and_current
    df = pd.read_parquet(REAL_PARQUET)
    baseline, current = split_baseline_and_current(df)
    report = MetricComputer(baseline).compute_all(current)
    assert report["overall_breach"] == "crit"
    assert report["n_critical"] >= 4
