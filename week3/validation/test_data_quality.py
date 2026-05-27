"""Tests for the week-3 data quality validator and the graceful-degradation
layer in week3/backend/data.py.

Runs with synthetic in-memory data so the suite passes in CI without needing
the 74 MB parquet on the runner. A separate `test_real_corrupted_parquet`
test runs against the real file if present, otherwise it's skipped.

From the repo root:
    pytest week3/validation/test_data_quality.py -v
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from week3.validation.check_data_quality import (
    CUTOFF,
    DataQualityValidator,
    split_baseline_and_current,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "PULocationID", "time_bucket", "trip_count",
    "hour", "minute", "dayofweek", "month",
    "is_weekend", "is_holiday", "cbd_pricing_active", "is_airport_zone",
]

def _make_clean_df(n_zones: int = 5, days: int = 20, start="2025-12-01",
                   holiday_rate: float = 0.05, seed: int = 7) -> pd.DataFrame:
    """Synthetic baseline-like data: clean, in-range, no duplicates.

    `holiday_rate` controls what fraction of *days* are flagged as holiday so
    the rate_drift check has a non-zero baseline to compare against.
    """
    rows = []
    rng = np.random.default_rng(seed=seed)
    zone_ids = [10 + i * 11 for i in range(n_zones)]
    start_ts = pd.Timestamp(start)
    # Pick a deterministic-ish set of holiday days from the window.
    holiday_days = set(rng.choice(days, size=max(1, int(days * holiday_rate)), replace=False).tolist())
    for zone in zone_ids:
        for d in range(days):
            is_h = 1 if d in holiday_days else 0
            for slot in range(96):  # 15-min slots per day
                ts = start_ts + pd.Timedelta(days=d, minutes=15 * slot)
                rows.append({
                    "PULocationID": zone,
                    "time_bucket": ts,
                    "trip_count": int(rng.integers(0, 60)),
                    "hour": ts.hour,
                    "minute": ts.minute,
                    "dayofweek": ts.dayofweek,
                    "month": ts.month,
                    "is_weekend": int(ts.dayofweek >= 5),
                    "is_holiday": is_h,
                    "cbd_pricing_active": int(rng.integers(0, 2)),
                    "is_airport_zone": int(zone in {1, 132, 138}),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def baseline_df() -> pd.DataFrame:
    return _make_clean_df()


@pytest.fixture
def validator(baseline_df) -> DataQualityValidator:
    return DataQualityValidator(baseline_df=baseline_df)


@pytest.fixture
def clean_current(baseline_df) -> pd.DataFrame:
    """A second clean window with the same distribution — should validate cleanly."""
    return _make_clean_df(start="2026-01-16", seed=42)


# ── Baseline behaviour ────────────────────────────────────────────────────────

class TestBaselineData:
    def test_clean_current_passes(self, validator, clean_current):
        result = validator.validate(clean_current)
        assert result["is_valid"], f"Clean data flagged as invalid: {result['issues']}"
        # Some medium-severity rate drift is allowed in synthetic data; just
        # make sure there are no critical/high blockers.
        blockers = [i for i in result["issues"]
                    if i["severity"] in {"critical", "high"}]
        assert blockers == [], f"Unexpected blockers: {blockers}"

    def test_empty_df_does_not_crash(self, validator):
        # An empty frame must produce schema issues, not raise.
        result = validator.validate(pd.DataFrame())
        assert not result["is_valid"]
        assert any(i["type"] == "missing_columns" for i in result["issues"])


# ── Issue-specific detection ──────────────────────────────────────────────────

class TestDataQualityIssues:
    def test_detect_negative_trip_count(self, validator, clean_current):
        """Issue 2 (lower half): trip_count can never be negative."""
        bad = clean_current.copy()
        bad.loc[bad.index[:5], "trip_count"] = -1
        result = validator.validate(bad)
        assert not result["is_valid"]
        types = [i["type"] for i in result["issues"]]
        assert "out_of_range_low" in types

    def test_detect_extreme_trip_count(self, validator, clean_current):
        """Issue 2 (upper half): impossible 99,999 trip counts."""
        bad = clean_current.copy()
        bad.loc[bad.index[:3], "trip_count"] = 99_999
        result = validator.validate(bad)
        assert not result["is_valid"]
        assert any(i["type"] == "out_of_range_high" and i["column"] == "trip_count"
                   for i in result["issues"])

    def test_detect_duplicate_rows(self, validator, clean_current):
        """Issue 1: full-row duplicates."""
        bad = pd.concat([clean_current, clean_current.head(50)], ignore_index=True)
        result = validator.validate(bad)
        assert not result["is_valid"]
        assert any(i["type"] == "duplicate_rows" for i in result["issues"])

    def test_detect_duplicate_keys(self, validator, clean_current):
        """Issue 1 variant: same (PULocationID, time_bucket) appears twice with diff values."""
        bad = clean_current.copy()
        twin = bad.head(20).copy()
        twin["trip_count"] = twin["trip_count"] + 1   # different content, same key
        bad = pd.concat([bad, twin], ignore_index=True)
        result = validator.validate(bad)
        types = [i["type"] for i in result["issues"]]
        assert "duplicate_keys" in types

    def test_detect_variance_collapse(self, validator, clean_current):
        """Issue 3: a 0/1 feature with all-1 values is a dead feature."""
        bad = clean_current.copy()
        bad["cbd_pricing_active"] = 1
        result = validator.validate(bad)
        assert any(i["type"] == "variance_collapse"
                   and i["column"] == "cbd_pricing_active"
                   for i in result["issues"])

    def test_detect_rate_drift_is_holiday(self, validator, clean_current):
        """Issue 4: is_holiday rate spike of 4x triggers a rate_drift flag."""
        bad = clean_current.copy()
        # Force ~50% of rows to be 'holiday' in the candidate window.
        idx = bad.sample(frac=0.5, random_state=1).index
        bad.loc[idx, "is_holiday"] = 1
        result = validator.validate(bad)
        assert any(i["type"] == "rate_drift" and i["column"] == "is_holiday"
                   for i in result["issues"])

    def test_missing_required_column(self, validator, clean_current):
        bad = clean_current.drop(columns=["trip_count"])
        result = validator.validate(bad)
        assert not result["is_valid"]
        assert any(i["type"] == "missing_columns" for i in result["issues"])


# ── Graceful degradation in data.py (the API loader) ──────────────────────────

class TestGracefulDegradation:
    """Tests that data.py never crashes and always logs when it cleans data."""

    def test_clean_upstream_dedupes_and_clips(self, caplog):
        """The _clean_upstream helper should remove dupes, clip out-of-range, log."""
        from week3.backend.data import _clean_upstream
        df = pd.DataFrame({
            "PULocationID": [10, 10, 20, 20],
            "time_bucket": pd.to_datetime(["2026-01-16 00:00"] * 4),
            "trip_count": [5, 5, -3, 99_999],
            "hour": [0, 0, 0, 0],
        })
        with caplog.at_level(logging.WARNING, logger="week3.backend.data"):
            out = _clean_upstream(df)
        # Dupes by (zone, time) removed; 4 rows -> 2.
        assert len(out) == 2
        # Clipping: -3 -> 0, 99999 -> 500
        assert out["trip_count"].min() >= 0
        assert out["trip_count"].max() <= 500
        # And it was logged.
        assert any("duplicate" in r.message.lower() for r in caplog.records)
        assert any("clipping" in r.message.lower() for r in caplog.records)

    def test_check_and_log_data_quality_never_crashes(self, monkeypatch, caplog):
        """If the parquet is missing, log a warning and return — don't raise."""
        from week3.backend import data as data_module
        monkeypatch.setattr(data_module, "CORRUPTED_DATA_PATH",
                            Path("/tmp/does-not-exist.parquet"))
        with caplog.at_level(logging.WARNING, logger="week3.backend.data"):
            data_module.check_and_log_data_quality()  # must NOT raise
        assert any("not found" in r.message.lower() for r in caplog.records)


# ── Optional: real-data smoke test (only runs if the parquet is on disk) ──────

REAL_PARQUET = Path(__file__).resolve().parents[1] / "data" / "demand_enriched_corrupted.parquet"


@pytest.mark.skipif(not REAL_PARQUET.exists(),
                    reason="Real corrupted parquet not present — skipping smoke test.")
def test_real_corrupted_parquet_flags_all_four_issue_types():
    df = pd.read_parquet(REAL_PARQUET)
    baseline, current = split_baseline_and_current(df)
    result = DataQualityValidator(baseline_df=baseline).validate(current)
    assert not result["is_valid"]
    types = {i["type"] for i in result["issues"]}
    # All four issue families from the report:
    assert "out_of_range_low" in types or "out_of_range_high" in types
    assert "duplicate_rows" in types or "duplicate_keys" in types
    assert "variance_collapse" in types
    assert "rate_drift" in types
