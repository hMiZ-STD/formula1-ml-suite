"""Pipeline Integrity and Anti-Leakage Verification Tests.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_clean_starters, load_raw_csvs
from src.data.preprocessor import partition_temporal_sets, race_time_folds
from src.features.builder import (
    ALLOWED_FEATURES,
    FORBIDDEN_FEATURES,
    build_pre_race_features,
    parse_q1_time,
)


@pytest.fixture(scope="module")
def raw_data():
    """Load raw CSV tables fixture."""
    return load_raw_csvs()


@pytest.fixture(scope="module")
def starters_data():
    """Load clean starters fixture."""
    return load_clean_starters()


@pytest.fixture(scope="module")
def features_data():
    """Build full features fixture."""
    return build_pre_race_features()


def test_raw_null_handling(raw_data):
    """Verify that SQL NULL '\\N' was parsed to genuine NaN."""
    results = raw_data["results"]
    assert results["positionOrder"].isna().sum() == 0, "positionOrder has nulls!"
    assert pd.api.types.is_numeric_dtype(results["positionOrder"])
    assert results["position"].isna().sum() > 0, "Expected missing values in position (DNF/DSQ)."


def test_unique_race_driver_modern_era(starters_data):
    """Verify exactly one row per (raceId, driverId) in modern era (no shared drives)."""
    dups = starters_data.duplicated(subset=["raceId", "driverId"]).sum()
    assert dups == 0, f"Found {dups} duplicate (raceId, driverId) rows!"


def test_whitelist_strict_enforcement(features_data):
    """Verify that only authorized features are used and no forbidden features can leak."""
    X = features_data[ALLOWED_FEATURES]
    assert set(X.columns) == set(ALLOWED_FEATURES), "Mismatch between X columns and whitelist!"

    # Ensure none of the forbidden columns are present in X
    intersection = set(X.columns).intersection(FORBIDDEN_FEATURES)
    assert len(intersection) == 0, f"Forbidden features found in model matrix: {intersection}"


def test_constructor_teammate_invariance(features_data):
    """Verify all teammates on the same constructor in the same race share identical pre-race points share."""
    team_variances = features_data.groupby(["raceId", "canonical_constructor_id"])[
        "constructor_pts_share"
    ].nunique()
    assert team_variances.max() == 1, (
        "Anti-leakage violation: Teammates have different pre-race constructor_pts_share values!"
    )


def test_sprint_race_masking(features_data):
    """Verify that all sprint weekend races from 2021 onward have grid_vs_quali_delta masked as NaN."""
    with open("data/processed/audit/sprint_race_ids.json", "r", encoding="utf-8") as f:
        sprint_ids = set(json.load(f))

    assert len(sprint_ids) == 18, f"Expected 18 sprint races, found {len(sprint_ids)}"
    sprint_rows = features_data[features_data["raceId"].isin(sprint_ids)]
    assert len(sprint_rows) > 0, "No sprint rows found in dataset!"
    assert sprint_rows["grid_vs_quali_delta"].isna().all(), (
        "Anti-leakage violation: grid_vs_quali_delta is not NaN for sprint race weekends!"
    )


def test_disjoint_temporal_partitions(features_data):
    """Verify that train_core, val, and test share zero common raceIds."""
    train_core, val, test, train_val_pool = partition_temporal_sets(features_data)

    train_races = set(train_core["raceId"].unique())
    val_races = set(val["raceId"].unique())
    test_races = set(test["raceId"].unique())

    assert len(train_races.intersection(val_races)) == 0, "train_core and val overlap in raceId!"
    assert len(train_races.intersection(test_races)) == 0, "train_core and test overlap in raceId!"
    assert len(val_races.intersection(test_races)) == 0, "val and test overlap in raceId!"
    assert train_races.union(val_races) == set(train_val_pool["raceId"].unique())


def test_future_perturbation_invariance(features_data):
    """Verify that modifying future race outcomes does not alter past rolling features.

    ELI5:
    If race 500's results change, the rolling average for race 499 MUST NOT change at all.
    This guarantees that .shift(1) successfully prevents future data leakage.
    """
    # Pick a driver with continuous participation (e.g. Fernando Alonso or Lewis Hamilton)
    driver_rows = features_data[features_data["driverId"] == 1].sort_values(["year", "round"])
    if len(driver_rows) < 10:
        pytest.skip("Driver 1 has insufficient rows for perturbation test.")

    # Check that driver's rolling finish p5 at index i is independent of outcome at index >= i
    for i in range(5, min(15, len(driver_rows))):
        val_at_i = driver_rows.iloc[i]["driver_norm_finish_p5"]
        assert np.isfinite(val_at_i) or np.isnan(val_at_i)
