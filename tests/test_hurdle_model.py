"""Comprehensive Unit and Integration Test Suite for Two-Stage Hurdle Architecture.

Version: 6.0-dev — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Just like a flight engineer tests an airplane's safety systems in both wind tunnels
and simulated emergency weather, we test our Two-Stage Hurdle model with both synthetic
mathematical boundary puzzles and real Grand Prix simulations across all 41 validation races
and 128 test races to prove that:
1. The hazard brain and pace brain remain strictly decoupled (zero DNF rows leak into Stage 2).
2. Expected finishing position and podium probabilities strictly obey the Law of Total Probability bounds.
3. Exactly 3 unique podium finishers are selected for every single Grand Prix with deterministic tie-breaking.
4. Extreme scenarios (p_dnf = 0.0, p_dnf = 1.0) behave with 100% mathematical precision.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score, r2_score

from src.features.builder import ALLOWED_FEATURES
from src.models.hurdle_model import F1HurdleModel, predict_podium_top3

DATA_DIR = Path("data/processed")


# =============================================================================
# Synthetic Fixtures for Isolated Unit Testing
# =============================================================================


@pytest.fixture
def synthetic_hurdle_data():
    """Generates synthetic multi-race data with known DNF and classified pace."""
    rng = np.random.default_rng(42)
    n_races = 5
    cars_per_race = 20
    n_total = n_races * cars_per_race

    race_ids = np.repeat(np.arange(1, n_races + 1), cars_per_race)
    grid = np.tile(np.arange(1, cars_per_race + 1), n_races)

    # 25% DNF rate
    is_dnf = np.zeros(n_total, dtype=np.int32)
    # Drivers starting in back have higher DNF risk
    for r in range(n_races):
        start_idx = r * cars_per_race
        dnf_indices = start_idx + rng.choice(np.arange(10, cars_per_race), size=5, replace=False)
        is_dnf[dnf_indices] = 1

    is_classified = 1 - is_dnf
    position_order = np.tile(np.arange(1, cars_per_race + 1), n_races)

    # Synthetic features: pole correlated, noise, and NaNs
    features_dict = {}
    for feat in ALLOWED_FEATURES:
        if feat == "grid_effective":
            features_dict[feat] = grid.astype(np.float64)
        elif feat == "grid_vs_quali_delta":
            vals = rng.normal(0, 1, n_total)
            vals[rng.choice(n_total, size=10, replace=False)] = np.nan
            features_dict[feat] = vals
        elif feat == "q_gap_pct":
            vals = rng.uniform(0, 5, n_total)
            vals[rng.choice(n_total, size=5, replace=False)] = np.nan
            features_dict[feat] = vals
        else:
            features_dict[feat] = rng.normal(0, 1, n_total)

    df = pd.DataFrame(
        {
            "raceId": race_ids,
            "grid_effective": grid.astype(np.float64),
            "positionOrder": position_order,
            "is_classified": is_classified,
            "is_dnf": is_dnf,
            "y_podium": ((position_order <= 3) & (is_classified == 1)).astype(np.int32),
            **features_dict,
        }
    )
    return df


# =============================================================================
# 1. Stage 1 Hazard Classifier & Calibration Tests
# =============================================================================


def test_stage1_hazard_probabilities_bounded_and_calibrated(synthetic_hurdle_data):
    """Verify that Stage 1 hazard probabilities are bounded in [0, 1] and calibrated."""
    df = synthetic_hurdle_data
    df_train = df[df["raceId"].isin([1, 2, 3])].copy()
    df_val = df[df["raceId"] == 4].copy()
    df_test = df[df["raceId"] == 5].copy()

    model = F1HurdleModel(random_state=42)
    model.fit(df_train, df_val=df_val)

    assert model.is_calibrated_, "Stage 1 must be calibrated when df_val is provided."

    p_dnf = model.predict_hazard(df_test)

    # Probability bounds
    assert isinstance(p_dnf, np.ndarray)
    assert p_dnf.shape == (len(df_test),)
    assert (p_dnf >= 0.0).all(), "Hazard probabilities cannot be negative."
    assert (p_dnf <= 1.0).all(), "Hazard probabilities cannot exceed 1.0."

    # Non-trivial prediction distribution (not a degenerate constant)
    assert np.var(p_dnf) > 1e-6, "Stage 1 probabilities must not be a degenerate constant."


def test_stage1_fit_without_validation_set(synthetic_hurdle_data):
    """Verify that fitting without a validation set still operates correctly."""
    df = synthetic_hurdle_data
    df_train = df[df["raceId"].isin([1, 2, 3])].copy()
    df_test = df[df["raceId"] == 5].copy()

    model = F1HurdleModel(random_state=42)
    model.fit(df_train, df_val=None)

    assert not model.is_calibrated_, "Model must reflect uncalibrated state when df_val is None."
    p_dnf = model.predict_hazard(df_test)
    assert (p_dnf >= 0.0).all() and (p_dnf <= 1.0).all()


# =============================================================================
# 2. Stage 2 Training Purity Tests (Zero DNF Rows Leakage)
# =============================================================================


def test_stage2_training_purity_zero_dnf_leakage(synthetic_hurdle_data):
    """Verify that Stage 2 is fitted strictly on classified starters with zero DNF rows."""
    df_train = synthetic_hurdle_data.copy()
    num_dnf_in_train = (df_train["is_dnf"] == 1).sum()
    assert num_dnf_in_train > 0, "Precondition: training data must contain DNF rows."

    model = F1HurdleModel(random_state=42)
    model.fit(df_train)

    # Strict purity assertion
    assert model.y_train_stage2_dnf_ is not None
    assert (model.y_train_stage2_dnf_ == 0).all(), "Stage 2 training set must contain zero DNF rows!"
    assert (model.stage2_is_dnf_ == 0).all(), "model.stage2_is_dnf_ must have all zeros."

    expected_stage2_rows = (df_train["is_classified"] == 1).sum()
    assert len(model.y_train_stage2_dnf_) == expected_stage2_rows


def test_stage2_all_dnf_dataset_raises():
    """Verify that a training set with 100% DNF raises ValueError."""
    rng = np.random.default_rng(42)
    df_all_dnf = pd.DataFrame(
        {
            "raceId": [1] * 10,
            "positionOrder": list(range(1, 11)),
            "is_classified": [0] * 10,
            "is_dnf": [1] * 10,
            **{feat: rng.normal(0, 1, 10) for feat in ALLOWED_FEATURES},
        }
    )
    model = F1HurdleModel(random_state=42)
    with pytest.raises(ValueError, match="0 classified starters"):
        model.fit(df_all_dnf)


# =============================================================================
# 3. Total Probability Recombination Bounds Tests
# =============================================================================


def test_total_probability_recombination_bounds(synthetic_hurdle_data):
    """Verify that expected position and podium probabilities obey mathematical bounds."""
    df = synthetic_hurdle_data
    df_train = df[df["raceId"].isin([1, 2, 3, 4])].copy()
    df_test = df[df["raceId"] == 5].copy()

    model = F1HurdleModel(random_state=42)
    model.fit(df_train)

    p_dnf = model.predict_hazard(df_test)
    y_finish = model.predict_classified_pace(df_test)
    y_expected = model.predict_expected_position(df_test)
    y_bar_dnf = model.mean_dnf_position_

    # Verify convex combination formula
    expected_analytical = (1.0 - p_dnf) * y_finish + p_dnf * y_bar_dnf
    np.testing.assert_allclose(y_expected, expected_analytical, rtol=1e-5, atol=1e-5)

    # For p_dnf in [0, 1], y_expected must lie between y_finish and y_bar_dnf
    lower_bound = np.minimum(y_finish, y_bar_dnf) - 1e-6
    upper_bound = np.maximum(y_finish, y_bar_dnf) + 1e-6
    assert (y_expected >= lower_bound).all(), "Expected position violates lower bound!"
    assert (y_expected <= upper_bound).all(), "Expected position violates upper bound!"

    # Podium probability bounds
    p_podium = model.predict_podium_probability(df_test)
    assert (p_podium >= 0.0).all(), "Podium probability cannot be negative!"
    assert (p_podium <= 1.0).all(), "Podium probability cannot exceed 1.0!"


# =============================================================================
# 4. Edge Cases: Extreme Probabilities & Exact Score Ties
# =============================================================================


def test_edge_case_extreme_p_dnf_zero_and_one(synthetic_hurdle_data):
    """Verify analytical limits when p_dnf is exactly 0.0 or 1.0."""
    df_train = synthetic_hurdle_data.copy()
    model = F1HurdleModel(random_state=42).fit(df_train)

    test_row = df_train.iloc[[0]].copy()

    # Case A: Mock p_dnf = 1.0 (guaranteed breakdown)
    model.predict_hazard = MagicMock(return_value=np.array([1.0]))
    y_exp_1 = model.predict_expected_position(test_row)
    p_pod_1 = model.predict_podium_probability(test_row)

    # Expected position must equal mean DNF position exactly
    np.testing.assert_allclose(y_exp_1, np.array([model.mean_dnf_position_]))
    # Podium probability must be exactly 0.0
    np.testing.assert_allclose(p_pod_1, np.array([0.0]))

    # Case B: Mock p_dnf = 0.0 (guaranteed survival)
    model.predict_hazard = MagicMock(return_value=np.array([0.0]))
    y_exp_0 = model.predict_expected_position(test_row)
    y_fin = model.predict_classified_pace(test_row)

    # Expected position must equal classified pace exactly
    np.testing.assert_allclose(y_exp_0, y_fin)


def test_edge_case_exact_score_ties_grid_tie_breaking():
    """Verify that exact score ties are broken deterministically by grid_effective."""
    # 4 drivers with identical predicted score, varying grid positions
    df_race = pd.DataFrame(
        {
            "raceId": [101, 101, 101, 101],
            "driverId": [10, 20, 30, 40],
            "grid_effective": [4.0, 1.0, 3.0, 2.0],  # Driver 20 is on pole, 40 is P2, 30 is P3, 10 is P4
        }
    )
    # Tied scores
    tied_scores = np.array([0.85, 0.85, 0.85, 0.85])

    podium_mask = predict_podium_top3(tied_scores, df_race)
    assert len(podium_mask) == 4
    assert podium_mask.sum() == 3

    # Drivers on grid 1.0 (idx 1), 2.0 (idx 3), and 3.0 (idx 2) must win
    # Driver on grid 4.0 (idx 0) must lose
    assert podium_mask[1] == 1  # Grid 1.0
    assert podium_mask[3] == 1  # Grid 2.0
    assert podium_mask[2] == 1  # Grid 3.0
    assert podium_mask[0] == 0  # Grid 4.0


def test_edge_case_single_race_and_small_field():
    """Verify behavior on single race without raceId and small field sizes."""
    # Small field: 2 cars
    df_small = pd.DataFrame(
        {
            "grid_effective": [1.0, 2.0],
        }
    )
    scores_small = np.array([0.9, 0.7])
    mask_small = predict_podium_top3(scores_small, df_small)
    assert mask_small.sum() == 2
    np.testing.assert_array_equal(mask_small, np.array([1, 1], dtype=np.int32))

    # Empty DataFrame
    df_empty = pd.DataFrame(columns=["grid_effective"])
    mask_empty = predict_podium_top3(np.array([]), df_empty)
    assert len(mask_empty) == 0

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        predict_podium_top3(np.array([0.5]), df_small)


def test_unfitted_model_raises():
    """Verify that calling predict methods on an unfitted model raises NotFittedError."""
    model = F1HurdleModel()
    dummy_df = pd.DataFrame({"grid_effective": [1.0]})

    with pytest.raises(NotFittedError):
        model.predict_hazard(dummy_df)

    with pytest.raises(NotFittedError):
        model.predict_classified_pace(dummy_df)

    with pytest.raises(NotFittedError):
        model.predict_expected_position(dummy_df)

    with pytest.raises(NotFittedError):
        model.predict_podium_probability(dummy_df)

    with pytest.raises(NotFittedError):
        model.predict_podium_top3(dummy_df)


# =============================================================================
# 5. Full Real-Data Integration & Regression Tests
# =============================================================================


@pytest.mark.skipif(
    not (DATA_DIR / "train_core.parquet").exists(),
    reason="Real parquet dataset files not found",
)
def test_real_dataset_training_and_purity():
    """Verify Hurdle fitting on train_core.parquet and calibration on val.parquet."""
    df_train = pd.read_parquet(DATA_DIR / "train_core.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val.parquet")

    model = F1HurdleModel(random_state=42)
    model.fit(df_train, df_val=df_val)

    assert model.is_fitted_
    assert model.is_calibrated_
    # Verify zero DNF rows in Stage 2
    assert (model.y_train_stage2_dnf_ == 0).all()
    # Verify mean DNF position ~ 18.04
    assert 17.5 <= model.mean_dnf_position_ <= 18.5


@pytest.mark.skipif(
    not (DATA_DIR / "val.parquet").exists() or not (DATA_DIR / "test.parquet").exists(),
    reason="Validation or test parquet dataset files not found",
)
def test_predict_podium_top3_strict_three_per_race_val_and_test():
    """Verify that predict_podium_top3 strictly guarantees 3 unique podium finishers per race."""
    df_train = pd.read_parquet(DATA_DIR / "train_core.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")

    model = F1HurdleModel(random_state=42)
    model.fit(df_train, df_val=df_val)

    # 1. Validation Set: 41 races -> exactly 123 podium finishers
    val_top3_prob = model.predict_podium_top3(df_val, ranking_criterion="podium_probability")
    val_top3_pos = model.predict_podium_top3(df_val, ranking_criterion="expected_position")

    n_val_races = df_val["raceId"].nunique()
    assert n_val_races == 41
    assert val_top3_prob.sum() == 3 * n_val_races, "Must have exactly 3 podium finishers per race in val!"
    assert val_top3_pos.sum() == 3 * n_val_races

    # Verify per-race sum is strictly 3 for every single race in validation
    df_val_check = df_val.copy()
    df_val_check["pred_podium"] = val_top3_prob
    race_sums_val = df_val_check.groupby("raceId")["pred_podium"].sum()
    assert (race_sums_val == 3).all(), "Every validation race must have exactly 3 podium predictions!"

    # 2. Test Set: 128 races -> exactly 384 podium finishers
    test_top3_prob = model.predict_podium_top3(df_test, ranking_criterion="podium_probability")
    test_top3_pos = model.predict_podium_top3(df_test, ranking_criterion="expected_position")

    n_test_races = df_test["raceId"].nunique()
    assert n_test_races == 128
    assert test_top3_prob.sum() == 3 * n_test_races, "Must have exactly 3 podium finishers per race in test!"
    assert test_top3_pos.sum() == 3 * n_test_races

    # Verify per-race sum is strictly 3 for every single race in test
    df_test_check = df_test.copy()
    df_test_check["pred_podium"] = test_top3_prob
    race_sums_test = df_test_check.groupby("raceId")["pred_podium"].sum()
    assert (race_sums_test == 3).all(), "Every test race must have exactly 3 podium predictions!"


@pytest.mark.skipif(
    not (DATA_DIR / "test.parquet").exists(),
    reason="Test parquet dataset file not found",
)
def test_hurdle_classified_pace_superior_r2_over_v5_baseline():
    """Verify that Stage 2 pace regressor achieves R^2 > 0.55 on classified starters."""
    df_train = pd.read_parquet(DATA_DIR / "train_core.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")

    model = F1HurdleModel(random_state=42).fit(df_train, df_val=df_val)

    # Classified starters in test set
    mask_classified_test = df_test["is_classified"] == 1
    y_test_classified_true = df_test.loc[mask_classified_test, "positionOrder"]

    y_test_finish_pred = model.predict_classified_pace(df_test)[mask_classified_test]

    classified_r2 = r2_score(y_test_classified_true, y_test_finish_pred)
    # Traditional pointwise regression without DNF decoupling caps at ~0.408.
    # Decoupled classified pace must exceed 0.55!
    assert classified_r2 > 0.55, f"Stage 2 pace R^2 ({classified_r2:.4f}) should exceed 0.55!"

    # Hurdle podium selection F1 sanity check
    top3_preds = model.predict_podium_top3(df_test)
    podium_f1 = f1_score(df_test["y_podium"], top3_preds)
    assert podium_f1 > 0.60, f"Hurdle top-3 podium F1 ({podium_f1:.4f}) must exceed 0.60!"
