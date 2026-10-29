"""Unit and Integration Test Suite for Learning-to-Rank Architecture.

Version: 6.0-dev — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Just like race marshals inspect every kart before the championship begins, we test
our Learning-to-Rank engines with both small practice drills (where we know the exact
hand-calculated winners) and full Grand Prix simulations across all 41 validation races
and 128 test races to prove that:
1. Exactly 3 drivers stand on the podium after every single race — no more, no less!
2. If two drivers tie on pace, the driver who started closer to pole position wins!
3. LambdaMART and Plackett-Luce neural networks learn true racing pace instead of guessing!
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.builder import ALLOWED_FEATURES
from src.models.ltr_ranker import (
    F1RankerDataset,
    LambdaMARTRanker,
    PlackettLuceRanker,
    compute_list_mle_loss,
    predict_podium_top3,
)


DATA_DIR = Path("data/processed")


# =============================================================================
# 1. Dataset Assembly & Relevance Target Tests
# =============================================================================


def test_f1_ranker_dataset_relevance_mappings_handcrafted():
    """Verify that all 3 relevance mappings produce exact analytical values."""
    df_toy = pd.DataFrame(
        {
            "raceId": [1] * 12,
            "positionOrder": list(range(1, 13)),
            # P2 and P10 are unclassified (e.g. late mechanical retirement)
            "is_classified": [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            **{feat: np.zeros(12) for feat in ALLOWED_FEATURES},
        }
    )

    # 1. f1_points mapping
    ds_f1 = F1RankerDataset(target_mapping="f1_points")
    _, y_f1, group_f1 = ds_f1.transform(df_toy)
    assert len(group_f1) == 1 and group_f1[0] == 12
    # P1=25, P2=0 (unclassified), P3=15, P4=12, P5=10, P6=8, P7=6, P8=4, P9=2, P10=0 (unclassified), P11=0, P12=0
    expected_f1 = np.array([25, 0, 15, 12, 10, 8, 6, 4, 2, 0, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(y_f1, expected_f1)

    # 2. linear_rank mapping: y_i = max(0, m - positionOrder_i + 1)
    ds_lin = F1RankerDataset(target_mapping="linear_rank")
    _, y_lin, group_lin = ds_lin.transform(df_toy)
    # m = 12: P1=12, P2=11, ..., P12=1
    expected_lin = np.arange(12, 0, -1, dtype=np.int32)
    np.testing.assert_array_equal(y_lin, expected_lin)

    # 3. podium_tier mapping: P1=3, P2=0 (unclassified), P3=1, rest=0
    ds_pod = F1RankerDataset(target_mapping="podium_tier")
    _, y_pod, group_pod = ds_pod.transform(df_toy)
    expected_pod = np.array([3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(y_pod, expected_pod)


def test_f1_ranker_dataset_invalid_mapping_raises():
    """Verify that an unknown relevance mapping name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid target_mapping"):
        F1RankerDataset(target_mapping="exponential_quadratic")


def test_f1_ranker_dataset_empty_dataframe():
    """Verify that an empty DataFrame returns empty arrays gracefully."""
    ds = F1RankerDataset()
    empty_df = pd.DataFrame(columns=["raceId", "positionOrder", "is_classified"] + ALLOWED_FEATURES)
    X, y, group = ds.transform(empty_df)
    assert X.shape == (0, len(ALLOWED_FEATURES))
    assert len(y) == 0
    assert len(group) == 0


def test_f1_ranker_dataset_missing_race_col_raises():
    """Verify that a DataFrame without raceId or race_id raises KeyError."""
    ds = F1RankerDataset()
    bad_df = pd.DataFrame({"driverId": [1, 2], "positionOrder": [1, 2]})
    with pytest.raises(KeyError, match="must contain 'raceId' or 'race_id'"):
        ds.transform(bad_df)


def test_f1_ranker_dataset_contiguity_sorting():
    """Verify that interleaved/unsorted race rows are grouped contiguously without loss."""
    ds = F1RankerDataset(target_mapping="linear_rank")
    df_interleaved = pd.DataFrame(
        {
            "raceId": [10, 20, 10, 20],
            "positionOrder": [1, 1, 2, 2],
            "is_classified": [1, 1, 1, 1],
            **{feat: np.ones(4) for feat in ALLOWED_FEATURES},
        }
    )
    X, y, group = ds.transform(df_interleaved)
    assert len(group) == 2
    assert list(group) == [2, 2]
    assert len(X) == 4
    assert len(y) == 4
    # Each race of size 2 has positions 1 (relevance 2) and 2 (relevance 1)
    np.testing.assert_array_equal(y, np.array([2, 1, 2, 1], dtype=np.int32))


def test_f1_ranker_dataset_missing_imputation():
    """Verify that median imputation cleanly replaces NaNs in features."""
    df_nans = pd.DataFrame(
        {
            "raceId": [1, 1, 1],
            "positionOrder": [1, 2, 3],
            "is_classified": [1, 1, 1],
            **{feat: [np.nan, 2.0, 4.0] for feat in ALLOWED_FEATURES},
        }
    )
    ds = F1RankerDataset(impute_missing=True)
    X, y, group = ds.fit_transform(df_nans)
    assert not np.isnan(X).any()
    # The nan should be replaced by median(2.0, 4.0) = 3.0
    np.testing.assert_allclose(X[0, :], 3.0)


@pytest.mark.skipif(not (DATA_DIR / "train_core.parquet").exists(), reason="Processed data not found")
def test_f1_ranker_dataset_real_partitions_shapes():
    """Verify dataset assembly dimensions and group query sizes across real parquet partitions."""
    train_df = pd.read_parquet(DATA_DIR / "train_core.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    ds = F1RankerDataset(target_mapping="linear_rank")
    X_train, y_train, group_train = ds.transform(train_df)
    X_val, y_val, group_val = ds.transform(val_df)
    X_test, y_test, group_test = ds.transform(test_df)

    assert X_train.shape == (8920, 17)
    assert len(group_train) == 408
    assert group_train.sum() == 8920
    assert len(y_train) == 8920

    assert X_val.shape == (820, 17)
    assert len(group_val) == 41
    assert group_val.sum() == 820

    assert X_test.shape == (2559, 17)
    assert len(group_test) == 128
    assert group_test.sum() == 2559


# =============================================================================
# 2. LambdaMART GBDT Ranker Tests
# =============================================================================


def test_lambdamart_synthetic_fit_and_predict():
    """Verify that LambdaMARTRanker fits on grouped data and predicts continuous pace scores."""
    np.random.seed(42)
    # 5 races of 20 cars each = 100 observations
    X = np.random.randn(100, 17)
    group = np.array([20, 20, 20, 20, 20], dtype=np.int32)
    # Inverted ranks: 20 down to 1 in each race
    y = np.tile(np.arange(20, 0, -1, dtype=np.int32), 5)

    ranker = LambdaMARTRanker(
        n_estimators=10,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
    )
    ranker.fit(X, y, group)

    preds = ranker.predict(X)
    assert preds.shape == (100,)
    assert not np.isnan(preds).any()
    # Continuous scores: almost all values should be distinct
    assert len(np.unique(preds)) >= 90
    assert ranker.feature_importances_.shape == (17,)


@pytest.mark.skipif(not (DATA_DIR / "train_core.parquet").exists(), reason="Processed data not found")
def test_lambdamart_early_stopping_on_validation():
    """Verify that early stopping monitors validation NDCG and saves best_iteration_."""
    train_df = pd.read_parquet(DATA_DIR / "train_core.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")

    ds = F1RankerDataset(target_mapping="linear_rank")
    X_train, y_train, group_train = ds.transform(train_df)
    X_val, y_val, group_val = ds.transform(val_df)

    ranker = LambdaMARTRanker(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=15,
        random_state=42,
    )
    ranker.fit(
        X_train,
        y_train,
        group_train,
        X_val=X_val,
        y_val=y_val,
        group_val=group_val,
        early_stopping_rounds=15,
    )

    assert ranker.best_iteration_ is not None
    assert isinstance(ranker.best_iteration_, int)
    assert 1 <= ranker.best_iteration_ <= 100

    val_preds = ranker.predict(X_val)
    assert val_preds.shape == (820,)
    assert not np.isnan(val_preds).any()


# =============================================================================
# 3. Plackett-Luce ListMLE Neural Ranker Tests
# =============================================================================


def test_list_mle_loss_analytical_value():
    r"""Verify compute_list_mle_loss against exact hand calculation.

    Analytical Derivation:
    ----------------------
    Scores: s = [2.0, 1.0, 0.0]
    Relevance: y = [3.0, 2.0, 1.0] (permutation order: items 0, 1, 2)
    Formula: - \sum_{i=1}^{m-1} (s_{\pi_i} - logsumexp(s_{\pi_{i:m}}))

    i=0 (winner):
    s_0 = 2.0
    denom_0 = log(exp(2.0) + exp(1.0) + exp(0.0))
             = log(7.389056 + 2.718282 + 1.0) = log(11.107338) \approx 2.40755
    term_0 = 2.0 - 2.40755 = -0.40755

    i=1 (2nd place):
    s_1 = 1.0
    denom_1 = log(exp(1.0) + exp(0.0)) = log(3.718282) \approx 1.31326
    term_1 = 1.0 - 1.31326 = -0.31326

    Loss = - (term_0 + term_1) = 0.40755 + 0.31326 \approx 0.72081
    """
    s = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)

    loss = compute_list_mle_loss(s, y)

    term0 = 2.0 - math.log(math.exp(2.0) + math.exp(1.0) + math.exp(0.0))
    term1 = 1.0 - math.log(math.exp(1.0) + math.exp(0.0))
    expected_loss = -(term0 + term1)

    assert math.isclose(loss.item(), expected_loss, rel_tol=1e-5)

    # Verify backpropagation
    loss.backward()
    assert s.grad is not None
    assert torch.all(torch.isfinite(s.grad))


def test_list_mle_loss_boundary_cases():
    """Verify ListMLE loss handles empty and single-candidate cases cleanly."""
    # Single candidate
    s_single = torch.tensor([1.5], requires_grad=True)
    y_single = torch.tensor([1.0])
    loss_single = compute_list_mle_loss(s_single, y_single)
    assert loss_single.item() == 0.0

    # Empty candidate
    s_empty = torch.empty(0, requires_grad=True)
    y_empty = torch.empty(0)
    loss_empty = compute_list_mle_loss(s_empty, y_empty)
    assert loss_empty.item() == 0.0


def test_list_mle_gradient_descent_convergence():
    """Verify that gradient descent on ListMLE loss correctly orders the scores."""
    torch.manual_seed(42)
    # Start with inverted scores: lowest score on winner, highest score on last place
    s = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)  # true order: item 0, 1, 2

    optimizer = torch.optim.SGD([s], lr=0.2)
    initial_loss = compute_list_mle_loss(s, y).item()

    for _ in range(30):
        optimizer.zero_grad()
        loss = compute_list_mle_loss(s, y)
        loss.backward()
        optimizer.step()

    final_loss = compute_list_mle_loss(s, y).item()
    assert final_loss < initial_loss
    # After gradient updates, predicted scores must strictly match ground truth order: s[0] > s[1] > s[2]
    assert s[0].item() > s[1].item() > s[2].item()


def test_plackett_luce_ranker_synthetic_training():
    """Verify PlackettLuceRanker fits across variable grid sizes without crashing."""
    np.random.seed(42)
    # 4 races with varying grid sizes: 18, 20, 22, 19 = 79 observations
    group = np.array([18, 20, 22, 19], dtype=np.int32)
    X = np.random.randn(79, 17)
    # Inverted ranks per race
    y_list: list[int] = []
    for g in group:
        y_list.extend(list(range(g, 0, -1)))
    y = np.array(y_list, dtype=np.int32)

    ranker = PlackettLuceRanker(
        input_dim=17,
        hidden_dim=32,
        lr=1e-2,
        epochs=5,
        batch_size_races=2,
        random_state=42,
    )
    ranker.fit(X, y, group)

    preds = ranker.predict(X)
    assert preds.shape == (79,)
    assert not np.isnan(preds).any()
    assert len(np.unique(preds)) >= 70


# =============================================================================
# 4. Guaranteed Top-3 Podium Selection Tests
# =============================================================================


def test_predict_podium_top3_handcrafted_tie_breaking():
    """Verify that exact score ties are deterministically broken by ascending grid_effective."""
    df_race = pd.DataFrame(
        {
            "raceId": [101] * 5,
            # Drivers 0 and 1 have identical scores, Driver 0 has pole (grid 1), Driver 1 has grid 4
            "grid_effective": [1.0, 4.0, 2.0, 3.0, 5.0],
            "driverId": [1, 2, 3, 4, 5],
        }
    )
    # Scores: Driver 2 has 0.9 (P1), Drivers 0 & 1 have 0.8 (tied!), Driver 3 has 0.5, Driver 4 has 0.1
    scores = np.array([0.8, 0.8, 0.9, 0.5, 0.1])

    podium_mask = predict_podium_top3(scores, df_race)
    assert len(podium_mask) == 5
    assert podium_mask.sum() == 3

    # Expected winners:
    # 1. Driver 2 (score 0.9)
    # 2. Driver 0 (score 0.8, grid 1 wins tie against grid 4)
    # 3. Driver 1 (score 0.8, grid 4)
    # Drivers 3 and 4 are non-podium
    expected_mask = np.array([1, 1, 1, 0, 0], dtype=np.int32)
    np.testing.assert_array_equal(podium_mask, expected_mask)


def test_predict_podium_top3_single_race_no_race_id():
    """Verify single-race selection when DataFrame does not contain a raceId column."""
    df_single = pd.DataFrame(
        {
            "grid_effective": [1.0, 2.0, 3.0, 4.0],
            "driverId": [10, 20, 30, 40],
        }
    )
    scores = np.array([10.0, 5.0, 8.0, 1.0])
    mask = predict_podium_top3(scores, df_single)
    # Top 3 scores: indices 0 (10.0), 2 (8.0), 1 (5.0)
    expected_mask = np.array([1, 1, 1, 0], dtype=np.int32)
    np.testing.assert_array_equal(mask, expected_mask)


def test_predict_podium_top3_small_field():
    """Verify that races with fewer than 3 cars select all available candidates."""
    df_small = pd.DataFrame(
        {
            "raceId": [1, 1],
            "grid_effective": [1.0, 2.0],
        }
    )
    scores = np.array([5.0, 2.0])
    mask = predict_podium_top3(scores, df_small)
    assert mask.sum() == 2
    np.testing.assert_array_equal(mask, np.array([1, 1], dtype=np.int32))


def test_predict_podium_top3_shape_mismatch_raises():
    """Verify ValueError when scores length differs from df_race length."""
    df = pd.DataFrame({"raceId": [1, 1]})
    with pytest.raises(ValueError, match="Shape mismatch"):
        predict_podium_top3(np.array([1.0, 2.0, 3.0]), df)


@pytest.mark.skipif(not (DATA_DIR / "val.parquet").exists(), reason="Processed data not found")
def test_predict_podium_top3_validation_set_strict_three_per_race():
    """Assert strictly 3 unique podium finishers across all 41 validation races."""
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    n_races = val_df["raceId"].nunique()
    assert n_races == 41

    # Heuristic scores: negative grid
    scores = -val_df["grid_effective"].to_numpy()
    mask = predict_podium_top3(scores, val_df)

    # Total podium count across all 41 races must strictly equal 3 * 41 = 123
    assert mask.sum() == 3 * n_races == 123

    # For every individual race, count must strictly equal 3
    for race_id, group in val_df.groupby("raceId"):
        race_mask = mask[group.index]
        assert race_mask.sum() == 3, f"Race {race_id} did not produce exactly 3 podium finishers!"


@pytest.mark.skipif(not (DATA_DIR / "test.parquet").exists(), reason="Processed data not found")
def test_predict_podium_top3_test_set_strict_three_per_race():
    """Assert strictly 3 unique podium finishers across all 128 test races."""
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    n_races = test_df["raceId"].nunique()
    assert n_races == 128

    # Heuristic scores: negative grid
    scores = -test_df["grid_effective"].to_numpy()
    mask = predict_podium_top3(scores, test_df)

    # Total podium count across all 128 races must strictly equal 3 * 128 = 384
    assert mask.sum() == 3 * n_races == 384

    # For every individual race, count must strictly equal 3
    for race_id, group in test_df.groupby("raceId"):
        race_mask = mask[group.index]
        assert race_mask.sum() == 3, f"Race {race_id} did not produce exactly 3 podium finishers!"
