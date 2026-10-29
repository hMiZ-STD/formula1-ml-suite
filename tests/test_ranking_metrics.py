"""Unit and Integration Tests for Race-Aligned Ranking Metrics.

Version: 6.0-dev — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Just like a teacher double-checks exam answer keys using simple math problems before
grading hundreds of students, we test our ranking metrics using small, crystal-clear
toy races where we can calculate the exact score by hand on paper!
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ranking_metrics import (
    CaseInsensitiveDict,
    compute_dcg_at_k,
    compute_kendall_tau,
    compute_mrr,
    compute_ndcg_at_k,
    evaluate_race_ranking,
    map_finish_positions_to_relevance,
)


# =============================================================================
# Tier 1: Core Functional Tests (Hand-Crafted Analytical Parity)
# =============================================================================


def test_compute_dcg_at_k_handcrafted():
    r"""Verify DCG@K against exact analytical hand-calculated values.

    Analytical Derivation:
    ----------------------
    Relevance vector: y = [3.0, 2.0, 1.0, 0.0]
    Formula: DCG@K = \sum_{r=1}^K (2^{y_r} - 1) / log2(r + 1)

    r=1: (2^3 - 1) / log2(2) = 7.0 / 1.0 = 7.0
    r=2: (2^2 - 1) / log2(3) = 3.0 / 1.58496250072 = 1.892789260714
    r=3: (2^1 - 1) / log2(4) = 1.0 / 2.0 = 0.5
    r=4: (2^0 - 1) / log2(5) = 0.0 / log2(5) = 0.0

    DCG@1 = 7.0
    DCG@2 = 7.0 + 1.892789260714 = 8.892789260714
    DCG@3 = 8.892789260714 + 0.5 = 9.392789260714
    DCG@4 = 9.392789260714 + 0.0 = 9.392789260714
    """
    rel = np.array([3.0, 2.0, 1.0, 0.0])

    expected_dcg1 = 7.0
    expected_dcg2 = 7.0 + 3.0 / math.log2(3.0)
    expected_dcg3 = expected_dcg2 + 1.0 / math.log2(4.0)
    expected_dcg4 = expected_dcg3 + 0.0

    assert math.isclose(compute_dcg_at_k(rel, k=1), expected_dcg1, rel_tol=1e-9)
    assert math.isclose(compute_dcg_at_k(rel, k=2), expected_dcg2, rel_tol=1e-9)
    assert math.isclose(compute_dcg_at_k(rel, k=3), expected_dcg3, rel_tol=1e-9)
    assert math.isclose(compute_dcg_at_k(rel, k=4), expected_dcg4, rel_tol=1e-9)


def test_compute_ndcg_at_k_perfect_ranking():
    """Verify NDCG@K equals strictly 1.0 under a perfect descending score ranking."""
    y_true_relevance = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    y_pred_scores = np.array([100.0, 80.0, 60.0, 40.0, 20.0])

    for k in (1, 2, 3, 5, 10):
        ndcg = compute_ndcg_at_k(y_true_relevance, y_pred_scores, k=k)
        assert math.isclose(ndcg, 1.0, abs_tol=1e-9)


def test_compute_ndcg_at_k_inverted_ranking():
    r"""Verify NDCG@K matches exact mathematical calculation under fully inverted scores.

    Analytical Derivation:
    ----------------------
    True relevance: y = [3.0, 2.0, 1.0, 0.0]
    Scores (inverted): s = [1.0, 5.0, 8.0, 10.0] -> ranked order: items [3, 2, 1, 0]
    Ranked relevance: y_ranked = [0.0, 1.0, 2.0, 3.0]

    DCG@3:
    r=1: (2^0 - 1) / log2(2) = 0.0
    r=2: (2^1 - 1) / log2(3) = 1.0 / log2(3) \approx 0.63092975357
    r=3: (2^2 - 1) / log2(4) = 3.0 / 2.0 = 1.5
    DCG@3 = 2.130929753571

    IDCG@3 (ideal order [3, 2, 1, 0]):
    r=1: (2^3 - 1) / 1 = 7.0
    r=2: (2^2 - 1) / log2(3) = 3.0 / log2(3) \approx 1.892789260714
    r=3: (2^1 - 1) / 2 = 0.5
    IDCG@3 = 9.392789260714

    NDCG@3 = 2.130929753571 / 9.392789260714 \approx 0.2268686855867
    """
    y_true_relevance = np.array([3.0, 2.0, 1.0, 0.0])
    y_pred_scores = np.array([1.0, 5.0, 8.0, 10.0])

    dcg3 = 0.0 + 1.0 / math.log2(3.0) + 3.0 / math.log2(4.0)
    idcg3 = 7.0 + 3.0 / math.log2(3.0) + 1.0 / math.log2(4.0)
    expected_ndcg3 = dcg3 / idcg3

    result = compute_ndcg_at_k(y_true_relevance, y_pred_scores, k=3)
    assert math.isclose(result, expected_ndcg3, rel_tol=1e-9)


def test_compute_ndcg_at_k_partial_swap():
    r"""Verify NDCG@K when P1 and P2 are swapped.

    Relevance: [3.0, 2.0, 1.0, 0.0]
    Scores: [8.0, 10.0, 5.0, 1.0] -> ranked order: items [1, 0, 2, 3]
    Ranked relevance: [2.0, 3.0, 1.0, 0.0]

    DCG@3 = (2^2 - 1)/1 + (2^3 - 1)/log2(3) + (2^1 - 1)/2 = 3.0 + 7.0/log2(3) + 0.5 = 7.916508275
    IDCG@3 = 9.392789260714
    NDCG@3 = 7.916508275 / 9.392789260714 \approx 0.842828205
    """
    y_true_relevance = np.array([3.0, 2.0, 1.0, 0.0])
    y_pred_scores = np.array([8.0, 10.0, 5.0, 1.0])

    dcg3 = 3.0 + 7.0 / math.log2(3.0) + 0.5
    idcg3 = 7.0 + 3.0 / math.log2(3.0) + 0.5
    expected_ndcg3 = dcg3 / idcg3

    result = compute_ndcg_at_k(y_true_relevance, y_pred_scores, k=3)
    assert math.isclose(result, expected_ndcg3, rel_tol=1e-9)


def test_compute_mrr_exact_values():
    """Verify Mean Reciprocal Rank on winner positioned at 1st, 2nd, 3rd, 5th, and absent."""
    y_true_winner = np.array([1, 0, 0, 0, 0])

    # Winner ranked 1st -> 1/1 = 1.0
    scores_1st = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    assert math.isclose(compute_mrr(y_true_winner, scores_1st), 1.0, rel_tol=1e-9)

    # Winner ranked 2nd -> 1/2 = 0.5
    scores_2nd = np.array([40.0, 50.0, 30.0, 20.0, 10.0])
    assert math.isclose(compute_mrr(y_true_winner, scores_2nd), 0.5, rel_tol=1e-9)

    # Winner ranked 3rd -> 1/3 \approx 0.333333333
    scores_3rd = np.array([30.0, 50.0, 40.0, 20.0, 10.0])
    assert math.isclose(compute_mrr(y_true_winner, scores_3rd), 1.0 / 3.0, rel_tol=1e-9)

    # Winner ranked 5th -> 1/5 = 0.2
    scores_5th = np.array([10.0, 50.0, 40.0, 30.0, 20.0])
    assert math.isclose(compute_mrr(y_true_winner, scores_5th), 0.2, rel_tol=1e-9)

    # No winner labeled -> 0.0
    y_no_winner = np.array([0, 0, 0, 0, 0])
    assert compute_mrr(y_no_winner, scores_1st) == 0.0


def test_compute_kendall_tau_exact_values():
    """Verify Kendall's Tau correlation on perfect, inverted, and partial orderings."""
    true_ranks = np.array([1, 2, 3, 4])

    # Perfect ranking: highest score has rank 1
    scores_perfect = np.array([40.0, 30.0, 20.0, 10.0])
    assert math.isclose(compute_kendall_tau(true_ranks, scores_perfect), 1.0, abs_tol=1e-9)

    # Inverted ranking: lowest score has rank 1
    scores_inverted = np.array([10.0, 20.0, 30.0, 40.0])
    assert math.isclose(compute_kendall_tau(true_ranks, scores_inverted), -1.0, abs_tol=1e-9)

    # Single swap between P1 and P2 (true: [1, 2, 3, 4], pred order: [2, 1, 3, 4]):
    # Pairs = 6. Concordant = 5, Discordant = 1 -> Tau = (5 - 1) / 6 = 4/6 = 0.666666667
    scores_one_swap = np.array([30.0, 40.0, 20.0, 10.0])
    expected_tau = (5.0 - 1.0) / 6.0
    assert math.isclose(compute_kendall_tau(true_ranks, scores_one_swap), expected_tau, rel_tol=1e-9)


def test_relevance_mappings():
    """Verify linear_rank, f1_points, and podium_tier mappings and DNF handling."""
    positions = np.array([1, 2, 3, 4, 10, 11])
    is_classified = np.array([1, 1, 1, 1, 1, 0])

    # 1. linear_rank: y_i = max(0, 6 - pos + 1)
    rel_linear = map_finish_positions_to_relevance(positions, mapping="linear_rank")
    np.testing.assert_array_equal(rel_linear, np.array([6.0, 5.0, 4.0, 3.0, 0.0, 0.0]))

    # 2. f1_points: P1=25, P2=18, P3=15, P4=12, P10=1, P11=0
    rel_points = map_finish_positions_to_relevance(positions, is_classified, mapping="f1_points")
    np.testing.assert_array_equal(rel_points, np.array([25.0, 18.0, 15.0, 12.0, 1.0, 0.0]))

    # Unclassified driver in podium position must receive 0 points
    classified_dnf = np.array([1, 0, 1, 1, 1, 1])
    rel_points_dnf = map_finish_positions_to_relevance(positions, classified_dnf, mapping="f1_points")
    assert rel_points_dnf[1] == 0.0  # P2 retired -> 0 points

    # 3. podium_tier: P1=3, P2=2, P3=1, rest=0
    rel_tier = map_finish_positions_to_relevance(positions, is_classified, mapping="podium_tier")
    np.testing.assert_array_equal(rel_tier, np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0]))

    # Invalid mapping raises ValueError
    with pytest.raises(ValueError, match="Unknown relevance mapping"):
        map_finish_positions_to_relevance(positions, mapping="invalid_scheme")


# =============================================================================
# Tier 2: Boundary Value Analysis & Edge Cases
# =============================================================================


def test_dcg_boundary_cases():
    """Verify DCG behavior under k <= 0, empty inputs, all-zeros, and k > m."""
    rel = np.array([3.0, 2.0, 1.0])

    # k <= 0
    assert compute_dcg_at_k(rel, k=0) == 0.0
    assert compute_dcg_at_k(rel, k=-5) == 0.0

    # empty array
    assert compute_dcg_at_k(np.empty(0), k=3) == 0.0

    # all-zero relevance
    assert compute_dcg_at_k(np.array([0.0, 0.0, 0.0]), k=3) == 0.0

    # k > m clamps at m
    dcg_clamp = compute_dcg_at_k(rel, k=10)
    dcg_exact = compute_dcg_at_k(rel, k=3)
    assert math.isclose(dcg_clamp, dcg_exact, rel_tol=1e-9)


def test_ndcg_boundary_cases():
    """Verify NDCG behavior on all-zero relevance, empty inputs, k > m, and mismatched shapes."""
    # All-zero relevance -> IDCG=0 -> returns 0.0 (no division by zero)
    y_zeros = np.array([0.0, 0.0, 0.0])
    scores = np.array([10.0, 5.0, 1.0])
    assert compute_ndcg_at_k(y_zeros, scores, k=3) == 0.0

    # k <= 0 or empty input
    assert compute_ndcg_at_k(y_zeros, scores, k=0) == 0.0
    assert compute_ndcg_at_k(np.empty(0), np.empty(0), k=3) == 0.0

    # k > m evaluates properly up to m
    y_rel = np.array([2.0, 1.0])
    s = np.array([10.0, 5.0])
    assert math.isclose(compute_ndcg_at_k(y_rel, s, k=10), 1.0, rel_tol=1e-9)

    # Shape mismatch raises ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_ndcg_at_k(np.array([1.0, 2.0]), np.array([1.0]), k=2)


def test_mrr_boundary_cases():
    """Verify MRR on empty arrays, no winner labeled, and mismatched shapes."""
    assert compute_mrr(np.empty(0), np.empty(0)) == 0.0
    assert compute_mrr(np.array([0, 0, 0]), np.array([10.0, 5.0, 1.0])) == 0.0

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_mrr(np.array([1, 0]), np.array([10.0, 5.0, 1.0]))


def test_kendall_tau_boundary_cases():
    """Verify Kendall's Tau on single item, empty inputs, all tied scores, and mismatched shapes."""
    # Length < 2 returns 1.0
    assert compute_kendall_tau(np.empty(0), np.empty(0)) == 1.0
    assert compute_kendall_tau(np.array([1]), np.array([10.0])) == 1.0

    # All tied scores return 0.0 (not NaN)
    ranks = np.array([1, 2, 3, 4])
    tied_scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_kendall_tau(ranks, tied_scores) == 0.0

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_kendall_tau(np.array([1, 2]), np.array([10.0]))


# =============================================================================
# Tier 3: Invariance & Property Testing
# =============================================================================


def test_score_translation_and_scale_invariance():
    """Verify NDCG, MRR, and Kendall's Tau are strictly invariant to monotonic score scaling."""
    y_true_relevance = np.array([3.0, 2.0, 1.0, 0.0])
    y_true_winner = np.array([1, 0, 0, 0])
    y_true_ranks = np.array([1, 2, 3, 4])
    base_scores = np.array([4.2, 3.1, 1.8, 0.5])

    # Monotonic shift: s' = a * s + b with a > 0
    shifted_scores = 5.0 * base_scores - 12.3

    # NDCG invariant
    ndcg_base = compute_ndcg_at_k(y_true_relevance, base_scores, k=3)
    ndcg_shifted = compute_ndcg_at_k(y_true_relevance, shifted_scores, k=3)
    assert math.isclose(ndcg_base, ndcg_shifted, rel_tol=1e-9)

    # MRR invariant
    mrr_base = compute_mrr(y_true_winner, base_scores)
    mrr_shifted = compute_mrr(y_true_winner, shifted_scores)
    assert math.isclose(mrr_base, mrr_shifted, rel_tol=1e-9)

    # Kendall's Tau invariant
    tau_base = compute_kendall_tau(y_true_ranks, base_scores)
    tau_shifted = compute_kendall_tau(y_true_ranks, shifted_scores)
    assert math.isclose(tau_base, tau_shifted, rel_tol=1e-9)


# =============================================================================
# Tier 4: Real-World Multi-Race Integration & Contract Scenarios
# =============================================================================


def test_evaluate_race_ranking_mock_multi_race():
    """Verify evaluate_race_ranking across multiple races with known analytical outcomes."""
    # Construct 2 races with 4 cars each:
    # Race 1: Perfect prediction (NDCG=1.0, MRR=1.0, Tau=1.0, Top-3 P=1.0, R=1.0, F1=1.0)
    # Race 2: Winner is predicted 2nd, podium has 2/3 correct
    df = pd.DataFrame(
        {
            "raceId": [101, 101, 101, 101, 102, 102, 102, 102],
            "positionOrder": [1, 2, 3, 4, 1, 2, 3, 4],
            "pred_score": [
                40.0, 30.0, 20.0, 10.0,  # Race 1: perfect
                30.0, 40.0, 10.0, 20.0,  # Race 2: items [1, 0, 3, 2] -> predicted podium [1, 0, 3] (items 0, 1 actual podium -> 2 hits)
            ],
            "grid_effective": [1, 2, 3, 4, 1, 2, 3, 4],
            "is_classified": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    results = evaluate_race_ranking(df, score_col="pred_score", k_values=(1, 3))

    # Race 1 MRR = 1.0, Race 2 MRR = 0.5 -> Mean MRR = 0.75
    assert math.isclose(results["mrr"], 0.75, rel_tol=1e-9)

    # Race 1 Podium F1 = 1.0 (3/3)
    # Race 2 Podium: actual {0, 1, 2}, predicted {1, 0, 3} -> 2 hits -> P=2/3, R=2/3, F1=2/3
    # Mean Podium F1 = (1.0 + 2/3) / 2 = 5/6 \approx 0.833333
    assert math.isclose(results["podium_f1"], (1.0 + 2.0 / 3.0) / 2.0, rel_tol=1e-9)
    assert math.isclose(results["podium_precision"], (1.0 + 2.0 / 3.0) / 2.0, rel_tol=1e-9)
    assert math.isclose(results["podium_recall"], (1.0 + 2.0 / 3.0) / 2.0, rel_tol=1e-9)

    # NDCG@1: Race 1=1.0, Race 2: P1 item has score 30, item with score 40 has true rank 2
    # In linear rank for 4 cars: item 0 rel=4, item 1 rel=3.
    # At k=1: Race 2 predicted item 1 (rel=3). Ideal item 0 (rel=4).
    # NDCG@1 (Race 2) = (2^3 - 1) / (2^4 - 1) = 7 / 15 \approx 0.466667
    # Mean NDCG@1 = (1.0 + 7/15) / 2 = 22 / 30 \approx 0.733333
    assert math.isclose(results["ndcg@1"], (1.0 + 7.0 / 15.0) / 2.0, rel_tol=1e-6)


def test_evaluate_race_ranking_tie_breaking_with_grid():
    """Verify deterministic tie-breaking using grid_effective (closer to pole wins)."""
    # Drivers B and C have identical prediction scores (25.0).
    # Driver B started P2, Driver C started P4. Driver B must be placed ahead of C.
    df = pd.DataFrame(
        {
            "raceId": [1, 1, 1, 1],
            "positionOrder": [1, 2, 4, 3],
            "pred_score": [30.0, 25.0, 25.0, 10.0],
            "grid_effective": [1, 2, 4, 3],
        }
    )

    results = evaluate_race_ranking(df, score_col="pred_score")
    # Top 3 predicted: items 0, 1, 2 (Driver 1, Driver 2, Driver 4)
    # Actual podium: items 0, 1, 3 (positions 1, 2, 3) -> 2 hits -> F1 = 2/3
    assert math.isclose(results["podium_f1"], 2.0 / 3.0, rel_tol=1e-6)


def test_evaluate_race_ranking_case_insensitive_access():
    """Verify CaseInsensitiveDict allows lowercase, uppercase, and title-case lookups."""
    df = pd.DataFrame(
        {
            "raceId": [1, 1, 1],
            "positionOrder": [1, 2, 3],
            "score": [10.0, 8.0, 6.0],
        }
    )

    res = evaluate_race_ranking(df, score_col="score")
    assert isinstance(res, dict)
    assert isinstance(res, CaseInsensitiveDict)

    # Check key equivalence
    assert res["ndcg@3"] == res["NDCG@3"]
    assert res["mrr"] == res["MRR"]
    assert res["kendall_tau"] == res["Kendall_Tau"] == res["KENDALL_TAU"]
    assert res["podium_f1"] == res["Podium_F1"] == res["PODIUM_F1"]

    # Check .get() functionality
    assert res.get("NDCG@3") == res["ndcg@3"]
    assert res.get("non_existent", -1.0) == -1.0

    # Check `in` operator
    assert "NDCG@3" in res
    assert "not_a_key" not in res


def test_evaluate_race_ranking_missing_columns_raises():
    """Verify KeyError is raised when required columns are absent."""
    df_missing_race = pd.DataFrame({"positionOrder": [1, 2], "score": [1.0, 2.0]})
    with pytest.raises(KeyError, match="raceId"):
        evaluate_race_ranking(df_missing_race, score_col="score")

    df_missing_score = pd.DataFrame({"raceId": [1, 1], "positionOrder": [1, 2]})
    with pytest.raises(KeyError, match="missing score column"):
        evaluate_race_ranking(df_missing_score, score_col="score")

    df_missing_target = pd.DataFrame({"raceId": [1, 1], "score": [1.0, 2.0]})
    with pytest.raises(KeyError, match="missing target column"):
        evaluate_race_ranking(df_missing_target, score_col="score")


def test_evaluate_race_ranking_empty_dataframe():
    """Verify empty DataFrame returns structured zero dictionary without error."""
    df_empty = pd.DataFrame(columns=["raceId", "positionOrder", "score"])
    res = evaluate_race_ranking(df_empty, score_col="score", k_values=(1, 3, 10))

    assert res["ndcg@1"] == 0.0
    assert res["ndcg@3"] == 0.0
    assert res["ndcg@10"] == 0.0
    assert res["mrr"] == 0.0
    assert res["kendall_tau"] == 0.0
    assert res["podium_f1"] == 0.0


def test_evaluate_race_ranking_small_field():
    """Verify races with fewer than 3 cars evaluate cleanly without index out of bounds."""
    df_small = pd.DataFrame(
        {
            "raceId": [1, 1],
            "positionOrder": [1, 2],
            "score": [10.0, 5.0],
        }
    )
    res = evaluate_race_ranking(df_small, score_col="score", k_values=(1, 3))
    assert res["ndcg@3"] == 1.0
    assert res["mrr"] == 1.0
    assert res["kendall_tau"] == 1.0
    assert res["podium_precision"] == 1.0
    assert res["podium_recall"] == 1.0
    assert res["podium_f1"] == 1.0


def test_evaluate_race_ranking_real_test_partition():
    """Verify ranking metrics against real modern test split (2019-2024, 128 races)."""
    test_path = Path("data/processed/test.parquet")
    if not test_path.exists():
        pytest.skip(f"Test split {test_path} not found.")

    df_test = pd.read_parquet(test_path)
    assert len(df_test) == 2559
    assert df_test["raceId"].nunique() == 128

    # Using inverted grid position as a realistic baseline predictor
    df_test["grid_heuristic_score"] = -df_test["grid_effective"]

    metrics = evaluate_race_ranking(
        df_test,
        score_col="grid_heuristic_score",
        target_col="positionOrder",
        k_values=(1, 3, 10),
        relevance_mapping="linear_rank",
    )

    # Pole position and top starting positions are highly predictive in F1:
    # Empirical bounds: NDCG@3 in [0.65, 0.85], MRR in [0.60, 0.80], Kendall Tau in [0.40, 0.60]
    assert 0.60 <= metrics["ndcg@3"] <= 0.85
    assert 0.55 <= metrics["mrr"] <= 0.80
    assert 0.40 <= metrics["kendall_tau"] <= 0.60
    assert 0.55 <= metrics["podium_f1"] <= 0.80
