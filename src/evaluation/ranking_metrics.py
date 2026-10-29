"""Race-Aligned Ranking Metrics Module for Formula 1.

Version: 6.0-dev — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Imagine a Formula 1 race as a school sports day where 20 runners race for trophies.
Instead of asking "Is this runner fast in general?", race ranking checks who crossed
the finish line 1st, 2nd, and 3rd!

1. NDCG (Normalized Discounted Cumulative Gain):
   Gives the biggest rewards for getting the podium medals right! Getting the gold
   medalist right gives a huge score boost, getting silver or bronze gives a good boost,
   while swapping 17th and 18th place barely matters at all.
2. MRR (Mean Reciprocal Rank):
   Checks how quickly we found the gold medalist! If our predicted winner actually won,
   we get 1/1 = 1.0 (perfect!). If our predicted winner ended up in 2nd place, we get
   1/2 = 0.5. If they crossed the line in 4th, we get 1/4 = 0.25.
3. Kendall's Tau:
   Checks if our overall finish list matches the real race order without any mix-ups!
   It looks at every pair of drivers (like Lewis vs. Max) and checks: "Did our model
   correctly predict who finished ahead of whom?"
4. Top-3 Podium Precision, Recall, and F1:
   Checks how many of the 3 drivers we placed on the imaginary podium actually sprayed
   champagne on the real podium after the race!
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats


class CaseInsensitiveDict(dict):
    """Dictionary supporting case-insensitive key lookups while preserving canonical keys.

    Allows callers to access ranking metric results via either lowercase
    (e.g., 'ndcg@3', 'mrr') or uppercase/title case (e.g., 'NDCG@3', 'MRR', 'Kendall_Tau').
    """

    def __getitem__(self, key: Any) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            key_lower = key.lower()
            for k in self:
                if isinstance(k, str) and k.lower() == key_lower:
                    return super().__getitem__(k)
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            key_lower = key.lower()
            return any(isinstance(k, str) and k.lower() == key_lower for k in self)
        return False

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


# Canonical 2010-2024 F1 championship points mapping
F1_POINTS_MAP: dict[int, float] = {
    1: 25.0,
    2: 18.0,
    3: 15.0,
    4: 12.0,
    5: 10.0,
    6: 8.0,
    7: 6.0,
    8: 4.0,
    9: 2.0,
    10: 1.0,
}

# Podium tier relevance mapping (Gold=3, Silver=2, Bronze=1, others=0)
PODIUM_TIER_MAP: dict[int, float] = {
    1: 3.0,
    2: 2.0,
    3: 1.0,
}


def map_finish_positions_to_relevance(
    positions: np.ndarray,
    is_classified: np.ndarray | None = None,
    mapping: str = "linear_rank",
) -> np.ndarray:
    """Map race finishing positions to non-negative numerical relevance scores.

    Parameters
    ----------
    positions : np.ndarray
        Array of integer or float finishing positions (1-based, positionOrder).
    is_classified : np.ndarray or None, default=None
        Optional binary indicator array (1 for classified finishers, 0 for DNF/unclassified).
        If provided, unclassified drivers receive 0 relevance under 'f1_points'
        and 'podium_tier' mappings.
    mapping : str, default='linear_rank'
        Relevance formulation strategy:
        - 'linear_rank': Inverted rank: y_i = max(0, field_size - position_i + 1).
        - 'f1_points': Standard 2010+ F1 points (25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0).
        - 'podium_tier': Podium tiers (P1=3, P2=2, P3=1, rest=0).

    Returns
    -------
    np.ndarray
        1D array of float64 relevance values matching input length.

    Raises
    ------
    ValueError
        If mapping name is unrecognized.
    """
    pos_arr = np.asarray(positions, dtype=np.float64).ravel()
    m = len(pos_arr)
    if m == 0:
        return np.empty(0, dtype=np.float64)

    classified_mask = (
        np.asarray(is_classified).ravel() == 1
        if is_classified is not None
        else np.ones(m, dtype=bool)
    )

    if mapping == "linear_rank":
        # y_i = max(0, m - positionOrder_i + 1)
        relevance = np.maximum(0.0, float(m) - pos_arr + 1.0)
    elif mapping == "f1_points":
        relevance = np.zeros(m, dtype=np.float64)
        for i in range(m):
            if classified_mask[i]:
                p = int(pos_arr[i])
                relevance[i] = F1_POINTS_MAP.get(p, 0.0)
    elif mapping == "podium_tier":
        relevance = np.zeros(m, dtype=np.float64)
        for i in range(m):
            if classified_mask[i]:
                p = int(pos_arr[i])
                relevance[i] = PODIUM_TIER_MAP.get(p, 0.0)
    else:
        raise ValueError(
            f"Unknown relevance mapping: '{mapping}'. "
            "Supported mappings are ('linear_rank', 'f1_points', 'podium_tier')."
        )

    return relevance


def compute_dcg_at_k(relevance: np.ndarray, k: int) -> float:
    r"""Compute Discounted Cumulative Gain up to rank threshold K (DCG@K).

    Mathematical Formulation:
    -------------------------
    $$\text{DCG}@K(\boldsymbol{y}) = \sum_{r=1}^{\min(K, m)} \frac{2^{y_{(r)}} - 1}{\log_2(r + 1)}$$

    where:
    - $y_{(r)}$ is the relevance of the candidate placed at rank $r$.
    - $r \in \{1, 2, \dots, \min(K, m)\}$ is the 1-based rank position.
    - $m = \text{len}(relevance)$ is the number of candidates in the query.
    - If $k \le 0$ or $m = 0$, the function returns $0.0$.

    Parameters
    ----------
    relevance : np.ndarray
        1D array of relevance values for items already sorted by predicted rank
        (item at index 0 is predicted rank 1, index 1 is predicted rank 2, etc.).
    k : int
        Rank cutoff threshold (e.g., 1, 3, 10). Clamped at array length $m$.

    Returns
    -------
    float
        Discounted cumulative gain value.
    """
    rel = np.asarray(relevance, dtype=np.float64).ravel()
    m = len(rel)
    if k <= 0 or m == 0:
        return 0.0

    cutoff = min(k, m)
    rel_k = rel[:cutoff]

    # 1-based ranks: 1, 2, ..., cutoff
    ranks = np.arange(1, cutoff + 1, dtype=np.float64)

    # Gains: 2^{y_{(r)}} - 1
    gains = np.power(2.0, rel_k) - 1.0

    # Discounts: log2(r + 1)
    discounts = np.log2(ranks + 1.0)

    return float(np.sum(gains / discounts))


def compute_ndcg_at_k(
    y_true_relevance: np.ndarray,
    y_pred_scores: np.ndarray,
    k: int = 3,
) -> float:
    r"""Compute Normalized Discounted Cumulative Gain at rank threshold K (NDCG@K).

    Mathematical Formulation:
    -------------------------
    $$\text{NDCG}@K = \frac{\text{DCG}@K(\boldsymbol{y})}{\text{IDCG}@K(\boldsymbol{y}^*)}$$

    where:
    - $\boldsymbol{y}$ is the true relevance vector permuted according to descending
      predicted scores $\hat{\boldsymbol{s}}$.
    - $\boldsymbol{y}^*$ is the ideal relevance vector sorted in descending order.
    - $\text{IDCG}@K = \text{DCG}@K(\boldsymbol{y}^*)$ is the Ideal Discounted Cumulative Gain.
    - If $\text{IDCG}@K = 0$ (all items have zero relevance), returns $0.0$.
    - $K$ is clamped at the total number of items $m$.

    Parameters
    ----------
    y_true_relevance : np.ndarray
        Ground truth relevance scores for all candidates in the race query.
    y_pred_scores : np.ndarray
        Model predicted scores (higher score indicates higher predicted rank / faster pace).
    k : int, default=3
        Rank cutoff threshold.

    Returns
    -------
    float
        NDCG@K score in range [0.0, 1.0].

    Raises
    ------
    ValueError
        If y_true_relevance and y_pred_scores have different lengths.
    """
    y_true = np.asarray(y_true_relevance, dtype=np.float64).ravel()
    scores = np.asarray(y_pred_scores, dtype=np.float64).ravel()

    if len(y_true) != len(scores):
        raise ValueError(
            f"Shape mismatch: y_true_relevance has length {len(y_true)}, "
            f"but y_pred_scores has length {len(scores)}."
        )

    m = len(y_true)
    if k <= 0 or m == 0:
        return 0.0

    # Permute items by predicted scores descending (stable sort preserves order on ties)
    ranked_order = np.argsort(-scores, kind="mergesort")
    ranked_relevance = y_true[ranked_order]
    dcg = compute_dcg_at_k(ranked_relevance, k=k)

    # Ideal DCG: best possible permutation is sorting true relevance descending
    ideal_relevance = np.sort(y_true)[::-1]
    idcg = compute_dcg_at_k(ideal_relevance, k=k)

    if idcg <= 0.0:
        return 0.0

    ndcg = dcg / idcg
    return float(np.clip(ndcg, 0.0, 1.0))


def compute_mrr(y_true_winner: np.ndarray, y_pred_scores: np.ndarray) -> float:
    r"""Compute Reciprocal Rank of the race winner.

    Mathematical Formulation:
    -------------------------
    $$\text{RR} = \frac{1}{\text{rank}(\text{winner})}$$

    where $\text{rank}(\text{winner}) \in \{1, 2, \dots, m\}$ is the 1-based predicted
    rank of the true race winner (the item where $y_{\text{true\_winner}} = 1$).
    If no driver is labeled as the winner, or if the input is empty, returns $0.0$.

    Parameters
    ----------
    y_true_winner : np.ndarray
        Binary indicator array where 1 identifies the race winner and 0 identifies others.
    y_pred_scores : np.ndarray
        Model predicted continuous scores (higher score indicates higher predicted rank).

    Returns
    -------
    float
        Reciprocal rank value in range [0.0, 1.0].

    Raises
    ------
    ValueError
        If y_true_winner and y_pred_scores have different lengths.
    """
    winner_mask = np.asarray(y_true_winner).ravel() == 1
    scores = np.asarray(y_pred_scores, dtype=np.float64).ravel()

    if len(winner_mask) != len(scores):
        raise ValueError(
            f"Shape mismatch: y_true_winner has length {len(winner_mask)}, "
            f"but y_pred_scores has length {len(scores)}."
        )

    if len(winner_mask) == 0 or not np.any(winner_mask):
        return 0.0

    # Stable descending sort by predicted score
    ranked_order = np.argsort(-scores, kind="mergesort")
    sorted_winner_mask = winner_mask[ranked_order]

    winner_positions = np.flatnonzero(sorted_winner_mask)
    if len(winner_positions) == 0:
        return 0.0

    rank = int(winner_positions[0]) + 1  # 1-based rank
    return float(1.0 / rank)


def compute_kendall_tau(y_true_ranks: np.ndarray, y_pred_scores: np.ndarray) -> float:
    r"""Compute Kendall's Tau pairwise rank correlation between true and predicted race order.

    Mathematical Formulation:
    -------------------------
    $$\tau = \frac{C - D}{\frac{1}{2} m (m - 1)}$$

    where:
    - $C$ is the count of concordant pairs (both orderings agree on who finished ahead).
    - $D$ is the count of discordant pairs (orderings disagree).
    - $m = \text{len}(y\_true\_ranks)$ is the field size.
    - In $y_{\text{true\_ranks}}$, smaller numbers represent better finishes (P1=1, P2=2).
    - In $y_{\text{pred\_scores}}$, higher scores represent better finishes.
    - Returns $1.0$ if length < 2 (trivially concordant single driver / empty race).
    - Returns $0.0$ if all predicted scores are tied (no pairwise ranking signal).

    Parameters
    ----------
    y_true_ranks : np.ndarray
        Ground truth finishing position ranks (e.g. positionOrder: 1, 2, ..., m).
    y_pred_scores : np.ndarray
        Model predicted scores (higher score indicates higher predicted finishing rank).

    Returns
    -------
    float
        Kendall's Tau correlation coefficient in range [-1.0, 1.0].

    Raises
    ------
    ValueError
        If y_true_ranks and y_pred_scores have different lengths.
    """
    true_ranks = np.asarray(y_true_ranks, dtype=np.float64).ravel()
    scores = np.asarray(y_pred_scores, dtype=np.float64).ravel()

    if len(true_ranks) != len(scores):
        raise ValueError(
            f"Shape mismatch: y_true_ranks has length {len(true_ranks)}, "
            f"but y_pred_scores has length {len(scores)}."
        )

    if len(true_ranks) < 2:
        return 1.0

    # Negate scores so smaller values correspond to better finishes (matching true_ranks)
    res = scipy.stats.kendalltau(true_ranks, -scores)
    tau = res.statistic if hasattr(res, "statistic") else res[0]

    if np.isnan(tau):
        # Tied scores yield NaN in scipy due to 0 variance in predicted ranks
        return 0.0

    return float(np.clip(tau, -1.0, 1.0))


def evaluate_race_ranking(
    df_test: pd.DataFrame,
    score_col: str,
    target_col: str = "positionOrder",
    k_values: tuple[int, ...] = (1, 3, 10),
    relevance_mapping: str = "linear_rank",
) -> dict[str, float]:
    r"""Evaluate race-level ranking metrics averaged across all races in a DataFrame.

    Groups the dataset by raceId, computes NDCG@K for each specified $K$, Mean Reciprocal
    Rank (MRR) of the winner, Kendall's Tau correlation, and Top-3 Podium Precision,
    Recall, and F1 per race, then averages across all races.

    Top-3 Podium Selection:
    -----------------------
    - Predicted Top 3: The 3 drivers with highest `score_col` per race.
      Deterministic tie-breaking uses `grid_effective` ascending (closer to pole wins ties)
      if available, or stable index sort otherwise.
    - Actual Top 3: Drivers with `target_col` <= 3.

    Podium Metrics:
    $$\text{Precision}@3 = \frac{|\hat{P} \cap P^*|}{|\hat{P}|}, \quad \text{Recall}@3 = \frac{|\hat{P} \cap P^*|}{|P^*|}, \quad F_1@3 = \frac{2 \cdot P \cdot R}{P + R}$$

    Parameters
    ----------
    df_test : pd.DataFrame
        DataFrame containing evaluation races. Must contain a race identifier ('raceId' or 'race_id'),
        the prediction score column (`score_col`), and the target rank column (`target_col`).
    score_col : str
        Column name for model prediction scores (higher score indicates higher predicted rank).
    target_col : str, default='positionOrder'
        Column name for true finishing positions (1 = winner, 2 = second, etc.).
    k_values : tuple[int, ...], default=(1, 3, 10)
        Tuple of cut-off thresholds for NDCG@K evaluation.
    relevance_mapping : str, default='linear_rank'
        Strategy for translating finishing positions to relevance targets:
        'linear_rank', 'f1_points', or 'podium_tier'.

    Returns
    -------
    dict[str, float]
        Case-insensitive dictionary containing mean metrics across all races:
        - 'ndcg@1', 'ndcg@3', 'ndcg@10' (or other k in k_values)
        - 'mrr'
        - 'kendall_tau'
        - 'podium_precision'
        - 'podium_recall'
        - 'podium_f1'

    Raises
    ------
    KeyError
        If required columns are missing from df_test.
    ValueError
        If relevance_mapping is unrecognized.
    """
    if "raceId" in df_test.columns:
        race_col = "raceId"
    elif "race_id" in df_test.columns:
        race_col = "race_id"
    else:
        raise KeyError("df_test must contain 'raceId' or 'race_id' column.")

    if score_col not in df_test.columns:
        raise KeyError(f"df_test is missing score column: '{score_col}'")
    if target_col not in df_test.columns:
        raise KeyError(f"df_test is missing target column: '{target_col}'")

    if df_test.empty:
        empty_res: dict[str, float] = {f"ndcg@{k}": 0.0 for k in k_values}
        empty_res.update(
            {
                "mrr": 0.0,
                "kendall_tau": 0.0,
                "podium_precision": 0.0,
                "podium_recall": 0.0,
                "podium_f1": 0.0,
            }
        )
        return CaseInsensitiveDict(empty_res)

    has_grid = "grid_effective" in df_test.columns
    has_classified = "is_classified" in df_test.columns

    race_ndcg: dict[int, list[float]] = {k: [] for k in k_values}
    race_mrr: list[float] = []
    race_tau: list[float] = []
    race_prec: list[float] = []
    race_rec: list[float] = []
    race_f1: list[float] = []

    # Group by race while preserving natural race grouping
    for _, df_race in df_test.groupby(race_col, sort=False):
        m = len(df_race)
        if m == 0:
            continue

        positions = df_race[target_col].to_numpy(dtype=np.float64)
        scores = df_race[score_col].to_numpy(dtype=np.float64)
        classified = (
            df_race["is_classified"].to_numpy() if has_classified else None
        )

        # 1. Relevance Mapping & NDCG@K
        y_true_relevance = map_finish_positions_to_relevance(
            positions, is_classified=classified, mapping=relevance_mapping
        )
        for k in k_values:
            ndcg_k = compute_ndcg_at_k(y_true_relevance, scores, k=k)
            race_ndcg[k].append(ndcg_k)

        # 2. MRR (True winner is position == 1)
        y_true_winner = (positions == 1.0).astype(int)
        mrr = compute_mrr(y_true_winner, scores)
        race_mrr.append(mrr)

        # 3. Kendall's Tau
        tau = compute_kendall_tau(positions, scores)
        race_tau.append(tau)

        # 4. Top-3 Podium Selection
        # If grid_effective available, break ties by starting grid (closer to pole wins)
        if has_grid:
            grid_eff = df_race["grid_effective"].to_numpy(dtype=np.float64)
            # Secondary key first in lexsort tuple:
            order = np.lexsort((grid_eff, -scores))
        else:
            order = np.argsort(-scores, kind="mergesort")

        top_k_podium = min(3, m)
        pred_podium_indices = set(order[:top_k_podium])
        actual_podium_indices = set(np.flatnonzero(positions <= 3.0))

        hits = len(pred_podium_indices & actual_podium_indices)
        prec = hits / len(pred_podium_indices) if len(pred_podium_indices) > 0 else 0.0
        rec = (
            hits / len(actual_podium_indices)
            if len(actual_podium_indices) > 0
            else 0.0
        )
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

        race_prec.append(prec)
        race_rec.append(rec)
        race_f1.append(f1)

    # Average metrics across all evaluated races
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = (
            float(np.mean(race_ndcg[k])) if len(race_ndcg[k]) > 0 else 0.0
        )

    metrics["mrr"] = float(np.mean(race_mrr)) if len(race_mrr) > 0 else 0.0
    metrics["kendall_tau"] = float(np.mean(race_tau)) if len(race_tau) > 0 else 0.0
    metrics["podium_precision"] = (
        float(np.mean(race_prec)) if len(race_prec) > 0 else 0.0
    )
    metrics["podium_recall"] = (
        float(np.mean(race_rec)) if len(race_rec) > 0 else 0.0
    )
    metrics["podium_f1"] = float(np.mean(race_f1)) if len(race_f1) > 0 else 0.0

    return CaseInsensitiveDict(metrics)
