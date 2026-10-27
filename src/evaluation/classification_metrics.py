"""Classification Evaluation Metrics and Statistical Inference Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
When grading a student's test:
1. Accuracy alone is misleading! If 86% of drivers do NOT finish on the podium,
   a lazy student who simply guesses "Nobody finishes on the podium!" gets 86% accuracy,
   yet understands absolutely zero about Formula 1!
2. Precision asks: "When the model predicted a driver would stand on the podium, were they right?"
3. Recall asks: "Out of all drivers who actually stood on the podium, how many did the model catch?"
4. F1-score balances precision and recall so neither can cheat.
5. In Formula 1, drivers in the same race share the same rain, safety cars, and crashes!
   If we resample random individual rows during bootstrap confidence intervals, we break that reality.
   Therefore, we use Race-Aligned Block Bootstrap: we resample entire races together so all 20 drivers
   stay in their true race environment!
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    is_decision_score: bool = False,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """Find decision threshold on validation set that maximizes F1 score.

    ELI5:
    Instead of assuming 0.5 is the best cutoff for saying "Yes, they will reach the podium!",
    we test many cutoffs on past validation data to find the sweet spot that catches the most
    podiums while avoiding false alarms.

    Parameters
    ----------
    y_true : np.ndarray
        True binary target labels (0 or 1).
    scores : np.ndarray
        Continuous prediction scores (probabilities or SVM decision_function output).
    is_decision_score : bool, default=False
        If True, sweep quantiles of unbounded decision scores (for SVM).
        If False, sweep linearly spaced thresholds between 0.01 and 0.99.
    n_thresholds : int, default=200
        Number of candidate thresholds to evaluate.

    Returns
    -------
    tuple[float, float]
        Tuple of (best_threshold, best_f1_score).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    if is_decision_score:
        thresholds = np.quantile(scores, np.linspace(0.01, 0.99, n_thresholds))
    else:
        thresholds = np.linspace(0.01, 0.99, n_thresholds)

    best_thresh = float(thresholds[0])
    best_f1 = -1.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(t)

    return best_thresh, best_f1


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute standard classification evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).
    scores : np.ndarray
        Continuous prediction scores.

    Returns
    -------
    dict[str, float]
        Dictionary of accuracy, precision, recall, f1, roc_auc, and pr_auc.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = float(np.mean(y_true))

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def bootstrap_auc_ci(
    df: pd.DataFrame,
    y_true_col: str,
    score_col: str,
    race_col: str = "raceId",
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate 95% Confidence Interval for ROC-AUC using Race-Aligned Block Bootstrap.

    ELI5:
    We pretend to run the racing championship 1,000 times by randomly sampling races
    with replacement. For each simulated championship, we calculate the ROC-AUC.
    The middle 95% of those scores gives us our reliable confidence interval!

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing race identifiers, true targets, and model prediction scores.
    y_true_col : str
        Column name of binary ground truth label.
    score_col : str
        Column name of prediction score.
    race_col : str, default='raceId'
        Column name identifying the race group.
    n_boot : int, default=1000
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level (0.05 gives 95% CI).
    random_seed : int, default=42
        Seed for pseudo-random number generator.

    Returns
    -------
    tuple[float, float, float]
        Tuple of (mean_auc, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(random_seed)
    unique_races = np.unique(df[race_col].to_numpy())
    n_races = len(unique_races)

    # Group index positions by raceId for high-speed sampling
    race_to_indices: dict[Any, np.ndarray] = {
        r: np.flatnonzero(df[race_col].to_numpy() == r) for r in unique_races
    }

    y_all = df[y_true_col].to_numpy().astype(int)
    scores_all = df[score_col].to_numpy()

    boot_aucs: list[float] = []
    max_attempts = n_boot * 3
    attempts = 0

    while len(boot_aucs) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled_races = rng.choice(unique_races, size=n_races, replace=True)
        sample_idx = np.concatenate([race_to_indices[r] for r in sampled_races])

        y_sample = y_all[sample_idx]
        if len(np.unique(y_sample)) < 2:
            # Degenerate sample with only 1 class; redraw
            continue

        scores_sample = scores_all[sample_idx]
        auc = roc_auc_score(y_sample, scores_sample)
        boot_aucs.append(float(auc))

    boot_arr = np.array(boot_aucs)
    ci_lower = float(np.percentile(boot_arr, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_arr, 100.0 * (1.0 - alpha / 2.0)))
    mean_auc = float(np.mean(boot_arr))

    return mean_auc, ci_lower, ci_upper
