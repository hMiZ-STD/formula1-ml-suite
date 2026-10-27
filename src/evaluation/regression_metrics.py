"""Regression Evaluation Metrics and Robustness Assessment Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
When predicting finishing position in Formula 1:
1. Mean Squared Error (MSE):
   Measures the average squared difference between predicted position and actual finish.
   If you predict 4th and they finish 6th, the error is 2, and 2 squared is 4.
2. Root Mean Squared Error (RMSE):
   Takes the square root of MSE so the units are back in regular race positions!
   "Our model is off by an average of 2.8 finishing spots."
3. R-squared (R²):
   Compares our model to a dumb baseline that simply guesses the average finish position (around 10th or 11th).
   An R² of 0.40 means our model explains 40% of the variance in race outcomes.
4. Classified-Only R² (Robustness Check):
   In Formula 1, cars crash or break down unexpectedly (DNFs). Ergast automatically ranks
   broken cars at the very back of the pack. To prove our model didn't just learn who broke down,
   we recompute R² strictly on cars that survived and crossed the finish line!
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_classified: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MSE, RMSE, R² on all starters and classified-only subpopulation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth finishing positions (positionOrder).
    y_pred : np.ndarray
        Model predicted finishing positions.
    is_classified : np.ndarray or None, default=None
        Binary mask (1 for classified finishers, 0 for DNF).

    Returns
    -------
    dict[str, float]
        Dictionary of mse, rmse, r2, and classified_only_r2.
    """
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    classified_r2 = float("nan")
    if is_classified is not None:
        mask = np.asarray(is_classified).astype(bool)
        if mask.sum() > 1:
            classified_r2 = float(r2_score(y_true[mask], y_pred[mask]))

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "classified_only_r2": classified_r2,
    }
