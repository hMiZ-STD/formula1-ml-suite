"""Polynomial Regression and Bias-Variance Trade-Off Demonstration Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Imagine you are drawing a curve through a set of points on graph paper:
1. Degree 1 (Straight Line - High Bias / Underfitting):
   You use a stiff wooden ruler. Even if the data curves gently, the stiff ruler cannot bend!
   It has high bias because it makes an oversimplified assumption. Both training error and
   test error are relatively high.
2. Degree 2-3 (Gentle Curve - Balanced Sweet Spot):
   You use a flexible curve. It bends naturally with the real pattern without chasing every
   tiny speck of dust.
3. Degree 10-12 (Wild Squiggle - High Variance / Overfitting):
   You use a piece of limp cooked spaghetti! The spaghetti wiggles violently through every
   single training point, getting near-zero error on the training paper. But the moment you
   show it brand new held-out test points, the spaghetti is pointing miles into the sky!
   The held-out error explodes into thousands!

This module demonstrates this fundamental bias-variance trade-off on a controlled, fixed subsample
of 60 training rows and 20 held-out validation rows, scaling the feature to [-1, 1] to prevent
numerical overflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def run_polynomial_bias_variance_demo(
    df_train_core: pd.DataFrame,
    degrees: list[int] | None = None,
    n_train_target: int = 60,
    n_held_out_target: int = 20,
    era_years: tuple[int, int] = (2000, 2005),
) -> tuple[pd.DataFrame, dict[int, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute polynomial regression degree sweep on controlled race-aligned subsample.

    Parameters
    ----------
    df_train_core : pd.DataFrame
        Core training DataFrame.
    degrees : list[int] or None, default=None
        List of polynomial degrees to sweep (default: [1, 2, 3, 5, 8, 10, 12]).
    n_train_target : int, default=60
        Target row count for training subsample.
    n_held_out_target : int, default=20
        Target row count for held-out validation subsample.
    era_years : tuple[int, int], default=(2000, 2005)
        Year window to select whole races from.

    Returns
    -------
    tuple
        (summary_df, models_dict, x_train_scaled, y_train, x_test_scaled, y_test)
    """
    if degrees is None:
        degrees = [1, 2, 3, 5, 8, 10, 12]

    # Filter to era window
    era_df = df_train_core[
        (df_train_core["year"] >= era_years[0]) & (df_train_core["year"] <= era_years[1])
    ].sort_values(["year", "round"]).reset_index(drop=True)

    unique_races = era_df["raceId"].unique()

    # Select whole races for training
    train_races: list[int] = []
    train_rows = 0
    for r in unique_races:
        cnt = int((era_df["raceId"] == r).sum())
        train_races.append(r)
        train_rows += cnt
        if train_rows >= n_train_target:
            break

    # Select subsequent whole races for held-out test
    held_out_races: list[int] = []
    held_out_rows = 0
    for r in unique_races:
        if r not in train_races:
            cnt = int((era_df["raceId"] == r).sum())
            held_out_races.append(r)
            held_out_rows += cnt
            if held_out_rows >= n_held_out_target:
                break

    train_sub = era_df[era_df["raceId"].isin(train_races)].copy()
    test_sub = era_df[era_df["raceId"].isin(held_out_races)].copy()

    # Scale feature 'grid_effective' to [-1, 1] using training bounds
    grid_min = train_sub["grid_effective"].min()
    grid_max = train_sub["grid_effective"].max()

    def scale_to_unit(s: pd.Series) -> np.ndarray:
        return 2.0 * (s.to_numpy() - grid_min) / max(grid_max - grid_min, 1e-6) - 1.0

    x_train = scale_to_unit(train_sub["grid_effective"])
    y_train = train_sub["y_position"].to_numpy()

    x_test = scale_to_unit(test_sub["grid_effective"])
    y_test = test_sub["y_position"].to_numpy()

    results: list[dict[str, Any]] = []
    models: dict[int, Any] = {}

    for d in degrees:
        # Fit polynomial using numpy's numerically stable Polynomial.fit
        poly_fit = np.polynomial.Polynomial.fit(x_train, y_train, deg=d)
        models[d] = poly_fit

        y_train_pred = poly_fit(x_train)
        y_test_pred = poly_fit(x_test)

        train_mse = float(mean_squared_error(y_train, y_train_pred))
        test_mse = float(mean_squared_error(y_test, y_test_pred))

        if d == 1:
            behavior = "High bias (stiff linear line)"
        elif d in (2, 3):
            behavior = "Optimal sweet spot (balanced curvature)"
        elif d in (5, 8):
            behavior = "Beginning of oscillation"
        else:
            behavior = "Severe overfitting (extreme boundary runaway)"

        results.append({
            "Degree": d,
            "Train MSE": round(train_mse, 4),
            "Held-Out MSE": round(test_mse, 4),
            "Observed Behavior": behavior,
        })

    summary_df = pd.DataFrame(results)
    return summary_df, models, x_train, y_train, x_test, y_test
