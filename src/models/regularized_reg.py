"""Regularized Linear Regression (Ridge and Lasso) and Feature Sparsity Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
When we expand our features into degree-2 polynomials (multiplying features by each other,
like grid_position * team_points), we create dozens of new features!
Some of these interaction features are helpful, but many are redundant or noisy.

1. Ordinary Least Squares (OLS):
   Tries to fit all features equally without constraint. When features are correlated,
   OLS coefficients can blow up to huge numbers (+1,000 and -1,000) that cancel each other out!
2. Ridge Regression (L2 Regularization):
   Adds a penalty proportional to the sum of squared weights: alpha * sum(w_j^2).
   It acts like an elastic rubber band pulling all coefficients toward zero, shrinking
   them smoothly to prevent any single feature from dominating.
3. Lasso Regression (L1 Regularization):
   Adds a penalty proportional to the absolute value: alpha * sum(|w_j|).
   Lasso acts like a pair of sharp pruning shears! It doesn't just shrink useless coefficients—it
   forces them to be EXACTLY zero, performing automatic feature selection and revealing the
   leanest, most important predictive signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.data.preprocessor import race_time_folds

CONTINUOUS_FEATURES: list[str] = [
    "grid_effective",
    "driver_norm_finish_p5",
    "driver_dnf_rate_p5",
    "q_gap_pct",
    "teammate_form_delta",
    "constructor_pts_share",
    "driver_age_years",
    "quali_position_norm",
    "rounds_completed",
]

BINARY_FLAGS: list[str] = [
    "grid_zero_flag",
    "q1_data_available",
    "is_rookie_window",
    "new_constructor_flag",
]


def build_poly2_transformer() -> ColumnTransformer:
    """Create a ColumnTransformer applying degree-2 expansion to continuous features only."""
    return ColumnTransformer(
        transformers=[
            (
                "poly",
                PolynomialFeatures(degree=2, include_bias=False),
                list(range(len(CONTINUOUS_FEATURES))),
            ),
            (
                "pass_binary",
                "passthrough",
                list(range(len(CONTINUOUS_FEATURES), len(CONTINUOUS_FEATURES) + len(BINARY_FLAGS))),
            ),
        ]
    )


def create_regularized_pipeline(model: Any) -> Pipeline:
    """Create complete Pipeline: Imputer -> Scaler -> Degree-2 Expansion -> Estimator."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("poly2", build_poly2_transformer()),
        ("regressor", model),
    ])


def tune_ridge_and_lasso(
    df_train: pd.DataFrame,
    alphas: np.ndarray | None = None,
    random_seed: int = 42,
) -> tuple[GridSearchCV, GridSearchCV, Pipeline, pd.DataFrame]:
    """Tune Ridge and Lasso models on degree-2 expanded features and produce Table 4.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training set DataFrame (e.g. train_core or train_val_pool).
    alphas : np.ndarray or None, default=None
        Array of candidate alpha penalty parameters.
    random_seed : int, default=42
        Random seed for Lasso coordinate descent.

    Returns
    -------
    tuple
        (ridge_grid, lasso_grid, ols_pipeline, table4_df)
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 30)

    feature_cols = CONTINUOUS_FEATURES + BINARY_FLAGS
    df_clean = df_train.reset_index(drop=True)
    X = df_clean[feature_cols]
    y = df_clean["y_position"].to_numpy()

    cv_folds = race_time_folds(df_clean, n_val_seasons=8)

    # 1. Fit OLS Pipeline
    ols_pipe = create_regularized_pipeline(LinearRegression())
    ols_pipe.fit(X, y)

    # 2. Tune Ridge
    ridge_pipe = create_regularized_pipeline(Ridge(random_state=random_seed))
    ridge_grid = GridSearchCV(
        ridge_pipe,
        {"regressor__alpha": alphas},
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    ridge_grid.fit(X, y)

    # 3. Tune Lasso
    lasso_pipe = create_regularized_pipeline(
        Lasso(max_iter=10_000, random_state=random_seed, tol=1e-4)
    )
    lasso_grid = GridSearchCV(
        lasso_pipe,
        {"regressor__alpha": alphas},
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    lasso_grid.fit(X, y)

    # Extract feature names after degree-2 expansion
    poly_feature_names = (
        ols_pipe.named_steps["poly2"]
        .named_transformers_["poly"]
        .get_feature_names_out(CONTINUOUS_FEATURES)
        .tolist()
    )
    all_expanded_names = poly_feature_names + BINARY_FLAGS

    ols_coefs = ols_pipe.named_steps["regressor"].coef_
    ridge_coefs = ridge_grid.best_estimator_.named_steps["regressor"].coef_
    lasso_coefs = lasso_grid.best_estimator_.named_steps["regressor"].coef_

    table4_rows: list[dict[str, Any]] = []
    for name, c_ols, c_ridge, c_lasso in zip(
        all_expanded_names, ols_coefs, ridge_coefs, lasso_coefs
    ):
        table4_rows.append({
            "Feature": name,
            "OLS Coeff": round(float(c_ols), 4),
            "Ridge Coeff": round(float(c_ridge), 4),
            "Lasso Coeff": round(float(c_lasso), 4),
            "Lasso Zeroed?": "YES" if abs(c_lasso) < 1e-5 else "NO",
        })

    table4_df = pd.DataFrame(table4_rows)
    return ridge_grid, lasso_grid, ols_pipe, table4_df
