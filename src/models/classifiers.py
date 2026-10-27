"""Classical Supervised Classification Models Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
1. K-Nearest Neighbors (KNN):
   "Tell me who your neighbors are, and I'll tell you who you are."
   To predict if a driver will reach the podium today, KNN looks back in history to find
   the K most similar race situations (same starting grid, similar qualifying pace,
   similar team strength). If most of those similar situations ended with a trophy,
   KNN predicts a podium!
2. Decision Tree:
   Like playing a game of '20 Questions':
   - "Did you qualify in the top 3?" -> YES -> "Is your team in the top 2 in points?" -> YES -> Podium!
   - We tune max_depth and ccp_alpha (cost-complexity pruning) so the tree doesn't grow so huge
     that it memorizes random historical flukes (overfitting).
3. Support Vector Machine (SVM with RBF kernel):
   Imagine dropping drivers onto a 2D sheet of rubber. If the podium finishers and non-podium
   finishers are mixed together like marble swirls, the RBF kernel bends and stretches the rubber
   sheet into 3D space until a clean sheet of glass (the hyperplane) can separate them cleanly!
   Because SVM scales quadratically with dataset size, we perform hyperparameter grid search
   on a race-aligned sample of the newest ~5,000 training rows, then refit the winner on the
   full dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocessor import race_time_folds
from src.features.builder import ALLOWED_FEATURES


def create_classifier_pipeline(
    model_name: str,
    estimator: Any,
) -> Pipeline:
    """Create an anti-leakage Scikit-Learn Pipeline with Median Imputer and Standard Scaler.

    Parameters
    ----------
    model_name : str
        Step name for the estimator (e.g. 'knn', 'dt', 'svm').
    estimator : Any
        Scikit-learn classification estimator instance.

    Returns
    -------
    Pipeline
        Configured pipeline with imputer, scaler, and estimator.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        (model_name, estimator),
    ])


def tune_knn(
    df_train_core: pd.DataFrame,
    feature_cols: list[str] = ALLOWED_FEATURES,
    n_neighbors_range: range = range(1, 31),
    random_seed: int = 42,
) -> tuple[GridSearchCV, Pipeline]:
    """Tune K-Nearest Neighbors classifier using race-aligned cross-validation.

    Parameters
    ----------
    df_train_core : pd.DataFrame
        Core training DataFrame (1994–2016).
    feature_cols : list[str]
        List of authorized feature columns.
    n_neighbors_range : range, default=range(1, 31)
        Range of K values to explore.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[GridSearchCV, Pipeline]
        Fitted GridSearchCV object and best pipeline.
    """
    df_clean = df_train_core.reset_index(drop=True)
    X = df_clean[feature_cols]
    y = df_clean["y_podium"].to_numpy()

    pipe = create_classifier_pipeline(
        "knn",
        KNeighborsClassifier(weights="distance"),
    )
    cv_folds = race_time_folds(df_clean, n_val_seasons=8)

    param_grid = {"knn__n_neighbors": list(n_neighbors_range)}
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X, y)
    return grid_search, grid_search.best_estimator_


def tune_decision_tree(
    df_train_core: pd.DataFrame,
    feature_cols: list[str] = ALLOWED_FEATURES,
    max_depth_range: range = range(1, 21),
    ccp_alphas: list[float] | None = None,
    random_seed: int = 42,
) -> tuple[GridSearchCV, Pipeline]:
    """Tune Decision Tree classifier using race-aligned cross-validation.

    Parameters
    ----------
    df_train_core : pd.DataFrame
        Core training DataFrame (1994–2016).
    feature_cols : list[str]
        List of authorized feature columns.
    max_depth_range : range, default=range(1, 21)
        Range of maximum tree depths.
    ccp_alphas : list[float] or None, default=None
        Cost-complexity pruning parameter values.
    random_seed : int, default=42
        Random seed for tree initialization.

    Returns
    -------
    tuple[GridSearchCV, Pipeline]
        Fitted GridSearchCV object and best pipeline.
    """
    if ccp_alphas is None:
        ccp_alphas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]

    df_clean = df_train_core.reset_index(drop=True)
    X = df_clean[feature_cols]
    y = df_clean["y_podium"].to_numpy()

    pipe = create_classifier_pipeline(
        "dt",
        DecisionTreeClassifier(class_weight="balanced", random_state=random_seed),
    )
    cv_folds = race_time_folds(df_clean, n_val_seasons=8)

    param_grid = {
        "dt__max_depth": list(max_depth_range),
        "dt__ccp_alpha": ccp_alphas,
    }
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X, y)
    return grid_search, grid_search.best_estimator_


def tune_svm(
    df_train_core: pd.DataFrame,
    feature_cols: list[str] = ALLOWED_FEATURES,
    target_sample_size: int = 5000,
    random_seed: int = 42,
) -> tuple[GridSearchCV, Pipeline]:
    """Tune SVM classifier on a race-aligned subsample of newest training races.

    ELI5:
    SVM requires solving a matrix problem that grows with the square of the rows.
    Evaluating 16 parameter pairs across 8 CV folds on 9,000 rows would take huge time.
    Instead, we select the newest ~5,000 rows of training data (moving backward in time
    by complete races so no race is sliced), find the best C and gamma, and then fit
    the winning parameters on the complete dataset.

    Parameters
    ----------
    df_train_core : pd.DataFrame
        Core training DataFrame (1994–2016).
    feature_cols : list[str]
        List of authorized feature columns.
    target_sample_size : int, default=5000
        Approximate number of rows to select from newest races.
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[GridSearchCV, Pipeline]
        Fitted GridSearchCV object and best pipeline.
    """
    df_sorted = df_train_core.sort_values(["year", "round"]).reset_index(drop=True)
    unique_races = df_sorted["raceId"].unique()

    # Accumulate whole races backward from the most recent race
    selected_races: list[int] = []
    accumulated_rows = 0
    for r in reversed(unique_races):
        selected_races.append(r)
        accumulated_rows += int((df_sorted["raceId"] == r).sum())
        if accumulated_rows >= target_sample_size:
            break

    df_sub = df_sorted[df_sorted["raceId"].isin(selected_races)].sort_values(
        ["year", "round"]
    ).reset_index(drop=True)

    X_sub = df_sub[feature_cols]
    y_sub = df_sub["y_podium"].to_numpy()

    pipe = create_classifier_pipeline(
        "svm",
        SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=random_seed),
    )
    cv_folds = race_time_folds(df_sub, n_val_seasons=5)

    param_grid = {
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__gamma": [0.001, 0.01, 0.1, "scale"],
    }
    grid_search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_sub, y_sub)
    return grid_search, grid_search.best_estimator_
