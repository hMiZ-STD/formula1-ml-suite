"""Data Partitioning, Temporal Cross-Validation, and Parquet Management Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Imagine you are training a chess player to predict future tournament games:
1. You CANNOT test them by giving them the 2024 World Championship games, then training
   them on the 2010 games, and then testing them on 2015! That would let them travel through
   time and peek at the future.
2. In Formula 1, technology and rules change every single season (hybrid engines, aerodynamics).
   To prove our AI actually understands racing rather than just memorizing dates, we train on
   older seasons (1994–2016), tune hyperparameters and early stopping on middle seasons (2017–2018),
   and lock the model in a vault! Then we evaluate it ONCE on the modern hybrid era (2019–2024).
3. In cross-validation, an entire race must stay together in either training or validation.
   You cannot have Hamilton's result in training while his teammate Russell's result in the exact
   same race is in validation — that would leak weather and track conditions!

This module provides race-aligned cross-validation fold generators, season-aligned data splitters,
and Parquet caching utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROCESSED_DATA_DIR = Path("data/processed")


def race_time_folds(
    df: pd.DataFrame,
    n_val_seasons: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate race-aligned, temporally strictly expanding cross-validation folds.

    ELI5 Explanation:
    We group races by year so no single race is cut in half.
    Fold 1: Train on years up to 2008, validate on 2009.
    Fold 2: Train on years up to 2009, validate on 2010.
    ...
    Fold 8: Train on years up to 2015, validate on 2016.
    Notice that the past always predicts the future, never the other way around!
    We use positional indices (np.flatnonzero) so Scikit-Learn never gets confused by index labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'year' column. Must have a clean or reset index.
    n_val_seasons : int, default=8
        Number of most-recent chronological seasons in the training set used as validation folds.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) positional integer arrays for Scikit-Learn GridSearchCV.
    """
    # Guarantee positional alignment regardless of original index state
    df_reset = df.reset_index(drop=True)
    years = df_reset["year"].to_numpy()
    seasons = np.unique(years)

    val_seasons = seasons[-n_val_seasons:]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for val_year in val_seasons:
        train_idx = np.flatnonzero(years < val_year)
        val_idx = np.flatnonzero(years == val_year)

        if train_idx.size > 0 and val_idx.size > 0:
            folds.append((train_idx, val_idx))

    return folds


def partition_temporal_sets(
    df_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into season-aligned train_core, val, test, and train_val_pool.

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame with full pre-race features and 'year' column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple of (train_core, val, test, train_val_pool) DataFrames.
    """
    train_core = df_features[df_features["year"] <= 2016].copy().reset_index(drop=True)
    val = df_features[
        (df_features["year"] >= 2017) & (df_features["year"] <= 2018)
    ].copy().reset_index(drop=True)
    test = df_features[df_features["year"] >= 2019].copy().reset_index(drop=True)
    train_val_pool = df_features[df_features["year"] <= 2018].copy().reset_index(drop=True)

    return train_core, val, test, train_val_pool


def save_processed_parquets(
    train_core: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    career_clustering: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save processed feature sets to Parquet format.

    Parameters
    ----------
    train_core : pd.DataFrame
        Core training set (1994–2016).
    val : pd.DataFrame
        Validation set (2017–2018).
    test : pd.DataFrame
        Test set (2019–2024).
    career_clustering : pd.DataFrame
        Career-level aggregated dataset (1950–2024).
    output_dir : Path, default=Path("data/processed")
        Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_core.to_parquet(output_dir / "train_core.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    career_clustering.to_parquet(output_dir / "career_clustering.parquet", index=False)


def write_experiment_freeze(
    filepath: Path,
    hyperparameters: dict[str, Any],
    thresholds: dict[str, Any],
    random_seeds: dict[str, int],
    cnn_stopped_epoch: int | None = None,
    allowed_features: list[str] | None = None,
) -> None:
    """Write EXPERIMENT_FREEZE.yaml prior to evaluating any model on the test set.

    ELI5 Explanation:
    Once an experiment is frozen, nobody is allowed to change the hyperparameters,
    tune thresholds, or fiddle with features! It is like sealing the exam papers in an
    envelope before handing them to the examiner.

    Parameters
    ----------
    filepath : Path
        Destination path (typically EXPERIMENT_FREEZE.yaml in project root).
    hyperparameters : dict[str, Any]
        Dictionary of tuned hyperparameters from cross-validation.
    thresholds : dict[str, Any]
        Decision thresholds tuned on the validation set.
    random_seeds : dict[str, int]
        Dictionary of random seeds used across numpy, torch, and sklearn.
    cnn_stopped_epoch : int or None, default=None
        Epoch at which CNN early stopping triggered.
    allowed_features : list[str] or None, default=None
        List of authorized feature names.
    """
    import hashlib
    from datetime import datetime

    feat_hash = ""
    if allowed_features:
        feat_hash = hashlib.sha256("".join(sorted(allowed_features)).encode()).hexdigest()

    freeze_payload = {
        "freeze_timestamp": datetime.now().isoformat(),
        "supervised_scope": "1994-2024",
        "train_val_pool_boundary": "last race of 2018",
        "test_boundary": "first race of 2019",
        "feature_set_hash": feat_hash,
        "hyperparameters": hyperparameters,
        "thresholds": thresholds,
        "random_seeds": random_seeds,
        "cnn_stopped_epoch": cnn_stopped_epoch,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(freeze_payload, f, default_flow_style=False, sort_keys=False)
