"""Race-Aligned Cross-Validation Fold Tests.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import race_time_folds


def test_race_time_folds_properties():
    """Verify race_time_folds returns disjoint positional indices with strict chronological order."""
    # Synthetic dataset spanning 5 seasons with multiple races per season
    data = []
    for year in range(2010, 2015):
        for race in range(1, 10):
            for driver in range(1, 21):
                data.append({"year": year, "raceId": year * 100 + race, "driverId": driver})
    df = pd.DataFrame(data)

    folds = race_time_folds(df, n_val_seasons=3)
    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"

    n_rows = len(df)
    for i, (train_idx, val_idx) in enumerate(folds):
        # 1. Bounds check
        assert train_idx.min() >= 0 and train_idx.max() < n_rows
        assert val_idx.min() >= 0 and val_idx.max() < n_rows

        # 2. Disjoint check
        overlap = set(train_idx).intersection(set(val_idx))
        assert len(overlap) == 0, f"Fold {i} has overlapping indices between train and val!"

        # 3. Chronological direction: all training years strictly < validation years
        train_years = df.iloc[train_idx]["year"].to_numpy()
        val_years = df.iloc[val_idx]["year"].to_numpy()
        assert train_years.max() < val_years.min(), (
            f"Fold {i} temporal violation: train max year {train_years.max()} >= val min year {val_years.min()}"
        )

        # 4. Whole-race containment: No raceId straddles train and val
        train_races = set(df.iloc[train_idx]["raceId"].unique())
        val_races = set(df.iloc[val_idx]["raceId"].unique())
        assert len(train_races.intersection(val_races)) == 0, (
            f"Fold {i} race straddle: raceId appears in both train and val!"
        )
