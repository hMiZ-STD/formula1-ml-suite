"""Grid Encoding Verification Tests.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.loader import load_clean_starters


@pytest.fixture(scope="module")
def starters():
    return load_clean_starters()


def test_grid_effective_strictly_positive(starters):
    """Verify effective grid is strictly >= 1 for all starters (0 is eliminated)."""
    assert (starters["grid_effective"] >= 1.0).all(), (
        "Found grid_effective < 1! Pit lane starts were not re-indexed."
    )


def test_grid_zero_flag_definition(starters):
    """Verify grid_zero_flag is 1 exactly when original grid was 0."""
    expected_flags = (starters["grid"] == 0).astype(int)
    assert (starters["grid_zero_flag"] == expected_flags).all()


def test_pit_lane_start_placed_at_back(starters):
    """Verify drivers with original grid == 0 are placed behind the normal grid."""
    zero_rows = starters[starters["grid"] == 0]
    for _, row in zero_rows.head(20).iterrows():
        season = row["year"]
        season_normal_max = starters[(starters["year"] == season) & (starters["grid"] > 0)]["grid"].max()
        assert row["grid_effective"] > season_normal_max, (
            f"Zero-grid row not placed behind normal grid: effective={row['grid_effective']}, max={season_normal_max}"
        )
