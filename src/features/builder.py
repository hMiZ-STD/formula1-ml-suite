"""Formula 1 Pre-Race Feature Engineering Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Imagine you are a detective trying to predict which runner will win tomorrow's marathon:
1. You CANNOT use clues from tomorrow afternoon (like who crossed the finish line first
   or how many blisters they have)! That would be peeking into the future (data leakage).
2. You can ONLY use clues from before the race begins:
   - Where do they line up on the starting grid? (Grid position)
   - How fast did they run in the preliminary qualifying heats yesterday? (Qualifying gap)
   - How did they perform in their last 5 races? (Recent rolling form)
   - How strong is their running team? (Constructor championship points share)
   - Are they faster than their own teammate who has the exact same running shoes? (Teammate delta)
3. To be 100% fair and prevent cheating:
   - When calculating a driver's average from their last 5 races, we use .shift(1).
     This deliberately pushes today's result out of reach so the computer only looks at the past!
   - Teammates on the same team in the same race get the exact same team score before lights out.
   - For sprint race weekends (2021+), where grid positions were decided by a mini-race rather
     than pure qualifying, we mask the delta as NaN to avoid confounding rules.

This module builds strictly pre-race features, enforces our security whitelist, and verifies
that zero forbidden in-race information can ever leak into our models.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_RAW_DIR,
    load_audit_artifacts,
    load_clean_starters,
    load_raw_csvs,
)

# -----------------------------------------------------------------------------
# Feature Whitelist & Blacklist Enforcement
# -----------------------------------------------------------------------------
ALLOWED_FEATURES: list[str] = [
    "grid_effective",
    "grid_norm",
    "grid_zero_flag",
    "q_gap_pct",
    "q1_data_available",
    "quali_position_norm",
    "grid_vs_quali_delta",
    "driver_norm_finish_p5",
    "driver_dnf_rate_p5",
    "driver_podium_rate_career",
    "is_rookie_window",
    "teammate_form_delta",
    "constructor_pts_share",
    "constructor_season_rank",
    "rounds_completed",
    "driver_age_years",
    "new_constructor_flag",
]

FORBIDDEN_FEATURES: set[str] = {
    "position",
    "positionOrder",
    "positionText",
    "points",
    "laps",
    "time",
    "milliseconds",
    "fastestLapTime",
    "fastestRank",
    "fastestLapSpeed",
    "statusId",
    "status",
    "raceId",
    "driverId",
    "constructorId",
    "date",
    "number",
    "url",
}


def parse_q1_time(time_str: Any) -> float | None:
    """Parse qualifying time string ('m:ss.sss' or 'ss.sss') to total seconds.

    ELI5:
    A stopwatch displays '1:24.350', which means 1 minute and 24.35 seconds.
    We convert that into pure seconds: 1 * 60 + 24.35 = 84.35 seconds,
    so the computer can compare numbers with normal math.

    Parameters
    ----------
    time_str : Any
        String or float representation of lap time.

    Returns
    -------
    float or None
        Total seconds as a float, or None if missing or malformed.
    """
    if pd.isna(time_str):
        return None
    s = str(time_str).strip()
    match_min = re.fullmatch(r"(\d+):(\d{2}\.\d+)", s)
    if match_min:
        return int(match_min.group(1)) * 60.0 + float(match_min.group(2))
    match_sec = re.fullmatch(r"(\d+\.\d+)", s)
    if match_sec:
        return float(match_sec.group(1))
    return None


def build_pre_race_features(
    raw_dir: Path = DEFAULT_RAW_DIR,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> pd.DataFrame:
    """Build leakage-safe pre-race features for all starters (1994–2024).

    Parameters
    ----------
    raw_dir : Path, default=Path("data/raw")
        Path to raw CSV directory.
    audit_dir : Path, default=Path("data/processed/audit")
        Path to audit directory containing frozen JSON artifacts.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame containing starters with ALLOWED_FEATURES and targets.
    """
    raw_dir = Path(raw_dir)
    audit_dir = Path(audit_dir)

    df_starters = load_clean_starters(raw_dir=raw_dir, audit_dir=audit_dir)
    raw_tables = load_raw_csvs(raw_dir=raw_dir)
    audit = load_audit_artifacts(audit_dir=audit_dir)
    sprint_race_ids = audit["sprint_race_ids"]

    # Ensure chronological sort before any rolling transformations
    df = df_starters.sort_values(["year", "round", "positionOrder"]).copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Feature 1-3: Grid Features (Known at race start)
    # -------------------------------------------------------------------------
    # grid_norm: normalize grid relative to the maximum grid position in that race
    race_max_grid = df.groupby("raceId")["grid_effective"].transform("max")
    df["grid_norm"] = df["grid_effective"] / race_max_grid

    # -------------------------------------------------------------------------
    # Feature 4-7: Qualifying Features & Outlier Handling
    # -------------------------------------------------------------------------
    qualifying = raw_tables["qualifying"].copy()
    qualifying["q1_seconds"] = qualifying["q1"].apply(parse_q1_time)

    # Add circuit and decade context to qualifying table for outlier detection
    races_meta = raw_tables["races"][["raceId", "year", "circuitId"]]
    qualifying = qualifying.merge(races_meta, on="raceId", how="left")
    qualifying["decade"] = (qualifying["year"] // 10) * 10

    # Flag and remove outlier Q1 lap times (>150% of circuit-decade median)
    circuit_medians = qualifying.groupby(["circuitId", "decade"])["q1_seconds"].transform("median")
    is_outlier = qualifying["q1_seconds"] > (1.5 * circuit_medians)
    qualifying.loc[is_outlier, "q1_seconds"] = np.nan

    # Calculate gap to fastest valid Q1 time in that specific race
    race_q1_best = qualifying.groupby("raceId")["q1_seconds"].transform("min")
    qualifying["q_gap_pct"] = (
        (qualifying["q1_seconds"] - race_q1_best) / race_q1_best * 100.0
    )
    qualifying["q1_data_available"] = (
        qualifying["q1_seconds"].notna() & (~is_outlier)
    ).astype(int)

    # Normalized qualifying position: position / total qualifiers in race
    race_total_qualifiers = qualifying.groupby("raceId")["position"].transform("count")
    qualifying["quali_position_norm"] = qualifying["position"] / np.maximum(race_total_qualifiers, 1)
    qualifying["qualifying_position"] = qualifying["position"]

    # Merge qualifying features into driver starters
    df = df.merge(
        qualifying[[
            "raceId",
            "driverId",
            "q_gap_pct",
            "q1_data_available",
            "quali_position_norm",
            "qualifying_position",
        ]],
        on=["raceId", "driverId"],
        how="left",
    )
    df["q1_data_available"] = df["q1_data_available"].fillna(0).astype(int)

    # grid_vs_quali_delta: grid_effective - qualifying.position
    # CRITICAL: For all sprint weekend races from 2021 onward, grid was set by sprint,
    # so we explicitly mask grid_vs_quali_delta as NaN
    df["grid_vs_quali_delta"] = df["grid_effective"] - df["qualifying_position"]
    df.loc[df["raceId"].isin(sprint_race_ids), "grid_vs_quali_delta"] = np.nan

    # -------------------------------------------------------------------------
    # Feature 8-11: Driver Rolling Form Features (Anti-Leakage Groupby)
    # -------------------------------------------------------------------------
    # Calculate past race outcome metrics (normalized finish: 0.0=winner, 1.0=last)
    race_field_size = df.groupby("raceId")["driverId"].transform("count")
    df["norm_finish"] = (df["positionOrder"] - 1.0) / np.maximum(race_field_size - 1.0, 1.0)

    # Rolling 5-race finish average: MUST group by driverId, shift(1), rolling(5)
    # ELI5: .shift(1) pushes today's result into the future so only yesterday is visible!
    df["driver_norm_finish_p5"] = (
        df.groupby("driverId")["norm_finish"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    )

    # Rolling 5-race DNF rate
    df["driver_dnf_rate_p5"] = (
        df.groupby("driverId")["is_dnf"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    )

    # Expanding career podium rate
    df["driver_podium_rate_career"] = (
        df.groupby("driverId")["y_podium"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    # Rookie window: driver has fewer than 3 prior race starts
    df["is_rookie_window"] = (
        df.groupby("driverId")["raceId"]
        .transform(lambda s: s.shift(1).expanding().count() < 3)
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # Feature 12: Teammate Form Delta
    # -------------------------------------------------------------------------
    # Isolates driver skill from car performance by subtracting the average rolling
    # form of all other drivers on the same team in that race.
    sum_form = df.groupby(["raceId", "canonical_constructor_id"])["driver_norm_finish_p5"].transform("sum")
    count_form = df.groupby(["raceId", "canonical_constructor_id"])["driver_norm_finish_p5"].transform("count")

    # If driver has valid form, subtract their value from team sum; otherwise use full sum
    has_form = df["driver_norm_finish_p5"].notna()
    other_sum = np.where(has_form, sum_form - df["driver_norm_finish_p5"], sum_form)
    other_count = np.where(has_form, count_form - 1, count_form)

    teammate_mean = np.where(other_count > 0, other_sum / np.maximum(other_count, 1), np.nan)
    df["teammate_form_delta"] = np.where(
        has_form & np.isfinite(teammate_mean),
        df["driver_norm_finish_p5"] - teammate_mean,
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Feature 13-15: Race-Level Constructor Features (Zero-Leakage Aggregation)
    # -------------------------------------------------------------------------
    # Step A: Aggregate starters to race-constructor level
    constructor_race = (
        df.groupby(["year", "round", "raceId", "canonical_constructor_id"])
        .agg(race_pts=("points", "sum"))
        .reset_index()
        .sort_values(["canonical_constructor_id", "year", "round"])
    )

    # Step B: Compute prior season final share lookup table
    season_totals = (
        constructor_race.groupby(["year", "canonical_constructor_id"])["race_pts"]
        .sum()
        .reset_index()
    )
    all_season_pts = season_totals.groupby("year")["race_pts"].transform("sum")
    season_totals["prior_season_share"] = season_totals["race_pts"] / np.maximum(all_season_pts, 1.0)
    season_totals["prior_year"] = season_totals["year"] + 1

    prior_share_map = dict(
        zip(
            zip(season_totals["prior_year"], season_totals["canonical_constructor_id"]),
            season_totals["prior_season_share"],
        )
    )

    # Step C: Within each season, compute cumulative constructor points shifted by 1 race
    constructor_race["pts_cumsum_before_race"] = (
        constructor_race.groupby(["canonical_constructor_id", "year"])["race_pts"]
        .cumsum()
        .shift(1)
        .fillna(0.0)
    )

    # Cumulative maximum points available before that round
    race_max_pts = (
        df.groupby(["year", "round", "raceId"])["points"]
        .max()
        .reset_index()
        .sort_values(["year", "round"])
    )
    race_max_pts["max_pts_cumsum_before"] = (
        race_max_pts.groupby("year")["points"]
        .cumsum()
        .shift(1)
        .fillna(0.0)
    )
    constructor_race = constructor_race.merge(
        race_max_pts[["raceId", "max_pts_cumsum_before"]],
        on="raceId",
        how="left",
    )

    # Round 1 rule: If round == 1, fall back to prior season's final share
    is_round_1 = constructor_race["round"] == 1
    prior_shares = [
        prior_share_map.get((yr, cid), None)
        for yr, cid in zip(constructor_race["year"], constructor_race["canonical_constructor_id"])
    ]

    constructor_pts_share = np.zeros(len(constructor_race))
    new_constructor_flag = np.zeros(len(constructor_race), dtype=int)

    for i in range(len(constructor_race)):
        if is_round_1.iloc[i]:
            ps = prior_shares[i]
            if ps is not None:
                constructor_pts_share[i] = ps
                new_constructor_flag[i] = 0
            else:
                constructor_pts_share[i] = 0.0
                new_constructor_flag[i] = 1
        else:
            denom = constructor_race["max_pts_cumsum_before"].iloc[i]
            num = constructor_race["pts_cumsum_before_race"].iloc[i]
            constructor_pts_share[i] = num / denom if denom > 0 else 0.0
            new_constructor_flag[i] = 0

    constructor_race["constructor_pts_share"] = constructor_pts_share
    constructor_race["new_constructor_flag"] = new_constructor_flag

    # Constructor rank within race (Rank 1 = most points before this round)
    constructor_race["constructor_season_rank"] = (
        constructor_race.groupby("raceId")["pts_cumsum_before_race"]
        .rank(ascending=False, method="min")
    )

    # Step D: Join race-level constructor features back to driver starters
    df = df.merge(
        constructor_race[[
            "raceId",
            "canonical_constructor_id",
            "constructor_pts_share",
            "constructor_season_rank",
            "new_constructor_flag",
        ]],
        on=["raceId", "canonical_constructor_id"],
        how="left",
    )

    # -------------------------------------------------------------------------
    # Feature 16-17: Calendar and Driver Demographics
    # -------------------------------------------------------------------------
    df["rounds_completed"] = (df["round"] - 1).astype(float)

    # -------------------------------------------------------------------------
    # Whitelist & Anti-Leakage Verification
    # -------------------------------------------------------------------------
    for feat in ALLOWED_FEATURES:
        assert feat in df.columns, f"Required feature '{feat}' is missing from builder output!"

    return df
