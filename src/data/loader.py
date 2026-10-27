"""Formula 1 Data Ingestion and Validation Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Think of loading data like shopping for fresh ingredients before cooking a 5-star meal.
If you buy rotten vegetables, bad milk, or expired eggs, your dinner will taste terrible,
no matter how great a chef you are!

In Formula 1 history:
1. Missing values were written as the weird code '\\N' instead of standard blanks.
   If we forget to translate '\\N' into real Python missing values (NaN), Python thinks
   it's regular text, which breaks all math!
2. In the 1950s, drivers used to jump into their teammate's car midway through the race
   (shared drives), meaning multiple rows have the exact same driver and race!
   Starting in 1994, shared drives were banned and electronic qualifying times began.
3. If a driver crashed in morning practice and never took the start of the race,
   putting them on the grid would be like grading a student who was absent from school!

This module safely imports all raw CSV files, cleans the missing codes, filters out
non-starters using our verified audit lists, and guarantees clean, trustworthy data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# -----------------------------------------------------------------------------
# Module Constants & Paths
# -----------------------------------------------------------------------------
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_AUDIT_DIR = Path("data/processed/audit")

# 1994 marks the start of the modern electronic telemetry era and consistent qualifying.
# Pre-1994 shared drives corrupt race-level targets (e.g. up to 7 podiums in 1 race).
SUPERVISED_START_YEAR: int = 1994

# Career-level clustering requires a driver to have had a meaningful career
# to avoid skewing distributions with one-off replacement drivers.
MIN_CAREER_STARTS: int = 20


def load_raw_csvs(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all Ergast F1 CSV files safely with raw string '\\N' handling.

    ELI5:
    We open each table with the rule na_values=[r"\\N"] so Python immediately recognizes
    the text '\\N' as empty/missing data rather than confusing it with real text.

    Parameters
    ----------
    raw_dir : Path, default=Path("data/raw")
        Path to directory containing the raw Ergast CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping table name (without .csv) to its loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If required CSV files are missing from the raw directory.
    AssertionError
        If critical columns fail numeric dtype validation.
    """
    raw_dir = Path(raw_dir)
    tables: dict[str, pd.DataFrame] = {}

    csv_names = [
        "circuits",
        "constructors",
        "constructor_results",
        "constructor_standings",
        "drivers",
        "driver_standings",
        "lap_times",
        "pit_stops",
        "qualifying",
        "races",
        "results",
        "seasons",
        "sprint_results",
        "status",
    ]

    for name in csv_names:
        file_path = raw_dir / f"{name}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required raw CSV file: {file_path}")
        # Literal two-character string r"\N" is Ergast's SQL NULL representation
        tables[name] = pd.read_csv(file_path, na_values=[r"\N"])

    # -------------------------------------------------------------------------
    # Dtype & Integrity Assertions
    # -------------------------------------------------------------------------
    results = tables["results"]
    assert is_numeric_dtype(results["positionOrder"]), (
        "Fatal: results['positionOrder'] is not numeric! Literal '\\N' was not parsed."
    )
    assert is_numeric_dtype(results["grid"]), (
        "Fatal: results['grid'] is not numeric! Check CSV null parsing."
    )
    assert results["positionOrder"].isna().sum() == 0, (
        "Fatal: results['positionOrder'] contains missing values. Expected 0 nulls."
    )

    return tables


def load_audit_artifacts(audit_dir: Path = DEFAULT_AUDIT_DIR) -> dict[str, Any]:
    """Load frozen audit artifacts produced by Table 1 Data Audit.

    Parameters
    ----------
    audit_dir : Path, default=Path("data/processed/audit")
        Directory containing frozen JSON audit artifacts.

    Returns
    -------
    dict[str, Any]
        Dictionary with non_starter_status_ids, sprint_race_ids, and constructor_canonical_map.
    """
    audit_dir = Path(audit_dir)
    with open(audit_dir / "non_starter_status_ids.json", "r", encoding="utf-8") as f:
        non_starters = set(json.load(f))

    with open(audit_dir / "sprint_race_ids.json", "r", encoding="utf-8") as f:
        sprint_race_ids = set(json.load(f))

    with open(audit_dir / "constructor_canonical_map.json", "r", encoding="utf-8") as f:
        constructor_map = json.load(f)

    return {
        "non_starter_status_ids": non_starters,
        "sprint_race_ids": sprint_race_ids,
        "constructor_canonical_map": constructor_map,
    }


def load_clean_starters(
    raw_dir: Path = DEFAULT_RAW_DIR,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> pd.DataFrame:
    """Load, filter, and validate supervised race starters (1994–2024).

    ELI5 Explanation:
    1. We only look at 1994 onward because in earlier decades, drivers could share cars
       mid-race, which meant multiple people shared 1st or 2nd place!
    2. We remove drivers who withdrew or didn't qualify before the race started.
       Real racers who crashed on Turn 1 (0 laps completed) ARE kept, because they started!
    3. If a driver starts from the pit lane (grid == 0), we don't treat them as starting in
       pole position (0 < 1)! Instead, we give them grid_zero_flag = 1 and set their
       effective grid position to the very back of the pack.

    Parameters
    ----------
    raw_dir : Path
        Path to raw CSV directory.
    audit_dir : Path
        Path to audit directory containing frozen JSON artifacts.

    Returns
    -------
    pd.DataFrame
        Cleaned starters DataFrame containing 1994–2024 race entries with targets and flags.
    """
    tables = load_raw_csvs(raw_dir)
    audit = load_audit_artifacts(audit_dir)

    results = tables["results"]
    races = tables["races"]
    status = tables["status"]
    constructors = tables["constructors"]
    drivers = tables["drivers"]

    # Step 1: Merge primary result records with race metadata
    df = results.merge(
        races[["raceId", "year", "round", "circuitId", "name", "date"]],
        on="raceId",
        how="left",
    ).merge(
        status[["statusId", "status"]],
        on="statusId",
        how="left",
    ).merge(
        constructors[["constructorId", "constructorRef"]],
        on="constructorId",
        how="left",
    ).merge(
        drivers[["driverId", "driverRef", "dob"]],
        on="driverId",
        how="left",
    )

    # Step 2: Supervised temporal scope constraint (1994–2024)
    df_supervised = df[df["year"] >= SUPERVISED_START_YEAR].copy()

    # Integrity assertion: Exactly one row per (raceId, driverId) in modern era
    assert df_supervised.duplicated(subset=["raceId", "driverId"]).sum() == 0, (
        "Duplicate (raceId, driverId) detected! Shared drives should not exist in post-1994."
    )

    # Step 3: Non-starter filtering using frozen statusId exclusion list
    non_starters = audit["non_starter_status_ids"]
    df_starters = df_supervised[~df_supervised["statusId"].isin(non_starters)].copy()

    # Step 4: Map canonical constructor lineage
    constructor_map = audit["constructor_canonical_map"]
    df_starters["canonical_constructor_id"] = (
        df_starters["constructorId"]
        .astype(str)
        .map(constructor_map)
        .fillna(df_starters["constructorRef"])
    )

    # Step 5: Grid zero handling and effective grid derivation
    # Grid == 0 in Ergast encodes pit lane starts or unspecified rear starts.
    # We create an explicit binary indicator and place the effective grid behind the field.
    df_starters["grid_zero_flag"] = (df_starters["grid"] == 0).astype(int)

    # Find the maximum non-zero grid assigned in that season
    season_max_grid = (
        df_starters.groupby("year")["grid"]
        .transform(lambda s: s.replace(0, np.nan).max())
    )
    # Effective grid: keep genuine grid if > 0, else place at season_max + 1
    df_starters["grid_effective"] = df_starters["grid"].where(
        df_starters["grid"] > 0,
        season_max_grid.fillna(22.0) + 1.0,
    )

    # Step 6: Target variable derivation
    # Binary Classification Target: Podium finisher (positionOrder <= 3 AND classified finisher)
    # Ergast statusId == 1 means "Finished", while statusIds 11-19 denote "+N Laps" classified finishers.
    # Drivers with DNF (mechanical, crash, etc.) or DSQ receive y_podium = 0.
    is_classified = (df_starters["statusId"] == 1) | (
        df_starters["status"].str.startswith("+") & df_starters["status"].str.endswith("Lap")
    ) | (
        df_starters["status"].str.startswith("+") & df_starters["status"].str.endswith("Laps")
    )
    df_starters["is_classified"] = is_classified.astype(int)
    df_starters["is_dnf"] = (~is_classified).astype(int)

    df_starters["y_podium"] = (
        (df_starters["positionOrder"] <= 3) & (df_starters["is_classified"] == 1)
    ).astype(int)

    # Continuous Regression Target: positionOrder (confirmed 100% complete)
    df_starters["y_position"] = df_starters["positionOrder"].astype(float)

    # Convert date strings to datetime for chronological sorting
    df_starters["race_date"] = pd.to_datetime(df_starters["date"])
    df_starters["dob_date"] = pd.to_datetime(df_starters["dob"])
    df_starters["driver_age_years"] = (
        (df_starters["race_date"] - df_starters["dob_date"]).dt.days / 365.25
    )

    # Sort strictly chronologically by year, round, and finishing order
    df_starters = df_starters.sort_values(
        ["year", "round", "positionOrder"]
    ).reset_index(drop=True)

    return df_starters


def load_career_clustering_data(
    raw_dir: Path = DEFAULT_RAW_DIR,
    min_starts: int = MIN_CAREER_STARTS,
) -> pd.DataFrame:
    """Load full historical dataset (1950–2024) aggregated to driver-career level.

    ELI5 Explanation:
    To group Formula 1 drivers by their career achievements (like "World Champions",
    "Podium Contenders", "Midfield Workhorses", or "Backmarkers"), we cannot just look
    at total points, because modern F1 awards 25 points for a win, whereas 1950 only
    awarded 8 points!
    Instead, we calculate normalized metrics:
    - win_share: percentage of starts they won
    - podium_share: percentage of starts they stood on the podium
    - pts_share: average percentage of available points scored per race
    - norm_avg_finish: field-size normalized finishing position
    - dnf_adj: driver DNF rate adjusted for the car failure rate of their era

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw Ergast CSV files.
    min_starts : int, default=20
        Minimum career race starts required for inclusion.

    Returns
    -------
    pd.DataFrame
        Career-level aggregated DataFrame with one row per driver.
    """
    tables = load_raw_csvs(raw_dir)
    results = tables["results"]
    races = tables["races"]
    status = tables["status"]
    drivers = tables["drivers"]

    df = results.merge(
        races[["raceId", "year", "round"]],
        on="raceId",
        how="left",
    ).merge(
        status[["statusId", "status"]],
        on="statusId",
        how="left",
    ).merge(
        drivers[["driverId", "driverRef", "forename", "surname"]],
        on="driverId",
        how="left",
    )

    # Exclude non-starters
    non_starters = {73, 77, 78, 81, 89, 97}
    df = df[~df["statusId"].isin(non_starters)].copy()

    # Field size per race
    field_sizes = df.groupby("raceId")["driverId"].transform("count")
    df["norm_finish"] = (df["positionOrder"] - 1.0) / np.maximum(field_sizes - 1.0, 1.0)

    # Points share per race: race points / maximum points awarded to winner that race
    race_max_pts = df.groupby("raceId")["points"].transform("max")
    df["pts_share"] = df["points"] / np.maximum(race_max_pts, 1.0)

    # DNF flag and era DNF rate
    is_classified = (df["statusId"] == 1) | (
        df["status"].str.startswith("+") & df["status"].str.endswith("Lap")
    ) | (
        df["status"].str.startswith("+") & df["status"].str.endswith("Laps")
    )
    df["is_dnf"] = (~is_classified).astype(float)
    race_dnf_rate = df.groupby("raceId")["is_dnf"].transform("mean")
    df["dnf_adj"] = df["is_dnf"] - race_dnf_rate

    df["is_win"] = (df["positionOrder"] == 1).astype(float)
    df["is_podium"] = (df["positionOrder"] <= 3).astype(float)

    # Teammate delta calculation over career
    sum_norm = df.groupby(["raceId", "constructorId"])["norm_finish"].transform("sum")
    count_norm = df.groupby(["raceId", "constructorId"])["norm_finish"].transform("count")
    other_sum = sum_norm - df["norm_finish"]
    other_count = count_norm - 1
    teammate_norm_mean = np.where(other_count > 0, other_sum / np.maximum(other_count, 1), np.nan)
    df["race_teammate_delta"] = np.where(
        np.isfinite(teammate_norm_mean),
        df["norm_finish"] - teammate_norm_mean,
        np.nan,
    )

    # Group by driver across their entire career
    careers = df.groupby(["driverId", "driverRef", "forename", "surname"]).agg(
        total_starts=("raceId", "count"),
        total_wins=("is_win", "sum"),
        total_podiums=("is_podium", "sum"),
        win_share=("is_win", "mean"),
        podium_share=("is_podium", "mean"),
        pts_share=("pts_share", "mean"),
        norm_avg_finish=("norm_finish", "mean"),
        dnf_adj=("dnf_adj", "mean"),
        teammate_delta=("race_teammate_delta", "mean"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ).reset_index()

    # Fill NaN teammate_delta with 0.0 (drivers who had no teammates)
    careers["teammate_delta"] = careers["teammate_delta"].fillna(0.0)

    # Filter to drivers with at least min_starts
    careers_filtered = careers[careers["total_starts"] >= min_starts].copy()
    careers_filtered["driver_name"] = (
        careers_filtered["forename"] + " " + careers_filtered["surname"]
    )

    return careers_filtered.reset_index(drop=True)
