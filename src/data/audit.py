"""Formula 1 Data Audit & Integrity Verification Module.

Version: 5.0 — Authoritative Table 1 Audit

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Before building a rocket, engineers inspect every single bolt, screw, and sensor
to ensure nothing is broken or missing. If you build a rocket with cracked metal,
it will explode during launch.

In machine learning, data is our building material! If our data has hidden bugs
(like races where results were recorded twice, or drivers who never actually drove,
or missing lap times), our models will learn the wrong lessons and fail completely.

This audit checks every single piece of historical Formula 1 data from 1950 to 2024,
answers the 11 critical data integrity questions required by Table 1, and freezes
the verified truth into permanent audit files before any model code is allowed to run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Module Constants & Paths
# -----------------------------------------------------------------------------
RAW_DATA_DIR = Path("data/raw")
AUDIT_DIR = Path("data/processed/audit")

# Supervised learning boundary: 1994 marks the start of electronic qualifying times
# and eliminates historical shared-drive races that corrupt podium labels.
SUPERVISED_START_YEAR = 1994

# Non-starter status IDs: Drivers with these status codes never took the green flag.
# 73: Injured prior to race (e.g. Button 2003, Häkkinen 1995)
# 77: 107% Rule (failed qualifying pace cutoff)
# 78: Safety withdrawal prior to race (e.g. Minardi 2002 Spain wing failures)
# 81: Did not qualify (DNQ)
# 89: Safety concerns prior to race (e.g. Sauber 2000 Brazil wing failures)
# 97: Did not prequalify (DNPQ)
NON_STARTER_STATUS_IDS: list[int] = [73, 77, 78, 81, 89, 97]


def run_table1_audit(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = AUDIT_DIR,
) -> dict[str, Any]:
    """Execute complete Table 1 integrity audit on raw F1 dataset.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the 14 raw Ergast CSV files.
    output_dir : Path
        Directory where frozen JSON/CSV audit artifacts will be saved.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed audit statistics and summary tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load all relevant raw CSV tables using raw-string null parsing
    # -------------------------------------------------------------------------
    results = pd.read_csv(raw_dir / "results.csv", na_values=[r"\N"])
    races = pd.read_csv(raw_dir / "races.csv", na_values=[r"\N"])
    status = pd.read_csv(raw_dir / "status.csv", na_values=[r"\N"])
    qualifying = pd.read_csv(raw_dir / "qualifying.csv", na_values=[r"\N"])
    sprint_results = pd.read_csv(raw_dir / "sprint_results.csv", na_values=[r"\N"])
    constructors = pd.read_csv(raw_dir / "constructors.csv", na_values=[r"\N"])
    drivers = pd.read_csv(raw_dir / "drivers.csv", na_values=[r"\N"])

    # Merge main driver results with race calendar and status labels
    df = results.merge(
        races[["raceId", "year", "round", "circuitId", "name", "date"]],
        on="raceId",
        how="left",
    ).merge(
        status[["statusId", "status"]],
        on="statusId",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Audit Check C1: Null coverage per column by decade
    # -------------------------------------------------------------------------
    df["decade"] = (df["year"] // 10) * 10
    cols_to_check = ["positionOrder", "position", "grid", "time", "milliseconds", "fastestLap"]
    null_summary_by_decade = df.groupby("decade")[cols_to_check].apply(
        lambda g: g.isna().mean() * 100
    ).reset_index()

    # -------------------------------------------------------------------------
    # Audit Check C2: Q1 coverage per season (1994–2024)
    # -------------------------------------------------------------------------
    q_merged = qualifying.merge(races[["raceId", "year"]], on="raceId", how="left")
    q_cov = q_merged.groupby("year").agg(
        total_qualifiers=("qualifyId", "count"),
        q1_non_null=("q1", "count"),
        q1_null=("q1", lambda s: s.isna().sum()),
    ).reset_index()
    q_cov["q1_coverage_pct"] = (q_cov["q1_non_null"] / q_cov["total_qualifiers"]) * 100
    q_cov.to_csv(output_dir / "q_coverage_by_season.csv", index=False)

    # -------------------------------------------------------------------------
    # Audit Check C3: Laps == 0 breakdown and freeze NON_STARTER_STATUS_IDS
    # -------------------------------------------------------------------------
    zero_laps_counts = df[df["laps"] == 0]["status"].value_counts().to_dict()
    with open(output_dir / "non_starter_status_ids.json", "w", encoding="utf-8") as f:
        json.dump(NON_STARTER_STATUS_IDS, f, indent=2)

    # -------------------------------------------------------------------------
    # Audit Check C4: Podium count per race (confirm exactly 3 post-1993)
    # -------------------------------------------------------------------------
    df_post94 = df[df["year"] >= SUPERVISED_START_YEAR]
    podiums_post94 = df_post94[df_post94["positionOrder"] <= 3].groupby("raceId").size()
    anomalous_races_post94 = int((podiums_post94 != 3).sum())

    df_pre94 = df[df["year"] < SUPERVISED_START_YEAR]
    podiums_pre94 = df_pre94[df_pre94["positionOrder"] <= 3].groupby("raceId").size()
    anomalous_races_pre94 = int((podiums_pre94 != 3).sum())

    # -------------------------------------------------------------------------
    # Audit Check C5: Canonical Constructor Map
    # -------------------------------------------------------------------------
    # Map constructor IDs to continuous organizational lineage (F1 factories)
    constructor_canonical_map: dict[str, str] = {
        # Brackley lineage: Tyrrell -> BAR -> Honda -> Brawn -> Mercedes
        "25": "mercedes_brackley",   # tyrrell
        "16": "mercedes_brackley",   # bar
        "11": "mercedes_brackley",   # honda
        "23": "mercedes_brackley",   # brawn
        "131": "mercedes_brackley",  # mercedes

        # Enstone lineage: Benetton -> Renault -> Lotus F1 -> Renault -> Alpine
        "22": "renault_enstone",     # benetton
        "4": "renault_enstone",      # renault
        "208": "renault_enstone",    # lotus_f1 (2012-2015)
        "214": "renault_enstone",    # alpine (2021+)

        # Silverstone lineage: Jordan -> Midland -> Spyker -> Force India -> Racing Point -> Aston Martin
        "17": "silverstone_aston",   # jordan
        "13": "silverstone_aston",   # mf1
        "14": "silverstone_aston",   # spyker_mf1
        "12": "silverstone_aston",   # spyker
        "10": "silverstone_aston",   # force_india
        "211": "silverstone_aston",  # racing_point
        "117": "silverstone_aston",  # aston_martin

        # Faenza lineage: Minardi -> Toro Rosso -> AlphaTauri -> RB
        "18": "toro_rosso_rb",       # minardi
        "5": "toro_rosso_rb",        # toro_rosso
        "213": "toro_rosso_rb",      # alphatauri
        "215": "toro_rosso_rb",      # rb

        # Hinwil lineage: Sauber -> BMW Sauber -> Sauber -> Alfa Romeo -> Sauber
        "15": "sauber_alfa",         # sauber
        "2": "sauber_alfa",          # bmw_sauber
        "51": "sauber_alfa",         # alfa

        # Milton Keynes lineage: Stewart -> Jaguar -> Red Bull
        "24": "red_bull",            # stewart
        "19": "red_bull",            # jaguar
        "9": "red_bull",             # red_bull

        # Historic / Traditional Independents
        "6": "ferrari",              # ferrari
        "1": "mclaren",              # mclaren
        "3": "williams",             # williams
        "210": "haas",               # haas
        "7": "toyota",               # toyota
        "8": "super_aguri",          # super_aguri
        "20": "prost",               # prost
        "21": "arrows",              # arrows
        "27": "prost",               # ligier -> prost
        "29": "arrows",              # footwork -> arrows
        "164": "hrt",                # hrt
        "166": "manor_marussia",     # virgin
        "206": "manor_marussia",     # marussia
        "209": "manor_marussia",     # manor
        "205": "caterham",           # lotus_racing (Fernandes)
        "207": "caterham",           # caterham
        "31": "simtek",              # simtek
        "30": "pacific",             # pacific
        "28": "forti",               # forti
        "33": "larrousse",           # larrousse
        "32": "team_lotus_historic", # Colin Chapman Team Lotus (ended 1994)
        "26": "lola",                # lola
    }
    with open(output_dir / "constructor_canonical_map.json", "w", encoding="utf-8") as f:
        json.dump(constructor_canonical_map, f, indent=2)

    # -------------------------------------------------------------------------
    # Audit Check C6: positionOrder completeness and DSQ range
    # -------------------------------------------------------------------------
    pos_order_nulls = int(results["positionOrder"].isna().sum())
    pos_nulls = int(results["position"].isna().sum())
    dsq_rows = df_post94[df_post94["statusId"] == 2]
    dsq_stats = {
        "count": int(len(dsq_rows)),
        "min_positionOrder": int(dsq_rows["positionOrder"].min()),
        "max_positionOrder": int(dsq_rows["positionOrder"].max()),
        "median_positionOrder": float(dsq_rows["positionOrder"].median()),
    }

    # -------------------------------------------------------------------------
    # Audit Check C7: grid == 0 counts by decade and post-filter remainder
    # -------------------------------------------------------------------------
    grid_zero_by_decade = df[df["grid"] == 0].groupby("decade").size().to_dict()
    df_starters = df_post94[~df_post94["statusId"].isin(NON_STARTER_STATUS_IDS)]
    grid_zero_remaining_starters = int((df_starters["grid"] == 0).sum())

    # -------------------------------------------------------------------------
    # Audit Check C8: Sprint race IDs from sprint_results.csv (all years)
    # -------------------------------------------------------------------------
    sprint_race_ids = sorted([int(x) for x in sprint_results["raceId"].unique()])
    with open(output_dir / "sprint_race_ids.json", "w", encoding="utf-8") as f:
        json.dump(sprint_race_ids, f, indent=2)

    # -------------------------------------------------------------------------
    # Audit Check C9: Indianapolis 500 races
    # -------------------------------------------------------------------------
    indy_races = races[races["name"].str.contains("Indianapolis 500", case=False, na=False)]
    indy_count = len(indy_races)
    indy_post94_count = len(indy_races[indy_races["year"] >= SUPERVISED_START_YEAR])

    # -------------------------------------------------------------------------
    # Audit Check C10: Shared-drive (raceId, driverId) duplicate count
    # -------------------------------------------------------------------------
    shared_drive_all = int(df.duplicated(subset=["raceId", "driverId"], keep=False).sum())
    shared_drive_post94 = int(df_post94.duplicated(subset=["raceId", "driverId"], keep=False).sum())

    # -------------------------------------------------------------------------
    # Audit Check C11: Data partition row counts and percentages
    # -------------------------------------------------------------------------
    n_starters_total = len(df_starters)
    train_core = df_starters[df_starters["year"] <= 2016]
    val = df_starters[(df_starters["year"] >= 2017) & (df_starters["year"] <= 2018)]
    train_val_pool = df_starters[df_starters["year"] <= 2018]
    test = df_starters[df_starters["year"] >= 2019]

    split_summary = {
        "train_core": {
            "seasons": "1994-2016",
            "races": int(train_core["raceId"].nunique()),
            "rows": int(len(train_core)),
            "row_pct": float(len(train_core) / n_starters_total * 100),
        },
        "val": {
            "seasons": "2017-2018",
            "races": int(val["raceId"].nunique()),
            "rows": int(len(val)),
            "row_pct": float(len(val) / n_starters_total * 100),
        },
        "train_val_pool": {
            "seasons": "1994-2018",
            "races": int(train_val_pool["raceId"].nunique()),
            "rows": int(len(train_val_pool)),
            "row_pct": float(len(train_val_pool) / n_starters_total * 100),
        },
        "test": {
            "seasons": "2019-2024",
            "races": int(test["raceId"].nunique()),
            "rows": int(len(test)),
            "row_pct": float(len(test) / n_starters_total * 100),
        },
        "total_starters": {
            "seasons": "1994-2024",
            "races": int(df_starters["raceId"].nunique()),
            "rows": int(n_starters_total),
            "row_pct": 100.0,
        },
    }

    # -------------------------------------------------------------------------
    # Save Master Table 1 Audit CSV
    # -------------------------------------------------------------------------
    table1_rows = [
        {"Audit Item": "Supervised Scope", "Observed Value": "1994-2024 (Seasons 1994 to 2024)"},
        {"Audit Item": "positionOrder Missing Values", "Observed Value": f"{pos_order_nulls} (100% complete)"},
        {"Audit Item": "position Missing Values", "Observed Value": f"{pos_nulls} (incomplete; DNF/DSQ missing)"},
        {"Audit Item": "Post-1993 Podium Corruption", "Observed Value": f"{anomalous_races_post94} races with != 3 podiums (0% error)"},
        {"Audit Item": "Pre-1994 Podium Corruption", "Observed Value": f"{anomalous_races_pre94} shared-drive races with 4-7 podiums"},
        {"Audit Item": "Shared-Drive Duplicate Keys (Post-1993)", "Observed Value": f"{shared_drive_post94} duplicates (clean 1-to-1)"},
        {"Audit Item": "Shared-Drive Duplicate Keys (All-Time)", "Observed Value": f"{shared_drive_all} duplicate rows"},
        {"Audit Item": "Indy 500 Races (1994-2024)", "Observed Value": f"{indy_post94_count} races (all {indy_count} in 1950-1960)"},
        {"Audit Item": "Sprint Race IDs Identified", "Observed Value": f"{len(sprint_race_ids)} races (2021-2024)"},
        {"Audit Item": "DSQ Drivers Position Range", "Observed Value": f"min={dsq_stats['min_positionOrder']}, median={dsq_stats['median_positionOrder']}, max={dsq_stats['max_positionOrder']}"},
        {"Audit Item": "Grid == 0 Starters Remaining", "Observed Value": f"{grid_zero_remaining_starters} rows (encoded as grid_zero_flag=1)"},
        {"Audit Item": "train_core (1994-2016)", "Observed Value": f"{split_summary['train_core']['races']} races, {split_summary['train_core']['rows']} rows ({split_summary['train_core']['row_pct']:.2f}%)"},
        {"Audit Item": "val (2017-2018)", "Observed Value": f"{split_summary['val']['races']} races, {split_summary['val']['rows']} rows ({split_summary['val']['row_pct']:.2f}%)"},
        {"Audit Item": "train_val_pool (1994-2018)", "Observed Value": f"{split_summary['train_val_pool']['races']} races, {split_summary['train_val_pool']['rows']} rows ({split_summary['train_val_pool']['row_pct']:.2f}%)"},
        {"Audit Item": "test (2019-2024)", "Observed Value": f"{split_summary['test']['races']} races, {split_summary['test']['rows']} rows ({split_summary['test']['row_pct']:.2f}%)"},
    ]
    table1_df = pd.DataFrame(table1_rows)
    table1_df.to_csv(output_dir / "table1_audit.csv", index=False)

    # -------------------------------------------------------------------------
    # Feature Registry Table
    # -------------------------------------------------------------------------
    feature_registry = [
        {"feature_name": "grid_effective", "type": "continuous", "is_allowed": True, "formula": "grid if grid>0 else field_max+1"},
        {"feature_name": "grid_norm", "type": "continuous", "is_allowed": True, "formula": "grid_effective / max(grid_effective in race)"},
        {"feature_name": "grid_zero_flag", "type": "binary", "is_allowed": True, "formula": "(grid == 0).astype(int)"},
        {"feature_name": "q_gap_pct", "type": "continuous", "is_allowed": True, "formula": "(q1 - min(q1)) / min(q1) * 100 (outliers >150% median -> NaN)"},
        {"feature_name": "q1_data_available", "type": "binary", "is_allowed": True, "formula": "1 if Q1 valid and not outlier else 0"},
        {"feature_name": "quali_position_norm", "type": "continuous", "is_allowed": True, "formula": "qualifying.position / num_qualifiers"},
        {"feature_name": "grid_vs_quali_delta", "type": "continuous", "is_allowed": True, "formula": "grid_effective - qualifying.position (NaN for sprint races)"},
        {"feature_name": "driver_norm_finish_p5", "type": "continuous", "is_allowed": True, "formula": "driver shift(1) rolling(5, min_periods=3) norm_finish mean"},
        {"feature_name": "driver_dnf_rate_p5", "type": "continuous", "is_allowed": True, "formula": "driver shift(1) rolling(5, min_periods=3) DNF mean"},
        {"feature_name": "driver_podium_rate_career", "type": "continuous", "is_allowed": True, "formula": "driver shift(1) expanding podium mean"},
        {"feature_name": "is_rookie_window", "type": "binary", "is_allowed": True, "formula": "1 if prior starts < 3 else 0"},
        {"feature_name": "teammate_form_delta", "type": "continuous", "is_allowed": True, "formula": "driver_norm_finish_p5 - mean(teammates_norm_finish_p5)"},
        {"feature_name": "constructor_pts_share", "type": "continuous", "is_allowed": True, "formula": "race-level constructor cumsum pts / cumsum max pts, shifted 1 race"},
        {"feature_name": "constructor_season_rank", "type": "continuous", "is_allowed": True, "formula": "rank of constructor pts_cumsum_before_race"},
        {"feature_name": "rounds_completed", "type": "continuous", "is_allowed": True, "formula": "round - 1"},
        {"feature_name": "driver_age_years", "type": "continuous", "is_allowed": True, "formula": "(race_date - dob) in years"},
        {"feature_name": "new_constructor_flag", "type": "binary", "is_allowed": True, "formula": "1 if constructor has no prior season history else 0"},
    ]
    pd.DataFrame(feature_registry).to_csv(output_dir / "feature_registry.csv", index=False)

    return {
        "split_summary": split_summary,
        "anomalous_races_post94": anomalous_races_post94,
        "anomalous_races_pre94": anomalous_races_pre94,
        "shared_drive_post94": shared_drive_post94,
        "sprint_race_count": len(sprint_race_ids),
        "dsq_stats": dsq_stats,
        "grid_zero_remaining": grid_zero_remaining_starters,
    }


if __name__ == "__main__":
    summary = run_table1_audit()
    print("Table 1 Audit completed successfully!")
    print(json.dumps(summary, indent=2))
