# MEMORY.md — Persistent Decision Log
## Formula 1 ML Suite
### Current Version: v5.0 — ALL PRIOR VERSIONS SUPERSEDED

> [!CAUTION]
> This file reflects v5.0 only. Any decision not listed here should be treated as unresolved and must be confirmed before implementation. If you see a reference to `time_gap_seconds`, `9999`, `laps > 0`, `TimeSeriesSplit(gap=...)`, `q_delta_to_pole`, `pit_lane_start` (without the "zero_flag" rename), or CIFAR-10 anywhere in the code or notebooks — that is a v3/v4 artifact. Stop and flag it.

---

## Section 1: Environment

| Item | Value |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.7.0 |
| Scikit-learn | 1.5.2 |
| NumPy | 2.2.6 |
| Pandas | 2.2.3 |
| Seaborn | 0.13.2 |
| Matplotlib | 3.10.3 |
| ReportLab | 5.0.1 |
| kagglehub | 1.0.2 |
| Workspace | `d:\ML-Project\` (Windows, PowerShell) |
| Dataset cache | `C:\Users\hamiz\.cache\kagglehub\datasets\rohanrao\formula-1-world-championship-1950-2020\versions\24` |
| Deadline | October 31, 2026 |

---

## Section 2: Authoritative Technical Decisions (v5.0)

### Data Scope
- **Supervised experiments (classification, regression)**: 1994–2024 only.
  - Reason: Pre-1994 has shared-drive rows that produce 4–7 podium labels in 18 races (label corruption). Qualifying data only starts 1994.
- **Career clustering (K-Means, PCA)**: Full 1950–2024 history.
  - Caveat documented: clusters describe established drivers (≥20 starts), not all entrants.

### Missing Values
- **All CSVs loaded with `na_values=[r"\N"]`** — the literal two-character string `\N` is Ergast's NULL encoding.
- Never use `na_values=['\N']` (Python escape sequence differs from raw string in edge cases).

### Non-Starter Filter
- **Status-based only** — exact `statusId` set produced by Table 1 audit, saved to `data/processed/audit/non_starter_status_ids.json`.
- `laps == 0` is **NOT** used as primary filter — first-lap crashers have `laps == 0` but genuinely started the race.
- DSQ (Disqualified) drivers ARE starters. They get `y_podium = 0`. Included in regression target.

### Regression Target (DEFINITIVE)
- **`positionOrder` for all starters (1994–2024).**
- `positionOrder` confirmed complete — zero nulls in the dataset.
- No sentinel value. No `time_gap_seconds`. No `milliseconds`. No `9999`.
- DNF results already rank at the back of `positionOrder` by Ergast convention.
- Robustness row in Table 3a: R² recomputed on classified-finishers-only subpopulation.

### Classification Target (DEFINITIVE)
- **`y_podium = 1` if `positionOrder <= 3` AND driver is classified finisher, else 0.**
- Applied to 1994–2024 only (avoids shared-drive label corruption).

### Data Split (DEFINITIVE)
```
train_core    = 1994–2016 seasons
val           = 2017–2018 seasons
test          = 2019–2024 seasons
train_val_pool = train_core ∪ val = 1994–2018
```
- **Final sklearn models** refit on `train_val_pool` after hyperparameters and thresholds are frozen.
- **CNN** stays on `train_core` (val is the early-stop anchor, cannot be added back).
- Split is season-aligned (no race split mid-championship).
- Table 1 reports race counts and row percentages for each set. train_val_pool ≈ 80% of rows.

### CV Folds (DEFINITIVE)
- Use `race_time_folds()` from `src/data/preprocessor.py`.
- Returns **positional indices** via `np.flatnonzero()` on a reset-index DataFrame.
- No `TimeSeriesSplit(gap=...)` anywhere in the project.
- Val blocks = single seasons; train always precedes val chronologically; no race straddles boundary.

### Grid Encoding (DEFINITIVE)
- `grid == 0` is renamed to `grid_zero_flag = 1` — NOT `pit_lane_start` (ambiguous per R1 audit).
- `grid_effective` = original `grid` if > 0, else `max_grid_that_season + 1` (placed last in field).
- `grid_norm = grid_effective / max(grid_effective in that race)`.
- All uses of `grid` in model features use `grid_effective`, never raw `grid`.

### Qualifying Gap (DEFINITIVE)
- `q_gap_pct` = Q1 driver time minus Q1 fastest time in that race, divided by Q1 fastest, × 100.
- Q1 times > 150% of circuit-era median are flagged as outliers and treated as NaN.
- Never compare a Q1-eliminated driver's time to a Q3 pole time.
- `quali_position_norm` = `qualifying.position / total_qualifiers_in_race` (separate ordinal feature).
- `q1_data_available` = 1 if Q1 time valid and non-outlier, else 0.

### Sprint Weekend Masking (DEFINITIVE)
- `grid_vs_quali_delta = NaN` for **all** sprint weekend races from 2021 onward.
- Sprint raceIds identified from `sprint_results.csv` — all years, not hardcoded.
- Frozen in `data/processed/audit/sprint_race_ids.json`.
- Total: 3 (2021) + 3 (2022) + 6 (2023) + 6 (2024) = 18 sprint raceIds.

### Constructor Features (DEFINITIVE)
- Aggregated to `(raceId, canonical_constructor_id)` level first.
- Cumulative sum shifted by one **race** (not one row) to prevent within-race leakage.
- All drivers on same constructor in same race have **identical** `constructor_pts_share`.
- Round 1 of each season: use prior-season final share if constructor existed, else 0.0 + `new_constructor_flag = 1`.
- Never use `constructor_standings.csv` or `driver_standings.csv` as pre-race features.

### Rolling Features (DEFINITIVE)
- All rolling features computed as: `df.groupby('driverId')[feature].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())`
- The `groupby` is mandatory — without it, values cross driver boundaries.
- `min_periods=3` so rookies genuinely get NaN (imputer handles it).

### Teammate Delta (DEFINITIVE)
- `teammate_form_delta = driver_norm_finish_p5 - mean(norm_finish_p5 of ALL other drivers on same canonical constructor in same race)`.
- For >2 drivers on same team: mean of ALL others, not arbitrary pairwise.
- NaN if no teammate has prior history → imputer handles.

### Imputation & Scaling (DEFINITIVE)
- `SimpleImputer(strategy="median", add_indicator=True)` is **step 0** in every sklearn `Pipeline`.
- `StandardScaler()` is **step 1** in every sklearn `Pipeline`.
- Imputer and scaler fitted on `train_core` only inside GridSearchCV folds.
- Final model fitted on `train_val_pool` using the same pipeline structure.
- No manual imputation or scaling outside of pipelines.
- CNN: imputer/scaler fitted on `train_core`; same statistics applied to val and test sequence inputs.

### CNN Early Stopping (DEFINITIVE)
- **Early stop on val BCE loss** (patience=10 epochs). NOT val F1.
- After training stops: sweep thresholds [0.01–0.99] once on val to maximize F1.
- That threshold is frozen in `EXPERIMENT_FREEZE.yaml` before test evaluation.

### SVM Threshold (DEFINITIVE)
- SVM uses `decision_function()`, not `predict_proba()`.
- Threshold swept over `np.quantile(val_decision_scores, np.linspace(0.01, 0.99, 199))`.
- NOT swept over [0.1, 0.9] probability grid.

### GD Parity Tolerance (DEFINITIVE)
- `relative_excess = (J_GD - J_lstsq) / max(1.0, |J_lstsq|) < 1e-6`
- NOT absolute `< 1e-3`.

### Polynomial Regression (DEFINITIVE)
- Feature (`grid_effective`) scaled to [-1, 1] before degree expansion.
- `PolynomialFeatures` applied to **continuous features only** (not binary flags).
- OLS column in Table 4 noted as minimum-norm solution (rank-deficient after expansion).
- Demo subsample is pedagogical only — NOT included in Table 3a regression benchmark.

### K-Means Clustering (DEFINITIVE)
- **One row per driver career**. Minimum 20 total starts (1950–2024).
- k-means++ initialization with 15 restarts for elbow/silhouette selection.
- Single shared-centroid init (for parity test with sklearn only).
- Empty-cluster rule: reassign centroid to point farthest from its current centroid.
- Parity only claimed for runs where no cluster empties.

### PCA (DEFINITIVE)
- Same standardized matrix as K-Means.
- `np.linalg.eigh` (guaranteed real outputs for symmetric covariance).
- **Full loading matrix reported** (all 6 features × all 6 PCs) — not just top 3.
- Sign convention: flip each eigenvector so largest-magnitude loading is positive.

### Standard Dataset Appendices (DEFINITIVE)
- **Appendix A**: `load_digits` → Q1 classification (KNN, Tree, SVM, 2D-CNN on `(N,1,8,8)`)
- **Appendix B**: `load_diabetes` → Q2 regression (OLS, Ridge, Lasso)
- **Appendix C**: `load_iris` → Q3 K-Means + PCA
- **CIFAR-10 removed entirely** from all plans, code, and references.

### Experiment Freeze (DEFINITIVE)
- `EXPERIMENT_FREEZE.yaml` written before any `predict(X_test)` call.
- Contains: split boundaries, feature set hash, hyperparameters, thresholds, seeds, CNN stopped epoch.
- After this file is written, no model or preprocessing changes are permitted.

---

## Section 3: Change Log

| Date | Version | Summary |
|---|---|---|
| 2026-10-10 | v1.0 | Initial plan created |
| 2026-10-10 | v2.0 | Chronological split, anti-leakage pipeline, bootstrap CI |
| 2026-10-10 | v3.0 | Race-aligned folds, classification target, CNN dual-branch |
| 2026-10-10 | v4.0 | 24 changes: positionOrder target, status-based starter filter, explicit race folds, constructor pts share, PCA loadings, 4-panel Figure 4 |
| 2026-10-27 | v5.0 | 24 changes from two expert reviews: 1994+ supervised scope, season-aligned split, shared-drive fix, race-level constructor agg, Q1-only gap + outlier detection, sprint mask all years, CNN early stop on BCE loss, positional race folds, grid_zero_flag rename, SVM quantile threshold, GD relative tolerance, final refit on train_val_pool, EXPERIMENT_FREEZE.yaml, CIFAR-10 removed, standard dataset appendices for all 3 questions, driverId-grouped rolling, teammate mean-of-others |

---

## Section 4: SUPERSEDED DECISIONS — DO NOT IMPLEMENT

The following appeared in v3.0 or earlier and are **incorrect**. If you see these in code, flag and fix:

| Old Decision | Correct v5.0 Decision |
|---|---|
| Regression target: `time_gap_seconds` | Regression target: `positionOrder` |
| DNF sentinel: `9999.0` | No sentinel. positionOrder complete for all starters |
| Starter filter: `laps > 0` | Starter filter: status-based only (from audit JSON) |
| CV: `TimeSeriesSplit(n_splits=5, gap=...)` | CV: `race_time_folds()` returning positional indices |
| Qualifying feature: `q_delta_to_pole` | Qualifying feature: `q_gap_pct` (Q1-relative only) |
| Grid flag: `pit_lane_start` | Grid flag: `grid_zero_flag` (renamed — not unambiguously pit lane) |
| CNN early stop: val F1 | CNN early stop: val BCE loss |
| GD tolerance: absolute `1e-3` | GD tolerance: relative `< 1e-6 × max(1.0, |J_lstsq|)` |
| CIFAR-10 appendix | Removed; use `load_digits`, `load_diabetes`, `load_iris` |
| Supervised scope: 1950–2024 | Supervised scope: 1994–2024 |
| Split: "80th-percentile date" | Split: season-aligned end-of-2018 |
| Final model on train_core only | Final sklearn models on train_val_pool; CNN stays on train_core |
| Plain `.shift(1).rolling()` without groupby | `.groupby('driverId')[...].transform(lambda s: s.shift(1).rolling(...))` |
| Standalone constructor cumsum on driver rows | Race-level constructor agg then join back to driver rows |
| Sprint mask: 2021–2022 only | Sprint mask: all raceIds in sprint_results.csv (2021–2024, 18 races) |
| SVM threshold: [0.1, 0.9] sweep | SVM threshold: quantile sweep of decision_function scores |
| Teammate: pairwise join to single teammate | Teammate: mean of ALL other drivers on same constructor in same race |
