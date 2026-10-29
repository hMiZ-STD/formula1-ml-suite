# BRIEFING — 2026-10-29T15:02:00Z

## Mission
Deeply inspect existing ML codebase in `d:\ML-Project` (data loading/splitting, race identifiers, finishes/DNF tracking, baseline and pointwise v5.0 models, EXPERIMENT_FREEZE.yaml, visualization conventions, and shared interfaces) to support the v6 LTR and Hurdle Architecture design.

## 🔒 My Identity
- Archetype: explorer
- Roles: teamwork_explorer
- Working directory: d:\ML-Project\.agents\teamwork\explorer_survey_2\
- Original parent: baa442d4-0226-4bdc-8a4b-922c0f4e6855
- Milestone: codebase_exploration_and_survey

## 🔒 Key Constraints
- Read-only investigation — do NOT modify source code in src/, tests/, or config files
- Strict 100% backward compatibility with v5.0
- Keep progress.md updated
- Produce comprehensive handoff.md following 5-component protocol

## Current Parent
- Conversation ID: baa442d4-0226-4bdc-8a4b-922c0f4e6855
- Updated: 2026-10-29T15:02:00Z

## Investigation State
- **Explored paths**:
  - `data/processed/train_core.parquet` (8,920 rows, 408 races, 1994–2016)
  - `data/processed/val.parquet` (820 rows, 41 races, 2017–2018)
  - `data/processed/test.parquet` (2,559 rows, 128 races, 2019–2024)
  - `src/data/loader.py`, `src/data/preprocessor.py`, `src/features/builder.py`
  - `src/models/classifiers.py`, `src/models/cnn_classifier.py`, `src/models/regularized_reg.py`
  - `src/evaluation/classification_metrics.py`, `src/evaluation/regression_metrics.py`, `src/evaluation/standard_datasets.py`
  - `src/run_experiments.py`, `src/visualization/plots.py`, `EXPERIMENT_FREEZE.yaml`
  - `tests/` (23 passing tests confirmed via pytest)
- **Key findings**:
  - Parquet rows are pre-sorted chronologically: `["year", "round", "positionOrder"]`.
  - Every race forms a contiguous block of rows with `positionOrder` monotonically increasing from 1 to field size (18-24).
  - Race groups for LightGBM LGBMRanker can be extracted directly via `df.groupby("raceId", sort=False).size().to_numpy()`.
  - DNF rate shifts dramatically across eras: 31.98% in train_core, 21.95% in val, 14.46% in test; mean DNF positionOrder is ~18.0.
  - Exactly 3 podium finishers exist per race (`y_podium == 1`). Pointwise static thresholding yields variable podium counts; post-hoc `rank() <= 3` forces 3 podiums and achieves F1 = 0.6641 (KNN), 0.6849 (DT), 0.6562 (SVM).
  - Visualization requires 300 DPI, seaborn "whitegrid" theme, saving to `reports/figures/`.
  - Zero-breaking-change rule requires keeping all v5.0 files intact and adding modular v6 files.
- **Unexplored areas**: None within the exploration scope.

## Key Decisions Made
- All evidence compiled with exact file paths, line numbers, and verification commands.
- Synthesize all findings into structured `handoff.md` with 5 required protocol components.

## Artifact Index
- `DISPATCH.md` — Initial dispatch prompt
- `BRIEFING.md` — Persistent situational awareness
- `progress.md` — Liveness heartbeat and milestone checklist
- `handoff.md` — Comprehensive survey report
