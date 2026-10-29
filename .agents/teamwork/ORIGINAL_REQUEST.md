# Original User Request

## Initial Request — 2026-10-29T14:51:19Z

Implement the Learning-to-Rank (LTR) and Two-Stage Hurdle Architecture in the Formula 1 Machine Learning Engineering Suite for production, comparing before and after results side by side with neat comments, child-friendly analogies, and 100% backward compatibility with v5.0.

Working directory: d:\ML-Project
Integrity mode: development

Reference: C:/Users/hamiz/.gemini/antigravity/brain/20fe5515-5671-4c16-be43-e50f0230ca78/plan_ltr_and_hurdle_architecture.md

## Requirements

### R1. Environment and Version Control
Create and checkout the Git branch `feat/ltr-and-hurdle-architecture`. Add `"lightgbm>=4.0.0"` to `pyproject.toml` and install it, ensuring zero breaking changes to the existing v5.0 baseline.

### R2. Race-Aligned Ranking Metrics
Implement `src/evaluation/ranking_metrics.py` calculating race-level NDCG@1, NDCG@3, NDCG@10, Mean Reciprocal Rank (MRR), Kendall's Tau, and top-3 podium Precision, Recall, and F1 across races.

### R3. Learning-to-Rank (LTR) Architecture
Create `src/models/ltr_ranker.py` providing:
- Grouped race dataset assembly with configurable relevance targets (`f1_points`, `linear_rank`, `podium_tier`), benchmarking all three mappings on validation data and freezing the winner in `EXPERIMENT_FREEZE.yaml`.
- LambdaMART ranker wrapping `lightgbm.LGBMRanker` (`objective='lambdarank'`) with validation early stopping.
- Plackett-Luce ListMLE neural ranker implemented in PyTorch.
- `predict_podium_top3` method strictly guaranteeing exactly 3 podium finishers per race with no duplicates.

### R4. Two-Stage Hurdle Architecture
Create `src/models/hurdle_model.py` decoupling reliability hazard from pure on-track pace:
- Stage 1: DNF hazard classifier predicting P(is_dnf = 1 | X).
- Stage 2: Classified pace ranker/regressor trained strictly on classified starters (`is_classified == 1`).
- Total probability formulation combining Stage 1 and Stage 2 for expected finishing position and podium probability.
- Top-3 guaranteed podium selection.

### R5. Experiment Orchestration and Side-by-Side Comparison
Create `src/run_v6_experiments.py` evaluating:
- Baselines (Constructor Rank, Grid Heuristic).
- Pointwise v5.0 models (KNN, Decision Tree, SVM, Dual-Branch 1D-CNN with post-hoc `rank() <= 3`).
- LambdaMART LTR and Plackett-Luce neural ranker.
- Two-Stage Hurdle architecture.
Produce `reports/tables/table8_ranking_hurdle_benchmark.csv` and a formatted Markdown table comparing before vs after results side-by-side. Update `EXPERIMENT_FREEZE.yaml` under the v6 section.

### R6. Diagnostic Visualizations
Implement `plot_fig12_ranking_and_hurdle_eval()` in `src/visualization/plots.py` to generate `reports/figures/fig12_ranking_and_hurdle_eval.png` (300 DPI, 4 panels: NDCG@K comparison, Hurdle DNF Brier calibration curve, Podium F1 comparison, and regression residual comparison).

### R7. Test Suite Expansion
Implement automated tests in `tests/test_ranking_metrics.py`, `tests/test_ltr_ranker.py`, and `tests/test_hurdle_model.py`. Ensure 100% of all existing and new tests pass.

### R8. Code Quality and Documentation
Use clear, intuitive comments with child-friendly ELI5 analogies, strict type hints, and docstrings across all new modules. Update the project walkthrough artifact upon completion.

## Acceptance Criteria

### Automated Tests
- [ ] `pytest tests/` passes 100% with all existing tests intact and new test coverage for ranking metrics, LTR, and Hurdle models.

### Output Artifacts
- [ ] `reports/tables/table8_ranking_hurdle_benchmark.csv` and Markdown table are produced containing side-by-side comparison across baselines, pointwise models, LTR models, and Hurdle models.
- [ ] `reports/figures/fig12_ranking_and_hurdle_eval.png` is generated at 300 DPI with 4 diagnostic panels.
- [ ] `EXPERIMENT_FREEZE.yaml` is updated with v6 ranking and hurdle configurations and metrics.
- [ ] `predict_podium_top3` returns exactly 3 podium finishers for every test race with zero ties or duplicate drivers.
- [ ] Git working tree has all changes committed on branch `feat/ltr-and-hurdle-architecture`.
