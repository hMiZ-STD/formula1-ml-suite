## 2026-10-29T14:52:48Z
You are Explorer 1 (teamwork_spec_miner).
Your working directory is: d:\ML-Project\.agents\teamwork\explorer_survey_1\
You must read and strictly adhere to:
- d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md
- Reference plan: C:/Users/hamiz/.gemini/antigravity/brain/20fe5515-5671-4c16-be43-e50f0230ca78/plan_ltr_and_hurdle_architecture.md

Your task:
1. Thoroughly read and analyze the reference plan file (C:/Users/hamiz/.gemini/antigravity/brain/20fe5515-5671-4c16-be43-e50f0230ca78/plan_ltr_and_hurdle_architecture.md).
2. Examine `pyproject.toml`, git status/branches in `d:\ML-Project`.
3. Extract and document all specifications for:
   - R1: Environment, git branch `feat/ltr-and-hurdle-architecture`, `lightgbm>=4.0.0` dependency.
   - R2: Race-level ranking metrics (NDCG@1, NDCG@3, NDCG@10, MRR, Kendall's Tau, top-3 Precision/Recall/F1). Detail mathematical formulas, edge cases (ties, DNFs, empty sets).
   - R3: LTR Architecture in `src/models/ltr_ranker.py`: grouped race dataset format, relevance target mappings (`f1_points`, `linear_rank`, `podium_tier`), LambdaMART (`lightgbm.LGBMRanker`), Plackett-Luce ListMLE in PyTorch, `predict_podium_top3` contract (strictly 3 unique drivers).
   - R4: Two-Stage Hurdle Architecture in `src/models/hurdle_model.py`: Stage 1 DNF hazard classifier P(is_dnf=1|X), Stage 2 classified pace ranker/regressor, total probability combination formula, top-3 selection.
   - R5: Benchmark orchestration (`src/run_v6_experiments.py`), `reports/tables/table8_ranking_hurdle_benchmark.csv`, Markdown table format, `EXPERIMENT_FREEZE.yaml` updates.
   - R6: Visualizations (`plot_fig12_ranking_and_hurdle_eval()`, 4-panel 300 DPI layout in `reports/figures/fig12_ranking_and_hurdle_eval.png`).
   - R7: Tests to write (`test_ranking_metrics.py`, `test_ltr_ranker.py`, `test_hurdle_model.py`).
   - R8: Documentation, child-friendly ELI5 analogies, type hints, docstrings, project walkthrough updates.
4. Write a comprehensive, structured report in:
   `d:\ML-Project\.agents\teamwork\explorer_survey_1\handoff.md`
5. Keep your `progress.md` updated in your working directory.
6. When complete, send a message to parent reporting completion and summary.
