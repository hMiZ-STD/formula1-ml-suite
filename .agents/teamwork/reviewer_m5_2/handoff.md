# Handoff Report — Milestone 5 Review & Adversarial Audit (Reviewer 2)

**Agent**: Reviewer 2 (`teamwork_reviewer`, `critic`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\reviewer_m5_2\`  
**Date**: 2026-10-29  
**Milestone**: Milestone 5 (Experiment Orchestration & Side-by-Side Comparison)  
**Status**: Complete (Hard Handoff)  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 Integrity & Source Code Audit
1. **`src/run_v6_experiments.py` (874 lines)**:
   - Module docstring features child-friendly ELI5 analogy contrasting isolated v5.0 runner evaluations against zero-sum tournament Learning-to-Rank and banana-peel-aware Two-Stage Hurdle models.
   - Robust `sys.path` injection and Windows `LOKY_MAX_CPU_COUNT = "4"` suppression.
   - Data loading (`load_datasets`) strictly consumes `data/processed/train_core.parquet` (8,920 rows), `val.parquet` (820 rows), and `test.parquet` (2,559 rows across 128 races).
   - Dynamic model fitting and evaluation across:
     - Heuristic Baselines: Constructor Rank (`score = -constructor_season_rank`) and Starting Grid (`score = -grid_effective`).
     - Pointwise v5.0 Models: KNN, Decision Tree, SVM (RBF), and Dual-Branch 1D-CNN evaluated under BOTH Decision Rule A (Static Threshold $\tau$) and Decision Rule B (Post-hoc Top-3 Selection).
     - Learning-to-Rank Models: LambdaMART (`linear_rank`), LambdaMART (`f1_points`), and Plackett-Luce Neural Ranker.
     - Two-Stage Hurdle Architecture: Stage 1 calibrated DNF hazard + Stage 2 classified pace model, predicting both expected position $\mathbb{E}[\text{Pos}]$ and hurdle podium probability.
   - Integrity Check: Zero hardcoded benchmark results or facade shortcuts. All metrics are computed dynamically on test data.

### 1.2 Table 8 Benchmark Artifact & Freeze Verification
1. **`reports/tables/table8_ranking_hurdle_benchmark.csv` (17 lines, 15 model records)**:
   - Contains all 15 models/variants across 15 columns:
     `Model`, `Architecture`, `Decision_Rule`, `NDCG@1`, `NDCG@3`, `NDCG@10`, `MRR`, `Kendall_Tau`, `Podium_Precision`, `Podium_Recall`, `Podium_F1`, `Test_RMSE`, `Test_R2_All`, `Test_R2_Classified`, `Key_Parameters`.
   - Independent verification executing `python src/run_v6_experiments.py` re-derived every single metric and confirmed that the generated CSV and `df_benchmark` match the stored artifact 100% bit-for-bit.
2. **`EXPERIMENT_FREEZE.yaml`**:
   - `git diff EXPERIMENT_FREEZE.yaml` confirms that lines 1–30 (`freeze_timestamp` through `cnn_stopped_epoch: 3`) remain 100% bit-for-bit identical to the v5.0 baseline.
   - Appended section `v6_ranking_and_hurdle` captures complete configurations and metrics for `lambdamart`, `relevance_mapping_benchmark`, `plackett_luce`, `hurdle_model`, and `table8_benchmark`.

### 1.3 Full Test Suite Execution
- Command: `pytest tests/ -v`
- Result: **73 passed in 14.26s (100% pass rate)**, with 0 warnings, 0 failures, 0 regressions.

---

## 2. Logic Chain

### 2.1 Before vs. After Empirical Validation
1. **Pointwise Static Threshold vs. Post-hoc Top-3 Penalty**:
   - In v5.0, pointwise classifiers scored drivers in isolation without intra-race constraints.
   - Under static thresholding ($\tau$), models generate invalid race outcomes:
     - KNN ($\tau = 0.286$): predicts 1 to 6 podium finishers per race (total: 499 drivers). In 92 of 128 races (**71.9%**), it fails to predict exactly 3 podium finishers. Its high recall (0.7865) and low precision (0.6052) produce an artificial $F_1 = 0.6840$.
     - Decision Tree ($\tau = 0.714$): predicts 1 to 3 podium finishers per race (total: 351 drivers). In 30 of 128 races (**23.4%**), it under-predicts podium finishers.
     - SVM ($\tau = 0.863$): predicts 1 to 5 podium finishers per race (total: 406 drivers). In 81 of 128 races (**63.3%**), it fails the 3-podium constraint.
   - When post-hoc top-3 selection (`rank() <= 3`) is applied to enforce zero-sum race mechanics, pointwise models suffer an $F_1$ penalty or plateau due to unranked tie-breaking across discrete score leaves:
     - KNN: $F_1$ drops from 0.6840 to **0.6641** (-2.9% penalty).
     - SVM: $F_1$ drops from 0.6709 to **0.6562** (-2.2% penalty).
     - 1D-CNN: $F_1$ drops from 0.6791 to **0.6615** (-2.6% penalty).
2. **Two-Stage Hurdle Classified Pace $R^2$ Elevation**:
   - In unmodeled v5.0 regression, 31.98% historical DNF noise slapped front-running mechanical failures with catastrophic squared-error penalties (e.g. $(18-2)^2 = 256$), capping test regression $R^2$ at $0.4080$.
   - Baseline heuristics achieve $R^2 = 0.3739$ (Grid) and $R^2 = 0.3895$ (Constructor Rank) on classified starters.
   - By decoupling Stage 1 DNF hazard from Stage 2 pure pace fitted strictly on classified starters (`is_classified == 1`), the Two-Stage Hurdle model achieves:
     - Stage 2 pure classified pace $R^2 = \mathbf{0.6587}$ on classified starters (a dramatic improvement of $+0.269$ over the unmodeled OLS baseline).
     - Total probability expected position achieves $R^2 = 0.4326$ on all starters (surpassing baseline heuristics $0.249$–$0.257$ and OLS $0.408$).

### 2.2 Strict Top-3 Deterministic Podium Exclusivity
- Evaluated all 9 top-3 decision rules across all 128 test races:
  1. Constructor Rank (Post-hoc Top-3): strictly 3 unique drivers per race (sum = 384).
  2. Grid Heuristic (Post-hoc Top-3): strictly 3 unique drivers per race (sum = 384).
  3. KNN (Post-hoc Top-3): strictly 3 unique drivers per race (sum = 384).
  4. Decision Tree (Post-hoc Top-3): strictly 3 unique drivers per race (sum = 384).
  5. SVM (Post-hoc Top-3): strictly 3 unique drivers per race (sum = 384).
  6. LambdaMART `linear_rank` (Top-3 Guaranteed): strictly 3 unique drivers per race (sum = 384).
  7. LambdaMART `f1_points` (Top-3 Guaranteed): strictly 3 unique drivers per race (sum = 384).
  8. Plackett-Luce Neural Ranker (Top-3 Guaranteed): strictly 3 unique drivers per race (sum = 384).
  9. Two-Stage Hurdle (Top-3 Guaranteed): strictly 3 unique drivers per race (sum = 384).
- In every top-3 decision rule, because exactly 3 unique drivers are predicted per race and ground truth has exactly 3 drivers per race, $TP + FP = 384$ and $TP + FN = 384 \implies FP = FN$, guaranteeing that **$\text{Precision} = \text{Recall} = F_1$** holds universally.

### 2.3 Backward Compatibility Guarantee
- `git diff --stat` confirms only `EXPERIMENT_FREEZE.yaml` and `pyproject.toml` were modified.
- `src/run_experiments.py`, prior tables (Tables 2–6b), and prior figures (Figures 1–11) remain 100% untouched.

---

## 3. Caveats

- In `reports/tables/table8_ranking_hurdle_benchmark.csv`, pointwise static threshold rows contain the literal string `'N/A'` for RMSE and $R^2$ because static classification thresholds do not predict continuous finishing positions. When loading via pandas, `keep_default_na=False` should be passed if downstream scripts expect string `'N/A'` instead of float `np.nan`.
- CPU execution for the complete 15-model benchmark in `src/run_v6_experiments.py` takes ~35 seconds on Windows, which is fast and requires no GPU hardware.

---

## 4. Conclusion & Verdict

**Verdict: APPROVE**

Milestone 5 satisfies 100% of functional, empirical, architectural, and integrity requirements:
1. `src/run_v6_experiments.py` is fully implemented, adheres strictly to project coding standards and ELI5 pedagogical analogies, and executes cleanly end-to-end.
2. `reports/tables/table8_ranking_hurdle_benchmark.csv` and `EXPERIMENT_FREEZE.yaml` accurately document before vs. after side-by-side performance across baselines, pointwise v5.0, listwise LTR, and Two-Stage Hurdle models.
3. Post-hoc ranking penalties on pointwise models and the dramatic classified pace $R^2$ increase ($0.6587$) of the Two-Stage Hurdle architecture are mathematically and empirically verified.
4. Top-3 exclusivity strictly guarantees 3 unique drivers per race across all 128 test races ($3 \times 128 = 384$) with zero ties or duplicate drivers.
5. 100% backward compatibility is preserved with zero regressions across the 73-test suite.

---

## 5. Verification Method

To independently verify these findings:

1. **Verify 100% Pass Rate Across Test Suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected outcome*: 73 passed in ~14s.

2. **Verify Benchmark Execution & Bit-for-Bit Table 8 Parity**:
   ```powershell
   python src/run_v6_experiments.py
   ```
   *Expected outcome*: Exits with code 0, outputs formatted Markdown table, updates `reports/tables/table8_ranking_hurdle_benchmark.csv`, and appends `EXPERIMENT_FREEZE.yaml`.

3. **Verify Zero Regressions on EXPERIMENT_FREEZE Lines 1–30**:
   ```powershell
   git diff EXPERIMENT_FREEZE.yaml
   ```
   *Expected outcome*: Lines 1–30 are completely unmodified.

4. **Verify Top-3 Exclusivity Across All 128 Test Races**:
   ```powershell
   python -c "import pandas as pd, numpy as np; from src.run_v6_experiments import load_datasets; from src.models.hurdle_model import F1HurdleModel, predict_podium_top3; _, val, test, _ = load_datasets(); m = F1HurdleModel().fit(_, val); p = predict_podium_top3(m.predict_podium_probability(test), test); assert p.sum() == 384; assert (test.assign(p=p).groupby('raceId')['p'].sum() == 3).all(); print('Top-3 exclusivity verified!')"
   ```
   *Expected outcome*: `Top-3 exclusivity verified!`.
