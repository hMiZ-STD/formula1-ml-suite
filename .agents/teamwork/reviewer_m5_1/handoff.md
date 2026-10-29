# Review & Adversarial Challenge Report — Milestone 5 (Experiment Orchestration & Side-by-Side Comparison)

**Reviewer**: Reviewer 1 (`teamwork_reviewer` / `critic`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\reviewer_m5_1\`  
**Date**: 2026-10-29  
**Milestone**: Milestone 5 — Experiment Orchestration & Side-by-Side Comparison  
**Verdict**: **APPROVE**  
**Overall Risk Assessment**: **LOW**

---

## 1. Observation

1. **Codebase Inspection (`src/run_v6_experiments.py`)**:
   - File length: 874 lines.
   - Pedagogical ELI5 child-friendly analogy present at lines 5–27:
     > "Welcome to the Grand Prix Showdown! This experiment script pits all our racing brains against each other on the 2019-2024 test track: the old pointwise models from v5.0 that graded runners in isolation, versus our new v6.0 tournament-style Learning-to-Rank algorithms and banana-peel-aware Two-Stage Hurdle models. We put their scorecards side by side to see who truly masters Formula 1!"
   - Windows optimization: Line 33 sets `os.environ["LOKY_MAX_CPU_COUNT"] = "4"` to suppress core-counting subprocess warnings.
   - Robust path handling: Lines 39–41 dynamically resolve `PROJECT_ROOT` and inject into `sys.path`.
   - Comprehensive model suite orchestration across all four specified paradigms:
     1. **Baselines**: Constructor Rank Heuristic (`-constructor_season_rank`) and Grid Heuristic (`-grid_effective`).
     2. **Pointwise v5.0 Models**: KNN ($K=30$), Decision Tree ($\text{max\_depth}=4$), SVM (RBF, $C=100.0, \gamma=0.001$), and Dual-Branch 1D-CNN (sequence + static fusion). Each evaluated under two decision policies:
        - Decision Rule A: Static threshold $\tau$ from `EXPERIMENT_FREEZE.yaml` ($\tau_{\text{knn}}=0.286, \tau_{\text{dt}}=0.714, \tau_{\text{svm}}=0.863, \tau_{\text{cnn}}=0.803$).
        - Decision Rule B: Post-hoc top-3 selection (`groupby('raceId')[score].rank(ascending=False, method='first') <= 3`).
     3. **Learning-to-Rank Models**:
        - `LambdaMART (linear_rank)` with validation early stopping.
        - `LambdaMART (f1_points)` with validation early stopping.
        - `Plackett-Luce Neural Ranker` optimizing ListMLE loss.
     4. **Two-Stage Hurdle Architecture**:
        - Stage 1 calibrated DNF hazard model + Stage 2 classified pace model.
        - Primary evaluation via hurdle podium probability (`score_col="hurdle_podium_prob"`).
        - Variant evaluation via expected finishing position (`score_col="hurdle_neg_exp_pos"`).
   - Dynamic metric calculation: Line 173 calls `evaluate_race_ranking()` from `src.evaluation.ranking_metrics`, and lines 182–196 dynamically calculate `precision_score`, `recall_score`, `f1_score`, `root_mean_squared_error`, and `r2_score`. No hardcoded or dummy metrics exist.

2. **Benchmark Execution Probe (`python src/run_v6_experiments.py`)**:
   - Command: `python src/run_v6_experiments.py`
   - Exit Code: `0`
   - Stdout output: Formatted Markdown comparison table printed cleanly.
   - Artifact generated: `reports/tables/table8_ranking_hurdle_benchmark.csv` (17 lines, 15 model rows + 1 header row, 15 columns).

3. **`EXPERIMENT_FREEZE.yaml` Verification**:
   - `git diff EXPERIMENT_FREEZE.yaml` confirmed:
     - Lines 1–30 (`freeze_timestamp` through `cnn_stopped_epoch: 3`) remain 100% bit-for-bit identical to the v5.0 git HEAD commit `f4be121`.
     - Appended `v6_ranking_and_hurdle` section cleanly populates:
       - `lambdamart` (winning mapping `linear_rank`, validation and test metrics).
       - `relevance_mapping_benchmark` (validation metrics for `linear_rank`, `f1_points`, `podium_tier`).
       - `plackett_luce` (neural hyperparameters and test metrics).
       - `hurdle_model` (Stage 1 and Stage 2 estimators, mean DNF position $18.0358$, test metrics).
       - `table8_benchmark` (complete 15-model benchmark records).

4. **Automated Test Suite Verification**:
   - Command: `pytest tests/ -v --tb=short`
   - Result: `73 passed in 27.44s (100% pass rate)` with 0 warnings, 0 failures, 0 regressions.

---

## 2. Logic Chain

1. **Integrity & Authenticity Verification**:
   - Direct inspection of `src/run_v6_experiments.py` confirmed that all models are trained on genuine training partitions (`train_core` or `train_val_pool`) and evaluated on the genuine test partition (2,559 samples across 128 races from 2019 to 2024).
   - There are no hardcoded scores, facade mocks, or shortcuts. All metric evaluations use scikit-learn and project ranking metric routines on live prediction arrays.
2. **Pedagogical and Engineering Standards Verification**:
   - The ELI5 analogy in `src/run_v6_experiments.py` clearly explains the transition from pointwise time-trials in isolation to zero-sum tournament ranking and banana-peel hazard decoupling.
   - Comprehensive NumPy docstrings and strict type annotations (`typing.Any`, `tuple`, `pd.DataFrame`, `np.ndarray`, `float | str`) are present across all 7 functions.
3. **Architectural Comparison Completeness**:
   - Table 8 captures 15 distinct model/rule combinations across 15 columns:
     - Baselines: 2 rows (Constructor Rank, Grid Heuristic).
     - Pointwise v5.0: 8 rows (KNN, DT, SVM, 1D-CNN under both static $\tau$ and post-hoc `rank() <= 3`).
     - LTR: 3 rows (LambdaMART `linear_rank`, LambdaMART `f1_points`, Plackett-Luce Neural Ranker).
     - Hurdle: 2 rows (Two-Stage Hurdle with podium probability, Two-Stage Hurdle with expected position).
   - This directly demonstrates the core machine learning engineering insight:
     - Static $\tau$ yields high recall at the cost of precision and violates the intra-race rule (predicting variable numbers of podium finishers per race).
     - Post-hoc top-3 on pointwise models suffers from discrete leaf probability ties.
     - LTR and Hurdle models enforce zero-sum competition, guaranteeing exactly 3 unique drivers per race while elevating NDCG and eliminating DNF squared-error tail penalties.
4. **Zero-Breaking-Change & Freeze Preservation**:
   - The update function in `src/run_v6_experiments.py` explicitly slices `original_lines[:30]`, guaranteeing that lines 1–30 of `EXPERIMENT_FREEZE.yaml` are perpetually protected from accidental corruption.

---

## 3. Caveats

- In `update_experiment_freeze`, the sections `lambdamart`, `relevance_mapping_benchmark`, `plackett_luce`, and `hurdle_model` carry the validation-derived frozen benchmark parameters established in Milestones 3 and 4, while `table8_benchmark` carries the dynamic test set results. This dual structure is intended and correct for an experiment freeze record.
- No implementation code was modified by this reviewer agent (read-only compliance).

---

## 4. Conclusion

Milestone 5 is thoroughly implemented, empirically validated, and fully compliant with all project requirements. The code exhibits high pedagogical clarity, strict typing, clean error handling, and complete reproducibility.

**Verdict**: **APPROVE**

---

## 5. Verification Method

To independently verify the Milestone 5 deliverables:
1. Run the master experiment runner:
   ```powershell
   python src/run_v6_experiments.py
   ```
   *Expected result*: Process exits with code 0, prints Table 8 Markdown table, writes `reports/tables/table8_ranking_hurdle_benchmark.csv`, and updates `EXPERIMENT_FREEZE.yaml`.
2. Verify freeze integrity on lines 1–30:
   ```powershell
   git diff EXPERIMENT_FREEZE.yaml
   ```
   *Expected result*: Lines 1–30 are identical to v5.0 git HEAD; diff only begins after line 30.
3. Run test regression suite:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected result*: 73 passed in ~10–25s with 100% pass rate.

---

## 6. Adversarial Challenge & Stress-Testing

### Challenge 1: Intra-Race Zero-Sum Precision/Recall Parity
- **Hypothesis**: In Table 8, why are `Podium_Precision`, `Podium_Recall`, and `Podium_F1` strictly identical for all models evaluated under "Post-hoc Top-3 Selection" or "Top-3 Guaranteed Selection"?
- **Stress Analysis**:
  - Test partition contains exactly 128 races, with exactly 3 ground truth podium finishers per race: Total True Positives in ground truth = $128 \times 3 = 384$.
  - When a model's decision rule strictly selects exactly 3 unique drivers per race, the total predicted positives is also strictly $128 \times 3 = 384$.
  - $\text{Precision} = TP / 384$, $\text{Recall} = TP / 384$, $\text{F1} = 2(P \cdot R)/(P + R) = TP / 384$.
  - Therefore, $P = R = F1$ is a rigorous mathematical identity for fixed-quota top-3 selection.
  - In contrast, under static threshold $\tau$, KNN predicts 499 drivers, yielding $\text{Precision}=0.6052$ and $\text{Recall}=0.7865$ ($\text{F1}=0.6840$).
- **Status**: PASSED (Confirmed mathematical necessity).

### Challenge 2: Test Data Leakage during Pointwise and LTR Training
- **Hypothesis**: Could `evaluate_pointwise_v5`, `evaluate_ltr_models`, or `evaluate_hurdle_model` leak 2019–2024 test data into model fitting?
- **Stress Analysis**:
  - Code inspection shows `train_val_pool` (1994–2018) is exclusively passed to `knn_pipe.fit`, `dt_pipe.fit`, and `svm_pipe.fit`.
  - `train_core` (1994–2016) and `val` (2017–2018) are passed to `train_cnn_classifier`, `LambdaMARTRanker.fit`, `PlackettLuceRanker.fit`, and `F1HurdleModel.fit`.
  - `test` (2019–2024) is strictly used as an evaluation target via `.predict()`, `.predict_proba()`, or `.transform()`.
- **Status**: PASSED (Strict zero temporal leakage).

### Challenge 3: Windows Console Glyph Rendering and Subprocess Stability
- **Hypothesis**: In Windows PowerShell/cmd environments, could unicode symbols (e.g. `²`) or multiprocessing fork issues cause script termination?
- **Stress Analysis**:
  - `os.environ["LOKY_MAX_CPU_COUNT"] = "4"` prevents Loky core-counting crashes.
  - Markdown table uses standard ASCII `Test R^2 (All)` and `Test R^2 (Classified)` with caret notation, preventing code page replacement crashes.
- **Status**: PASSED.
