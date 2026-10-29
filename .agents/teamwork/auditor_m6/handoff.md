# Milestone 6 Forensic Integrity Audit Report

**Work Product**: Milestone 6 Diagnostic Visualizations (`src/visualization/plots.py`, `reports/figures/fig12_ranking_and_hurdle_eval.png`, `src/run_v6_experiments.py`, `tests/test_plots.py`)  
**Auditor**: Forensic Integrity Auditor (`auditor_m6`)  
**Parent Agent**: `baa442d4-0226-4bdc-8a4b-922c0f4e6855`  
**Date**: 2026-10-29  
**Branch**: `feat/ltr-and-hurdle-architecture`  
**Verdict**: **CLEAN**

---

## Forensic Audit Report Summary

| Audit Dimension | Standard / Invalidation Rule | Observed Result | Verdict |
|---|---|---|---|
| **1. Source Code Logic & Facades** | No static base64 strings, no embedded binary image payloads, no bypass wrappers | `src/visualization/plots.py` (lines 588–972) constructs genuine 4-panel matplotlib / seaborn subplots using live arrays and dynamic metrics | **PASS** |
| **2. Dynamic Artifact Generation** | Figure 12 must be generated dynamically from scratch; must reflect perturbed data | Deleted `fig12_ranking_and_hurdle_eval.png`; recreated via CLI (4800x3600 px, 300 DPI, 1.06 MB). Perturbed input data altered binary output | **PASS** |
| **3. Backward Compatibility & Non-Tampering** | Zero modification/deletion of `plot_fig1` through `plot_fig11`; baseline tests untouched | `git diff main -- src/visualization/plots.py` has 0 changes in lines 40–568; `git diff main -- tests/` is clean | **PASS** |
| **4. Independent Test Execution** | `pytest tests/ -v` must execute and pass 100% without tampering | 76 passed in 52.50s with 0 warnings and 0 failures | **PASS** |

---

## 1. Observation

### Observation 1.1: Genuine Matplotlib & Seaborn Subplot Implementation
In `src/visualization/plots.py` (lines 588–972):
- Line 673: `fig, axes = plt.subplots(2, 2, figsize=(16, 12))` initializes the genuine 2x2 grid.
- **Panel A** (lines 675–716): Iterates over canonical models from `df_table8` (`Grid Heuristic`, `KNN`, `Decision Tree`, `SVM (RBF)`, `Dual-Branch 1D-CNN`, `LambdaMART`, `Plackett-Luce`, `Two-Stage Hurdle`), extracts `NDCG@1`, `NDCG@3`, `NDCG@10`, and plots clustered bars via `ax_a.bar(x_a - w_a, ndcg1_vals, ...)`, `ax_a.bar(x_a, ndcg3_vals, ...)`, `ax_a.bar(x_a + w_a, ndcg10_vals, ...)`.
- **Panel B** (lines 718–768): Dynamically queries `p_dnf = hurdle_model.predict_hazard(test_df)`, computes `brier_score_loss(y_true_dnf, p_dnf)`, calculates quantile calibration bins via `calibration_curve(y_true_dnf, p_dnf, n_bins=8, strategy="quantile")`, plots the diagonal reference `[0.0, 0.45]`, and renders the calibration curve with annotated Brier score (`0.1281`).
- **Panel C** (lines 770–874): Extracts test podium F1 for pointwise static vs. post-hoc top-3 and tournament ranking models from `df_table8`, plots comparative bars, and dynamically computes and annotates the ranking penalty deltas ($\Delta = -0.020$ for KNN, $-0.015$ for SVM, $-0.018$ for 1D-CNN).
- **Panel D** (lines 876–950): Dynamically fits `LinearRegression().fit(x_tr, y_tr)` on `train_core`, evaluates test residuals for unmodeled OLS, Hurdle Expected Position $\mathbb{E}[\text{Pos}]$, and Hurdle Stage 2 Classified Pace ($R^2 = 0.6587$), and plots continuous kernel density estimates using `sns.kdeplot`.
- Lines 960–963: Executes `plt.tight_layout()`, `plt.savefig(output_path, dpi=300)`, and `plt.close()`.
- Lines 969–970: Provides standalone CLI execution: `if __name__ == "__main__": plot_fig12_ranking_and_hurdle_eval()`.

No base64 string decoding, no pre-baked binary image copying, and no mocked return values exist in the file.

### Observation 1.2: Dynamic Artifact Generation & Data Sensitivity
1. **Deletion and Dynamic Recreation**:
   - Deleted `reports/figures/fig12_ranking_and_hurdle_eval.png` (`os.remove(target)` confirmed `not target.exists()`).
   - Invoked `python src/visualization/plots.py`.
   - Tool output:
     ```
     STDOUT: [Figure Produced] Saved Figure 12 diagnostic plot to: reports\figures\fig12_ranking_and_hurdle_eval.png (300 DPI)
     Return code: 0
     Exists after: True
     Format: PNG
     Dimensions: (4800, 3600)
     DPI: (299.9994, 299.9994)
     Bytes: 1063071
     ```
2. **Adversarial Data Sensitivity Verification**:
   - Perturbed Table 8 NDCG values to a constant `0.55` and re-executed `plot_fig12_ranking_and_hurdle_eval(df_table8=t8_mod, output_path=out_test)`.
   - Output comparison:
     ```
     Original size: 1063071 bytes
     Modified size: 1059514 bytes
     Are bytes identical?: False
     Adversarial data sensitivity test: PASSED
     ```
   This confirms empirical data dependency: changing input data directly alters the rendered figure payload.

### Observation 1.3: Git Diff Verification on Baseline Functions and Tests
1. **Existing Plot Functions (`src/visualization/plots.py`)**:
   - Executed `git diff main -- src/visualization/plots.py | Select-String -Pattern "@@"`.
   - Exactly 2 hunks detected:
     - Hunk 1 (`@@ -19,14 +19,29 @@`): Added top-level project root path configuration and standard sklearn metric imports (`calibration_curve`, `LinearRegression`, `brier_score_loss`, `r2_score`).
     - Hunk 2 (`@@ -568,3 +583,389 @@`): Appended `plot_fig12_ranking_and_hurdle_eval` and CLI execution block.
   - Zero changes, zero deletions, and zero regressions between lines 40 and 568 containing `plot_fig1_confusion_matrices` through `plot_fig11_pca_3d`.
2. **Baseline Tests (`tests/`)**:
   - Executed `git diff main -- tests/`. Output was empty (0 modifications to baseline tests).
   - `git ls-tree main tests/` shows 7 files: `test_grid_encoding.py`, `test_kmeans_scratch.py`, `test_linear_gd.py`, `test_pca_scratch.py`, `test_pipeline_integrity.py`, `test_qualifying_parser.py`, `test_race_folds.py`. All 7 are byte-for-byte identical to `main`.
   - Milestone 6 added non-intrusive tests in `tests/test_plots.py` verifying Figure 12 resolution, DPI, explicit arguments, and backward compatibility.

### Observation 1.4: Independent Test Execution
- Executed `pytest tests/ -v --tb=short` independently in a clean environment session.
- Output:
  ```
  collected 76 items
  ...
  ============================= 76 passed in 52.50s =============================
  ```
- 100% pass rate across all 76 unit, integration, and regression tests.

---

## 2. Logic Chain

1. **Premise 1 (Authenticity)**: A visualization module passes forensic integrity checks if and only if it generates figures dynamically from live statistical calculations and raw data arrays, rather than emitting static pre-rendered image payloads or hardcoded values.
   - Observation 1.1 reveals genuine matplotlib calls (`plt.subplots`, `ax.bar`, `ax.plot`, `sns.kdeplot`, `plt.savefig`).
   - Observation 1.2 demonstrates that deleting the figure file results in dynamic re-rendering with genuine execution logs, and perturbing input metrics directly changes the binary output. Therefore, the implementation is authentic and non-fabricated.

2. **Premise 2 (Non-Regression and Backward Compatibility)**: A feature branch satisfies baseline non-tampering requirements if and only if existing baseline assets are neither deleted, degraded, nor bypassed.
   - Observation 1.3 shows that all 11 existing figure generation functions (`plot_fig1` through `plot_fig11`) and all 7 baseline test suites from `main` remain completely unmodified.
   - Observation 1.4 proves that all baseline pipeline and model tests continue to pass 100%.

3. **Premise 3 (Orchestration Integration)**: Figure 12 must be integrated into the master benchmark runner without compromising benchmark integrity.
   - Observation 1.1 and inspection of `src/run_v6_experiments.py` confirm that `run_v6_benchmarks()` automatically invokes `plot_fig12_ranking_and_hurdle_eval(df_table8=df_table8, test_df=test, train_df=train_core, val_df=val)` immediately following Table 8 benchmark assembly and `EXPERIMENT_FREEZE.yaml` synchronization.

4. **Conclusion**: All forensic checks pass without exception. The work product is genuine, robust, and clean.

---

## 3. Caveats

- **Figure Rendering Time**: Generating Figure 12 from scratch takes approximately 10–15 seconds because Panel D computes continuous Kernel Density Estimates across test residuals and fits an OLS model, while Panel B evaluates Platt-calibrated DNF hazards. This compute latency is an empirical marker of genuine on-the-fly calculation.
- No other caveats.

---

## 4. Conclusion

**Verdict: CLEAN**

Milestone 6 (Diagnostic Visualizations) has successfully cleared all forensic anti-cheating and integrity gate checks:
1. `src/visualization/plots.py` implements genuine publication-grade matplotlib and seaborn logic across all 4 panels of Figure 12.
2. `reports/figures/fig12_ranking_and_hurdle_eval.png` is generated dynamically at 300 DPI (4800x3600 pixels) and is sensitive to live data inputs.
3. 100% backward compatibility is preserved: zero modifications were made to `plot_fig1` through `plot_fig11` or baseline tests.
4. Independent test execution confirms 76/76 tests passing cleanly.

Milestone 6 is approved for gate progression to Milestone 7.

---

## 5. Verification Method

To independently re-verify this verdict:

1. **Verify Figure 12 Dynamic Recreation**:
   ```powershell
   python -c "
   import os, subprocess
   from pathlib import Path
   from PIL import Image
   fig = Path('reports/figures/fig12_ranking_and_hurdle_eval.png')
   if fig.exists(): os.remove(fig)
   subprocess.run(['python', 'src/visualization/plots.py'], check=True)
   assert fig.exists()
   with Image.open(fig) as im:
       assert im.format == 'PNG'
       assert im.size == (4800, 3600)
       assert int(round(im.info['dpi'][0])) == 300
   print('VERIFICATION: PASS')
   "
   ```

2. **Verify Non-Tampering via Git Diff**:
   ```powershell
   git diff main -- tests/
   git diff main -- src/visualization/plots.py
   ```
   *Expected*: `tests/` diff is empty; `plots.py` diff only shows added imports and `plot_fig12_ranking_and_hurdle_eval` appended at the bottom.

3. **Verify Independent Test Suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected*: `76 passed in <60s`.

*Invalidation conditions*:
- Any static payload, base64 decoding, or bypass of matplotlib/seaborn.
- `reports/figures/fig12_ranking_and_hurdle_eval.png` failing to recreate dynamically upon deletion.
- Any modification or deletion of `plot_fig1` through `plot_fig11` or baseline tests.
- Any failure in `pytest tests/`.
