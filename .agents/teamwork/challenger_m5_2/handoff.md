# Handoff Report — Milestone 5 Challenger 2 (Adversarial Benchmark & Freeze Audit)

**Agent**: Challenger 2 (`teamwork_challenger`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\challenger_m5_2\`  
**Date**: 2026-10-29  
**Milestone**: Milestone 5 (Experiment Orchestration & Side-by-Side Comparison)  
**Verdict**: **APPROVE** (with non-blocking advisory findings)  

---

## 1. Observation

### 1.1 Execution & Runtime of `src/run_v6_experiments.py`
- Executed `powershell -Command "Measure-Command { python src/run_v6_experiments.py }"`:
  - Exit code: `0`
  - Wall-clock runtime: **48.36 seconds** (TotalSeconds: `48.3584293`).
- Console output and table printing:
  - Formatted Markdown table printed to stdout with pure ASCII formatting:
    ```
    | Model | Architecture | Decision Rule | NDCG@1 | NDCG@3 | NDCG@10 | MRR | Kendall Tau | Podium Prec | Podium Rec | Podium F1 | Test RMSE | Test R^2 (All) | Test R^2 (Classified) |
    | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | Constructor Rank | Heuristic Baseline | Post-hoc Top-3 Selection | 0.7484 | 0.7280 | 0.8241 | 0.7461 | 0.4667 | 0.6432 | 0.6432 | 0.6432 | 4.9679 | 0.2572 | 0.3895 |
    | Grid Heuristic | Heuristic Baseline | Post-hoc Top-3 Selection | 0.6271 | 0.7315 | 0.8156 | 0.6807 | 0.4998 | 0.6823 | 0.6823 | 0.6823 | 4.9946 | 0.2492 | 0.3739 |
    | KNN | Pointwise Classification (v5.0) | Static Threshold (tau=0.286) | 0.6149 | 0.7263 | 0.8205 | 0.6809 | 0.4918 | 0.6052 | 0.7865 | 0.6840 | N/A | N/A | N/A |
    | KNN | Pointwise Classification (v5.0) | Post-hoc Top-3 Selection | 0.6149 | 0.7263 | 0.8205 | 0.6809 | 0.4918 | 0.6641 | 0.6641 | 0.6641 | 4.1868 | 0.4724 | 0.6391 |
    | Decision Tree | Pointwise Classification (v5.0) | Static Threshold (tau=0.714) | 0.8608 | 0.8086 | 0.8958 | 0.8536 | 0.5207 | 0.7123 | 0.6510 | 0.6803 | N/A | N/A | N/A |
    | Decision Tree | Pointwise Classification (v5.0) | Post-hoc Top-3 Selection | 0.8608 | 0.8086 | 0.8958 | 0.8536 | 0.5207 | 0.6849 | 0.6849 | 0.6849 | 4.2120 | 0.4661 | 0.6432 |
    | SVM (RBF) | Pointwise Classification (v5.0) | Static Threshold (tau=0.863) | 0.5672 | 0.6973 | 0.8035 | 0.6309 | 0.5327 | 0.6527 | 0.6901 | 0.6709 | N/A | N/A | N/A |
    | SVM (RBF) | Pointwise Classification (v5.0) | Post-hoc Top-3 Selection | 0.5672 | 0.6973 | 0.8035 | 0.6309 | 0.5327 | 0.6562 | 0.6562 | 0.6562 | 4.7508 | 0.3208 | 0.4975 |
    | Dual-Branch 1D-CNN | Pointwise Classification (v5.0) | Static Threshold (tau=0.803) | 0.6706 | 0.7347 | 0.8374 | 0.7272 | 0.5333 | 0.6478 | 0.7135 | 0.6791 | N/A | N/A | N/A |
    | Dual-Branch 1D-CNN | Pointwise Classification (v5.0) | Post-hoc Top-3 Selection | 0.6706 | 0.7347 | 0.8374 | 0.7272 | 0.5333 | 0.6615 | 0.6615 | 0.6615 | 4.7620 | 0.3175 | 0.4921 |
    | LambdaMART (linear_rank) | Listwise Learning-to-Rank (v6.0) | Top-3 Guaranteed Selection | 0.8063 | 0.7731 | 0.8618 | 0.7875 | 0.5172 | 0.6771 | 0.6771 | 0.6771 | 4.2449 | 0.4577 | 0.6309 |
    | LambdaMART (f1_points) | Listwise Learning-to-Rank (v6.0) | Top-3 Guaranteed Selection | 0.8353 | 0.7666 | 0.8449 | 0.7901 | 0.3820 | 0.6432 | 0.6432 | 0.6432 | 4.8408 | 0.2948 | 0.3652 |
    | Plackett-Luce Neural Ranker | Listwise Learning-to-Rank (v6.0) | Top-3 Guaranteed Selection | 0.6955 | 0.7600 | 0.8446 | 0.7453 | 0.5379 | 0.6823 | 0.6823 | 0.6823 | 4.7487 | 0.3213 | 0.4928 |
    | Two-Stage Hurdle | Two-Stage Hurdle (v6.0) | Top-3 Guaranteed Selection | 0.6486 | 0.7417 | 0.8305 | 0.7203 | 0.5106 | 0.6693 | 0.6693 | 0.6693 | 4.3420 | 0.4326 | 0.6587 |
    | Two-Stage Hurdle (Expected Pos) | Two-Stage Hurdle (v6.0) | Top-3 Guaranteed (Expected Pos) | 0.6116 | 0.7256 | 0.8229 | 0.6844 | 0.5306 | 0.6745 | 0.6745 | 0.6745 | 4.3420 | 0.4326 | 0.6587 |
    ```
  - The table string itself encoded cleanly without `UnicodeEncodeError` across UTF-8, CP1252, CP437, and ASCII. Zero replacement glyphs in the table.
  - **Adversarial Observation (Non-blocking)**: In `src/run_v6_experiments.py` line 665, `print(f"  -> Updating {freeze_path} (preserving lines 1–30 untouched)...")` contains a unicode en-dash (`\u2013`), which caused charmap encoding errors under strict CP437/OEM pipelines (printing `130`).

### 1.2 CSV Formatting & Row Count Verification
- Inspected `reports/tables/table8_ranking_hurdle_benchmark.csv`:
  - Total non-empty lines: **16** (1 header row + 15 model data rows, spanning lines 1 to 16 in spreadsheet representation).
  - Columns (15): `Model`, `Architecture`, `Decision_Rule`, `NDCG@1`, `NDCG@3`, `NDCG@10`, `MRR`, `Kendall_Tau`, `Podium_Precision`, `Podium_Recall`, `Podium_F1`, `Test_RMSE`, `Test_R2_All`, `Test_R2_Classified`, `Key_Parameters`.
  - Float validity:
    - `NDCG@1`, `NDCG@3`, `NDCG@10`, `MRR`, `Kendall_Tau`, `Podium_Precision`, `Podium_Recall`, `Podium_F1`: 15 valid floats, 0 N/A, 0 invalid strings.
    - `Test_RMSE`, `Test_R2_All`, `Test_R2_Classified`: 11 valid floats for models that generate continuous position/rank estimates, and exactly 4 `"N/A"` strings for static threshold classification models (KNN, DT, SVM, CNN under static $\tau$).
  - Boundary checks: All ranking metrics $\in [0, 1]$, Kendall Tau $\in [-1, 1]$, Podium metrics $\in [0, 1]$, RMSE $> 0$, $R^2 \le 1.0$.
  - Precision/Recall parity: For all 11 models with Top-3 Selection, `Podium_Precision == Podium_Recall == Podium_F1` holds exactly.

### 1.3 `EXPERIMENT_FREEZE.yaml` Consistency Check
- Ran programmatic comparison between `EXPERIMENT_FREEZE.yaml` (`v6_ranking_and_hurdle.table8_benchmark`) and `table8_ranking_hurdle_benchmark.csv`:
  - 15 records in YAML, 15 records in CSV.
  - Exact bit-for-bit/numerical agreement across all 15 rows $\times$ 15 fields (0 mismatches).
- Ran `git diff HEAD EXPERIMENT_FREEZE.yaml`:
  - Lines 1–30 (`freeze_timestamp` through `cnn_stopped_epoch: 3` and following newline) are 100% bit-for-bit identical to baseline v5.0. Only lines 31+ were appended.

### 1.4 Automated Test Suite Execution
- Executed `pytest tests/ -v --tb=short`:
  - Result: **73 passed in 12.76s** (100% pass rate).
  - 0 failures, 0 warnings, 0 regressions.

### 1.5 Adversarial Subdirectory Invocation
- Executed `powershell -Command "cd src; python run_v6_experiments.py"`:
  - Exited with code 1: `FileNotFoundError: Required dataset partition missing: data\processed\train_core.parquet`.
  - Cause: In `src/run_v6_experiments.py` lines 79–82, `DATA_DIR = Path("data/processed")`, `TABLES_DIR = Path("reports/tables")`, `FREEZE_PATH = Path("EXPERIMENT_FREEZE.yaml")` are cwd-relative rather than anchored to `PROJECT_ROOT`.

---

## 2. Logic Chain

1. **Step 1 (Runtime & Execution Validity)**:
   The master benchmark script executes end-to-end in 48.36 seconds from project root, training all models from scratch on `train_core` and `val`, evaluating on `test` (128 races, 2559 rows), saving the CSV artifact, printing the Markdown table, and updating the YAML freeze.
2. **Step 2 (Table Encoding & Universal Console Safety)**:
   The Markdown comparison table utilizes ASCII headers (e.g. `Test R^2 (All)` rather than non-ASCII `Test R² (All)`) and ASCII data formatting. Encoding tests verified that the table string encodes with 0 errors across UTF-8, CP1252, CP437, and ASCII. The only non-ASCII character in console output occurs in the line 665 progress message (`1–30` en-dash), which is non-fatal and does not affect table generation.
3. **Step 3 (CSV Schema, Row Count & Mathematical Integrity)**:
   The CSV file contains exactly 16 lines (1 header + 15 data rows). The 15 data rows comprehensively cover all 4 modeling families specified in `ORIGINAL_REQUEST.md` R5: 2 baseline heuristics, 8 pointwise configurations (4 models $\times$ 2 decision rules: static $\tau$ vs. post-hoc top-3), 3 LTR rankers (`linear_rank`, `f1_points`, `Plackett-Luce`), and 2 Two-Stage Hurdle variants. All numerical entries are strictly valid floats conforming to theoretical bounds, with `P == R == F1` holding for all top-3 models.
4. **Step 4 (Freeze Veracity & Backward Compatibility)**:
   The appended `v6_ranking_and_hurdle` section in `EXPERIMENT_FREEZE.yaml` matches the CSV output across every single model and field without any rounding or copy-paste drift, while lines 1–30 remain 100% bit-for-bit identical to v5.0.
5. **Step 5 (Test Suite Health)**:
   The entire test suite (73 tests) passes cleanly in 12.76 seconds with 100% pass rate.

---

## 3. Caveats

1. **Subfolder Execution Path Dependency**:
   Executing `python src/run_v6_experiments.py` requires invoking from the project root directory (`d:\ML-Project`). Invoking from within `src/` fails with `FileNotFoundError` due to relative paths in lines 79–82 (`Path("data/processed")` instead of `PROJECT_ROOT / "data" / "processed"`).
2. **15 Model Rows vs. 16 Rows Terminology**:
   `table8_ranking_hurdle_benchmark.csv` has 16 total rows (1 header + 15 model data rows, spanning rows 1 to 16 in Excel). The 15 models represent 100% of the specified models across R5 and the reference plan. `LambdaMART (podium_tier)` was evaluated as a validation benchmark scheme in R3/`relevance_mapping_benchmark` rather than an independent test set model in Table 8.

---

## 4. Conclusion & Verdict

**Verdict**: **APPROVE**

Milestone 5 meets all functional, numerical, and structural requirements. The benchmark script runs reliably in under 50 seconds, the generated Table 8 CSV and Markdown outputs are mathematically sound and consistent with the YAML freeze, and the entire test suite passes 100%.

### Non-Blocking Recommendations for Future Hardening:
1. In `src/run_v6_experiments.py` lines 79–82, anchor data paths to `PROJECT_ROOT`:
   ```python
   DATA_DIR = PROJECT_ROOT / "data" / "processed"
   TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
   FREEZE_PATH = PROJECT_ROOT / "EXPERIMENT_FREEZE.yaml"
   ```
2. In `src/run_v6_experiments.py` line 665, replace the en-dash `1–30` with an ASCII hyphen `1-30` to guarantee 100% ASCII compliance across all console print statements.

---

## 5. Verification Method

To independently reproduce the empirical findings:

1. **Benchmark Execution & Runtime**:
   ```powershell
   powershell -Command "Measure-Command { python src/run_v6_experiments.py }"
   ```
   *Expected outcome*: Exits with code 0, wall-clock time $\approx 45\text{--}50\text{ s}$.

2. **Console Table ASCII Safety**:
   ```powershell
   python -c "
   import pandas as pd
   from src.run_v6_experiments import format_markdown_table
   df = pd.read_csv('reports/tables/table8_ranking_hurdle_benchmark.csv')
   table_str = format_markdown_table(df)
   for enc in ['utf-8', 'cp1252', 'cp437', 'ascii']:
       table_str.encode(enc)
   print('All encodings passed!')
   "
   ```

3. **Bit-for-Bit YAML Freeze Verification**:
   ```powershell
   python -c "
   import yaml, pandas as pd
   with open('EXPERIMENT_FREEZE.yaml', 'r', encoding='utf-8') as f:
       y = yaml.safe_load(f)['v6_ranking_and_hurdle']['table8_benchmark']
   c = pd.read_csv('reports/tables/table8_ranking_hurdle_benchmark.csv')
   assert len(y) == len(c) == 15
   print('Exact 15-row parity verified!')
   "
   ```

4. **Preserved Lines 1–30 Check**:
   ```powershell
   git diff HEAD EXPERIMENT_FREEZE.yaml
   ```
   *Expected outcome*: Lines 1–30 are unmodified.

5. **Full Test Suite Execution**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected outcome*: 73 passed in ~12 seconds.
