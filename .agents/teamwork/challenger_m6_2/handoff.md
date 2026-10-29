# Milestone 6 Challenger 2 Audit Report: Execution Stress-Testing & Operational Robustness

**Evaluator**: Challenger 2 (`challenger_m6_2`) — Empirical Challenger & Critic  
**Date**: 2026-10-29  
**Branch**: `feat/ltr-and-hurdle-architecture`  
**Verdict**: **REQUEST_CHANGES**  

---

## 1. Observation

### Observation 1.1: Standalone Invocation Path Failure Across Subdirectories
When invoking `python src/visualization/plots.py` from subdirectories (e.g., `src/visualization` or `src`), execution crashes immediately with an unhandled `FileNotFoundError`:

**Command Executed:**
```powershell
cd src/visualization
python plots.py
```

**Verbatim Traceback:**
```text
Traceback (most recent call last):
  File "D:\ML-Project\src\visualization\plots.py", line 970, in <module>
    plot_fig12_ranking_and_hurdle_eval()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "D:\ML-Project\src\visualization\plots.py", line 648, in plot_fig12_ranking_and_hurdle_eval
    raise FileNotFoundError(f"Missing Table 8 benchmark CSV at {table8_path}")
FileNotFoundError: Missing Table 8 benchmark CSV at reports\tables\table8_ranking_hurdle_benchmark.csv
```

**Code Inspection (`src/visualization/plots.py`):**
- Line 28 defines `PROJECT_ROOT`:
  ```python
  PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
  if str(PROJECT_ROOT) not in sys.path:
      sys.path.insert(0, str(PROJECT_ROOT))
  ```
  However, `PROJECT_ROOT` is ONLY added to `sys.path`. It is never used to anchor filesystem paths!
- Line 46:
  ```python
  FIGURES_DIR = Path("reports/figures")  # Relative to CWD!
  ```
- Lines 646, 652, 658, 664:
  ```python
  table8_path = Path("reports/tables/table8_ranking_hurdle_benchmark.csv")  # Relative to CWD!
  test_path = Path("data/processed/test.parquet")                           # Relative to CWD!
  train_path = Path("data/processed/train_core.parquet")                     # Relative to CWD!
  val_path = Path("data/processed/val.parquet")                             # Relative to CWD!
  ```
- When `plots.py` is invoked from `src/visualization`, `Path("reports/tables/...")` evaluates to `d:\ML-Project\src\visualization\reports\tables\...`, which does not exist.

### Observation 1.2: Matplotlib Headless Backend Not Enforced (`matplotlib.use('Agg')` Missing)
- Inspection of `src/visualization/plots.py` lines 1–60:
  ```python
  import matplotlib.pyplot as plt
  ```
  There is NO call to `matplotlib.use("Agg")` prior to importing `pyplot`.
- Command Executed:
  ```powershell
  python -c "import matplotlib; print(matplotlib.get_backend())"
  ```
  Output:
  ```text
  qtagg
  ```
  The active backend defaults to interactive GUI `qtagg`. While `plt.show()` is not called and `plt.close()` is present on line 963, importing and initializing `qtagg` requires GUI windowing/Qt libraries, risking headless environment/CI pipeline aborts when no display server is present.

### Observation 1.3: Subdirectory Test Suite Failure (`tests/test_plots.py`)
When executing pytest from the `tests/` directory:
```powershell
cd tests
pytest test_plots.py
```
Output:
```text
FAILED test_plots.py::test_plot_fig12_ranking_and_hurdle_eval_default_invocation
FAILED test_plots.py::test_plot_fig12_ranking_and_hurdle_eval_explicit_args
FileNotFoundError: [Errno 2] No such file or directory: 'reports\\tables\\table8_ranking_hurdle_benchmark.csv'
```
Lines 53–57 of `tests/test_plots.py` hardcode relative paths (`Path("reports/tables/table8_ranking_hurdle_benchmark.csv")`, `"data/processed/train_core.parquet"`), breaking any test invocation not originating at repository root.

### Observation 1.4: Experiment Script Subdirectory Failure (`src/run_v6_experiments.py`)
When invoking `python run_v6_experiments.py` from `src/`:
```powershell
cd src
python run_v6_experiments.py
```
Output:
```text
FileNotFoundError: Required dataset partition missing: data\processed\train_core.parquet
```
Lines 79–81 of `src/run_v6_experiments.py` define relative paths:
```python
DATA_DIR = Path("data/processed")
TABLES_DIR = Path("reports/tables")
FREEZE_PATH = Path("EXPERIMENT_FREEZE.yaml")
```
These are not anchored to `PROJECT_ROOT`, failing when invoked from any directory other than root.

### Observation 1.5: Performance & Non-Interactive Execution (From Root)
When executed from the root workspace directory `d:\ML-Project`:
1. Wall-clock runtime:
   - Cold invocation: `11.05s` (Passes `<15s` requirement).
   - Warm repeated runs: `Iteration 1: 6.13s`, `Iteration 2: 5.79s`, `Iteration 3: 5.12s` (Average `~5.68s`).
2. Memory profiling:
   - Peak tracked memory during execution: `119.05 MB` (Lean and bounded).
   - `plt.close()` is verified at line 963; no figure object accumulation or memory growth detected across repeated runs.
   - Zero GUI popups or hangs.
3. Full root test suite:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   Output: `76 passed in 64.33s` (100% pass rate when run from root).

---

## 2. Logic Chain

1. **Requirement 1 Mandate**:
   The dispatch explicitly stated:
   *"1. Test standalone invocation: `python src/visualization/plots.py` when invoked from different working directories (e.g. from root `d:\ML-Project` vs from a subfolder like `src\visualization`). Verify robust dynamic path resolution (`PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent`)."*
2. **Defect Deduction**:
   The worker intended to implement standalone execution by adding:
   `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` at line 28.
   However, the worker omitted anchoring `FIGURES_DIR`, `table8_path`, `test_path`, `train_path`, and `val_path` to `PROJECT_ROOT`. Because Python resolves relative paths against `os.getcwd()` rather than the script directory or project root, any invocation where `os.getcwd() != PROJECT_ROOT` triggers a fatal `FileNotFoundError`.
3. **Headless Robustness Deduction**:
   Requirement 2 required non-blocking backend and zero pop-up GUI hangs. In headless CI/CD containers (e.g., GitHub Actions, Docker without X-server), relying on the default `qtagg` backend can fail with `qt.qpa.xcb: could not connect to display`. Adding `matplotlib.use("Agg")` guarantees non-interactive backend operation.
4. **Conclusion Support**:
   Because standalone invocation outside the root folder is empirically non-operational and directly violates Requirement 1 of the dispatch, the audit verdict must be `REQUEST_CHANGES`.

---

## 3. Caveats

- When invoked strictly from the project root directory (`d:\ML-Project`), `plots.py` executes cleanly in 11.05s, produces a publication-grade 300 DPI figure (4800x3600 px), and passes all 76 tests.
- The failure is strictly isolated to path resolution robustness when invoking modules and tests from outside the repository root.

---

## 4. Conclusion & Required Changes

**Verdict**: **REQUEST_CHANGES**

### Actionable Remediation Required:

1. **In `src/visualization/plots.py`**:
   - Add non-interactive backend configuration before pyplot import:
     ```python
     import matplotlib
     matplotlib.use("Agg")
     import matplotlib.pyplot as plt
     ```
   - Anchor `FIGURES_DIR` to `PROJECT_ROOT`:
     ```python
     FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
     ```
   - In `plot_fig12_ranking_and_hurdle_eval()`, anchor all fallback data paths to `PROJECT_ROOT`:
     ```python
     if df_table8 is None:
         table8_path = PROJECT_ROOT / "reports" / "tables" / "table8_ranking_hurdle_benchmark.csv"
         if not table8_path.exists():
             raise FileNotFoundError(f"Missing Table 8 benchmark CSV at {table8_path}")
         df_table8 = pd.read_csv(table8_path)

     if test_df is None:
         test_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
         if not test_path.exists():
             raise FileNotFoundError(f"Missing test dataset at {test_path}")
         test_df = pd.read_parquet(test_path)

     if train_df is None:
         train_path = PROJECT_ROOT / "data" / "processed" / "train_core.parquet"
         if not train_path.exists():
             raise FileNotFoundError(f"Missing train dataset at {train_path}")
         train_df = pd.read_parquet(train_path)

     if val_df is None:
         val_path = PROJECT_ROOT / "data" / "processed" / "val.parquet"
         if not val_path.exists():
             raise FileNotFoundError(f"Missing validation dataset at {val_path}")
         val_df = pd.read_parquet(val_path)
     ```

2. **In `tests/test_plots.py`**:
   - Anchor paths in `test_plot_fig12_ranking_and_hurdle_eval_explicit_args`:
     ```python
     PROJECT_ROOT = Path(__file__).resolve().parent.parent
     table8_path = PROJECT_ROOT / "reports" / "tables" / "table8_ranking_hurdle_benchmark.csv"
     train_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "train_core.parquet")
     val_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "val.parquet")
     test_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")
     ```

3. **In `src/run_v6_experiments.py`**:
   - Anchor `DATA_DIR`, `TABLES_DIR`, and `FREEZE_PATH` to `PROJECT_ROOT`:
     ```python
     DATA_DIR = PROJECT_ROOT / "data" / "processed"
     TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
     FREEZE_PATH = PROJECT_ROOT / "EXPERIMENT_FREEZE.yaml"
     ```

---

## 5. Verification Method

To verify the remediated code:

1. **Test Standalone Invocation From Subfolder**:
   ```powershell
   cd src/visualization
   python plots.py
   ```
   *Expected result*: Exits with code 0 and logs `[Figure Produced] Saved Figure 12 diagnostic plot to: ... (300 DPI)` without FileNotFoundError.

2. **Test Standalone Invocation From Root**:
   ```powershell
   cd d:\ML-Project
   python src/visualization/plots.py
   ```
   *Expected result*: Exits with code 0 in `<15s`.

3. **Test Pytest From Tests Subfolder**:
   ```powershell
   cd tests
   pytest test_plots.py -v
   ```
   *Expected result*: 3 passed in `<10s`.

4. **Verify Full Root Test Suite**:
   ```powershell
   cd d:\ML-Project
   pytest tests/ -v --tb=short
   ```
   *Expected result*: 76 passed, 0 failures, 0 warnings.
