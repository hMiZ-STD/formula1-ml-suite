# Comprehensive Testing Infrastructure Survey Report

**Author**: Explorer 3 (`teamwork_explorer`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\explorer_survey_3\`  
**Date**: 2026-10-29  
**Reference Request**: `d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md`

---

## 1. Observation

### 1.1 Git Status & Version Control
- **Command executed**: `git status; git branch -a; git tag -l`
- **Current branch**: `main` (tracked to `origin/main`, up to date)
- **Local branches**: `main`, `v1-stable`
- **Remote branches**: `remotes/origin/main`, `remotes/origin/v1-stable`
- **Git tags**: None (`git tag -l` returned 0 tags)
- **Working tree status**: Clean code repository. Only untracked directory is `.agents/` (agent metadata). Zero modified or staged code files.

### 1.2 Python Environment & Dependencies
- **Command executed**: `Get-Command python; python --version; python -c "import sys; print(sys.executable); print(sys.prefix)"`
- **Python Executable**: `C:\python\python.exe`
- **Python Version**: `3.13.7`
- **Virtual Environment**: System/base Python installation (`sys.prefix == C:\python`). There is no `.venv` or `venv` directory inside `d:\ML-Project`.
- **Key Installed Packages** (`python -m pip list`):
  - `torch`: `2.7.0` (PyTorch is installed and available for Plackett-Luce ListMLE ranker in R3)
  - `scikit-learn`: `1.5.2`
  - `pandas`: `2.2.3`
  - `numpy`: `2.2.6`
  - `scipy`: `1.17.1`
  - `pytest`: `9.1.1`
  - `pluggy`: `1.6.0`
  - `pyarrow`: `26.0.0`
  - `seaborn`: `0.13.2`
  - `matplotlib`: `3.10.3`
  - `reportlab`: `5.0.1`
  - `streamlit`: `1.66.1`
  - `pyyaml`: `6.0.3`
- **LightGBM Status**:
  - `python -c "import lightgbm"` returned: `LightGBM not installed: No module named 'lightgbm'`.
  - LightGBM is **NOT** installed in the environment.
  - `pyproject.toml` currently specifies `dependencies` (lines 11–24) without `lightgbm`.
  - Requirement R1 explicitly specifies: *"Create and checkout the Git branch `feat/ltr-and-hurdle-architecture`. Add `"lightgbm>=4.0.0"` to `pyproject.toml` and install it, ensuring zero breaking changes to the existing v5.0 baseline."*

### 1.3 Test Directory Structure & Files
The `tests/` directory contains 7 test modules and 1 bytecode cache directory:
```
d:\ML-Project\tests\
├── __pycache__\
├── test_grid_encoding.py       (41 lines, 3 tests)
├── test_kmeans_scratch.py      (106 lines, 3 tests)
├── test_linear_gd.py           (73 lines, 3 tests)
├── test_pca_scratch.py         (83 lines, 4 tests)
├── test_pipeline_integrity.py  (120 lines, 7 tests)
├── test_qualifying_parser.py   (37 lines, 2 tests)
└── test_race_folds.py          (51 lines, 1 test)
```
- Total test modules: **7**
- Total test functions: **23**
- There is **no `conftest.py`** in `tests/` or in the project root.

### 1.4 Test Configuration & Pytest Settings
In `d:\ML-Project\pyproject.toml` (lines 26–32):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
```
- **Markers**: None declared. Search for `@pytest.mark` returned 0 matches across the repository.
- **Coverage Tooling**: `pytest-cov` is **NOT** installed in pip, and no coverage options (`--cov`, `.coveragerc`) exist.
- **Custom config files**: No `pytest.ini`, `setup.cfg`, or `tox.ini` files exist.

### 1.5 Baseline Test Execution & Performance Metrics
Command executed: `pytest tests/ --durations=10`
- **Results**:
  - **Passed**: 23
  - **Failed**: 0
  - **Errors**: 0
  - **Total execution time**: ~11.05s to 12.74s
  - **Warnings**: 1 warning:
    ```
    tests/test_kmeans_scratch.py::test_kmeans_wcss_parity_with_sklearn
      C:\python\Lib\site-packages\joblib\externals\loky\backend\context.py:136: UserWarning: Could not find the number of physical cores for the following reason:
      [WinError 2] The system cannot find the file specified
      Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use.
        warnings.warn(
    ```
- **Execution Durations (Slowest 10)**:
  1. `2.85s setup tests/test_pipeline_integrity.py::test_whitelist_strict_enforcement` (calls `build_pre_race_features()`)
  2. `2.72s call tests/test_qualifying_parser.py::test_q_gap_pct_non_negative` (calls `build_pre_race_features()`)
  3. `1.17s setup tests/test_pipeline_integrity.py::test_unique_race_driver_modern_era` (calls `load_clean_starters()`)
  4. `0.93s setup tests/test_pipeline_integrity.py::test_raw_null_handling` (calls `load_raw_csvs()`)
  5. `0.76s setup tests/test_grid_encoding.py::test_grid_effective_strictly_positive` (calls `load_clean_starters()`)
  6. `0.18s call tests/test_kmeans_scratch.py::test_kmeans_wcss_parity_with_sklearn`
  7. `0.04s call tests/test_grid_encoding.py::test_pit_lane_start_placed_at_back`
  8. `0.03s call tests/test_pipeline_integrity.py::test_disjoint_temporal_partitions`
  9. `0.02s call tests/test_pipeline_integrity.py::test_constructor_teammate_invariance`
  10. `0.02s call tests/test_kmeans_scratch.py::test_kmeans_label_parity_up_to_permutation`

### 1.6 Existing Fixtures & Mock/Synthetic Race Data Patterns

#### Fixture Scoping
All 5 existing test files that define fixtures use `@pytest.fixture(scope="module")`:
- `tests/test_grid_encoding.py`: line 14 `@pytest.fixture(scope="module") def starters()`
- `tests/test_kmeans_scratch.py`: line 16 `@pytest.fixture(scope="module") def cluster_data()`
- `tests/test_linear_gd.py`: line 19 `@pytest.fixture(scope="module") def synthetic_regression_data()`
- `tests/test_pca_scratch.py`: line 16 `@pytest.fixture(scope="module") def standardized_data()`
- `tests/test_pipeline_integrity.py`: lines 25, 31, 37 `@pytest.fixture(scope="module") def raw_data()`, `starters_data()`, `features_data()`

#### Data Access Modes in Tests
1. **Mode A: Real Data Loaders (I/O & Feature Engineering)**
   - Used in `test_grid_encoding.py`, `test_pipeline_integrity.py`, `test_qualifying_parser.py`.
   - Calls `load_raw_csvs()`, `load_clean_starters()`, and `build_pre_race_features()`.
   - `build_pre_race_features()` processes raw CSVs, performs rolling window calculations, and takes ~2.7s–2.8s per invocation.
   - Alternatively, reading pre-saved parquet datasets (e.g. `pd.read_parquet("data/processed/val.parquet")`) takes only **0.069 seconds** for 820 rows across 41 races.
2. **Mode B: Synthetic/Mock Race Data Generation (In-Memory)**
   - Used in `tests/test_race_folds.py` (lines 17–23):
     ```python
     data = []
     for year in range(2010, 2015):
         for race in range(1, 10):
             for driver in range(1, 21):
                 data.append({"year": year, "raceId": year * 100 + race, "driverId": driver})
     df = pd.DataFrame(data)
     ```
   - This generates a clean, synthetic, multi-season race grid (5 seasons x 9 races x 20 drivers = 900 rows) executing in under 0.01 seconds.
3. **Mode C: Synthetic Mathematical & Statistical Fixtures**
   - `test_kmeans_scratch.py`: 3 2D Gaussian blobs (`np.random.default_rng(42)`).
   - `test_linear_gd.py`: `X = [1, X_raw]`, `y = X @ theta + noise`, `n=500, d=5`.
   - `test_pca_scratch.py`: correlated multivariate Gaussian matrix standardized with `StandardScaler()`.

### 1.7 Race Data Structure in Real Validation Split (`data/processed/val.parquet`)
- **Rows**: 820 rows.
- **Races**: 41 distinct `raceId`s (2017–2018 seasons).
- **Grid size**: Exactly 20 drivers per race (`df.groupby('raceId').size().min() == 20`, `max() == 20`).
- **Target relationships**:
  - `is_classified == 1 - is_dnf` (100% verified across train_core, val, and test).
  - `positionOrder == y_position` (100% verified across starters).
  - Exactly 3 podium finishers per race (`y_podium == 1` exactly 3 times per `raceId`, total 123 in `val.parquet`).
  - DNF rate: 180 / 820 = 21.95%.

---

## 2. Logic Chain

1. **Baseline Health**: The test suite currently passes 100% (23/23 tests) with a single external library warning from Joblib's `loky` backend.
2. **Warning Root Cause**:
   - `joblib.externals.loky` queries Windows system information to detect physical cores.
   - On Windows under certain process environments, it emits a `UserWarning`.
   - In `src/run_experiments.py` line 24, the codebase authors solved this with `os.environ["LOKY_MAX_CPU_COUNT"] = "4"`.
   - However, when pytest runs directly, `os.environ["LOKY_MAX_CPU_COUNT"]` is not set prior to joblib import.
   - Setting `os.environ["LOKY_MAX_CPU_COUNT"] = "4"` in a top-level `tests/conftest.py` will eliminate this warning cleanly without altering library internals.
3. **Performance Bottlenecks in Tests**:
   - The entire 23-test suite takes ~11–12 seconds, of which ~7.5 seconds is spent recomputing raw features (`build_pre_race_features()`, `load_clean_starters()`).
   - Pure algorithmic tests (scratch GD, scratch PCA, scratch K-Means, fold splitting) run in <0.2s total.
   - When developing the 3 new test modules (`test_ranking_metrics.py`, `test_ltr_ranker.py`, `test_hurdle_model.py`), tests must avoid re-running raw CSV pipelines.
   - Fast synthetic race generators (Mode B) and direct loading of `data/processed/val.parquet` (Mode A with parquet, 0.069s) will ensure the expanded test suite stays fast (<15s total).
4. **Missing Dependencies for R1–R8**:
   - `torch 2.7.0` is already installed and verified ready for the Plackett-Luce ListMLE ranker in R3.
   - `lightgbm` is missing in `C:\python\python.exe`.
   - Before implementing R3/R4/R7, `lightgbm>=4.0.0` must be installed via `pip install "lightgbm>=4.0.0"` and added to `pyproject.toml`.
5. **Git Branch & Hygiene**:
   - The repository is on `main`, clean, with no tags and no uncommitted code.
   - Per R1, the implementer must create and checkout `feat/ltr-and-hurdle-architecture` prior to making changes.

---

## 3. Caveats

1. **LightGBM Binary Wheel on Windows**:
   - LightGBM on Windows with Python 3.13 requires a pre-built wheel from PyPI. If installing `lightgbm>=4.0.0`, ensure the environment installs a compatible wheel (or Visual C++ runtime is available).
2. **Coverage Tooling**:
   - `pytest-cov` is not in `pyproject.toml` or installed in `pip`. If coverage metrics are needed, `pytest-cov` would have to be installed; currently standard `pytest tests/` is the authoritative acceptance check.
3. **Module Scope Fixtures**:
   - When creating tests that test model mutations or fit states, fixtures should remain stateless or return fresh model instances, while dataset fixtures can remain `scope="module"` for performance.

---

## 4. Conclusion

The testing infrastructure in `d:\ML-Project` is clean, functional, and well-structured, but requires three specific additions to support the upcoming Learning-to-Rank (LTR) and Two-Stage Hurdle Architecture:
1. **Dependency Installation**: Install `lightgbm>=4.0.0` into `C:\python\python.exe` and append to `pyproject.toml` dependencies.
2. **Test Infrastructure Enhancement**:
   - Add a shared `tests/conftest.py` setting `os.environ["LOKY_MAX_CPU_COUNT"] = "4"` to suppress the Loky warning.
   - In `conftest.py`, define reusable mock race fixtures:
     - `mock_single_race_df`: A clean 20-driver race with realistic `y_position`, `y_podium`, `is_dnf`, and `is_classified`.
     - `mock_multi_race_df`: A multi-race dataset (e.g. 5 races x 20 drivers = 100 rows) for grouping and NDCG aggregation.
     - `val_dataset`: Instant parquet loader fixture loading `data/processed/val.parquet` (`scope="module"`).
3. **Test Suite Expansion Design (R7)**:
   - `tests/test_ranking_metrics.py`: Test NDCG@1, @3, @10, MRR, Kendall's Tau, Podium Precision, Recall, F1 against analytical ground truth (perfect ranking, inverse ranking, tied ranking, multi-race mean).
   - `tests/test_ltr_ranker.py`: Test relevance target mappings (`f1_points`, `linear_rank`, `podium_tier`), LambdaMART ranker group fitting, Plackett-Luce neural ranker loss/gradient convergence, and strict podium top-3 output (guaranteeing exactly 3 non-duplicate drivers per race).
   - `tests/test_hurdle_model.py`: Test Stage 1 DNF hazard calibration/probabilities, Stage 2 classified ranker training on `is_classified == 1`, combined expected position formulation, and guaranteed top-3 podium selection.

---

## 5. Verification Method

### 5.1 Independent Test Suite Verification
Run the following PowerShell command in `d:\ML-Project`:
```powershell
pytest tests/ -v --tb=short --durations=10
```
**Expected Output**:
- 23 passed, 0 failed, 0 errors.
- Wall time between 10s and 14s.

### 5.2 Environment Verification Commands
Run the following command in PowerShell:
```powershell
python -c "import sys, torch, sklearn, pandas, numpy, pytest; print('Python:', sys.version); print('Torch:', torch.__version__); print('Sklearn:', sklearn.__version__); print('Pandas:', pandas.__version__); print('Pytest:', pytest.__version__)"
```
**Expected Output**:
- Python: `3.13.7`
- Torch: `2.7.0`
- Sklearn: `1.5.2`
- Pandas: `2.2.3`
- Pytest: `9.1.1`

### 5.3 LightGBM Verification Command (Post-Install)
```powershell
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
```
**Expected Output**:
- `LightGBM: 4.x.x`

### 5.4 Invalidation Conditions
- If any of the 23 baseline tests fail, the baseline code integrity has been compromised.
- If `pytest tests/` execution time exceeds 25 seconds, expensive re-computation is occurring without fixture caching.
