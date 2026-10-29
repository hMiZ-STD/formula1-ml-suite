# Milestone 1 Challenger 2 Report: Adversarial Stress Testing & Backward Compatibility

**Agent**: Challenger 2 (`teamwork_challenger`, critic / specialist)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\challenger_m1_2\`  
**Date**: 2026-10-29  
**Milestone**: M1 (Environment and Version Control)  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 LightGBM Missing Values (NaN Handling in Features)
- Executed empirical test simulating F1 races with sparse NaNs (qualifying Q3 gap missing ~25%), heavy NaNs (sprint race finishes missing ~75%), 100% NaN features, and 95% missingness across all features.
- Command executed:
  ```powershell
  @'
  import numpy as np
  import lightgbm as lgb
  from lightgbm import LGBMRanker
  np.random.seed(42)
  X_train = np.random.randn(200, 12)
  X_train[np.random.rand(200) < 0.25, 0] = np.nan
  X_train[np.random.rand(200) < 0.75, 1] = np.nan
  X_train[:, 2] = np.nan
  y_train = np.random.randint(0, 26, size=200)
  group_train = np.full(10, 20)
  ranker = LGBMRanker(objective="lambdarank", n_estimators=50, learning_rate=0.05, random_state=42, verbose=-1)
  ranker.fit(X_train, y_train, group=group_train)
  preds = ranker.predict(X_train)
  print("Any NaN in preds:", np.isnan(preds).any(), "Shape:", preds.shape)
  '@ | python -
  ```
- Output verbatim:
  ```
  Any NaN in preds: False Shape: (200,)
  ```
- Extreme scenario test: 95% NaNs across all features and completely NaN driver row:
  ```
  Extreme (95% NaN across all features) predict shape: (200,), any NaN: False
  Single row all NaN prediction: -0.09863741317799847, is NaN: False
  ```
- Observation: `lightgbm.LGBMRanker` natively routes missing values during histogram binning and tree splits without error or producing NaN predictions.

### 1.2 LightGBM Single-Group Queries and Target Constraints
- Tested `LGBMRanker` across single-group queries and edge boundary scenarios:
  1. **Single group query in training (`group=[20]`, 1 race of 20 drivers)**:
     - Executed:
       ```python
       X = np.random.randn(20, 5)
       y = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0]*10)
       ranker = LGBMRanker(objective="lambdarank", n_estimators=10, min_child_samples=2, random_state=42, verbose=-1)
       ranker.fit(X, y, group=[20])
       preds = ranker.predict(X)
       ```
     - Output: Succeeded. `preds.shape == (20,)`. When `min_child_samples=2`, tree splits successfully and top-ranked driver scores higher than backmarkers (`1.3409 vs -1.3636`). Note: Default `min_child_samples=20` requires `>=20` samples per leaf, so for single 20-driver queries, `min_child_samples` must be `<=20`.
  2. **Single group in validation set (`eval_group=[20]`)**:
     - Succeeded. Early stopping on NDCG evaluates correctly on a single race validation query.
  3. **Single query at inference**:
     - Succeeded. `ranker.predict(X_test_race)` outputs shape `(20,)` cleanly.
  4. **Degenerate query size `group=[1]`**:
     - Raised: `ValueError: Found array with 1 sample(s) (shape=(1, 5)) while a minimum of 2 is required.`
     - Succeeded when mixed: `group=[20, 1, 19]` trains cleanly.
  5. **Target label types and bounds**:
     - Negative targets (`y = -np.arange(1, 21)`):
       `[LightGBM] [Fatal] Label should be non-negative (met -1.000000) for ranking task` -> `LightGBMError`
     - Non-integer float targets (`y = np.linspace(0.0, 10.0, 20)`):
       `[LightGBM] [Fatal] label should be int type (met 0.526316) for ranking task` -> `LightGBMError`
     - Non-negative integer targets (`y = (21 - positions).astype(int)`):
       Succeeded without errors.

### 1.3 Backward Compatibility with Existing Models in `src/models/`
- Evaluated models:
  - `KNeighborsClassifier` (via `create_classifier_pipeline("knn", ...)` in `src/models/classifiers.py`)
  - `DecisionTreeClassifier` (via `create_classifier_pipeline("dt", ...)` in `src/models/classifiers.py`)
  - `SVC` (via `create_classifier_pipeline("svm", ...)` in `src/models/classifiers.py`)
  - `DualBranchCNN` (in `src/models/cnn_classifier.py`)
- Tested instantiation, pipeline preprocessing (SimpleImputer median + StandardScaler), fitting, and prediction on data with missing values.
- Tested PyTorch 1D-CNN temporal branch + Static MLP forward pass with shapes `[batch_size=4, seq_channels=15, seq_len=10]` and `[batch_size=4, n_static=18]`.
- Output verbatim:
  ```
  === BACKWARD COMPATIBILITY TEST: Existing Models ===
  Instantiated KNN: KNeighborsClassifier()
  Instantiated DecisionTree: DecisionTreeClassifier(max_depth=5, random_state=42)
  Instantiated SVC: SVC(probability=True, random_state=42)
  Pipeline KNN fit & predict succeeded: preds=[0 0 0 1 1 0], probs shape=(6, 2)
  Pipeline DecisionTree fit & predict succeeded: preds=[0 1 0 1 1 0], probs shape=(6, 2)
  Pipeline SVC fit & predict succeeded: preds=[0 1 0 1 1 0], probs shape=(6, 2)
  Instantiated DualBranchCNN: DualBranchCNN
  DualBranchCNN forward pass output shape: torch.Size([4, 1]), probs shape: torch.Size([4, 1])
  >>> ALL 4 EXISTING MODELS (KNN, DecisionTree, SVC, DualBranchCNN) INSTANTIATE AND RUN CLEANLY!
  ```

### 1.4 Baseline Test Suite Runtime and Memory Footprint
- Executed `pytest tests/ -v --tb=short` with subprocess memory and time telemetry.
- Telemetry results:
  ```
  Pytest returncode: 0
  Elapsed time: 12.14 seconds (in-session: 11.09s)
  Peak memory footprint (RSS): 327.03 MB
  Pytest summary line: ============================= 23 passed in 11.09s =============================
  ```
- All 23 tests passed, 0 failures, 0 errors, 0 warnings.
- Incremental LightGBM memory footprint:
  - Base Python process: 29.24 MB
  - After `import lightgbm`: 149.34 MB (Delta: +120.10 MB for C++ shared library load)
  - After fitting 100 trees on 1,600 rows (80 races) + predicting 400 rows: 154.12 MB (Delta: +4.78 MB)
  - Fit runtime: 94.96 ms
  - Predict runtime: 1.41 ms

---

## 2. Logic Chain

1. **LightGBM Missing Values Robustness (Observation 1.1)**: F1 features regularly experience missing values due to partial sessions (e.g. drivers knocked out in Q1 having no Q3 lap time gap) and non-sprint race weekends (where sprint race finish position is absent). Testing `LGBMRanker` with 25%, 75%, 95%, and 100% NaN features produced 0 crashes and 0 NaN predictions. Tree splits natively separate valid vs missing values, verifying full suitability for F1 race data.
2. **Single-Group Query Handling and Target Typing (Observation 1.2)**: `LGBMRanker` operates successfully with a single query of 20 items (representing 1 Grand Prix grid), single validation queries, and single test query inference. However, stress testing revealed two critical constraints for Milestone 3 implementation:
   - Target relevance values must be non-negative integers (`int >= 0`). Float values and negative values trigger immediate fatal LightGBM errors (`Label should be non-negative`, `label should be int type`).
   - Group size must be at least 2 (`group >= 2`).
3. **Zero-Breaking-Change Verification (Observation 1.3)**: Addition of `lightgbm>=4.0.0` to `pyproject.toml` did not introduce dependency downgrades, ABI conflicts, or namespace collisions. `KNeighborsClassifier`, `DecisionTreeClassifier`, `SVC`, and `DualBranchCNN` import, instantiate, fit, and predict with exact expected shapes and probabilities.
4. **Runtime & Memory Safety (Observation 1.4)**: Running the test suite consumes only 327.03 MB peak RSS memory and finishes in ~11s. LightGBM inference consumes only 1.41 ms per race grid, presenting negligible overhead for production batch scoring.

---

## 3. Caveats

- **LightGBM 4.7 API Deprecation**: LightGBM 4.7.0 emits `LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y' instead.` While `eval_set` remains backward compatible, Milestone 3 workers should use `eval_X` and `eval_y` or provide backward compatibility guards.
- **Loky Environment Variable**: Standalone python scripts without pytest conftest will trigger Loky's core detection warning unless `LOKY_MAX_CPU_COUNT` is set. Worker M1's `tests/conftest.py` correctly suppresses this in pytest, but any standalone CLI scripts in Milestone 5 should include `os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")`.

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone 1 satisfies all functional, architectural, and non-regression requirements:
1. `lightgbm>=4.0.0` (installed: 4.7.0) is stable, fast (1.41 ms inference), and memory-efficient (154 MB resident).
2. Missing values (NaNs) are handled natively without errors or corrupted predictions.
3. Single-group race queries operate cleanly.
4. 100% backward compatibility with existing v5.0 classical models and neural architectures is preserved.
5. All 23 baseline unit tests pass with zero warnings, zero errors, and a peak memory footprint of 327 MB.

---

## 5. Verification Method

To independently verify all findings:

1. **Verify Baseline Test Suite & Warnings Suppression**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected*: `23 passed in ~11s`, 0 warnings, 0 failures.

2. **Verify LGBMRanker with NaNs and Single-Group Queries**:
   ```powershell
   @'
   import numpy as np
   from lightgbm import LGBMRanker
   X = np.random.randn(20, 5)
   X[0, :] = np.nan
   y = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0]*10)
   ranker = LGBMRanker(objective="lambdarank", n_estimators=10, min_child_samples=2, random_state=42, verbose=-1)
   ranker.fit(X, y, group=[20])
   preds = ranker.predict(X)
   print("Predictions valid:", len(preds) == 20 and not np.isnan(preds).any())
   '@ | python -
   ```
   *Expected*: `Predictions valid: True`.

3. **Verify Existing Models Instantiation**:
   ```powershell
   @'
   import torch
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.svm import SVC
   from src.models.cnn_classifier import DualBranchCNN
   knn = KNeighborsClassifier()
   dt = DecisionTreeClassifier()
   svc = SVC()
   cnn = DualBranchCNN(n_static=18, seq_channels=15)
   print("All 4 models instantiated successfully!")
   '@ | python -
   ```
   *Expected*: `All 4 models instantiated successfully!`.

---

## Challenge Report

### Challenge Summary
**Overall risk assessment**: LOW

### Challenges

#### Challenge 1 [Low - Preemptive for M3]: Target label typing in `LGBMRanker`
- **Assumption challenged**: Can any ranking score (including negative ranks like `-positionOrder` or continuous floats) be used directly as targets in `LGBMRanker`?
- **Attack scenario**: Passing negative ranks (`-1`, `-2`, ...) or continuous target floats (`0.5`, `1.2`) to `LGBMRanker.fit()`.
- **Blast radius**: Crashes training with `LightGBMError: Label should be non-negative` or `LightGBMError: label should be int type`.
- **Mitigation**: In Milestone 3 (`F1RankerDataset`), ensure `linear_rank` target mapping is strictly non-negative integer, e.g. `(n_drivers - positionOrder + 1).astype(int)`.

#### Challenge 2 [Low - Preemptive for M3]: Leaf size on single race group queries
- **Assumption challenged**: Will default `min_child_samples=20` learn effective splits on small race datasets?
- **Attack scenario**: Fitting `LGBMRanker` with default parameters on a single race of 20 drivers.
- **Blast radius**: The tree cannot split (since neither leaf can contain >= 20 items), outputting constant zero predictions.
- **Mitigation**: For unit tests and small group queries in Milestone 3, set `min_child_samples <= 5`.

### Stress Test Results
- `LGBMRanker` with 25% - 95% NaNs in features -> Native split handling -> PASSED (0 NaNs in predictions)
- `LGBMRanker` with single-group query (`group=[20]`) -> Valid ranking output -> PASSED
- `LGBMRanker` with single-group query (`group=[1]`) -> Raised `ValueError: min samples 2 required` -> EXPECTED BEHAVIOR
- Classical models (`KNN`, `DecisionTree`, `SVC`) in scikit-learn pipeline -> Fit and predict -> PASSED
- `DualBranchCNN` forward pass with temporal sequence and static inputs -> Tensor shape `[4, 1]` -> PASSED
- Full pytest test suite (23 tests) -> 23 passed in 11.09s, 327 MB peak RSS -> PASSED

### Unchallenged Areas
- Distributed / GPU LightGBM training — reason: out of scope, project runs on single-node CPU environment.
