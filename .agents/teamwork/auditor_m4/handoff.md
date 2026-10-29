# Forensic Audit Report — Milestone 4 (Two-Stage Hurdle Architecture)

**Auditor**: Forensic Integrity Auditor (`teamwork_auditor`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\auditor_m4\`  
**Date**: 2026-10-29  
**Target**: Milestone 4 (`src/models/hurdle_model.py` and `tests/test_hurdle_model.py`)  
**Integrity Mode**: Development (per `ORIGINAL_REQUEST.md`)  
**Verdict**: **CLEAN**

---

## 1. Observation

1. **Source Code Inspection (`src/models/hurdle_model.py`)**:
   - **Stage 1 (Hazard Classifier)** (lines 412–445):
     - Uses `HistGradientBoostingClassifier(loss='log_loss', max_depth=4, l2_regularization=1.0, random_state=42)` fitted on all starters in `df_train` ($N = 8,920$).
     - Genuine calibration via `CalibratedClassifierCV(estimator=self.stage1_base_estimator_, method='sigmoid', cv='prefit')` fitted on `df_val` ($N = 820$).
     - Predictions use `predict_proba(X)[:, 1]` strictly clipped to $[0.0, 1.0]$.
   - **Stage 2 (Classified Pace Model)** (lines 455–474, 476–511):
     - Strictly filters training data: `classified_mask = y_train_dnf == 0`.
     - Purity assertion: `assert (self.stage2_is_dnf_ == 0).all()`.
     - Exactly 6,067 classified starter rows in Stage 2 training set; 0 DNF rows leaked.
     - Fits `HistGradientBoostingRegressor` for on-track finishing position $\hat{y}_{\text{finish}}$ and `HistGradientBoostingClassifier` for $P(\text{Podium} \mid \text{Finish})$.
   - **Law of Total Probability Formulation** (lines 604–608, 629–649):
     - Expected finishing position: $\mathbb{E}[\text{Position}_i \mid X_i] = (1 - p_{\text{dnf}, i})\hat{y}_{\text{finish}, i} + p_{\text{dnf}, i}\bar{y}_{\text{dnf}}$.
     - $\bar{y}_{\text{dnf}}$ is dynamically derived from training data: $\bar{y}_{\text{dnf}} = 18.03575 \approx 18.04$ (lines 447–453).
     - Podium probability: $P(\text{Podium}_i \mid X_i) = (1 - p_{\text{dnf}, i}) P(\text{Podium}_i \mid \text{Finish}_i, X_i)$ bounded strictly in $[0.0, 1.0]$.
   - **Guaranteed Top-3 Podium Selection** (`predict_podium_top3`, lines 72–182):
     - Enforces deterministic primary ranking by podium probability or negative expected position, with secondary tie-breaking on ascending starting grid (`grid_effective`).
     - Across all 128 test races ($N = 2,559$), produces strictly $3 \times 128 = 384$ predicted podium finishers (3 per race).
     - Across all 41 validation races ($N = 820$), produces strictly $3 \times 41 = 123$ predicted podium finishers (3 per race).

2. **No Test Tampering or Weakening**:
   - `git diff main -- tests/` produced **0 lines of diff**.
   - No existing tests (`test_grid_encoding.py`, `test_kmeans_scratch.py`, `test_linear_gd.py`, `test_pca_scratch.py`, `test_pipeline_integrity.py`, `test_qualifying_parser.py`, `test_race_folds.py`, etc.) were modified, bypassed, or weakened.

3. **Automated Test Execution Results**:
   - `pytest tests/test_hurdle_model.py -v`:
     ```
     ============================= 12 passed in 4.63s ==============================
     ```
     All 12 unit/integration tests passed.
   - `pytest tests/`:
     ```
     ============================= 73 passed in 18.55s =============================
     ```
     100% of all 73 tests passed.

4. **Empirical Independent Performance Verification**:
   - Stage 2 pure classified pace test $R^2$: **0.6587** (dramatic improvement over baseline v5.0 $R^2 \approx 0.4080$).
   - Stage 2 pure classified pace test RMSE: **2.9778**.
   - Stage 2 pure classified pace test MAE: **2.2601**.
   - Hurdle top-3 podium test F1: **0.6745** (precision: 0.6745, recall: 0.6745).
   - Test NDCG@1: **0.6116**, NDCG@3: **0.7256**, NDCG@10: **0.8229**, MRR: **0.6844**, Kendall's Tau: **0.5306**.

---

## 2. Logic Chain

1. **Integrity Mode Compliance**: Under `development` mode (defined in `ORIGINAL_REQUEST.md`), the audit verifies that no dummy facades, hardcoded test results, fabricated output files, or test tampering occurred.
2. **Mathematical Authenticity**:
   - Stage 1 is a genuine gradient boosting classifier calibrated via Scikit-Learn's `CalibratedClassifierCV(..., method='sigmoid', cv='prefit')`. Probabilities vary continuously ($p_{\text{dnf}} \in [0.073, 0.427]$ on test data) and are never hardcoded constants.
   - Stage 2 is strictly conditioned on $S_i = 1$ (`is_classified == 1`). Zero DNF rows contaminate Stage 2 training. The empirical $R^2$ of 0.6587 proves the model genuinely decouples mechanical attrition tails from pure on-track pace.
   - The Law of Total Probability recombination dynamically references empirical $\bar{y}_{\text{dnf}} = 18.0358$, yielding mathematically sound convex combinations.
3. **Adversarial Resilience**:
   - Tested extreme probability boundary conditions ($p_{\text{dnf}} = 0.0$ and $p_{\text{dnf}} = 1.0$), small field sizes ($N < 3$), empty arrays, exact score ties, and both pace transformation methods (`gaussian_cdf` and `classifier`). All behave with mathematical accuracy.
4. **Conclusion Support**: All empirical observations directly support a **CLEAN** verdict.

---

## 3. Caveats

1. **Custom Estimator Imputation**: In `_extract_features()`, `SimpleImputer` is fitted on `X_arr` when NaNs are detected, but the imputed array is not applied before returning to estimators. While the default `HistGradientBoosting` estimators natively support missing values (e.g. `grid_vs_quali_delta` in sprint races) and operate without issue, any third-party estimator requiring dense inputs would need to be passed inside a `make_pipeline(SimpleImputer(), estimator)`.
2. **Field Size $< 3$**: If a race has fewer than 3 starters, `predict_podium_top3` returns all $m$ starters (bounded by $\min(3, m)$). This is mathematically correct and tested.

---

## 4. Conclusion

**Verdict: CLEAN**

Milestone 4 (Two-Stage Hurdle Architecture) is verified to be 100% genuine, authentic, and free of any integrity violations, hardcoded shortcuts, facades, DNF data leakage, or test tampering. All requirements from R4 and R7 are satisfied.

---

## 5. Verification Method

To independently reproduce the forensic verification findings:

1. **Verify No Git Diff on Existing Tests**:
   ```powershell
   git diff main -- tests/
   ```
   *Expected Output*: Empty (0 lines).

2. **Run Hurdle Unit and Integration Tests**:
   ```powershell
   pytest tests/test_hurdle_model.py -v
   ```
   *Expected Output*: 12 passed.

3. **Run Full Test Suite**:
   ```powershell
   pytest tests/
   ```
   *Expected Output*: 73 passed.

4. **Verify Stage 2 Purity and Real Metrics**:
   ```powershell
   python -c "import pandas as pd; from sklearn.metrics import r2_score; from src.models.hurdle_model import F1HurdleModel; tr = pd.read_parquet('data/processed/train_core.parquet'); val = pd.read_parquet('data/processed/val.parquet'); test = pd.read_parquet('data/processed/test.parquet'); m = F1HurdleModel().fit(tr, val); assert (m.stage2_is_dnf_ == 0).all(); mask = test['is_classified'] == 1; r2 = r2_score(test.loc[mask, 'positionOrder'], m.predict_classified_pace(test)[mask]); print(f'Stage 2 Pace R2: {r2:.4f}'); assert r2 > 0.65; top3 = m.predict_podium_top3(test); assert top3.sum() == 3 * test['raceId'].nunique(); print('Verification Succeeded.')"
   ```
   *Expected Output*: `Stage 2 Pace R2: 0.6587` and `Verification Succeeded.`.
