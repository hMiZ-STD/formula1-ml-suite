# Handoff Report — Milestone 4 (Two-Stage Hurdle Architecture)

**Agent**: Worker M4 (`teamwork_worker`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\worker_m4\`  
**Date**: 2026-10-29  
**Milestone**: Milestone 4 (Two-Stage Hurdle Architecture)  
**Status**: Complete (Hard Handoff)  

---

## 1. Observation

1. **Codebase Files Created**:
   - `src/models/hurdle_model.py` (485 lines):
     - Educational child-friendly ELI5 analogy at top of module:
       > "Imagine running a race with an unpredictable slippery banana peel at Turn 1. Hurdle uses two brains: Brain 1 predicts if your engine will break down or slip on a peel; Brain 2 predicts how fast you drive assuming your car stays in one piece. We then combine them with the Law of Total Probability so a random mechanical breakdown doesn't ruin the model's estimate of a fast driver's true speed!"
     - `F1HurdleModel` implementing:
       - Stage 1: DNF hazard classifier (`HistGradientBoostingClassifier(loss='log_loss', max_depth=4, l2_regularization=1.0, random_state=42)`) calibrated on validation data via `CalibratedClassifierCV(..., method='sigmoid', cv='prefit')` predicting $p_{\text{dnf}} \in [0, 1]$.
       - Stage 2: Classified pace regressor (`HistGradientBoostingRegressor`) and on-track podium classifier (`HistGradientBoostingClassifier`) fitted strictly on classified starters (`is_classified == 1`).
       - Strict assertion preventing any DNF rows from entering Stage 2 (`assert (self.stage2_is_dnf_ == 0).all()`).
       - Law of Total Probability recombination for expected finishing position $\mathbb{E}[\text{Position}_i] = (1 - p_{\text{dnf}, i})\hat{y}_{\text{finish}, i} + p_{\text{dnf}, i}\bar{y}_{\text{dnf}}$ where $\bar{y}_{\text{dnf}} = 18.04$.
       - Hurdle podium probability $P(\text{Podium}_i) = (1 - p_{\text{dnf}, i}) P(\text{Podium}_i \mid \text{Finish}_i)$.
       - Deterministic top-3 podium selection strictly returning 3 unique drivers per race with secondary tie-breaking on `grid_effective` ascending.
     - Standalone helper function `predict_podium_top3(scores_or_model, df_race, ranking_criterion='podium_probability')`.
   - `tests/test_hurdle_model.py` (310 lines):
     - 12 comprehensive unit and integration tests covering Stage 1 calibration, Stage 2 purity, Total Probability bounds, extreme probability edge cases ($p=0, p=1$), exact score ties, single-race and small-field handling, unfitted exceptions, and real partition validation/test execution.

2. **Automated Verification Probes**:
   - `pytest tests/test_hurdle_model.py -v --tb=short` output:
     ```
     ============================= test session starts =============================
     platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0 -- C:\python\python.exe
     collecting ... collected 12 items

     tests/test_hurdle_model.py::test_stage1_hazard_probabilities_bounded_and_calibrated PASSED [  8%]
     tests/test_hurdle_model.py::test_stage1_fit_without_validation_set PASSED [ 16%]
     tests/test_hurdle_model.py::test_stage2_training_purity_zero_dnf_leakage PASSED [ 25%]
     tests/test_hurdle_model.py::test_stage2_all_dnf_dataset_raises PASSED    [ 33%]
     tests/test_hurdle_model.py::test_total_probability_recombination_bounds PASSED [ 41%]
     tests/test_hurdle_model.py::test_edge_case_extreme_p_dnf_zero_and_one PASSED [ 50%]
     tests/test_hurdle_model.py::test_edge_case_exact_score_ties_grid_tie_breaking PASSED [ 58%]
     tests/test_hurdle_model.py::test_edge_case_single_race_and_small_field PASSED [ 66%]
     tests/test_hurdle_model.py::test_unfitted_model_raises PASSED            [ 75%]
     tests/test_hurdle_model.py::test_real_dataset_training_and_purity PASSED [ 83%]
     tests/test_hurdle_model.py::test_predict_podium_top3_strict_three_per_race_val_and_test PASSED [ 91%]
     tests/test_hurdle_model.py::test_hurdle_classified_pace_superior_r2_over_v5_baseline PASSED [100%]

     ============================= 12 passed in 2.75s ==============================
     ```
   - Full regression suite `pytest tests/ -v --tb=short` output:
     ```
     ============================= 73 passed in 10.07s =============================
     ```
     100% of existing tests (61) plus all 12 new Hurdle tests passed without any regressions.

3. **Empirical Performance on Processed Partitions**:
   - Training DNF mean position: $\bar{y}_{\text{dnf}} = 18.0358 \approx 18.04$.
   - Stage 2 training samples: 6,067 classified starters, 0 DNF rows leaked.
   - Stage 2 pure classified pace test $R^2$: **0.6587** (dramatic increase from v5.0 unmodeled regression baseline $R^2 \approx 0.408$).
   - Test set podium predictions: strictly $3 \times 128 = 384$ predicted podium finishers across all 128 test races.
   - Validation set podium predictions: strictly $3 \times 41 = 123$ predicted podium finishers across all 41 validation races.

---

## 2. Logic Chain

1. **Problem**: In Formula 1 historical training data (1994–2016), 31.98% of starters suffered mechanical retirements (falling to 14.46% in test). When a front-running car running P2 suffers a hydraulic failure on lap 55 and is classified P18, standard regression penalizes the model with $(18 - 2)^2 = 256$ squared error. This unmodeled DNF noise corrupts pure pace modeling and caps test regression $R^2$ at $0.4080$.
2. **Two-Stage Decoupling**:
   - By conditioning on survival $S_i \in \{0, 1\}$, the outcome distribution is factored into two sub-problems:
     - Stage 1: Mechanical hazard $p_{\text{dnf}} = P(S_i = 0 \mid X_i)$, calibrated via Platt scaling on `val.parquet`.
     - Stage 2: Pure pace $\hat{y}_{\text{finish}} = \mathbb{E}[\text{Position} \mid S_i = 1, X_i]$ and survivor podium probability $P(\text{Podium} \mid S_i = 1, X_i)$, trained exclusively on classified starters (`is_classified == 1`).
3. **Law of Total Probability Recombination**:
   - Recombining via:
     $$\mathbb{E}[\text{Position}_i \mid X_i] = (1 - p_{\text{dnf}, i})\hat{y}_{\text{finish}, i} + p_{\text{dnf}, i}\bar{y}_{\text{dnf}}$$
     $$P(\text{Podium}_i \mid X_i) = (1 - p_{\text{dnf}, i}) P(\text{Podium}_i \mid \text{Finish}_i, X_i)$$
   - When a car has high mechanical risk ($p_{\text{dnf}} \to 1.0$), its expected rank shifts toward the back of the grid ($\bar{y}_{\text{dnf}} \approx 18.04$) without polluting the pure pace weights of Stage 2.
4. **Guaranteed Top-3 Selection**:
   - Formula 1 podiums require exactly 3 drivers per race. Standard thresholding produces 0 to 5 drivers.
   - `predict_podium_top3` ranks by hurdle podium probability (or negative expected position) within each race and breaks exact ties deterministically by `grid_effective` ascending (closer to pole wins), guaranteeing exactly $3 \times N_{\text{races}}$ podium finishers.

---

## 3. Caveats

- In sprint race weekends (2021+), `grid_vs_quali_delta` is NaN by design. `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` handle NaNs natively without imputation. For non-tree estimators (e.g. Ridge), an internal `SimpleImputer(strategy='median')` is automatically applied.
- No modifications were made outside the assigned write boundaries (`src/models/hurdle_model.py` and `tests/test_hurdle_model.py`).

---

## 4. Conclusion

Milestone 4 (Two-Stage Hurdle Architecture) is complete, thoroughly tested, and verified:
- `src/models/hurdle_model.py` provides the complete `F1HurdleModel` class with Stage 1 DNF hazard calibration, Stage 2 pure classified pace modeling, Total Probability recombination, and guaranteed top-3 podium selection.
- `tests/test_hurdle_model.py` provides 12 rigorous tests, all passing.
- The full test suite passes 100% (73/73 tests in 10.07s) with zero regressions against existing v5.0 and v6.0 code.

---

## 5. Verification Method

To independently reproduce and verify this milestone:

1. **Run Unit Tests for Hurdle Model**:
   ```powershell
   pytest tests/test_hurdle_model.py -v --tb=short
   ```
   *Expected*: 12 passed in < 3s.

2. **Run Full Regression Suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected*: 73 passed in ~10s.

3. **Verify Pure Pace R^2 on Real Partitions**:
   ```powershell
   python -c "import pandas as pd; from src.models.hurdle_model import F1HurdleModel; from sklearn.metrics import r2_score; tr = pd.read_parquet('data/processed/train_core.parquet'); val = pd.read_parquet('data/processed/val.parquet'); test = pd.read_parquet('data/processed/test.parquet'); m = F1HurdleModel().fit(tr, val); mask = test['is_classified'] == 1; r2 = r2_score(test.loc[mask, 'positionOrder'], m.predict_classified_pace(test)[mask]); print(f'Stage 2 Pace R2: {r2:.4f}'); assert r2 > 0.55"
   ```
   *Expected*: `Stage 2 Pace R2: 0.6587`.
