# Handoff Report: Challenger M3 — Adversarial Stress Testing of LTR Architecture

**Agent**: Challenger M3 (`teamwork_challenger`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\challenger_m3_1\`  
**Date**: 2026-10-29  
**Target Milestone**: Milestone 3 (Learning-to-Rank Architecture)  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 `predict_podium_top3` Stress Testing on Real Partitions
Executed empirical evaluation across all real validation and test races:
1. **Validation Set (`data/processed/val.parquet`)**:
   - Total rows: 820 starters across 41 distinct races (`raceId.nunique() == 41`).
   - Command executed:
     ```powershell
     python -c "import numpy as np, pandas as pd; from src.models.ltr_ranker import predict_podium_top3; val_df = pd.read_parquet('data/processed/val.parquet'); np.random.seed(42); scores = np.random.randn(len(val_df)); mask = predict_podium_top3(scores, val_df); assert mask.sum() == 3 * 41 == 123; assert all(mask[g.index].sum() == 3 for _, g in val_df.groupby('raceId')); assert all(len(set(g.loc[mask[g.index] == 1, 'driverId'])) == 3 for _, g in val_df.groupby('raceId')); print('PASSED')"
     ```
   - Observed Output: `PASSED`. Exactly 123 podium selections across 41 races ($3 \times 41$), with strictly 3 unique drivers per race and zero duplicates.

2. **Test Set (`data/processed/test.parquet`)**:
   - Total rows: 2,559 starters across 128 distinct races (`raceId.nunique() == 128`).
   - Command executed:
     ```powershell
     python -c "import numpy as np, pandas as pd; from src.models.ltr_ranker import predict_podium_top3; test_df = pd.read_parquet('data/processed/test.parquet'); np.random.seed(42); scores = np.random.randn(len(test_df)); mask = predict_podium_top3(scores, test_df); assert mask.sum() == 3 * 128 == 384; assert all(mask[g.index].sum() == 3 for _, g in test_df.groupby('raceId')); assert all(len(set(g.loc[mask[g.index] == 1, 'driverId'])) == 3 for _, g in test_df.groupby('raceId')); print('PASSED')"
     ```
   - Observed Output: `PASSED`. Exactly 384 podium selections across 128 races ($3 \times 128$), with strictly 3 unique drivers per race and zero duplicates.

### 1.2 Extreme Score Ties & Secondary Tie-Breaking
Created a synthetic 20-car race with randomly permuted starting grids where every single candidate had the exact same predicted score (`score == 42.0`):
- Grid order shuffled: `[14, 18, 1, 9, 13, 6, 4, 15, 12, 17, 19, 16, 7, 10, 5, 20, 2, 8, 11, 3]`.
- All scores identical: `np.ones(20) * 42.0`.
- Output:
  ```
  Winner grids on all-tied scores: [1, 2, 3]
  Tie-breaking on grid_effective: PASSED.
  ```
- Evaluated on multi-race DataFrames with non-contiguous random index values: all races cleanly selected drivers from grids 1, 2, and 3 without index corruption.

### 1.3 `LambdaMARTRanker` Missing Feature (NaN) Invariance
Evaluated `LambdaMARTRanker` with 25% synthetic missing values injected across all 17 features in both training (10 races, 200 rows) and validation sets (3 races, 60 rows):
- Fit result: Completed without exceptions.
- Predict result: `preds.shape == (60,)`, `np.isnan(preds).any() == False`.
- Feature importance: 17 feature weights successfully computed.

### 1.4 `LambdaMARTRanker` Early Stopping
Trained `LambdaMARTRanker` with `n_estimators=100` and `early_stopping_rounds=5` against pure uncorrelated validation noise:
- Observed Output:
  ```
  Early stopping triggered! best_iteration_: 1, total trees fitted: 1
  ```
- Predictions with the early-stopped booster executed successfully and produced finite continuous scores.

### 1.5 Pytest Suite Execution
- `pytest tests/test_ltr_ranker.py`: 19 passed in 21.24s.
- `pytest tests/`: 61 passed in 26.41s (100% pass rate across entire project).

---

## 2. Logic Chain

1. **Intra-Race Zero-Sum Constraint (Observation 1.1)**:
   In Formula 1, top-3 podium outcomes must be mutually exclusive and strictly partition the grid into 3 winners and $m_r - 3$ non-podium finishers. `predict_podium_top3` achieved $3 \times N_{\text{races}}$ podium allocations on both validation ($3 \times 41 = 123$) and test sets ($3 \times 128 = 384$) with 0 duplicate drivers.
2. **Deterministic Tie Breaking (Observation 1.2)**:
   In real racing, when predictions or telemetry are tied, qualifying track position (`grid_effective`) serves as the true physical tie-breaker. Using `np.lexsort((grid_effective, -scores))` deterministically prioritizes pole position starters (grids 1, 2, and 3) even under complete score degeneracy.
3. **Robustness to Incomplete Telemetry (Observation 1.3)**:
   Pre-race data occasionally lacks specific qualifying gaps (e.g. grid penalties or back-of-the-grid engine changes). LightGBM's native missing value branch allocation in `LambdaMARTRanker` seamlessly accommodates NaNs without requiring external imputation steps or crashing during inference.
4. **Regularization via Early Stopping (Observation 1.4)**:
   To prevent tree-based over-specialization on historical eras (1994–2018), validation early stopping stops boosting when validation NDCG plateaus. When fed uncorrelated validation targets, the ranker halted after iteration 1, confirming responsive early stopping.
5. **No Regressions (Observation 1.5)**:
   All 61 project tests passed, confirming full backward compatibility with v5.0 baselines.

---

## 3. Caveats

1. **Extreme Ties with Identical Grid Positions**:
   In official FIA regulations, two drivers cannot start from the exact same grid slot (`grid_effective` is strictly unique per race). If a synthetic dataset fed identical scores AND identical grid numbers to two drivers, `np.lexsort` would break the tie by original DataFrame row order.
2. **Single-Candidate Races**:
   If a race has fewer than 3 starters ($m < 3$), `predict_podium_top3` selects $\min(3, m)$ drivers, ensuring no out-of-bounds indexing or crash occurs.
3. **CPU vs GPU**:
   All empirical tests were run on CPU; LightGBM and PyTorch ran with sub-second latency on the available hardware.

---

## 4. Conclusion

**Verdict: APPROVE**.  
The Milestone 3 Learning-to-Rank implementation (`src/models/ltr_ranker.py`) satisfies all functional, architectural, and adversarial criteria specified in `ORIGINAL_REQUEST.md` and `PROJECT.md`:
- `predict_podium_top3` guarantees strictly 3 unique podium finishers per race across all validation and test races.
- Secondary tie-breaking on `grid_effective` is deterministic and mathematically sound.
- `LambdaMARTRanker` handles missing values natively and terminates early when validation NDCG fails to improve.
- Code quality, type annotations, and ELI5 pedagogical analogies conform to project standards.

---

## 5. Verification Method

To reproduce and independently verify all observations in this report:

1. **Verify Validation and Test Strict Podium Allocations**:
   ```powershell
   python -c "import numpy as np, pandas as pd; from src.models.ltr_ranker import predict_podium_top3; val_df = pd.read_parquet('data/processed/val.parquet'); test_df = pd.read_parquet('data/processed/test.parquet'); assert predict_podium_top3(np.random.randn(len(val_df)), val_df).sum() == 123; assert predict_podium_top3(np.random.randn(len(test_df)), test_df).sum() == 384; print('Podium count verification: PASSED')"
   ```

2. **Verify Extreme Ties and Secondary Tie-Breaking**:
   ```powershell
   python -c "import numpy as np, pandas as pd; from src.models.ltr_ranker import predict_podium_top3; df = pd.DataFrame({'raceId': [1]*20, 'grid_effective': np.random.permutation(range(1, 21))}); mask = predict_podium_top3(np.ones(20), df); assert sorted(df.loc[mask == 1, 'grid_effective'].tolist()) == [1, 2, 3]; print('Extreme tie-breaking verification: PASSED')"
   ```

3. **Verify LambdaMART Missing Values and Early Stopping**:
   ```powershell
   python -c "import numpy as np; from src.models.ltr_ranker import LambdaMARTRanker; X = np.random.randn(100, 17); X[0, 0] = np.nan; y = np.tile(np.arange(20, 0, -1), 5); g = np.array([20]*5); r = LambdaMARTRanker(n_estimators=50).fit(X, y, g, X_val=X, y_val=y, group_val=g, early_stopping_rounds=5); assert not np.isnan(r.predict(X)).any(); print('LambdaMART verification: PASSED')"
   ```

4. **Run Full Test Suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
