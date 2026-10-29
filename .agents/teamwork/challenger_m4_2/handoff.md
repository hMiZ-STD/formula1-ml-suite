# Handoff Report — Challenger 2 (Milestone 4: Two-Stage Hurdle Architecture)

**Agent**: Challenger 2 (`teamwork_challenger`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\challenger_m4_2\`  
**Milestone**: Milestone 4 (Two-Stage Hurdle Architecture)  
**Date**: 2026-10-29  
**Verdict**: **APPROVE**  

---

## 1. Observation

Direct empirical observations from test runs, verification probes, and source inspection:

1. **`predict_podium_top3` on Real Partitions**:
   - Validation set (`data/processed/val.parquet`): 820 rows, 41 unique races (`raceId`).
     - Ranking criterion `podium_probability`: `val_mask.sum() == 123` ($41 \times 3$). Every single race contains strictly 3 unique drivers, 0 duplicates.
     - Ranking criterion `expected_position`: `val_mask.sum() == 123` ($41 \times 3$). Every single race contains strictly 3 unique drivers, 0 duplicates.
   - Test set (`data/processed/test.parquet`): 2,559 rows, 128 unique races (`raceId`).
     - Ranking criterion `podium_probability`: `test_mask.sum() == 384` ($128 \times 3$). Every single race contains strictly 3 unique drivers, 0 duplicates.
     - Ranking criterion `expected_position`: `test_mask.sum() == 384` ($128 \times 3$). Every single race contains strictly 3 unique drivers, 0 duplicates.

2. **Extreme Ties Determinism Stress Test**:
   - Tested 20-driver race fields with identical Hurdle podium probability ($0.35$ and $0.50$) across seeds `[1, 42, 99, 777, 2026]` with shuffled `grid_effective` allocations ($1 \dots 20$):
     - In 100% of cases, `predict_podium_top3` selected exactly 3 drivers (`mask.sum() == 3`).
     - In 100% of cases, the selected drivers possessed `grid_effective` values `[1.0, 2.0, 3.0]` strictly.
   - Tested multi-race exact ties batch (10 races $\times$ 20 drivers = 200 drivers, identical scores $0.618$):
     - Total mask sum: strictly 30 ($10 \times 3$).
     - Every individual race selected drivers on grid positions 1, 2, and 3.

3. **High-Attrition Chaos Race Stress Test (70% Attrition)**:
   - Evaluated synthetic 20-driver Grand Prix where 14 drivers (70%) suffered high mechanical hazard ($p_{\text{dnf}} \in [0.70, 0.98]$) and 6 drivers were healthy finishers ($p_{\text{dnf}} \in [0.02, 0.10]$).
   - Training set empirical DNF mean finishing position: $\bar{y}_{\text{dnf}} = 18.0358 \approx 18.04$.
   - Healthy finishers:
     - Expected finishing positions ranged from $1.83$ to $8.82$ (all $< 9.0$).
     - Hurdle podium probabilities ranged from $0.045$ to $0.882$.
   - Retiring drivers:
     - Expected finishing positions ranged from $12.99$ to $17.92$ (mean $15.92$, all $> 12.5$).
     - Even a pole-sitting car with championship-winning pace ($\hat{y}_{\text{finish}} = 1.2$) but high failure hazard ($p_{\text{dnf}} = 0.70$) was penalized to $\mathbb{E}[\text{Pos}] = 12.99$ and $P(\text{Podium}) = 0.240$.
   - Field ordering:
     - Healthy finishers occupied all 6 top positions (ranks 1 to 6).
     - Retiring drivers were relegated to ranks 7 through 20.
     - `predict_podium_top3` selected 3 healthy finishers (`driverId`: `[1001, 1002, 1003]`) and strictly 0 retiring drivers.

4. **Test Suite Runtime Performance**:
   - `pytest tests/` output:
     ```
     ============================= 73 passed in 17.16s =============================
     ```
   - All 73 tests passed (100% success rate). Total execution time was 17.16s, comfortably meeting the $< 20.0\text{s}$ constraint.
   - Isolated hurdle suite (`pytest tests/test_hurdle_model.py`):
     ```
     ============================= 12 passed in 4.05s ==============================
     ```

---

## 2. Logic Chain

1. **Top-3 Selection Mathematical Integrity**:
   - Formula 1 regulations require exactly 3 podium finishers per race.
   - In `src/models/hurdle_model.py` (lines 159 & 176), `predict_podium_top3` uses `np.lexsort((grid_all, -scores_arr))` (and grouped `np.lexsort((race_grid, -race_scores))` for multi-race DataFrames).
   - Because `-scores_arr` is primary, higher continuous probabilities/scores are prioritized. When exact ties occur, `grid_all` ascending acts as the deterministic secondary key, selecting the driver starting closer to pole position.
   - Observation 1 proves that across 41 validation races (820 rows) and 128 test races (2,559 rows), the selection strictly sums to $3 \times N_{\text{races}}$ ($123$ and $384$ respectively) with zero driver duplication.
   - Observation 2 proves that under extreme ties where all 20 drivers have equal Hurdle scores, drivers starting P1, P2, and P3 are deterministically chosen without non-deterministic drift.

2. **High-Attrition Penalization Dynamics**:
   - Under the Law of Total Probability recombination:
     $$\mathbb{E}[\text{Position}_i] = (1 - p_{\text{dnf}, i})\hat{y}_{\text{finish}, i} + p_{\text{dnf}, i}\bar{y}_{\text{dnf}}$$
     $$P(\text{Podium}_i) = (1 - p_{\text{dnf}, i}) P(\text{Podium}_i \mid \text{Finish}_i)$$
   - When mechanical attrition strikes ($p_{\text{dnf}} \to 1.0$), the driver's expected position is pulled toward the back of the grid ($\bar{y}_{\text{dnf}} \approx 18.04$).
   - In a 70% attrition scenario (Observation 3), this penalty successfully suppresses fast but unreliable machinery ($\hat{y}_{\text{finish}} = 1.2 \to \mathbb{E}[\text{Pos}] = 12.99$), allowing reliable midfield runners ($\hat{y}_{\text{finish}} = 7.8, p_{\text{dnf}} = 0.10 \to \mathbb{E}[\text{Pos}] = 8.82$) to be elevated into top finish and podium contention.

3. **Performance & Regression Immunity**:
   - Full test suite passes 73/73 tests in 17.16s with zero regressions against v5.0 baselines or earlier v6.0 ranker modules.

---

## 3. Caveats

- In the extreme ties test, setting identical features while varying `grid_effective` directly in the feature DataFrame will cause `F1HurdleModel` to output slightly different probabilities because `grid_effective` is an explicit input feature in `ALLOWED_FEATURES` (better starting grid improves pace and podium probability). To test exact score tie-breaking on the model itself, identical Hurdle output probabilities were supplied directly or via mock, validating the secondary sorting mechanism independently.
- Sprint races (2021+) contain `NaN` in `grid_vs_quali_delta`, which is handled natively by `HistGradientBoosting`.

---

## 4. Conclusion

**Verdict: APPROVE**

The Hurdle Top-3 selection algorithm (`predict_podium_top3`) and the Two-Stage Hurdle Architecture (`F1HurdleModel`) fulfill all specified requirements:
- Strictly produces $3 \times N_{\text{races}}$ podium finishers on validation (123) and test (384) partitions with 0 duplicates.
- Secondary tie-breaking on `grid_effective` deterministically selects grid 1, 2, 3 during extreme ties.
- High-attrition chaos scenarios properly penalize retiring cars ($\mathbb{E}[\text{Pos}] \sim 16.0$) and elevate healthy finishers ($\mathbb{E}[\text{Pos}] < 9.0$).
- Full test suite passes 100% in 17.16s (< 20s).

---

## 5. Verification Method

To independently reproduce the empirical findings:

1. **Verify Full Test Suite Runtime**:
   ```powershell
   pytest tests/ --durations=10
   ```
   *Expected*: 73 passed in < 20s.

2. **Verify Hurdle Top-3 Selection on Partitions & Exact Ties**:
   ```powershell
   python -c @"
   import pandas as pd, numpy as np
   from src.models.hurdle_model import F1HurdleModel, predict_podium_top3
   val = pd.read_parquet('data/processed/val.parquet')
   test = pd.read_parquet('data/processed/test.parquet')
   tr = pd.read_parquet('data/processed/train_core.parquet')
   m = F1HurdleModel().fit(tr, val)
   v_mask = m.predict_podium_top3(val)
   t_mask = m.predict_podium_top3(test)
   assert v_mask.sum() == 123
   assert t_mask.sum() == 384
   # Extreme tie-breaking
   df = pd.DataFrame({'raceId': [1]*20, 'driverId': range(20), 'grid_effective': np.random.default_rng(42).permutation(range(1, 21)).astype(float)})
   mask = predict_podium_top3(np.full(20, 0.5), df)
   assert sorted(df.loc[mask == 1, 'grid_effective'].tolist()) == [1.0, 2.0, 3.0]
   print('Verification Passed Successfully!')
   "@
   ```
   *Expected*: `Verification Passed Successfully!`.
