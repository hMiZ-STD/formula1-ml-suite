# Handoff Report: Reviewer 2 — Milestone 3 (Learning-to-Rank Architecture)

**Agent**: Reviewer 2 (`teamwork_reviewer`, `reviewer_m3_2`)  
**Roles**: Reviewer, Critic  
**Working Directory**: `d:\ML-Project\.agents\teamwork\reviewer_m3_2\`  
**Date**: 2026-10-29  
**Target Milestone**: Milestone 3 (Learning-to-Rank Architecture)  
**Branch**: `feat/ltr-and-hurdle-architecture`  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 Integrity and Anti-Cheat Audit
- **Source Code Verification (`src/models/ltr_ranker.py`)**:
  - No hardcoded test results, expected outputs, or precomputed tables embedded in source code.
  - No facade logic or mock delegators:
    - `F1RankerDataset` processes real DataFrames dynamically and maps positions to relevance.
    - `LambdaMARTRanker` wraps real `lightgbm.LGBMRanker(objective='lambdarank')`.
    - `PlackettLuceRanker` is a fully functional 3-layer PyTorch neural network optimizing the exact ListMLE permutation loss via PyTorch autograd.
    - `predict_podium_top3` executes deterministic two-key lexicographical sorting (`np.lexsort`).
- **Data Parity Verification (`EXPERIMENT_FREEZE.yaml`)**:
  - Live recalculation of the validation benchmark on `data/processed/train_core.parquet` and `data/processed/val.parquet` confirmed exact parity to 16 decimal places:
    - `linear_rank` NDCG@1: Live = `0.8475617606472896`, Frozen = `0.8475617606472896`
    - `f1_points` NDCG@1: Live = `0.7574552152619594`, Frozen = `0.7574552152619594`
    - `podium_tier` NDCG@1: Live = `0.7491289198606272`, Frozen = `0.7491289198606272`
  - Zero fabricated verification outputs; zero self-certifying shortcuts.

### 1.2 Feature Alignment
- `src/models/ltr_ranker.py:40`: `from src.features.builder import ALLOWED_FEATURES`.
- `F1RankerDataset.__init__` defaults `feature_cols` to `list(ALLOWED_FEATURES)` (lines 117–119).
- Exactly 17 whitelist pre-race features are extracted in order.
- In `transform`, `df_work[self.feature_cols].to_numpy(dtype=np.float64)` cleanly extracts features.
- Optional median imputation (`SimpleImputer(strategy='median')`) handles NaNs (`grid_vs_quali_delta`, `q_gap_pct`) without lookahead leakage.

### 1.3 `LambdaMARTRanker` Validation Group Handling
- In `src/models/ltr_ranker.py:365-382`:
  ```python
  if X_val is not None and y_val is not None and group_val is not None:
      eval_set = [(X_val, y_val)]
      eval_group = [group_val]
      if early_stopping_rounds > 0:
          callbacks.append(
              lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
          )
  ```
  `eval_set` is passed as a list of tuples and `eval_group` is passed as `[group_val]`, conforming strictly to LightGBM's native API for grouped ranking models.
- Validation early stopping correctly sets `best_iteration_`.

### 1.4 `PlackettLuceRanker` Convergence, Dropout, and Numerical Stability
- Architecture: 3-layer MLP (`Linear(17, 64) -> ReLU() -> Dropout(0.2) -> Linear(64, 32) -> ReLU() -> Linear(32, 1)`).
- Dropout: Active during training (`self.net.train()`) and disabled during validation evaluation and inference (`self.net.eval()`).
- Loss: `compute_list_mle_loss` vectorizes Plackett-Luce ListMLE negative log-likelihood via upper-triangular masking and `torch.logsumexp`.
- Convergence: Gradient descent on inverted initial scores converges monotonically to ground truth ranking order.
- Numerical stability: Scale/translation invariance tested with extreme inputs $s \in [-10^4, 10^4]$; produced identical finite loss (`0.7207`) and identical gradients (`[-0.3345, -0.0243, 0.3589]`) with zero overflow or NaN.

### 1.5 Strict Top-3 Podium Guarantees
- `predict_podium_top3` returns a 1D `np.int32` array with values strictly in $\{0, 1\}$.
- Validation set: exactly 123 podium predictions across 41 races ($3 \times 41$), with each race having exactly 3 ones.
- Test set: exactly 384 podium predictions across 128 races ($3 \times 128$), with each race having exactly 3 ones.
- Exact score ties are deterministically resolved by ascending `grid_effective` (pole position wins ties).

### 1.6 Backward Compatibility and Test Execution
- `git status`:
  - `src/models/classifiers.py` and `src/models/cnn_classifier.py` are completely untouched.
  - `EXPERIMENT_FREEZE.yaml` was modified strictly append-only (lines 1–29 intact).
- Automated tests:
  - `pytest tests/test_ltr_ranker.py -v`: 19 passed in 22.39s.
  - `pytest tests/ -v`: 61 passed in 30.30s (100% pass rate).

---

## 2. Logic Chain

1. **Step 1 (Integrity Verification)**:
   Reviewed AST and implementation of `src/models/ltr_ranker.py` and benchmark metrics in `EXPERIMENT_FREEZE.yaml`. Validated live recalculation against raw parquet partitions. Confirmed absence of hardcoding, facade patterns, or data manipulation.
2. **Step 2 (Feature Alignment Verification)**:
   Inspected feature extraction against `ALLOWED_FEATURES` from `src.features.builder`. Verified that `F1RankerDataset` strictly limits input dimensions to the 17 whitelisted pre-race features, preventing data leakage.
3. **Step 3 (GBDT Listwise Contract Verification)**:
   Inspected LightGBM parameterization. Verified that `LambdaMARTRanker` handles validation group sizes `[group_val]` properly, uses `objective='lambdarank'`, and stops early on validation NDCG.
4. **Step 4 (Neural Permutation Architecture Verification)**:
   Verified Plackett-Luce ListMLE loss formulation and implementation. Verified that `torch.logsumexp` alongside upper-triangular indexing provides both mathematical correctness and numerical robustness against extreme values.
5. **Step 5 (Strict Podium Invariance Verification)**:
   Tested `predict_podium_top3` across synthetic tie conditions, single-race and multi-race DataFrames, non-standard index labels, and the complete 128-race test partition. Confirmed that every race produces strictly 3 unique podium finishers.
6. **Step 6 (Backward Compatibility Verification)**:
   Verified via `git diff` that no v5.0 models or pipeline code were modified. Verified that all 61 tests across all modules pass cleanly.

---

## 3. Caveats

1. **PlackettLuceRanker PyTorch Execution Device**:
   The `PlackettLuceRanker.__init__` signature does not currently expose a `device` parameter; tensors and the neural net default to CPU execution. For the F1 dataset (8,920 training rows), CPU training is extremely fast (~0.5s/epoch, ~5s total), avoiding GPU memory transfer overhead. If GPU training is desired in future releases, a `device: str = "cpu"` parameter should be added to `__init__`.
2. **Field Size Truncation ($m < 3$)**:
   If an anomalous race has fewer than 3 starting cars, `predict_podium_top3` produces $\min(3, m)$ podium finishers (e.g. 2 finishers if field size is 2), which is physically sound since 3 podium finishers cannot be selected from 2 cars.

---

## 4. Quality & Adversarial Review Details

### 4.1 Quality Review Summary
- **Verdict**: **APPROVE**
- **Findings**:
  - *Minor (Informational)*: `PlackettLuceRanker` runs on CPU by default and does not expose `device` argument in `__init__`. Not a defect, as CPU execution executes 10 epochs across 408 races in 5.11 seconds.
- **Verified Claims**:
  - `F1RankerDataset` features align with `ALLOWED_FEATURES` (17 columns) → Verified via AST and shape assertions → **PASS**
  - `LambdaMARTRanker` passes `eval_set` and `eval_group` and stops early → Verified via `best_iteration_` check → **PASS**
  - `PlackettLuceRanker` converges and handles dropout properly → Verified via autograd convergence test → **PASS**
  - `predict_podium_top3` guarantees $3 \times N_{\text{races}}$ binary podium mask → Verified on all 41 validation and 128 test races → **PASS**
  - Zero modifications to v5.0 baseline models → Verified via `git status` → **PASS**
  - Project test suite passing 61/61 → Verified via `pytest tests/ -v` → **PASS**
- **Coverage Gaps**: None.

### 4.2 Adversarial Review Summary
- **Overall Risk Assessment**: **LOW**
- **Stress-Test Results**:
  1. *Extreme Score Inputs in ListMLE*: $s = \pm 10,000$ tested against ground truth ranking. `torch.logsumexp` prevented underflow/overflow; finite loss and gradients produced identically. → **PASS**
  2. *Total Pace Score Tie*: 6 drivers with identical score 0.0. `predict_podium_top3` broke tie strictly by ascending `grid_effective` (starting positions 1, 2, 3 selected). → **PASS**
  3. *Non-Standard DataFrame Indexing*: Discontinuous row labels `[10, 20, 30, 40, ...]`. `predict_podium_top3` maintained positional alignment and exactly 3 podiums per race. → **PASS**
  4. *Missing `is_classified` Column*: Evaluated `F1RankerDataset` without classification column; defaulted gracefully to classifying all starters without throwing errors. → **PASS**
  5. *Interleaved Race IDs*: Evaluated DataFrame with interleaved races (`[10, 20, 10, 20]`). Group lengths and contiguity sorted stably. → **PASS**

---

## 5. Conclusion

Milestone 3 (Learning-to-Rank Architecture) is fully implemented, rigorously verified, and adheres to all project standards, interface contracts, and numerical stability requirements.
- Zero integrity violations detected.
- 100% backward compatibility maintained.
- All 19 Milestone 3 unit tests and all 61 full-suite tests pass.
- Final Verdict: **APPROVE**.

---

## 6. Verification Method

To independently reproduce this verification:

1. **Verify LTR Ranker Test Suite**:
   ```powershell
   pytest tests/test_ltr_ranker.py -v
   ```
   *Expected outcome*: 19 passed in ~22s.

2. **Verify Full Project Test Suite**:
   ```powershell
   pytest tests/ -v
   ```
   *Expected outcome*: 61 passed in ~30s.

3. **Verify Podium Top-3 Guarantees**:
   ```powershell
   python -c "import pandas as pd; from src.models.ltr_ranker import predict_podium_top3; df = pd.read_parquet('data/processed/test.parquet'); mask = predict_podium_top3(-df['grid_effective'], df); print('Total podiums:', mask.sum(), 'Expected:', 3*df['raceId'].nunique()); assert mask.sum() == 3*df['raceId'].nunique(); assert all(mask[g.index].sum() == 3 for _, g in df.groupby('raceId'))"
   ```
   *Expected outcome*: `Total podiums: 384 Expected: 384`.

4. **Verify Relevance Mapping Live Metric Parity**:
   ```powershell
   python -c "import pandas as pd, yaml; from src.models.ltr_ranker import F1RankerDataset, LambdaMARTRanker; from src.evaluation.ranking_metrics import evaluate_race_ranking; df_train = pd.read_parquet('data/processed/train_core.parquet'); df_val = pd.read_parquet('data/processed/val.parquet'); freeze = yaml.safe_load(open('EXPERIMENT_FREEZE.yaml'))['v6_ranking_and_hurdle']['relevance_mapping_benchmark']; [print(m, evaluate_race_ranking(df_val.assign(score=LambdaMARTRanker(random_state=42).fit(*F1RankerDataset(target_mapping=m).transform(df_train), *F1RankerDataset(target_mapping=m).transform(df_val), early_stopping_rounds=15).predict(F1RankerDataset(target_mapping=m).transform(df_val)[0])), score_col='score', target_col='positionOrder', relevance_mapping=m)['ndcg@1'], freeze[m]['val_ndcg_at_1']) for m in ['linear_rank', 'f1_points', 'podium_tier']]"
   ```
