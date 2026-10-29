# Forensic Audit Report: Milestone 3 — Learning-to-Rank Architecture

**Auditor**: Forensic Auditor (`teamwork_auditor`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\auditor_m3\`  
**Target Milestone**: Milestone 3 (Learning-to-Rank Architecture)  
**Integrity Mode**: `development` (per `ORIGINAL_REQUEST.md` line 8)  
**Verdict**: **CLEAN** (Zero Integrity Violations Detected)  

---

## 1. Observation

### 1.1 Source Code Verification (`src/models/ltr_ranker.py`)
Direct forensic inspection of `src/models/ltr_ranker.py` (784 lines) confirmed:
1. **`LambdaMARTRanker`**:
   - Genuinely wraps `lightgbm.LGBMRanker`:
     ```python
     self.model_ = lgb.LGBMRanker(
         objective="lambdarank",
         metric="ndcg",
         eval_at=[1, 3, 10],
         n_estimators=self.n_estimators,
         learning_rate=self.learning_rate,
         num_leaves=self.num_leaves,
         min_child_samples=self.min_child_samples,
         random_state=self.random_state,
         verbose=-1,
         **self.kwargs,
     )
     ```
   - In `.fit()`, genuine validation early stopping is passed via `eval_set=[(X_val, y_val)]`, `eval_group=[group_val]`, and `callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]`.
   - Feature importances and continuous prediction scores are retrieved authentically from `self.model_.predict(X)`.
   - No mock wrappers, hardcoded return values, or facade shortcuts are present.
2. **`PlackettLuceRanker`**:
   - Implements genuine 3-layer PyTorch MLP in `_PlackettLuceMLP`:
     `Linear(input_dim, hidden_dim) -> ReLU() -> Dropout(0.2) -> Linear(hidden_dim, 32) -> ReLU() -> Linear(32, 1)`.
   - `compute_list_mle_loss()` computes the true analytical Plackett-Luce ListMLE negative log-likelihood:
     $$\mathcal{L}(\boldsymbol{s}, \pi) = - \sum_{i=1}^{m-1} \left( s_{\pi_i} - \text{logsumexp}(s_{\pi_{i:m}}) \right)$$
     Vectorized with upper-triangular masking (`torch.triu`) and `torch.logsumexp`.
   - In `.fit()`, genuine Adam optimizer updates are executed: `(batch_loss / float(n_in_batch)).backward()`, `optimizer.step()`, tracking validation loss and rolling back best model weights.
   - Empirical neural weight verification executed by the auditor (`verify_plackett_luce.py`) showed active parameter divergence after 3 epochs:
     - `net.0.weight` diff: 12.993286
     - `net.0.bias` diff: 0.757679
     - `net.3.weight` diff: 25.975573
     - `net.3.bias` diff: 0.891394
     - `net.5.weight` diff: 0.895766
     - `net.5.bias` diff: 0.070829
     - Unique predictions on validation set: 820 / 820 distinct values.
3. **`F1RankerDataset`**:
   - Genuinely handles race contiguity, group queries (`size().to_numpy(dtype=np.int32)`), median imputation, and computes discrete relevance targets:
     - `'linear_rank'`: $y_i = \max(0, m - \text{positionOrder}_i + 1)$.
     - `'f1_points'`: championship scale {1: 25, 2: 18, 3: 15, ..., 10: 1}, 0 otherwise (classified finishers only).
     - `'podium_tier'`: {1: 3, 2: 2, 3: 1}, 0 otherwise (classified finishers only).
4. **`predict_podium_top3`**:
   - Genuinely sorts continuous scores descending with secondary deterministic tie-breaking on `grid_effective` ascending via `np.lexsort((grid, -scores))`.
   - Guarantees strictly 3 unique drivers per race across both single-race and multi-race DataFrames.

### 1.2 `EXPERIMENT_FREEZE.yaml` Verification
- Executed `git diff main -- EXPERIMENT_FREEZE.yaml`:
  - Verified lines 1–30 (the baseline v5.0 freeze: timestamp, supervised scope, feature set hash, hyperparameters, thresholds, seeds, cnn epoch) are **100% untouched and identical to main**.
  - Verified lines 31–86 are an append-only block recording `v6_ranking_and_hurdle`.
- Independent empirical reproduction of the relevance mapping benchmark (`verify_benchmark.py`) on `train_core.parquet` evaluated against `val.parquet`:
  - **`linear_rank`**:
    - Calculated vs Frozen:
      - `ndcg@1`: calc=0.847562, frozen=0.847562 (Match: True)
      - `ndcg@3`: calc=0.799942, frozen=0.799942 (Match: True)
      - `ndcg@10`: calc=0.895797, frozen=0.895797 (Match: True)
      - `mrr`: calc=0.817683, frozen=0.817683 (Match: True)
      - `kendall_tau`: calc=0.517901, frozen=0.517901 (Match: True)
      - `podium_f1`: calc=0.682927, frozen=0.682927 (Match: True)
  - **`f1_points`**:
    - `ndcg@1`: calc=0.757455, frozen=0.757455 (Match: True)
    - `ndcg@3`: calc=0.783662, frozen=0.783662 (Match: True)
    - `ndcg@10`: calc=0.855891, frozen=0.855891 (Match: True)
    - `mrr`: calc=0.819290, frozen=0.819290 (Match: True)
    - `kendall_tau`: calc=0.416889, frozen=0.416889 (Match: True)
    - `podium_f1`: calc=0.634146, frozen=0.634146 (Match: True)
  - **`podium_tier`**:
    - `ndcg@1`: calc=0.749129, frozen=0.749129 (Match: True)
    - `ndcg@3`: calc=0.784321, frozen=0.784321 (Match: True)
    - `ndcg@10`: calc=0.870098, frozen=0.870098 (Match: True)
    - `mrr`: calc=0.775000, frozen=0.775000 (Match: True)
    - `kendall_tau`: calc=0.514664, frozen=0.514664 (Match: True)
    - `podium_f1`: calc=0.707317, frozen=0.707317 (Match: True)
  - Conclusion: Zero discrepancy. Every frozen value is authentic and directly reproducible from real models.

### 1.3 Test Suite Integrity & Independent Execution
1. **No Tampering of Existing Tests**:
   - `git diff main -- tests/`: Zero modifications or deletions. All 7 baseline test files (23 tests) are intact.
   - `tests/test_ranking_metrics.py` (M2, 19 tests) is intact.
   - `tests/test_ltr_ranker.py` (M3, 19 tests) contains rigorous assertions:
     - Handcrafted mathematical checks for relevance targets with unclassified DNF handling.
     - Analytical loss calculations matching hand-calculated logs.
     - PyTorch autograd backpropagation check (`assert s.grad is not None`).
     - Gradient descent optimization test proving score inversion convergence.
     - LightGBM early stopping iteration validation.
     - Deterministic tie-breaking on `grid_effective`.
     - Full validation set (41 races) and test set (128 races) strict podium checks ($3 \times N$).
2. **Independent Test Execution**:
   - `pytest tests/test_ltr_ranker.py -v --tb=short`:
     `19 passed in 15.48s`
   - `pytest tests/ -v --tb=short`:
     `61 passed in 35.58s (100% passing)`

### 1.4 Prohibited Pattern Check (General Project Profile)
| # | Prohibited Pattern | Auditor Finding | Status |
|---|-------------------|-----------------|:------:|
| 1 | Hardcoded test results | None found; dynamic calculations throughout | **PASS** |
| 2 | Facade implementations | Genuine LightGBM & PyTorch neural architectures | **PASS** |
| 3 | Fabricated verification outputs | Pre-populated log search found only raw data files | **PASS** |
| 4 | Self-certifying tests | Tests use analytical ground truths and real partitions | **PASS** |
| 5 | Baseline test modification | Zero baseline tests altered or weakened | **PASS** |

---

## 2. Logic Chain

1. **Step 1 (Source Integrity)**:
   Inspection of `src/models/ltr_ranker.py` confirms that `LambdaMARTRanker` wraps `lightgbm.LGBMRanker` with objective `lambdarank` and metrics `ndcg`, while `PlackettLuceRanker` implements a complete PyTorch neural network optimizing the exact ListMLE likelihood formulation via autograd. There are no dummy return values, trivial stubs, or facades.
2. **Step 2 (Gradient & Learning Authenticity)**:
   Empirical testing of `PlackettLuceRanker` on the actual historical training dataset (`verify_plackett_luce.py`) proved that model parameters update across training epochs (cumulative weight shift > 38.0 across layers) and generate continuous, non-degenerate latent pace scores (820/820 unique values on validation data).
3. **Step 3 (Metric Freeze Authenticity)**:
   Independent re-execution of the relevance target benchmark on `train_core.parquet` against `val.parquet` produced metrics matching `EXPERIMENT_FREEZE.yaml` values to 6 decimal places across all three candidate mappings (`linear_rank`, `f1_points`, `podium_tier`). Lines 1–30 of `EXPERIMENT_FREEZE.yaml` were confirmed 100% identical to the `main` branch.
4. **Step 4 (Test Non-Degradation)**:
   Git diff verification against `main` proved that none of the 23 baseline test files or 19 M2 ranking metrics test files were deleted, weakened, or altered. Independent execution of `pytest tests/` passed all 61 tests cleanly.
5. **Step 5 (Verdict Synthesis)**:
   Because all forensic checks passed and zero prohibited patterns were observed, the deliverable satisfies all integrity standards under `development` mode.

---

## 3. Caveats

No caveats. All components of Milestone 3 were audited, verified against source code, executed independently, and checked against the baseline git repository.

---

## 4. Conclusion

**Verdict: CLEAN**.

Worker M3's implementation of Milestone 3 (Learning-to-Rank Architecture) is authentic, production-grade, mathematically sound, and fully backward-compatible with v5.0. It contains zero hardcoding, zero facades, zero test tampering, and all frozen benchmark metrics are verified. Milestone 3 is recommended for unconditional acceptance.

---

## 5. Verification Method

To reproduce and independently verify the findings in this audit report:

1. **Run Full Test Suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected outcome*: 61 passed in ~35s.

2. **Verify Baseline Test Integrity (Main vs Feature Branch)**:
   ```powershell
   git diff main -- tests/
   ```
   *Expected outcome*: Empty output (no modifications to baseline tests).

3. **Verify EXPERIMENT_FREEZE.yaml v5 Keys (Lines 1–30)**:
   ```powershell
   git diff main -- EXPERIMENT_FREEZE.yaml
   ```
   *Expected outcome*: Diffs only appear after line 30 in the `v6_ranking_and_hurdle` section.

4. **Verify Relevance Benchmark Replication**:
   ```powershell
   python .agents/teamwork/auditor_m3/verify_benchmark.py
   ```
   *Expected outcome*: All calculated metrics match frozen values to 6 decimal places.

5. **Verify Plackett-Luce Neural Gradient Updates**:
   ```powershell
   python .agents/teamwork/auditor_m3/verify_plackett_luce.py
   ```
   *Expected outcome*: Weight diffs > 0 across all layers; 820/820 unique predictions.
