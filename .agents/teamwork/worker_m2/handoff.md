# Milestone 2 Handoff Report: Race-Aligned Ranking Metrics

**Agent**: Worker M2 (`teamwork_worker`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\worker_m2\`  
**Milestone**: Milestone 2 (Race-Aligned Ranking Metrics)  
**Date**: 2026-10-29  

---

## 1. Observation

1. **Source Code Implementation (`src/evaluation/ranking_metrics.py`)**:
   - Total lines: 542 lines.
   - Module docstring features an educational child-friendly ELI5 analogy:
     > "Imagine a Formula 1 race as a school sports day where 20 runners race for trophies. Instead of asking 'Is this runner fast in general?', race ranking checks who crossed the finish line 1st, 2nd, and 3rd! NDCG gives the biggest rewards for getting the podium medals right, MRR checks how quickly we found the gold medalist, and Kendall's Tau checks if our overall finish list matches the real race order without any mix-ups!"
   - Functions implemented with strict Python type hints and mathematical formulas:
     - `map_finish_positions_to_relevance(positions, is_classified=None, mapping='linear_rank') -> np.ndarray`: Maps race finishing positions to non-negative numerical relevance scores under `'linear_rank'`, `'f1_points'`, or `'podium_tier'`. Unclassified/DNF drivers receive 0 points under points/podium mappings.
     - `compute_dcg_at_k(relevance: np.ndarray, k: int) -> float`: Evaluates $\sum_{r=1}^{\min(K, m)} \frac{2^{y_{(r)}} - 1}{\log_2(r + 1)}$.
     - `compute_ndcg_at_k(y_true_relevance: np.ndarray, y_pred_scores: np.ndarray, k: int = 3) -> float`: Computes $\frac{\text{DCG}@K}{\text{IDCG}@K}$. Returns `0.0` when IDCG == 0. Clamps $K$ at field size $m$.
     - `compute_mrr(y_true_winner: np.ndarray, y_pred_scores: np.ndarray) -> float`: Computes $1 / \text{rank}$ of the race winner where `y_true_winner == 1`. Returns `0.0` if no winner is labeled or array is empty.
     - `compute_kendall_tau(y_true_ranks: np.ndarray, y_pred_scores: np.ndarray) -> float`: Pairwise rank correlation via `scipy.stats.kendalltau(true_ranks, -scores)`. Returns `1.0` if length < 2; returns `0.0` on tied scores.
     - `evaluate_race_ranking(df_test: pd.DataFrame, score_col: str, target_col: str = "positionOrder", k_values: tuple[int, ...] = (1, 3, 10), relevance_mapping: str = "linear_rank") -> dict[str, float]`: Groups by `raceId`, evaluates metrics per race, computes Top-3 Precision/Recall/F1 with secondary tie-breaking on `grid_effective`, and returns a `CaseInsensitiveDict` averaging metrics across all races.

2. **Unit Test Suite Implementation (`tests/test_ranking_metrics.py`)**:
   - Total lines: 350 lines, 19 comprehensive test functions covering 4 testing tiers:
     - Tier 1: Hand-crafted analytical DCG, NDCG, MRR, Kendall's Tau, and relevance mapping parity.
     - Tier 2: Boundary value tests ($k \le 0$, $k > m$, empty arrays, all-zero relevance, all tied scores, single driver).
     - Tier 3: Invariance properties (translation and positive scale invariance of prediction scores).
     - Tier 4: Real-world mock multi-race DataFrames, tie-breaking on `grid_effective`, case-insensitive dictionary access, and evaluation on real 128-race test split (`data/processed/test.parquet`).

3. **Test Suite Execution Results**:
   - Command: `pytest tests/test_ranking_metrics.py -v --tb=short`
     - Output: `19 passed in 2.83s`.
   - Command: `pytest tests/ -v --tb=short`
     - Output: `42 passed in 11.16s`. (All 23 baseline tests passed + 19 new tests, 100% pass rate, zero regressions).

4. **Real-Data Baseline Verification**:
   - Evaluated inverted starting grid (`-grid_effective`) across all 128 races of `data/processed/test.parquet` (2,559 starters):
     - `NDCG@1`: 0.6271
     - `NDCG@3`: 0.7315
     - `NDCG@10`: 0.8156
     - `MRR`: 0.6807
     - `Kendall_Tau`: 0.4998
     - `Podium_Precision`: 0.6823
     - `Podium_Recall`: 0.6823
     - `Podium_F1`: 0.6823

---

## 2. Logic Chain

1. **Intra-Race Competition Alignment**:
   - In Formula 1, race outcomes are zero-sum permutations. Unlike pointwise classification or regression, ranking metrics evaluate driver position relative to all other starters within the same race.
   - Using exponential gain $2^y - 1$ and logarithmic discounting $\log_2(r + 1)$, NDCG assigns dominant weight to the podium positions (P1–P3) while penalizing errors in the midfield gently, aligning directly with LambdaMART's objective.
2. **Robustness & Edge-Case Safety**:
   - Races with non-standard field sizes ($m < 3$ or $m < K$) are safely handled by clamping $K = \min(K, m)$ in DCG and NDCG.
   - When all candidates have zero relevance (IDCG = 0), `compute_ndcg_at_k` returns `0.0` instead of raising `ZeroDivisionError`.
   - In races with all identical prediction scores, `scipy.stats.kendalltau` returns `NaN`; our implementation explicitly intercepts `NaN` and returns `0.0`, reflecting zero ranking signal.
3. **Dual Key Compatibility**:
   - By subclassing `dict` as `CaseInsensitiveDict`, `evaluate_race_ranking` satisfies callers expecting lowercase keys (`results['ndcg@3']`) and tabular benchmark reporters expecting uppercase keys (`results['NDCG@3']`, `results['MRR']`) without polluting dictionary iteration length.
4. **Zero-Regression Verification**:
   - Running the full test suite confirmed all 23 prior tests (`test_grid_encoding`, `test_kmeans_scratch`, `test_linear_gd`, `test_pca_scratch`, `test_pipeline_integrity`, `test_qualifying_parser`, `test_race_folds`) pass with zero degradation.

---

## 3. Caveats

1. **Relevance Mapping Sensitivity**:
   - `linear_rank` evaluates ordering across all $m$ starters uniformly.
   - `f1_points` and `podium_tier` assign zero relevance to P11+ and unclassified starters. For races where few drivers are classified (e.g. Monaco 1996), IDCG reflects only classified starters.
2. **Tie Breaking**:
   - Secondary tie-breaking in `evaluate_race_ranking` utilizes `grid_effective` ascending (closer to pole wins) if available. If `grid_effective` is absent, stable mergesort preserves original row ordering.

---

## 4. Conclusion

Milestone 2 is complete. `src/evaluation/ranking_metrics.py` provides production-grade, mathematically exact, and thoroughly tested ranking metrics for the Formula 1 Machine Learning Engineering Suite. All 42 unit tests pass 100% with zero regressions.

---

## 5. Verification Method

To independently verify the implementation:

1. **Run ranking metrics tests**:
   ```powershell
   pytest tests/test_ranking_metrics.py -v --tb=short
   ```
   *Expected*: 19 passed in ~3s.

2. **Run full regression test suite**:
   ```powershell
   pytest tests/ -v --tb=short
   ```
   *Expected*: 42 passed in ~12s with 0 failures, 0 errors.

3. **Verify real test set ranking evaluation**:
   ```powershell
   python -c "import pandas as pd; from src.evaluation.ranking_metrics import evaluate_race_ranking; df = pd.read_parquet('data/processed/test.parquet'); df['score'] = -df['grid_effective']; print(evaluate_race_ranking(df, 'score'))"
   ```
   *Expected*: Valid `CaseInsensitiveDict` output with mean `NDCG@3` $\approx 0.7315$ and `MRR` $\approx 0.6807$.
