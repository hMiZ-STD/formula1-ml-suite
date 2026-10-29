## 2026-10-29T15:18:41Z
You are Worker M2 (teamwork_worker).
Your working directory is: d:\ML-Project\.agents\teamwork\worker_m2\

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You must read and strictly adhere to:
- d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md
- d:\ML-Project\PROJECT.md
- Reference plan: C:/Users/hamiz/.gemini/antigravity/brain/20fe5515-5671-4c16-be43-e50f0230ca78/plan_ltr_and_hurdle_architecture.md
- Explorer survey report: d:\ML-Project\.agents\teamwork\explorer_survey_1\handoff.md

Your exclusive write ownership:
- `src/evaluation/ranking_metrics.py`
- `tests/test_ranking_metrics.py`

Your task — Milestone 2 (Race-Aligned Ranking Metrics):
1. Implement `src/evaluation/ranking_metrics.py`:
   - Include a neat child-friendly ELI5 analogy at the top of the module: e.g. "Imagine a Formula 1 race as a school sports day where 20 runners race for trophies. Instead of asking 'Is this runner fast in general?', race ranking checks who crossed the finish line 1st, 2nd, and 3rd. NDCG gives the biggest rewards for getting the podium medals right, MRR checks how quickly we found the gold medalist, and Kendall's Tau checks if our overall finish list matches the real race order without any mix-ups!"
   - Strict Python type hints and docstrings with mathematical formulas.
   - Implement:
     - `compute_dcg_at_k(relevance: np.ndarray, k: int) -> float`:
       $$\sum_{r=1}^{\min(K, m)} \frac{2^{y_{(r)}} - 1}{\log_2(r + 1)}$$
     - `compute_ndcg_at_k(y_true_relevance: np.ndarray, y_pred_scores: np.ndarray, k: int = 3) -> float`:
       $$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
       Return `0.0` if IDCG == 0. Clamp $K$ at array length $m$.
     - `compute_mrr(y_true_winner: np.ndarray, y_pred_scores: np.ndarray) -> float`:
       Reciprocal rank $1 / \text{rank}$ of the race winner (where `y_true_winner == 1`). Return `0.0` if no winner labeled.
     - `compute_kendall_tau(y_true_ranks: np.ndarray, y_pred_scores: np.ndarray) -> float`:
       Pairwise rank correlation using `scipy.stats.kendalltau`. Return `1.0` if length < 2.
     - `evaluate_race_ranking(df_test: pd.DataFrame, score_col: str, target_col: str = "positionOrder", k_values: tuple[int, ...] = (1, 3, 10), relevance_mapping: str = "linear_rank") -> dict[str, float]`:
       Groups by `raceId`, evaluates metrics per race, computes Top-3 Precision/Recall/F1, and averages across all races in the DataFrame.
2. Implement comprehensive unit tests in `tests/test_ranking_metrics.py`:
   - Test NDCG, MRR, Kendall's Tau on hand-crafted arrays with exact mathematical verification.
   - Test edge cases: perfect ranking, inverted ranking, tied scores, all-zero relevance, empty/single driver, K > grid size.
   - Test race-level evaluation with mock multi-race DataFrames.
3. Run tests:
   - Run `pytest tests/test_ranking_metrics.py -v --tb=short`
   - Run full test suite: `pytest tests/ -v --tb=short` (must pass 100% with zero regressions).
4. Write handoff report to:
   `d:\ML-Project\.agents\teamwork\worker_m2\handoff.md`
5. Send a message to parent when complete.
