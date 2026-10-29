## 2026-10-29T15:37:58Z
You are Worker M3 (teamwork_worker).
Your working directory is: d:\ML-Project\.agents\teamwork\worker_m3\

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

You must read and strictly adhere to:
- d:\ML-Project\.agents\teamwork\ORIGINAL_REQUEST.md
- d:\ML-Project\PROJECT.md
- Reference plan: C:/Users/hamiz/.gemini/antigravity/brain/20fe5515-5671-4c16-be43-e50f0230ca78/plan_ltr_and_hurdle_architecture.md
- Explorer survey reports:
  - d:\ML-Project\.agents\teamwork\explorer_survey_1\handoff.md
  - d:\ML-Project\.agents\teamwork\explorer_survey_2\handoff.md

Your exclusive write ownership:
- `src/models/ltr_ranker.py`
- `tests/test_ltr_ranker.py`
- `EXPERIMENT_FREEZE.yaml` (append v6 ranking section without altering v5.0 keys)

Your task — Milestone 3 (Learning-to-Rank Architecture):
1. Implement `src/models/ltr_ranker.py`:
   - Pedagogical child-friendly ELI5 analogy at top of module:
     "Instead of asking 'Is this runner good or bad?' in isolation, Learning-to-Rank holds a mini-tournament between all runners in the race, giving 100x more coaching attention to who gets the Gold, Silver, and Bronze medals rather than who finished 15th vs 16th!"
   - Strict Python type hints and mathematical docstrings.
   - `F1RankerDataset`:
     - Assembles race-contiguous feature matrix $X$ using `ALLOWED_FEATURES` from `src.features.builder`.
     - Handles missing feature values (`grid_vs_quali_delta`, `q_gap_pct`) cleanly via median imputation if needed for neural net or native tree handling.
     - Computes group query lengths: `group = df.groupby('raceId', sort=False).size().to_numpy()`.
     - Maps relevance targets with configurable mappings:
       - `'f1_points'`: championship scale {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}, 0 otherwise. Classified starters only (`is_classified == 1`).
       - `'linear_rank'`: $y_i = \max(0, m - \text{positionOrder}_i + 1)$.
       - `'podium_tier'`: {1: 3, 2: 2, 3: 1}, 0 otherwise. Classified starters only (`is_classified == 1`).
   - `LambdaMARTRanker`:
     - Wraps `lightgbm.LGBMRanker(objective='lambdarank', metric='ndcg', eval_at=[1, 3, 10], n_estimators=100, learning_rate=0.05, num_leaves=31, min_child_samples=15, random_state=42)`.
     - Note: Use `eval_set=[(X_val, y_val)]` and `eval_group=[group_val]` with early stopping (`stopping_rounds=15`) on validation NDCG.
     - Produces continuous latent pace scores.
   - `PlackettLuceRanker`:
     - Deep neural ranker in PyTorch.
     - 3-layer MLP: `Linear(17, 64) -> ReLU() -> Dropout(0.2) -> Linear(64, 32) -> ReLU() -> Linear(32, 1)`.
     - Plackett-Luce ListMLE permutation loss:
       $$\mathcal{L}(\boldsymbol{s}, \pi) = - \sum_{i=1}^{m-1} \left( s_{\pi_i} - \text{logsumexp}(s_{\pi_{i:m}}) \right)$$
     - Adam optimizer (`lr=1e-3`, `weight_decay=1e-4`), validation early stopping, handles variable grid sizes.
   - `predict_podium_top3(scores: np.ndarray | pd.Series, df_race: pd.DataFrame) -> np.ndarray`:
     - Strictly guarantees exactly 3 podium finishers per race (`y_pred == 1`).
     - Secondary tie-breaking on `grid_effective` ascending (closer to pole wins ties).
     - Handles both single-race and multi-race DataFrames. Output binary mask sums to $3 \times N_{\text{races}}$.
2. Relevance Target Benchmarking:
   - Benchmark all three relevance mappings (`f1_points`, `linear_rank`, `podium_tier`) on `data/processed/val.parquet` using `LambdaMARTRanker`.
   - Record validation NDCG@1, NDCG@3, and NDCG@10 for all three in the handoff report.
   - Freeze the winning mapping under `v6_ranking_and_hurdle` in `EXPERIMENT_FREEZE.yaml`.
3. Unit Test Suite in `tests/test_ltr_ranker.py`:
   - Test dataset shapes, group sizes, and relevance mapping correctness.
   - Test `LambdaMARTRanker` training, convergence, and early stopping.
   - Test `PlackettLuceRanker` forward pass, ListMLE loss gradient convergence, and inference.
   - Test `predict_podium_top3`: assert strictly 3 unique drivers per race across all 41 validation races and 128 test races.
4. Run verification:
   - Run `pytest tests/test_ltr_ranker.py -v --tb=short`
   - Run full regression suite: `pytest tests/ -v --tb=short` (must pass 100%).
5. Write detailed handoff report to `d:\ML-Project\.agents\teamwork\worker_m3\handoff.md`.
6. Send a message to parent when complete.
