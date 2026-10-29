# Specification Mining Report: Learning-to-Rank (LTR) & Two-Stage Hurdle Architecture

**Agent**: Explorer 1 (`teamwork_spec_miner`)  
**Working Directory**: `d:\ML-Project\.agents\teamwork\explorer_survey_1\`  
**Date**: 2026-10-29  
**Target Milestone**: v6.0-dev Implementation Planning  
**Branch**: `feat/ltr-and-hurdle-architecture`  

---

## 1. Observation

### 1.1 Direct Codebase & Environment Probes
1. **Git Repository Status**:
   - `git status` output: Currently on branch `main`, working tree clean (untracked `.agents/` directory present).
   - `git branch -a` shows branches `main` and `v1-stable`. Branch `feat/ltr-and-hurdle-architecture` does not yet exist.
2. **Current Python Dependencies**:
   - `pyproject.toml` lines 11–24:
     ```toml
     dependencies = [
         "numpy>=2.2.0",
         "pandas>=2.2.0",
         "scipy>=1.15.0",
         "scikit-learn>=1.5.0",
         "torch>=2.5.0",
         "seaborn>=0.13.0",
         "matplotlib>=3.10.0",
         "reportlab>=4.0.0",
         "pyarrow>=18.0.0",
         "streamlit>=1.40.0",
         "pyyaml>=6.0.0",
         "pytest>=8.0.0",
     ]
     ```
   - Python command `python -c "import lightgbm"` resulted in `ModuleNotFoundError: No module named 'lightgbm'`. LightGBM is not yet installed in the current environment (Python 3.13.7 on Windows x86_64).
3. **Existing Test Suite Baseline**:
   - `pytest tests/` executed successfully: 23 passed in 16.80s (`tests/test_grid_encoding.py`, `test_kmeans_scratch.py`, `test_linear_gd.py`, `test_pca_scratch.py`, `test_pipeline_integrity.py`, `test_qualifying_parser.py`, `test_race_folds.py`).
4. **Data Partitions & Contiguity**:
   - `train_core.parquet`: 8,920 rows across 408 races (1994–2016). DNF rate = 31.98% (`is_dnf == 1`: 2,853 rows). Mean DNF finishing position = 18.036.
   - `val.parquet`: 820 rows across 41 races (2017–2018). DNF rate = 21.95%.
   - `test.parquet`: 2,559 rows across 128 races (2019–2024). DNF rate = 14.46%.
   - Contiguity test: In all three parquet files, rows for each `raceId` are arranged in unbroken contiguous blocks (`changes == unique raceIds`).
5. **Feature Set Whitelist**:
   - `src/features/builder.py` lines 47–65 defines 17 `ALLOWED_FEATURES`:
     `grid_effective`, `grid_norm`, `grid_zero_flag`, `q_gap_pct`, `q1_data_available`, `quali_position_norm`, `grid_vs_quali_delta`, `driver_norm_finish_p5`, `driver_dnf_rate_p5`, `driver_podium_rate_career`, `is_rookie_window`, `teammate_form_delta`, `constructor_pts_share`, `constructor_season_rank`, `rounds_completed`, `driver_age_years`, `new_constructor_flag`.
   - `grid_vs_quali_delta` and `q_gap_pct` contain NaNs (e.g. for sprint races or when Q1 lap times are unavailable). Tree-based models (LightGBM, HistGradientBoosting) handle NaNs natively; neural networks (PyTorch Plackett-Luce) require median imputation.
6. **Existing Experiment Freeze (`EXPERIMENT_FREEZE.yaml`)**:
   - Records v5.0 thresholds: `knn: 0.2858`, `dt: 0.7142`, `svm: 0.8629`, `cnn: 0.8029`, `grid_heuristic: -3.1215`.
   - Records random seeds: numpy=42, torch=42, sklearn=42.

---

## 2. Logic Chain

1. **Problem Statement**:
   - In v5.0, pointwise classification with static decision thresholds ($\tau$) predicted erratic podium counts (0 to 5 per race). Simple post-hoc sorting (`rank() <= 3`) caused a 1.5–2.0 point F1 penalty due to discrete leaf probability ties and lack of pairwise ranking gradients.
   - In regression, historical DNF rate was 31.98% (falling to 14.46% in test). When a leading car suffers a mechanical retirement and is assigned position 18, pointwise squared error is $(18 - 2)^2 = 256$, capping test $R^2$ at $0.4080$.
2. **Listwise LTR Solution**:
   - Grouping training rows by `raceId` and passing group sizes allows LambdaMART (`lightgbm.LGBMRanker`) and Plackett-Luce ListMLE neural rankers to optimize NDCG directly.
   - Swaps at the front of the grid (P1–P3) dominate the LambdaMART gradient update $\lambda_{ij}$, producing continuous, non-tied latent pace scores.
   - Direct sorting on continuous latent pace guarantees strictly 3 unique podium finishers per race without arbitrary tie-breaking.
3. **Two-Stage Hurdle Solution**:
   - Decoupling mechanical reliability hazard from on-track pace via the Law of Total Probability:
     - Stage 1 models $P(\text{is\_dnf} = 1 \mid X)$ via calibrated classification.
     - Stage 2 models on-track pace $\hat{y}_{\text{finish}}$ or $P(\text{Podium} \mid \text{Finish})$ strictly on survivors (`is_classified == 1`).
     - Hurdle combination $\mathbb{E}[\text{Position}] = (1 - p_{\text{dnf}}) \cdot \hat{y}_{\text{finish}} + p_{\text{dnf}} \cdot \bar{y}_{\text{dnf}}$ absorbs the DNF penalty into the hazard term, eliminating catastrophic squared-error residuals on healthy cars.
4. **Architectural Separation**:
   - Adding modular files `src/evaluation/ranking_metrics.py`, `src/models/ltr_ranker.py`, `src/models/hurdle_model.py`, and `src/run_v6_experiments.py` ensures 100% backward compatibility with all v5.0 baselines, tables (1–7), and figures (1–11).

---

## 3. Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Environment | `lightgbm>=4.0.0` | Add LightGBM dependency for LambdaMART and GBDT estimators | `pyproject.toml` | Installed package `lightgbm` | Fail fast if version < 4.0.0 | Reference Plan §2, pyproject.toml |
| 2 | Environment | Git Branch `feat/ltr-and-hurdle-architecture` | Isolated feature development branch | Git repository | Clean branch checked out | Abort if uncommitted v5.0 diffs exist | ORIGINAL_REQUEST.md R1 |
| 3 | Metrics | `compute_dcg_at_k` | Discounted Cumulative Gain at rank cut-off $K$ | `relevance_array: np.ndarray`, `k: int` | `float` (DCG score) | Returns `0.0` if $k \le 0$ or array empty | Reference Plan §5.1, §6B |
| 4 | Metrics | `compute_ndcg_at_k` | Normalized DCG at rank cut-off $K$ comparing model ranking to ideal | `y_true_relevance: np.ndarray`, `y_pred_scores: np.ndarray`, `k: int` | `float` in $[0.0, 1.0]$ | Returns `0.0` if IDCG is 0 | Reference Plan §5.1, §6B |
| 5 | Metrics | `compute_mrr` | Mean Reciprocal Rank of the race winner ($1 / \text{rank}$) | `y_true_winner: np.ndarray`, `y_pred_scores: np.ndarray` | `float` in $(0.0, 1.0]$ | Returns `0.0` if no winner labeled in race | Reference Plan §6B |
| 6 | Metrics | `compute_kendall_tau` | Rank correlation ($\tau$) between predicted ranks and ground truth | `y_true_ranks: np.ndarray`, `y_pred_scores: np.ndarray` | `float` in $[-1.0, 1.0]$ | Returns `np.nan` or `1.0` if len < 2 | Reference Plan §6B |
| 7 | Metrics | `evaluate_race_ranking` | Race-level aggregation across all test races for NDCG@1, 3, 10, MRR, $\tau$, Top-3 P/R/F1 | `df_test: pd.DataFrame`, `score_col: str`, `target_col: str`, `k_values: tuple` | `dict[str, float]` containing mean metrics | Raises `KeyError` if required columns missing | Reference Plan §6B |
| 8 | LTR Dataset | `F1RankerDataset` | Assembles race-aligned contiguous feature matrices, group sizes, and relevance targets | `df: pd.DataFrame`, `target_mapping: str`, `feature_cols: list[str]` | `X: np.ndarray`, `y: np.ndarray`, `groups: np.ndarray` | Raises `ValueError` on invalid mapping name | Reference Plan §3, §6C |
| 9 | LTR Relevance | `f1_points` Mapping | F1 championship points: P1=25, P2=18, P3=15, P4=12, P5=10, P6=8, P7=6, P8=4, P9=2, P10=1, P11+=0 | `positionOrder`, `is_classified` | Non-negative integers $[0, 25]$ | Unclassified or P11+ receive 0 | Reference Plan §3 |
| 10 | LTR Relevance | `linear_rank` Mapping | Inverted rank $y_i = \max(0, m - \text{positionOrder}_i + 1)$ providing uniform gradient | `positionOrder`, `group_size` | Non-negative integers $[1, m]$ | Clamped at 0 for invalid inputs | Reference Plan §3 |
| 11 | LTR Relevance | `podium_tier` Mapping | Targeted podium tiers: P1=3, P2=2, P3=1, rest=0 | `positionOrder`, `is_classified` | Non-negative integers $[0, 3]$ | Unclassified receive 0 | Reference Plan §3 |
| 12 | LTR Model | `LambdaMARTRanker` | Wrapper around `lightgbm.LGBMRanker` (`objective='lambdarank'`) with early stopping | `X_train`, `y_train`, `group_train`, `X_val`, `y_val`, `group_val` | Trained model producing continuous latent pace $s_i$ | Validates group sum equals row count | Reference Plan §5.1, §6C |
| 13 | LTR Model | `PlackettLuceRanker` | PyTorch ListMLE neural ranker optimizing competitive elimination likelihood | `X_train`, `y_train_ranks`, `groups_train`, `epochs`, `lr` | Neural network predicting continuous pace $s_i$ | Numerical stability via `torch.logsumexp` | Reference Plan §5.2, §6C |
| 14 | LTR Contract | `predict_podium_top3` | Selects strictly 3 unique podium finishers per race with no duplicates or ties | `ranker: Any`, `df_race: pd.DataFrame` | Binary mask with exactly three 1s per race | Raises error if race has < 3 starters | Reference Plan §6C, ORIGINAL_REQUEST R3 |
| 15 | Hurdle Model | Stage 1 DNF Hazard Classifier | Predicts $P(\text{is\_dnf} = 1 \mid X)$ calibrated with isotonic regression or Platt scaling | `df[ALLOWED_FEATURES]`, `y_dnf = is_dnf` | Calibrated probabilities $p_{\text{dnf}} \in [0.0, 1.0]$ | Bounded within $[0, 1]$ | Reference Plan §5.3, §6D |
| 16 | Hurdle Model | Stage 2 Classified Pace Model | Models finish position $\hat{y}_{\text{finish}}$ or $P(\text{Podium} \mid \text{Finish})$ trained strictly on `is_classified == 1` | `df[df["is_classified"] == 1]` | Pure on-track finish position or podium proba | Fails if any DNF row leaks into training | Reference Plan §5.3, §6D |
| 17 | Hurdle Model | Hurdle Law of Total Probability Combination | Computes expected position $\mathbb{E}[\text{Pos}] = (1-p)\hat{y} + p\bar{y}_{\text{dnf}}$ and $P(\text{Podium}) = (1-p)P(\text{Podium}\mid\text{Finish})$ | Stage 1 $p_{\text{dnf}}$, Stage 2 $\hat{y}_{\text{finish}}$, $\bar{y}_{\text{dnf}}$ | Combined continuous expected rank and podium probability | Guaranteed mathematical bounds | Reference Plan §5.3, §6D |
| 18 | Orchestration | `src/run_v6_experiments.py` | Orchestrates side-by-side benchmark across baselines, pointwise models, LTR, and Hurdle | `train_core`, `val`, `test` parquets | `table8_ranking_hurdle_benchmark.csv`, updated `EXPERIMENT_FREEZE.yaml` | Validates exactly 3 podiums per race across all 128 test races | Reference Plan §6E |
| 19 | Reporting | Table 8 Side-by-Side Comparison | Comprehensive benchmark CSV and formatted Markdown table | Model predictions on test set | `reports/tables/table8_ranking_hurdle_benchmark.csv` | Overwrites cleanly without altering Tables 1–7 | Reference Plan §6E |
| 20 | Visualization | `plot_fig12_ranking_and_hurdle_eval` | 4-panel diagnostic plot at 300 DPI: NDCG@K, DNF Brier calibration curve, Podium F1, and residuals | Evaluation results and residuals | `reports/figures/fig12_ranking_and_hurdle_eval.png` | Strictly 300 DPI, whitegrid theme | Reference Plan §6E, ORIGINAL_REQUEST R6 |
| 21 | Verification | Expanded Test Suite | Automated tests for ranking metrics, LTR rankers, hurdle models | Test data in `tests/` | 100% pass across existing (23) and new tests | Fails on any contract violation | Reference Plan §6F, §7 |

---

## 4. Edge Cases

| # | Feature | Input | Observed / Required Behavior |
|---|---------|-------|-------------------------------|
| 1 | `compute_ndcg_at_k` | All relevance values are 0 in race ($IDCG = 0$) | When no items have positive relevance, return `0.0` (matching `sklearn.metrics.ndcg_score`). Do not divide by zero. |
| 2 | `compute_ndcg_at_k` | Cut-off $K > m$ (e.g. $K=10$ on a race with 8 cars) | Evaluate NDCG up to $\min(K, m)$ items. Never raise `IndexError`. |
| 3 | `compute_mrr` | No driver has `positionOrder == 1` in input slice | Return `0.0` gracefully instead of dividing by zero or raising `StopIteration`. |
| 4 | `compute_kendall_tau` | Race with fewer than 2 cars ($m < 2$) or all identical predicted scores | Return `1.0` (for $m \le 1$) or `0.0` with tie handling via `scipy.stats.kendalltau`. |
| 5 | `predict_podium_top3` | Exact score ties between 2 or more drivers (e.g. identical leaf probabilities in Decision Tree) | Enforce deterministic secondary tie-breaking by `grid_effective` ascending (driver starting closer to pole gets precedence), guaranteeing strictly 3 unique podium finishers. |
| 6 | `predict_podium_top3` | Multi-race DataFrame passed at once | Correctly group by `raceId`, select exactly 3 winners per race, and preserve original row index order. Output mask must have $\sum y_{\text{pred}} = 3 \times N_{\text{races}}$. |
| 7 | `F1RankerDataset` | Relevance mapping with unclassified starters in top 10 (e.g. Monaco 1996) | For `f1_points` and `podium_tier`, require `is_classified == 1` to score points/podium tier (unclassified drivers receive 0). For `linear_rank`, assign rank based on `positionOrder`. |
| 8 | `F1RankerDataset` | DataFrames with non-contiguous or unsorted `raceId` rows | Explicitly sort by `['raceId', 'positionOrder']` (or `raceId`) prior to splitting into $X, y$ and computing `group = df.groupby('raceId', sort=False).size()`, ensuring LightGBM group alignment. |
| 9 | `LambdaMARTRanker` | Early stopping when validation NDCG plateaus | Register early stopping callback with `stopping_rounds=15`, logging the best iteration and rolling back to best iteration for `.predict()`. |
| 10 | `PlackettLuceRanker` | Races with variable grid sizes ($m \in [18, 26]$) | Implement ListMLE loss with dynamic summation boundary $j \ge i$. Use `torch.logsumexp` to prevent numerical underflow/overflow. |
| 11 | `F1HurdleModel` | Starters with 100% predicted DNF probability ($p_{\text{dnf}} = 1.0$) | Formula yields $\mathbb{E}[\text{Pos}] = \bar{y}_{\text{dnf}} \approx 18.0$ and $P(\text{Podium}) = 0.0$. Obey total probability boundaries strictly. |
| 12 | `F1HurdleModel` | Feature missingness in `ALLOWED_FEATURES` (`q_gap_pct`, `grid_vs_quali_delta`) | Use `SimpleImputer(strategy='median')` or native tree handling (`HistGradientBoostingClassifier`, `LGBMRanker`) to prevent NaN crashes. |

---

## 5. Detailed Component Specifications (R1 through R8)

### R1: Environment, Version Control, and Dependencies
- **Git Branch**: Checkout or create branch `feat/ltr-and-hurdle-architecture` from current `main`.
- **`pyproject.toml`**: Add `"lightgbm>=4.0.0"` to `project.dependencies`.
- **Installation**: Run `pip install lightgbm>=4.0.0` inside the active environment.
- **Backward Compatibility**: Ensure all 23 existing unit tests pass before and after adding LightGBM.

### R2: Race-Aligned Ranking Metrics (`src/evaluation/ranking_metrics.py`)
1. **Mathematical Definitions**:
   - **Discounted Cumulative Gain (DCG@K)**:
     $$\text{DCG}@K(\boldsymbol{y}) = \sum_{r=1}^{\min(K, m)} \frac{2^{y_{(r)}} - 1}{\log_2(r + 1)}$$
     where $y_{(r)}$ is the relevance of the item placed at predicted rank $r$.
   - **Normalized DCG (NDCG@K)**:
     $$\text{NDCG}@K = \frac{\text{DCG}@K(\boldsymbol{y})}{\text{IDCG}@K(\boldsymbol{y}^*)}, \quad \text{where } \text{IDCG}@K = \text{DCG}@K(\boldsymbol{y}^*)$$
     with $\boldsymbol{y}^*$ denoting true relevance sorted in ideal descending order. If $\text{IDCG} = 0$, $\text{NDCG} = 0.0$.
   - **Mean Reciprocal Rank (MRR)**:
     $$\text{MRR} = \frac{1}{|R|} \sum_{r \in R} \frac{1}{\text{rank}_r(\text{winner})}$$
     where $\text{rank}_r(\text{winner})$ is the 1-based predicted rank of the true race winner (`positionOrder == 1`).
   - **Kendall's Tau ($\tau$)**:
     Rank correlation between predicted ranks and ground truth ranks:
     $$\tau = \frac{C - D}{\frac{1}{2} m (m - 1)}$$
     where $C$ is concordant pairs and $D$ is discordant pairs, computed via `scipy.stats.kendalltau(true_ranks, predicted_ranks)`.
   - **Top-3 Podium Metrics**:
     For predicted podium set $\hat{P}_r$ and actual podium set $P_r^*$:
     $$\text{Precision}@3 = \frac{|\hat{P}_r \cap P_r^*|}{3}, \quad \text{Recall}@3 = \frac{|\hat{P}_r \cap P_r^*|}{|P_r^*|}, \quad F_1@3 = \frac{2 \cdot P \cdot R}{P + R}$$
2. **Interface Functions**:
   ```python
   def compute_dcg_at_k(relevance_array: np.ndarray, k: int) -> float: ...
   def compute_ndcg_at_k(y_true_relevance: np.ndarray, y_pred_scores: np.ndarray, k: int = 3) -> float: ...
   def compute_mrr(y_true_winner: np.ndarray, y_pred_scores: np.ndarray) -> float: ...
   def compute_kendall_tau(y_true_ranks: np.ndarray, y_pred_scores: np.ndarray) -> float: ...
   def evaluate_race_ranking(
       df_test: pd.DataFrame,
       score_col: str,
       target_col: str = "positionOrder",
       k_values: tuple[int, ...] = (1, 3, 10),
       relevance_mapping: str = "linear_rank",
   ) -> dict[str, float]: ...
   ```

### R3: Listwise Learning-to-Rank Architecture (`src/models/ltr_ranker.py`)
1. **`F1RankerDataset`**:
   - Configurable relevance mappings:
     - `'f1_points'`: {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}, 0 otherwise. Classified starters only.
     - `'linear_rank'`: $y_i = \max(0, m - \text{positionOrder}_i + 1)$.
     - `'podium_tier'`: {1: 3, 2: 2, 3: 1}, 0 otherwise. Classified starters only.
   - Outputs: $X$ (`float32`), $y$ (`int32`), `groups` (`list[int]` or `np.ndarray`).
2. **`LambdaMARTRanker`**:
   - Encapsulates `lightgbm.LGBMRanker`:
     - `objective='lambdarank'`
     - `metric='ndcg'` with `eval_at=[1, 3, 10]`
     - `n_estimators=100`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=15`, `random_state=42`
     - Early stopping on validation NDCG (`val.parquet`) with `stopping_rounds=15`.
3. **`PlackettLuceRanker`**:
   - PyTorch module with a 3-layer MLP: `Linear(17, 64) -> ReLU() -> Dropout(0.2) -> Linear(64, 32) -> ReLU() -> Linear(32, 1)`.
   - ListMLE Permutation Loss:
     $$\mathcal{L}_{\text{ListMLE}}(\boldsymbol{s}, \pi) = - \sum_{i=1}^{m-1} \left( s_{\pi_i} - \text{logsumexp}(s_{\pi_{i:m}}) \right)$$
   - Optimization: Adam (`lr=1e-3`, `weight_decay=1e-4`), validation early stopping.
4. **`predict_podium_top3` Contract**:
   - Accepts any ranker or score column and a race dataframe.
   - Predicts scores, breaks any ties by `grid_effective` ascending, and flags the top 3 drivers per race (`y_pred == 1`).
   - Guarantees strictly 3 unique drivers per race.

### R4: Two-Stage Hurdle Architecture (`src/models/hurdle_model.py`)
1. **Decoupled Architecture**:
   - **Stage 1 (DNF Hazard Classifier)**:
     - Fits $P(\text{is\_dnf} = 1 \mid X)$ on all starters using `HistGradientBoostingClassifier(loss='log_loss', max_depth=4, l2_regularization=1.0, random_state=42)`.
     - Calibrated on `val.parquet` via `CalibratedClassifierCV(..., method='sigmoid', cv='prefit')`.
     - Outputs calibrated probabilities $p_{\text{dnf}} \in [0, 1]$.
   - **Stage 2 (Classified Pace Model)**:
     - Trained strictly on classified starters: `df[df["is_classified"] == 1]`.
     - Dual output support:
       - Regressor for on-track finish position $\hat{y}_{\text{finish}} = \mathbb{E}[\text{Position} \mid \text{Finish}, X]$.
       - Classifier or ranker for $P(\text{Podium} \mid \text{Finish}, X)$.
2. **Combination Formulas (Law of Total Probability)**:
   - Expected finishing position:
     $$\mathbb{E}[\text{Position}_i \mid X_i] = (1 - p_{\text{dnf}, i}) \cdot \hat{y}_{\text{finish}, i} + p_{\text{dnf}, i} \cdot \bar{y}_{\text{dnf}}$$
     where $\bar{y}_{\text{dnf}}$ is dynamically calculated on training data ($\approx 18.03$).
   - Hurdle podium probability:
     $$P(\text{Podium}_i \mid X_i) = (1 - p_{\text{dnf}, i}) \cdot P(\text{Podium}_i \mid \text{Finish}_i, X_i)$$
   - Guaranteed top-3 podium selection per race based on lowest $\mathbb{E}[\text{Position}]$ or highest $P(\text{Podium})$.

### R5: Experiment Orchestration & Benchmark (`src/run_v6_experiments.py`)
1. **Evaluated Models**:
   - Baselines: Grid Heuristic, Constructor Season Rank Heuristic.
   - Pointwise v5.0 Models: KNN, Decision Tree, SVM (RBF), Dual-Branch 1D-CNN (with both static $\tau$ and post-hoc `rank() <= 3`).
   - Listwise LTR Models: LambdaMART (`f1_points`), LambdaMART (`linear_rank`), Plackett-Luce Neural Ranker.
   - Two-Stage Hurdle Model: DNF Hazard + Classified Pace Regressor/Ranker.
2. **Artifact Outputs**:
   - `reports/tables/table8_ranking_hurdle_benchmark.csv`:
     Columns: `Model`, `Architecture`, `Decision_Rule`, `NDCG@1`, `NDCG@3`, `NDCG@10`, `MRR`, `Kendall_Tau`, `Podium_Precision`, `Podium_Recall`, `Podium_F1`, `Test_RMSE`, `Test_R2_All`, `Test_R2_Classified`, `Key_Parameters`.
   - Printed Markdown table formatting before vs after results side-by-side.
   - Update `EXPERIMENT_FREEZE.yaml` under `v6_ranking_and_hurdle` with the winning relevance mapping, hyperparameter configurations, and frozen evaluation metrics.

### R6: Diagnostic Visualizations (`reports/figures/fig12_ranking_and_hurdle_eval.png`)
- Implemented as `plot_fig12_ranking_and_hurdle_eval()` in `src/visualization/plots.py`.
- Canvas: 2x2 grid, 300 DPI, whitegrid styling.
- **Panel 1 (Top-Left)**: NDCG@K comparison (NDCG@1, NDCG@3, NDCG@10) across Pointwise vs LambdaMART vs Plackett-Luce vs Hurdle.
- **Panel 2 (Top-Right)**: Hurdle Stage 1 DNF Reliability Curve (calibration curve comparing mean predicted hazard to empirical DNF rate, with Brier score annotation).
- **Panel 3 (Bottom-Left)**: Test Podium F1 comparison (Static thresholding $\tau$ vs Post-hoc `rank() <= 3` vs LTR vs Hurdle).
- **Panel 4 (Bottom-Right)**: Regression residual error distribution ($y_i - \hat{y}_i$) comparing standard OLS against Hurdle expected position (illustrating elimination of the 256 squared-error tail).

### R7: Test Suite Expansion
1. `tests/test_ranking_metrics.py`:
   - Verifies DCG/NDCG analytical accuracy on hand-crafted vectors.
   - Verifies MRR, Kendall's Tau, and Top-3 Precision/Recall/F1.
   - Tests edge cases: ties, empty inputs, all-zero relevance, races with 1 driver.
2. `tests/test_ltr_ranker.py`:
   - Verifies `F1RankerDataset` shapes, group sums, and all 3 relevance mappings.
   - Verifies `LambdaMARTRanker` training, prediction, and early stopping.
   - Verifies `PlackettLuceRanker` forward pass, ListMLE loss computation, and backprop.
   - Asserts `predict_podium_top3` returns strictly 3 unique drivers per race across 128 test races.
3. `tests/test_hurdle_model.py`:
   - Asserts Stage 1 probabilities $p_{\text{dnf}} \in [0, 1]$.
   - Asserts Stage 2 is fitted on zero DNF rows (`assert (y_train_stage2_is_dnf == 0).all()`).
   - Asserts Hurdle expected positions and podium probabilities obey total probability bounds.
   - Asserts top-3 selection produces exactly 3 unique drivers per race.
- Total test target: 23 existing tests + new ranking & hurdle tests all passing 100%.

### R8: Code Quality, ELI5 Analogies, and Walkthrough Documentation
- Include child-friendly ELI5 analogies at the top of every module and class:
  - **Learning-to-Rank (LambdaMART)**: "Instead of asking 'Is this runner good or bad?' in isolation, LTR holds a mini-tournament between all runners in the race, giving 100x more coaching attention to who gets the Gold, Silver, and Bronze medals rather than who finished 15th vs 16th."
  - **Two-Stage Hurdle**: "Imagine running a race with an unpredictable slippery banana peel at Turn 1. Hurdle uses two brains: Brain 1 predicts if your engine will break down or slip on a peel; Brain 2 predicts how fast you drive assuming your car stays in one piece. We then combine them with the Law of Total Probability so a random mechanical breakdown doesn't ruin the model's estimate of a fast driver's true speed."
- Strict Python 3.11+ type hints (`from __future__ import annotations`, `np.ndarray`, `pd.DataFrame`).
- Update project walkthrough documentation upon milestone completion.

---

## 6. Caveats

1. **LightGBM Wheel Compatibility**:
   - `lightgbm` must be installed on Windows with Python 3.13. LightGBM wheels for Windows x86_64 are available up to 4.7.0. If binary installation fails on Python 3.13, a pre-compiled wheel or conda-forge binary can be used.
2. **Missing Feature Values in Allowed Features**:
   - `grid_vs_quali_delta` and `q_gap_pct` contain NaNs (masked for sprint races or when Q1 data is missing). PyTorch ListMLE models must include a median imputer in their pipeline, whereas LightGBM and HistGradientBoosting handle NaNs natively.
3. **No Code Implementation**:
   - As an Explorer/Spec Miner, zero production code has been modified in this turn. All findings and formulas are ready for Builder agents.

---

## 7. Conclusion

The specification for the Learning-to-Rank (LTR) and Two-Stage Hurdle Architecture is complete, mathematically rigorous, and verified against the existing v5.0 codebase.
All 8 requirements (R1–R8) have concrete interface definitions, mathematical formulas, edge case handlers, and verification criteria. Implementation can proceed with zero ambiguity.

---

## 8. Verification Method

To verify the findings of this report independently:
1. Verify existing test suite baseline:
   ```powershell
   pytest tests/ -v
   ```
   (Expected: 23 passed in ~16s).
2. Verify LightGBM missing status:
   ```powershell
   python -c "import lightgbm"
   ```
   (Expected: `ModuleNotFoundError`).
3. Verify data partition sizes and DNF statistics:
   ```powershell
   python -c "import pandas as pd; df = pd.read_parquet('data/processed/train_core.parquet'); print(df['is_dnf'].mean(), df['raceId'].nunique())"
   ```
   (Expected: DNF mean = 0.3198, 408 races).
4. Verify parquet contiguity:
   ```powershell
   python -c "import pandas as pd; df = pd.read_parquet('data/processed/test.parquet'); rids = df['raceId'].to_numpy(); print((rids[:-1] != rids[1:]).sum() + 1, df['raceId'].nunique())"
   ```
   (Expected: 128 contiguous blocks, 128 unique races).
