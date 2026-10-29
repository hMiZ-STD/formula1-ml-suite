# Project: Formula 1 Learning-to-Rank (LTR) & Two-Stage Hurdle Architecture (v6.0-dev)

## Architecture
The v6.0 architecture elevates the Formula 1 Machine Learning Engineering Suite from pointwise single-driver scoring to listwise race permutations and decoupled mechanical attrition modeling:
1. **Race-Aligned Ranking Metrics (`src/evaluation/ranking_metrics.py`)**: Computes race-level NDCG@1/3/10, MRR, Kendall's Tau, and Top-3 Podium Precision/Recall/F1, accounting for zero-sum intra-race competition.
2. **Learning-to-Rank Architecture (`src/models/ltr_ranker.py`)**: 
   - `F1RankerDataset`: Assembles race-contiguous feature matrices with query group sizes and configurable relevance targets (`f1_points`, `linear_rank`, `podium_tier`).
   - `LambdaMARTRanker`: GBDT listwise ranker wrapping `lightgbm.LGBMRanker` (`objective='lambdarank'`) with early stopping on validation NDCG.
   - `PlackettLuceRanker`: Deep neural ranker in PyTorch optimizing the Plackett-Luce ListMLE permutation likelihood via `torch.logsumexp`.
   - `predict_podium_top3`: Enforces deterministic top-3 podium selection strictly guaranteeing 3 unique drivers per race.
3. **Two-Stage Hurdle Architecture (`src/models/hurdle_model.py`)**:
   - Stage 1: Calibrated DNF hazard classifier predicting $P(\text{is\_dnf} = 1 \mid X)$.
   - Stage 2: Classified pace regressor/ranker trained exclusively on classified starters (`is_classified == 1`), eliminating catastrophic squared-error DNF tails.
   - Recombination via the Law of Total Probability:
     $$\mathbb{E}[\text{Pos}] = (1 - p_{\text{dnf}})\hat{y}_{\text{finish}} + p_{\text{dnf}}\bar{y}_{\text{dnf}}$$
     $$P(\text{Podium}) = (1 - p_{\text{dnf}}) P(\text{Podium} \mid \text{Finish})$$
4. **Master Benchmark & Diagnostics (`src/run_v6_experiments.py`, `src/visualization/plots.py`)**:
   - Orchestrates side-by-side evaluation across baselines, pointwise v5.0 models, LTR rankers, and Two-Stage Hurdle architectures.
   - Generates `reports/tables/table8_ranking_hurdle_benchmark.csv` and side-by-side Markdown comparison table.
   - Generates `reports/figures/fig12_ranking_and_hurdle_eval.png` (4-panel diagnostic plot at 300 DPI).
   - Updates `EXPERIMENT_FREEZE.yaml` under `v6_ranking_and_hurdle`.
5. **Zero-Breaking-Change Guarantee**: 100% backward compatibility with v5.0 baselines, code, tables (Tables 2–6b), and figures (Figures 1–11).

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Git Branch `feat/ltr-and-hurdle-architecture` | Create and checkout isolated feature development branch | M1 | ORIGINAL_REQUEST R1 |
| 2 | Dependency `lightgbm>=4.0.0` | Add LightGBM dependency to `pyproject.toml` and install in environment | M1 | ORIGINAL_REQUEST R1 |
| 3 | Loky Warning Suppression | Add `tests/conftest.py` setting `LOKY_MAX_CPU_COUNT` for clean test execution | M1 | Survey Explorer 3 |
| 4 | Discounted Cumulative Gain (`compute_dcg_at_k`) | Logarithmic rank discount computation up to cut-off K | M2 | Reference Plan §5.1 |
| 5 | Normalized DCG (`compute_ndcg_at_k`) | Normalized DCG against ideal permutation; handles IDCG=0 gracefully | M2 | ORIGINAL_REQUEST R2 |
| 6 | Mean Reciprocal Rank (`compute_mrr`) | Reciprocal rank of true race winner (`positionOrder == 1`) | M2 | ORIGINAL_REQUEST R2 |
| 7 | Kendall's Tau (`compute_kendall_tau`) | Pairwise concordant vs discordant rank correlation across race grid | M2 | ORIGINAL_REQUEST R2 |
| 8 | Race Ranking Evaluator (`evaluate_race_ranking`) | Race-level aggregation across test set for NDCG@1/3/10, MRR, Tau, Top-3 P/R/F1 | M2 | ORIGINAL_REQUEST R2 |
| 9 | LTR Race Dataset (`F1RankerDataset`) | Assembles contiguous race groups, queries, and relevance mappings | M3 | Reference Plan §3 |
| 10 | Target Mappings | Implements `f1_points`, `linear_rank`, and `podium_tier` target schemes | M3 | ORIGINAL_REQUEST R3 |
| 11 | LambdaMART Ranker (`LambdaMARTRanker`) | LightGBM GBDT ranker optimizing NDCG with validation early stopping | M3 | ORIGINAL_REQUEST R3 |
| 12 | Plackett-Luce Ranker (`PlackettLuceRanker`) | PyTorch ListMLE permutation neural ranker with `torch.logsumexp` stability | M3 | ORIGINAL_REQUEST R3 |
| 13 | Guaranteed Top-3 Selection (`predict_podium_top3`) | Selects strictly 3 unique drivers per race with secondary tie-breaking on `grid_effective` | M3 | ORIGINAL_REQUEST R3 |
| 14 | Hurdle Stage 1 DNF Hazard Classifier | Predicts $P(\text{is\_dnf} = 1 \mid X)$ calibrated via Platt scaling / isotonic regression | M4 | ORIGINAL_REQUEST R4 |
| 15 | Hurdle Stage 2 Classified Pace Model | Models pure on-track finish position and podium probability trained strictly on `is_classified == 1` | M4 | ORIGINAL_REQUEST R4 |
| 16 | Hurdle Total Probability Recombination | Law of Total Probability combining hazard and pace for expected rank and podium proba | M4 | ORIGINAL_REQUEST R4 |
| 17 | Hurdle Top-3 Guaranteed Selection | Top-3 podium selection strictly producing 3 unique drivers per race from Hurdle outputs | M4 | ORIGINAL_REQUEST R4 |
| 18 | Master Benchmark Script (`src/run_v6_experiments.py`) | Evaluates baselines, pointwise v5.0, LTR, and Hurdle models side-by-side | M5 | ORIGINAL_REQUEST R5 |
| 19 | Table 8 Benchmark Report | Generates `reports/tables/table8_ranking_hurdle_benchmark.csv` and Markdown table | M5 | ORIGINAL_REQUEST R5 |
| 20 | Experiment Freeze Update | Appends `v6_ranking_and_hurdle` configuration and metrics to `EXPERIMENT_FREEZE.yaml` | M5 | ORIGINAL_REQUEST R5 |
| 21 | Diagnostic Figure 12 (`plot_fig12_ranking_and_hurdle_eval`) | 4-panel diagnostic plot at 300 DPI in `reports/figures/fig12_ranking_and_hurdle_eval.png` | M6 | ORIGINAL_REQUEST R6 |
| 22 | Ranking Metrics Test Suite | Automated test coverage in `tests/test_ranking_metrics.py` | M7 / E2E Track | ORIGINAL_REQUEST R7 |
| 23 | LTR Ranker Test Suite | Automated test coverage in `tests/test_ltr_ranker.py` | M7 / E2E Track | ORIGINAL_REQUEST R7 |
| 24 | Hurdle Model Test Suite | Automated test coverage in `tests/test_hurdle_model.py` | M7 / E2E Track | ORIGINAL_REQUEST R7 |
| 25 | Documentation & Pedagogical Analogies | ELI5 child-friendly analogies, docstrings, typing, project walkthrough update | M7 | ORIGINAL_REQUEST R8 |
| 26 | Git Commit | Commit all changes cleanly on `feat/ltr-and-hurdle-architecture` | M7 | Acceptance Criteria |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Environment & Version Control | Git branch `feat/ltr-and-hurdle-architecture`, `lightgbm>=4.0.0` in `pyproject.toml` and installed, `tests/conftest.py` setup | none | DONE |
| M2 | Race-Aligned Ranking Metrics | `src/evaluation/ranking_metrics.py` implementing NDCG@1/3/10, MRR, Kendall's Tau, Top-3 P/R/F1, and unit tests | M1 | DONE |
| M3 | Learning-to-Rank Architecture | `src/models/ltr_ranker.py` (`F1RankerDataset`, `LambdaMARTRanker`, `PlackettLuceRanker`, `predict_podium_top3`) | M1, M2 | DONE |
| M4 | Two-Stage Hurdle Architecture | `src/models/hurdle_model.py` (`F1HurdleModel` Stage 1 DNF hazard, Stage 2 pace model, Total Probability combination, top-3 selection) | M1, M2 | DONE |
| M5 | Experiment Runner & Table 8 Benchmark | `src/run_v6_experiments.py`, `table8_ranking_hurdle_benchmark.csv`, update `EXPERIMENT_FREEZE.yaml` | M3, M4 | DONE |
| M6 | Diagnostic Visualizations | `plot_fig12_ranking_and_hurdle_eval()` in `src/visualization/plots.py`, `reports/figures/fig12_ranking_and_hurdle_eval.png` | M5 | DONE |
| M7 | Final Milestone: E2E Validation, Coverage Hardening, Documentation & Git Commit | Pass 100% E2E tests (Tiers 1-4), Tier 5 adversarial hardening, ELI5 analogies, walkthrough update, git commit | M2, M3, M4, M5, M6, E2E Track | DONE |

---

## Interface Contracts

### `src/evaluation/ranking_metrics.py`
```python
def compute_dcg_at_k(relevance: np.ndarray, k: int) -> float: ...
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

### `src/models/ltr_ranker.py`
```python
class F1RankerDataset:
    def __init__(self, target_mapping: str = "f1_points", feature_cols: list[str] | None = None): ...
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class LambdaMARTRanker:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, num_leaves: int = 31, random_state: int = 42): ...
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, group_val: np.ndarray | None = None,
            early_stopping_rounds: int = 15) -> LambdaMARTRanker: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

class PlackettLuceRanker:
    def __init__(self, input_dim: int = 17, hidden_dim: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 50, random_state: int = 42): ...
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, group_val: np.ndarray | None = None) -> PlackettLuceRanker: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

def predict_podium_top3(scores: np.ndarray | pd.Series, df_race: pd.DataFrame) -> np.ndarray: ...
```

### `src/models/hurdle_model.py`
```python
class F1HurdleModel:
    def __init__(self, stage1_estimator: Any | None = None, stage2_estimator: Any | None = None, random_state: int = 42): ...
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None) -> F1HurdleModel: ...
    def predict_hazard(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_classified_pace(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_expected_position(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_podium_probability(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_podium_top3(self, df: pd.DataFrame) -> np.ndarray: ...
```

---

## Code Layout
```
d:\ML-Project\
├── pyproject.toml                         # Project metadata & dependencies (add lightgbm>=4.0.0)
├── EXPERIMENT_FREEZE.yaml                 # Configuration & metric freeze (append v6_ranking_and_hurdle)
├── CODING_STANDARDS.md                    # Pedagogical & coding guidelines
├── src/
│   ├── evaluation/
│   │   ├── classification_metrics.py      # [v5.0 unchanged]
│   │   ├── regression_metrics.py          # [v5.0 unchanged]
│   │   └── ranking_metrics.py             # [NEW v6.0] NDCG, MRR, Kendall's Tau, Top-3 metrics
│   ├── models/
│   │   ├── classifiers.py                 # [v5.0 unchanged]
│   │   ├── cnn_classifier.py              # [v5.0 unchanged]
│   │   ├── ltr_ranker.py                  # [NEW v6.0] LambdaMART & Plackett-Luce neural rankers
│   │   └── hurdle_model.py                # [NEW v6.0] Two-Stage Hurdle Architecture
│   ├── visualization/
│   │   └── plots.py                       # [v6.0 update] add plot_fig12_ranking_and_hurdle_eval()
│   └── run_v6_experiments.py              # [NEW v6.0] Master experiment runner & benchmark
├── tests/
│   ├── conftest.py                        # [NEW v6.0] Loky warning suppression & shared mock fixtures
│   ├── test_ranking_metrics.py            # [NEW v6.0] Tests for ranking metrics & edge cases
│   ├── test_ltr_ranker.py                 # [NEW v6.0] Tests for LTR dataset, LambdaMART, ListMLE
│   └── test_hurdle_model.py               # [NEW v6.0] Tests for Stage 1 hazard, Stage 2 pace, combination
└── reports/
    ├── tables/
    │   └── table8_ranking_hurdle_benchmark.csv # [NEW v6.0] Side-by-side benchmark results
    └── figures/
        └── fig12_ranking_and_hurdle_eval.png   # [NEW v6.0] 4-panel diagnostic plot at 300 DPI
```
