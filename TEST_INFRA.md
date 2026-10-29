# E2E Test Infra: Formula 1 Learning-to-Rank & Two-Stage Hurdle Architecture

## Test Philosophy
- Requirement-driven, opaque-box testing derived strictly from `ORIGINAL_REQUEST.md`.
- No reliance on implementation internals; verifies mathematical bounds, output types, non-duplicate constraints, and edge case resilience.
- Methodology: 4-Tier Hierarchical Testing (Category-Partition, Boundary Value Analysis, Pairwise Combinations, Real-World Application Scenarios).

## Feature Inventory & Test Mapping
| # | Feature | Source | Tier 1 (Coverage >=5) | Tier 2 (Boundary >=5) | Tier 3 (Pairwise) | Tier 4 (Real-World) |
|---|---------|--------|:---------------------:|:---------------------:|:-----------------:|:-------------------:|
| 1 | Ranking Metrics (NDCG, MRR, Tau, Top-3 P/R/F1) | R2 | >= 5 tests | >= 5 tests | Yes | Yes |
| 2 | LTR Dataset & Target Mappings (f1_points, linear_rank, podium_tier) | R3 | >= 5 tests | >= 5 tests | Yes | Yes |
| 3 | LambdaMART GBDT Ranker | R3 | >= 5 tests | >= 5 tests | Yes | Yes |
| 4 | Plackett-Luce Neural Ranker (PyTorch ListMLE) | R3 | >= 5 tests | >= 5 tests | Yes | Yes |
| 5 | Strictly Guaranteed Top-3 Podium Selection | R3 | >= 5 tests | >= 5 tests | Yes | Yes |
| 6 | Two-Stage Hurdle Model (Stage 1 Hazard + Stage 2 Pace) | R4 | >= 5 tests | >= 5 tests | Yes | Yes |
| 7 | Hurdle Total Probability Combination | R4 | >= 5 tests | >= 5 tests | Yes | Yes |
| 8 | Master Benchmark Orchestration (Table 8 & Markdown) | R5 | >= 5 tests | >= 5 tests | Yes | Yes |
| 9 | Diagnostic Visualizations (Figure 12 at 300 DPI) | R6 | >= 5 tests | >= 5 tests | Yes | Yes |

## Test Architecture
- Test runner: `pytest tests/ -v --tb=short`
- Test files:
  - `tests/test_ranking_metrics.py`: DCG/NDCG analytical accuracy, MRR, Kendall's Tau, Top-3 P/R/F1, edge cases (ties, all-zero, single driver).
  - `tests/test_ltr_ranker.py`: `F1RankerDataset` query shapes, 3 relevance mappings, LambdaMART fit/predict, Plackett-Luce forward/loss/convergence, `predict_podium_top3` non-duplicate guarantee.
  - `tests/test_hurdle_model.py`: Stage 1 calibration $p_{\text{dnf}} \in [0, 1]$, Stage 2 training purity (zero DNFs), Total Probability recombination bounds, top-3 podium selection.
  - Existing 23 baseline tests in `tests/`: 100% pass guarantee.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Expected Outcome |
|---|----------|--------------------|------------------|
| 1 | Full 41-Race Validation Split Evaluation | Metrics, LTR, Hurdle | Zero crashes, exact 3 podiums/race, NDCG@3 > 0.70 |
| 2 | Full 128-Race Modern Era Test Split (2019–2024) | LambdaMART, Hurdle, Top-3 | Exactly 384 podium selections (3 x 128), zero duplicate drivers |
| 3 | Sprint Race Weekend with Missing Quali Times (NaNs) | LightGBM, Hurdle, SimpleImputer | Handled gracefully without crash or NaN score leakage |
| 4 | High-Attrition Chaos Race (e.g. 50%+ DNFs) | Hurdle Stage 1 & Stage 2 | Expected position penalizes DNFs, podium selected from healthy cars |
| 5 | Pre-2010 Variable Grid Size Race (e.g. 24 cars) | Plackett-Luce ListMLE, LambdaMART | Permutation loss dynamic boundaries succeed, group sum matches rows |

## Acceptance Criteria
- Full test suite passes: `pytest tests/` with 0 failures, 0 errors, 100% pass rate.
- `TEST_READY.md` generated upon test suite completion.
