# Formula 1 End-to-End Machine Learning Engineering Suite

**Version:** 6.0 — Learning-to-Rank & Two-Stage Hurdle Release (100% backward compatible with v5.0)  
**Academic Target:** IIIT Sri City Machine Learning Assignment  
**Championship History:** 1950–2024 (75 Seasons, 1,125 Grand Prix)  
**Supervised Scope:** Modern Era (1994–2024, 577 Grand Prix, 12,299 clean starters)

[![Python 3.13](https://img.shields.io/badge/Python-3.13.7-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C.svg)](https://pytorch.org/)
[![LightGBM 4.6](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn 1.5](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-76%20passed-brightgreen.svg)](tests/)
[![Report](https://img.shields.io/badge/ReportLab-PDF%20Compiled-red.svg)](reports/final_assignment_report.pdf)

---

## Overview

Formula 1 is an extreme predictive environment governed by temporal non-stationarity, aerodynamic regulation resets, and coupled zero-sum competition. Standard textbook ML workflows fail here due to:

1. **Data corruption** — Pre-1994 shared-car drives produce up to 7 "podium finishers" per race.
2. **Leakage** — Un-grouped rolling features or row-level constructor aggregations leak in-race telemetry into pre-race inputs.
3. **DNF noise** — Mechanical retirements inflate regression errors catastrophically: a leading car retiring on lap 52 from P1 becomes classified P18, generating $(18-1)^2 = 289$ squared error despite a perfect pace prediction.

This suite solves all three with a rigorous, anti-leakage temporal pipeline and two new v6.0 architectures.

---

## Architecture at a Glance

```
d:/ML-Project/
├── EXPERIMENT_FREEZE.yaml          # Sealed hyperparameters, thresholds, seeds (v5 + v6)
├── pyproject.toml                  # Package metadata and pytest config
│
├── data/
│   ├── raw/                        # 14 raw Ergast CSV tables (gitignored)
│   └── processed/
│       ├── audit/                  # Frozen audit artifacts (JSON + CSV)
│       ├── train_core.parquet      # 1994–2016 (8,920 starters, 408 races)
│       ├── val.parquet             # 2017–2018 (820 starters, 41 races)
│       ├── test.parquet            # 2019–2024 hybrid era (2,559 starters, 128 races)
│       └── career_clustering.parquet
│
├── src/
│   ├── data/
│   │   ├── audit.py                # Table 1 audit and artifact freezing
│   │   ├── loader.py               # Non-starter filtering, canonical constructor map
│   │   └── preprocessor.py        # Temporal splits, race-aligned CV folds
│   ├── features/
│   │   └── builder.py             # Leakage-safe pre-race feature engineering
│   ├── models/
│   │   ├── classifiers.py         # KNN, Decision Tree, SVM pipelines
│   │   ├── cnn_classifier.py      # Dual-Branch 1D-CNN (sequence + static fusion)
│   │   ├── ltr_ranker.py          # LambdaMART (LGBMRanker) + Plackett-Luce ListMLE [v6]
│   │   ├── hurdle_model.py        # Two-Stage DNF Hazard + Pace Ranker [v6]
│   │   ├── linear_gd.py           # Vectorized Gradient Descent from scratch
│   │   ├── regularized_reg.py     # Ridge & Lasso with degree-2 expansion
│   │   ├── polynomial_reg.py      # Polynomial bias-variance demo
│   │   ├── kmeans_scratch.py      # K-Means from scratch (pure NumPy)
│   │   └── pca_scratch.py         # PCA from scratch (sample covariance + eigh)
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── regression_metrics.py
│   │   ├── ranking_metrics.py     # NDCG@1/3/10, MRR, Kendall Tau, Podium F1 [v6]
│   │   └── standard_datasets.py   # load_digits, load_diabetes, load_iris appendices
│   ├── visualization/
│   │   └── plots.py               # Figures 1–12 (all 300 DPI)
│   ├── run_experiments.py         # Master experiment runner (v5.0)
│   └── run_v6_experiments.py      # v6 benchmark runner (LTR + Hurdle) [v6]
│
├── tests/                          # 76 automated tests (23 baseline + 53 new in v6)
├── notebooks/                      # 7 Jupyter notebooks
├── app/streamlit_app.py            # Interactive dashboard
├── reports/
│   ├── figures/                    # Figures 1–12 (PNG, 300 DPI)
│   ├── tables/                     # Tables 1–8 (CSV)
│   └── final_assignment_report.pdf # Full scientific PDF report (4.5 MB)
└── submission_code.zip             # Clean submission archive (7.27 MB, no raw CSVs)
```

---

## Mini-Projects (v5.0 Baseline)

### Mini-Project 1 — Classification: Predicting Podium Finishes (`y_podium`)

| Model | Test ROC-AUC | Test PR-AUC | Test F1 |
| :--- | :---: | :---: | :---: |
| Grid Heuristic (Baseline) | 0.915 | 0.660 | 0.682 |
| K-Nearest Neighbours | 0.936 | 0.701 | 0.684 |
| Decision Tree (Pruned) | 0.936 | 0.709 | 0.680 |
| Support Vector Machine (RBF) | 0.940 | 0.705 | 0.671 |
| Dual-Branch 1D-CNN | **0.937** | **0.719** | **0.679** |

### Mini-Project 2 — Regression: Predicting Finishing Position (`positionOrder`)

| Model | Test MSE | Test R² |
| :--- | :---: | :---: |
| Scratch Gradient Descent | 22.41 | 0.375 |
| OLS (Pseudo-inverse) | 19.67 | 0.408 |
| Ridge (degree-2) | 19.57 | 0.411 |
| Lasso (degree-2) | 19.57 | 0.411 |

### Mini-Project 3 — Unsupervised: Driver Career Archetypes (1950–2024)

K-Means from scratch (K=2, best silhouette 0.412) and PCA from scratch (PC1 explains 47.3% variance) applied to career-aggregated statistics for 857 drivers.

---

## v6.0 New Architectures — Before vs. After

### Why Pointwise Models Hit a Ceiling

Standard classification models (KNN, Decision Tree, SVM, CNN) treat each driver independently. In a 20-car race, they can predict 0 or 5 "podium finishers" — because they never see the field as a group. Post-hoc sorting drops F1 further due to probability ties.

**v6.0 fixes this with two architectures:**

### 1. Listwise Learning-to-Rank (LTR)

Groups all 20 cars per race and directly optimises NDCG (how well the model ranks the full starting grid). Automatically guarantees exactly 3 podium finishers per race.

- **LambdaMART** (`lightgbm.LGBMRanker`, `objective='lambdarank'`) with `linear_rank` relevance target — won the validation sweep (Val NDCG@1: 0.848, Val Podium F1: 0.683).
- **Plackett-Luce Neural Ranker** — PyTorch MLP minimising the ListMLE permutation log-likelihood loss.

### 2. Two-Stage Hurdle Architecture

Separates the question "will the car reach the finish line?" from "how fast was it?":
- **Stage 1** — DNF Hazard Classifier: predicts P(mechanical failure | X). Brier Score: 0.0883.
- **Stage 2** — Classified Pace Ranker: trained **only on cars that finished** (`is_classified == 1`), eliminating the squared-error penalty from engine blowups.

### Side-by-Side Benchmark (2019–2024 test set, 128 races)

| Model | NDCG@1 | NDCG@3 | NDCG@10 | Podium F1 | R² (Classified) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Grid Heuristic | 0.627 | 0.732 | 0.816 | 0.682 | 0.374 |
| Constructor Rank | 0.748 | 0.728 | 0.824 | 0.643 | 0.390 |
| **— v5.0 Pointwise (BEFORE) —** | | | | | |
| KNN (best threshold) | 0.615 | 0.726 | 0.821 | 0.684 | 0.639 |
| Decision Tree (best threshold) | 0.861 | 0.809 | 0.896 | 0.680 | 0.643 |
| SVM RBF (best threshold) | 0.567 | 0.697 | 0.804 | 0.671 | 0.498 |
| Dual-Branch 1D-CNN | 0.671 | 0.735 | 0.837 | 0.679 | 0.492 |
| **— v6.0 LTR (AFTER) —** | | | | | |
| LambdaMART (`linear_rank`) | **0.806** | 0.773 | **0.862** | 0.677 | 0.631 |
| Plackett-Luce Neural Ranker | 0.696 | **0.760** | 0.845 | **0.682** | 0.493 |
| **— v6.0 Hurdle (AFTER) —** | | | | | |
| Two-Stage Hurdle | 0.649 | 0.742 | 0.831 | 0.669 | **0.6587** |

> **Key result**: The Two-Stage Hurdle's Stage 2 (classified finishers only) achieves R² = **0.6587** vs the v5.0 OLS baseline of 0.4080 — a +61.4% improvement in explained variance by eliminating the DNF noise penalty.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/hMiZ-STD/formula1-ml-suite.git
cd formula1-ml-suite
pip install -r requirements.txt

# 2. Run full test suite (76 tests)
pytest tests/ -v

# 3. Run v5.0 experiment pipeline (classification + regression + unsupervised)
python -m src.run_experiments

# 4. Run v6.0 LTR + Hurdle benchmark (produces Table 8 & Figure 12)
python -m src.run_v6_experiments

# 5. Launch interactive Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## Temporal Anti-Leakage Design

The single most critical engineering constraint is **zero data leakage** across time:

- `train_core`: 1994–2016 (tuning, CV)
- `val`: 2017–2018 (threshold calibration, early stopping)
- `test`: 2019–2024 (evaluated exactly once, after all parameters are frozen in `EXPERIMENT_FREEZE.yaml`)

All rolling features use `.shift(1)` to push today's result out of reach. Constructor features are aggregated at race-round level and lagged by 1 race. Sprint weekend qualifying deltas are masked as `NaN` to avoid rule confounding.

---

## License & Academic Compliance

All raw telemetry is sourced from the historical Ergast / Kaggle Formula 1 World Championship dataset. Built as a pair-programmed machine learning engineering suite adhering to strict software engineering, mathematical rigour, and pedagogical standards.
