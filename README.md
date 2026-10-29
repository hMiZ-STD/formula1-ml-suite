# formula1-ml-suite

A production-grade machine learning pipeline for Formula 1 race outcome prediction.
Built on 30 years of Ergast data (1994–2024) with strict temporal anti-leakage guarantees,
listwise Learning-to-Rank, and a Two-Stage Hurdle architecture that separates mechanical
reliability from pure on-track race pace.

![pytest](https://img.shields.io/badge/pytest-76%20passed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C)

---

## The Problem

Standard machine learning treats every driver in a race as an independent observation.
That assumption breaks in Formula 1 for two fundamental reasons:

**Zero-sum competition.** There are exactly 3 podium trophies per race. A model that
evaluates drivers independently can confidently predict 5 of them will finish on the
podium — which is physically impossible. Threshold tuning patches over this symptom
without fixing the root cause.

**Mechanical DNF distortion.** In the 1994–2016 training era, 32% of cars retired with
mechanical failures. When a driver leads 50 laps and blows an engine on lap 51, they
are classified 18th. Standard regression penalises the model with a squared error of
$(18 - 1)^2 = 289$ even though the pace prediction was correct. This caps explained
variance at R² ≈ 0.41 regardless of model complexity.

Both problems are addressed in v6.0 — the first through listwise Learning-to-Rank,
the second through the Two-Stage Hurdle architecture.

---

## Quickstart

```bash
git clone https://github.com/hMiZ-STD/formula1-ml-suite.git
cd formula1-ml-suite
pip install -r requirements.txt

# Run full test suite
pytest tests/ -v

# v5.0 pipeline — classification, regression, unsupervised
python -m src.run_experiments

# v6.0 pipeline — LTR and Hurdle benchmark
python -m src.run_v6_experiments

# Interactive dashboard
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
formula1-ml-suite/
├── src/
│   ├── data/
│   │   ├── audit.py                 # Table 1 audit and artifact freezing
│   │   ├── loader.py                # Non-starter filtering, constructor lineage
│   │   └── preprocessor.py         # Temporal splits, race-aligned CV folds
│   ├── features/
│   │   └── builder.py              # Pre-race feature engineering (zero leakage)
│   ├── models/
│   │   ├── classifiers.py          # KNN, Decision Tree, SVM
│   │   ├── cnn_classifier.py       # Dual-Branch 1D-CNN (sequence + static)
│   │   ├── ltr_ranker.py           # LambdaMART + Plackett-Luce Neural Ranker
│   │   ├── hurdle_model.py         # Two-Stage DNF Hazard + Pace Ranker
│   │   ├── linear_gd.py            # Gradient descent from scratch
│   │   ├── regularized_reg.py      # Ridge and Lasso with degree-2 expansion
│   │   ├── kmeans_scratch.py       # K-Means from scratch (NumPy only)
│   │   └── pca_scratch.py          # PCA from scratch (sample covariance + eigh)
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── regression_metrics.py
│   │   ├── ranking_metrics.py      # NDCG@1/3/10, MRR, Kendall Tau, Podium F1
│   │   └── standard_datasets.py    # Appendix: digits, diabetes, iris
│   ├── visualization/
│   │   └── plots.py                # Figures 1–12 at 300 DPI
│   ├── run_experiments.py          # v5.0 master runner
│   └── run_v6_experiments.py       # v6.0 LTR + Hurdle runner
├── tests/                           # 76 automated tests
├── notebooks/                       # 7 Jupyter notebooks
├── app/streamlit_app.py             # Interactive dashboard
├── data/processed/                  # Parquet splits + audit artifacts
├── reports/
│   ├── figures/                     # fig1–fig12.png (300 DPI)
│   ├── tables/                      # table1–table8.csv
│   └── final_assignment_report.pdf
└── EXPERIMENT_FREEZE.yaml           # Frozen hyperparameters, seeds, thresholds
```

---

## Data Splits

All models obey a strict temporal boundary with no cross-contamination.

| Split | Seasons | Races | Starters | Role |
| --- | --- | --- | --- | --- |
| `train_core` | 1994–2016 | 408 | 8,920 | Hyperparameter tuning |
| `val` | 2017–2018 | 41 | 820 | Threshold calibration, early stopping |
| `test` | 2019–2024 | 128 | 2,559 | Final evaluation (evaluated once) |

Rolling features use `.shift(1)` throughout. Constructor aggregations are lagged by one
race. Sprint weekend qualifying deltas are masked as `NaN`. All decisions are frozen in
`EXPERIMENT_FREEZE.yaml` before the test set is touched.

---

## Results

### v5.0 — Classification (predicting podium finishes)

| Model | ROC-AUC | PR-AUC | F1 |
| --- | --- | --- | --- |
| Grid position baseline | 0.915 | 0.660 | 0.682 |
| K-Nearest Neighbours | 0.936 | 0.701 | 0.684 |
| Decision Tree (pruned) | 0.936 | 0.709 | 0.680 |
| Support Vector Machine (RBF) | 0.940 | 0.705 | 0.671 |
| Dual-Branch 1D-CNN | 0.937 | 0.719 | 0.679 |

### v5.0 — Regression (predicting finishing position)

| Model | Test MSE | Test R² |
| --- | --- | --- |
| Gradient Descent (scratch) | 22.41 | 0.375 |
| OLS (pseudo-inverse) | 19.67 | 0.408 |
| Ridge + degree-2 expansion | 19.57 | 0.411 |
| Lasso + degree-2 expansion | 19.57 | 0.411 |

### v6.0 — Learning-to-Rank and Hurdle (2019–2024 test set, 128 races)

| Model | NDCG@1 | NDCG@3 | NDCG@10 | Podium F1 | R² (classified) |
| --- | --- | --- | --- | --- | --- |
| Grid baseline | 0.627 | 0.732 | 0.816 | 0.682 | 0.374 |
| KNN (best threshold) | 0.615 | 0.726 | 0.821 | 0.684 | 0.639 |
| Decision Tree (best threshold) | 0.861 | 0.809 | 0.896 | 0.680 | 0.643 |
| SVM RBF (best threshold) | 0.567 | 0.697 | 0.804 | 0.671 | 0.498 |
| Dual-Branch 1D-CNN | 0.671 | 0.735 | 0.837 | 0.679 | 0.492 |
| LambdaMART (linear_rank) | 0.806 | 0.773 | 0.862 | 0.677 | 0.631 |
| Plackett-Luce Neural Ranker | 0.696 | 0.760 | 0.845 | 0.682 | 0.493 |
| Two-Stage Hurdle | 0.649 | 0.742 | 0.831 | 0.669 | **0.659** |

The Two-Stage Hurdle Stage 2 (trained on classified finishers only) achieves R² = 0.659
against the v5.0 OLS baseline of 0.408 — a 61% improvement in explained variance by
removing the mechanical DNF penalty from the training objective.

LambdaMART with `linear_rank` relevance won the validation sweep (NDCG@1: 0.848).
Every test race has exactly 3 predicted podium finishers — 384 total across 128 races,
zero ties, zero duplicates.

---

## Key Design Decisions

**Anti-leakage over accuracy.** We deliberately exclude pit stop data (missing before
2011) and Q2/Q3 times (missing before 2006) rather than impute them. An R² of 0.66
on clean data is more credible than 0.75 on imputed proxies.

**Race-aligned cross-validation.** All CV folds keep entire races intact in either
train or validation. Splitting a race at the driver level leaks weather and track
conditions between folds.

**Scratch implementations.** K-Means, PCA, and Gradient Descent are implemented in
pure NumPy with parity tests against scikit-learn to verify correctness.

**LambdaMART relevance target.** Three mappings were benchmarked on validation data:
F1 championship points (25-18-15...), linear inverted rank (20 down to 1), and podium
tier (3-2-1-0). Linear rank won because it provides a distinct gradient signal for
every grid position rather than collapsing positions 4–20 into a single zero.

---

## Tests

```
pytest tests/ -v
# 76 passed in 21.75s
```

| Suite | Tests | Coverage |
| --- | --- | --- |
| Pipeline integrity | 7 | Leakage, temporal partitions, whitelist enforcement |
| Grid encoding | 3 | Pit lane starts, zero-grid handling |
| Qualifying parser | 2 | Time string formats, gap calculation |
| Race folds | 1 | Temporal CV fold properties |
| Gradient descent | 3 | lstsq parity, divergence, learning rate bounds |
| K-Means scratch | 3 | WCSS parity, label parity, empty cluster recovery |
| PCA scratch | 4 | Eigenvalue parity, EVR, projection, loading matrix |
| Ranking metrics | 19 | DCG/NDCG analytical values, boundary cases, multi-race |
| LTR ranker | 19 | Group alignment, monotonic scores, top-3 guarantee |
| Hurdle model | 12 | Probability bounds, Stage 2 purity, combination logic |
| Plot generation | 3 | Figure 12 generation, backward compatibility |
