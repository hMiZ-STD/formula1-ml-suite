# Formula 1 End-to-End Machine Learning Engineering Suite
### Production-Grade Predictive Modeling, Deep Temporal Architectures & Unsupervised Career Discovery
**Author:** Pair Programming Team (User & AI)  
**Academic Target:** IIIT Sri City Machine Learning Assignment  
**Version:** 5.0 (Authoritative Technical Specification & Sealed Implementation)  
**Championship History:** 1950–2024 (75 Seasons, 1,125 Grand Prix)  
**Supervised Scope:** Modern Telemetry Era (1994–2024, 577 Grand Prix, 12,299 Clean Starters)  

[![Python 3.13](https://img.shields.io/badge/Python-3.13.7-blue.svg)](https://www.python.org/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn 1.5](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-23%20passed-brightgreen.svg)](tests/)
[![Report PDF](https://img.shields.io/badge/ReportLab-Final%20PDF%20Compiled-red.svg)](reports/final_assignment_report.pdf)

---

## 📖 Executive Summary & Engineering Architecture

Formula 1 racing represents an extreme predictive environment governed by temporal non-stationarity, aerodynamic evolution, strict regulatory resets, and strategic multi-agent interaction. Standard textbook modeling workflows fail catastrophically in this domain due to:
1. **Historical Data Corruption:** Pre-1994 shared-car drives produce up to 7 podium finishers in a single Grand Prix.
2. **Subtle Future Leakage:** Un-grouped rolling features or row-level constructor aggregations inadvertently leak in-race teammate telemetry into pre-race inputs.
3. **Imbalanced Attrition:** Extreme class imbalance (~15% podium rate) and mechanical DNFs skew regression targets if artificial sentinels (e.g. 9999 or time gaps) are introduced.

This repository implements a **leakage-free, mathematically proven, production-grade machine learning suite** addressing three core investigations:
* **Mini-Project 1 (Classification):** Predicting podium finishes (`y_podium`) using KNN, Pruned Decision Trees, Support Vector Machines (RBF), and a custom Dual-Branch 1D-CNN with validation-calibrated decision thresholds and race-level block bootstrap statistical inference.
* **Mini-Project 2 (Regression):** Predicting final finishing order (`positionOrder`) using first-principles Vectorized Gradient Descent with analytical stability proofs ($\alpha < 2/\lambda_{\max}$), polynomial degree complexity U-curves, and L1/L2 regularized feature selection (Ridge & Lasso).
* **Mini-Project 3 (Unsupervised Discovery):** Discovering historical driver career archetypes across 75 years (1950–2024) using K-Means Clustering and Principal Component Analysis (PCA) coded completely from scratch in pure NumPy.
* **Standard Dataset Appendices:** Complete replication on classic benchmark datasets (`load_digits`, `load_diabetes`, `load_iris`) to guarantee 100% formal compliance with the assignment brief.

---

## 🏛️ System Architecture & Directory Blueprint

```
d:/ML-Project/
├── FINAL_IMPLEMENTATION_PLAN.md    # Definitive Master Plan & Change Register (v5.0)
├── CODING_STANDARDS.md             # Software engineering, PEP 8, typing, and ELI5 rules
├── MEMORY.md                       # Persistent technical decision log
├── EXPERIMENT_FREEZE.yaml          # Sealed hyperparameters, thresholds, seeds, and hashes
├── README.md                       # High-level architecture and resume-grade documentation
├── pyproject.toml                  # Modern package metadata and pytest configurations
├── requirements.txt                # Exact dependency specifications
│
├── data/
│   ├── raw/                        # 14 raw Ergast CSV tables
│   └── processed/
│       ├── audit/                  # Frozen Table 1 artifacts (JSON & CSV)
│       │   ├── table1_audit.csv
│       │   ├── non_starter_status_ids.json
│       │   ├── sprint_race_ids.json
│       │   ├── q_coverage_by_season.csv
│       │   ├── constructor_canonical_map.json
│       │   └── feature_registry.csv
│       ├── train_core.parquet      # 1994–2016 seasons (8,920 starters, 408 races)
│       ├── val.parquet             # 2017–2018 seasons (820 starters, 41 races)
│       ├── test.parquet            # 2019–2024 hybrid era (2,559 starters, 128 races)
│       └── career_clustering.parquet # 1950–2024 full career aggregates (252 drivers)
│
├── src/
│   ├── data/
│   │   ├── audit.py                # 11-point Table 1 audit generator
│   │   ├── loader.py               # Raw string NULL parsing, dtypes, non-starter filter
│   │   └── preprocessor.py         # race_time_folds(), temporal splitters, freeze writer
│   ├── features/
│   │   └── builder.py              # Whitelist builder: race-level constructor agg, Q1 gap
│   ├── models/
│   │   ├── classifiers.py          # KNN, Decision Tree, SVM Pipelines with Imputer/Scaler
│   │   ├── cnn_classifier.py       # Dual-Branch 1D-CNN (Temporal sequence + static MLP)
│   │   ├── linear_gd.py            # Scratch vectorized GD with analytical stability proof
│   │   ├── polynomial_reg.py       # Degree sweep on [-1, 1] scaled continuous inputs
│   │   ├── regularized_reg.py      # Degree-2 continuous expansion + Ridge & Lasso
│   │   ├── kmeans_scratch.py       # Scratch K-Means with K-Means++ & empty-cluster recovery
│   │   └── pca_scratch.py          # Scratch PCA via sample covariance & np.linalg.eigh
│   ├── evaluation/
│   │   ├── classification_metrics.py  # Precision, Recall, F1, PR-AUC, Block Bootstrap CI
│   │   ├── regression_metrics.py      # MSE, RMSE, R², Classified-only R² robustness
│   │   └── standard_datasets.py       # Appendices A, B, C standard dataset runners
│   ├── visualization/
│   │   └── plots.py                # Publication-grade plotting routines (300 DPI)
│   └── run_experiments.py          # Master orchestrator executing entire pipeline
│
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_classification_benchmarks.ipynb
│   ├── 03_regression_overfitting.ipynb
│   ├── 04_unsupervised_kmeans_pca.ipynb
│   ├── appendix_A_digits_classification.ipynb
│   ├── appendix_B_diabetes_regression.ipynb
│   └── appendix_C_iris_clustering_pca.ipynb
│
├── tests/
│   ├── test_pipeline_integrity.py  # Whitelist, teammate invariance, sprint mask, perturbation
│   ├── test_grid_encoding.py       # grid_zero_flag and grid_effective bounds
│   ├── test_qualifying_parser.py   # Lap time parsing and outlier detection
│   ├── test_race_folds.py          # Positional index alignment & temporal integrity
│   ├── test_linear_gd.py           # Gradient descent closed-form parity & divergence
│   ├── test_kmeans_scratch.py      # WCSS parity with sklearn & empty-cluster recovery
│   └── test_pca_scratch.py         # Eigenvalue parity, EVR unity, sign alignment
│
├── app/
│   └── streamlit_app.py            # Interactive full-stack dashboard & driver lookup tool
│
└── reports/
    ├── figures/                    # 11 publication figures exported at 300 DPI
    ├── tables/                     # 7 master result tables in CSV format
    ├── generate_report.py          # ReportLab 5.0 automated PDF builder
    └── final_assignment_report.pdf # 4.5 MB comprehensive scientific report
```

---

## 🛡️ Anti-Leakage Protocol & Data Partitioning (Table 1)

All data preprocessing adheres to five non-negotiable guarantees:
1. **Supervised Temporal Scope (1994–2024):** Eliminates 18 historical races with corrupted podium counts (4 to 7 podium labels caused by mid-race shared drives). Modern electronic Q1 qualifying times start in 1994.
2. **Race-Level Constructor Aggregation:** Team championship points are aggregated to `(raceId, constructor)` level first, shifted by one whole race (`.shift(1)`), and then joined back to driver rows. All teammates in the same race share identical pre-race points shares.
3. **Q1-Only Qualifying Gap with Outlier Defense:** Drivers are evaluated against the fastest Q1 session time (`q_gap_pct`), preventing selection bias from Q2/Q3 eliminations. Telemetry logging errors (e.g. Mark Blundell's 16:42 lap) exceeding 150% of the circuit-era median are flagged and treated as missing.
4. **Sprint Weekend Masking:** All 18 sprint race weekends (2021–2024) have `grid_vs_quali_delta` explicitly masked as `NaN` because starting grids were determined by sprint races rather than qualifying sessions.
5. **Strict Three-Set Partitioning:**
   * **`train_core` (1994–2016):** 408 races (8,920 rows; 72.53%) — used for 8-fold race-aligned cross-validation.
   * **`val` (2017–2018):** 41 races (820 rows; 6.67%) — used strictly for 1D-CNN early stopping (val BCE loss) and threshold calibration.
   * **`train_val_pool` (1994–2018):** 449 races (9,740 rows; 79.19%) — final Scikit-Learn models refit on this pool before test evaluation.
   * **`test` (2019–2024):** 128 races (2,559 rows; 20.81%) — **evaluated exactly ONCE** after `EXPERIMENT_FREEZE.yaml` is written.

---

## 🏆 Key Results & Master Benchmarks

### 1. Classification Suite (Table 2 — Test Set 2019–2024)
*Target: `y_podium = 1` if positionOrder &le; 3 and classified finisher (Test Base Rate: 15.01%)*

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC [95% Bootstrap CI] | PR-AUC | Key Parameters |
|---|---|---|---|---|---|---|---|---|
| **Majority Class** | N/A | 84.99% | 0.0000 | 0.0000 | 0.0000 | 0.500 [0.500, 0.500] | 0.1501 | Constant Negative |
| **Grid Heuristic** | -3.12 | 90.43% | 0.6815 | 0.6797 | 0.6806 | 0.915 [0.895, 0.933] | 0.6602 | `score = -grid_effective` |
| **KNN** | 0.286 | 89.10% | 0.6052 | **0.7865** | 0.6840 | 0.928 [0.910, 0.944] | 0.6818 | $K=30$, `weights='distance'` |
| **Decision Tree** | 0.714 | **0.9082** | **0.7123** | 0.6510 | 0.6803 | 0.926 [0.910, 0.941] | 0.6382 | Max Depth 4, `ccp_alpha=0.0` |
| **SVM (RBF)** | 0.863 | 89.84% | 0.6527 | 0.6901 | 0.6709 | 0.934 [0.919, 0.949] | 0.7075 | $C=100.0, \gamma=0.001$ |
| **Dual-Branch 1D-CNN** | 0.803 | 89.88% | 0.6478 | 0.7135 | **0.6791** | **0.937 [0.922, 0.951]** | **0.7194** | 10-Race Sequence + Static MLP |

*Statistical Finding:* The Dual-Branch 1D-CNN achieved the highest ROC-AUC and PR-AUC by leveraging multi-race momentum. Decision Tree achieved highest precision (71.2%), while KNN achieved highest recall (78.7%). Overlapping 95% bootstrap CIs confirm robust performance across all non-linear algorithms.

---

### 2. Regression Suite (Table 3a — Test Set 2019–2024)
*Target: `positionOrder` for all starters (1994–2024; zero nulls, no artificial sentinels)*

| Model | Key Parameters | Train MSE | Test MSE | Test RMSE | Test R² (All Starters) | Test R² (Classified-Only) |
|---|---|---|---|---|---|---|
| **Predict-mean Baseline** | Mean = 11.42 | 40.6786 | 34.0727 | 5.8372 | -0.0254 | -0.1964 |
| **Grid-only OLS** | `positionOrder ~ grid` | 29.2422 | 20.8663 | 4.5680 | 0.3720 | 0.4323 |
| **OLS (lstsq)** | All features (min-norm) | 26.9528 | **19.6694** | **4.4350** | **0.4080** | **0.4884** |
| **GD Linear Reg** | $\alpha_{\text{safe}}=0.2698, 1k$ iter | 26.9534 | 19.6916 | 4.4375 | 0.4074 | 0.4866 |
| **Ridge (L2)** | $\alpha=100.0$, degree=2 | **26.9489** | 20.1199 | 4.4855 | 0.3945 | 0.4693 |
| **Lasso (L1)** | $\alpha=0.0117$, degree=2 | 26.9779 | 19.8300 | 4.4531 | 0.4032 | 0.4821 |

*Analytical Gradient Descent Stability Proof:*
For cost $J(\theta) = \frac{1}{2n} \|X\theta - y\|_2^2$, the Hessian is $H = \frac{1}{n} X^T X$. The largest eigenvalue $\lambda_{\max} = 5.9366$ establishes the theoretical stability boundary $\alpha_{\max} = \frac{2}{\lambda_{\max}} = 0.3369$. Our analytically chosen $\alpha_{\text{safe}} = 0.8 \times \alpha_{\max} = 0.2698$ achieved strict cost parity with closed-form least squares (relative excess $2.3 \times 10^{-7} < 10^{-6}$), while $\alpha_{\text{unsafe}} = 2.2 \times \alpha_{\max} = 0.7412$ diverged demonstrably within 15 iterations.

---

### 3. Unsupervised Driver Career Discovery (Table 5 & 6)
*Scope: Full 75-year championship history (1950–2024; 252 drivers with &ge; 20 career starts)*

* **Optimal K Selection (Table 5):** Evaluated $K=2..6$ over 15 K-Means++ restarts. $K=3$ selected as optimal (WCSS = 554.39, Silhouette = 0.4145), aligning with the distinct tiers of Formula 1 motorsport:
  * **Cluster 1 (Elite Champions & Title Contenders):** High win share (0.411 loading on PC1), podium share (0.455), and points share (0.456). Contains multi-champions: Hamilton, Schumacher, Senna, Verstappen, and Alonso.
  * **Cluster 2 (Midfield Workhorses & Specialists):** Consistent points scorers with balanced finishing averages and low DNF rates.
  * **Cluster 0 (Backmarker & Development Drivers):** Low points share and positive teammate deltas (routinely beaten by teammates).
* **PCA Variance Spectrum (Table 6):**
  * **PC1 ($\lambda=4.3466$, 72.16% variance):** Dominance & Trophy Vector (wins, podiums, points share).
  * **PC2 ($\lambda=0.8225$, 13.65% variance):** Attrition & Teammate Benchmark Vector (DNF adjustment, teammate delta).
  * **Cumulative Variance:** PC1 + PC2 capture **85.81%** of historical career variance.

---

## ⚡ Quickstart & Reproducibility Guide

### 1. Installation
Clone the repository and install the production dependencies:
```bash
git clone https://github.com/hamiz/f1-ml-suite.git
cd f1-ml-suite
pip install -r requirements.txt
```

### 2. Run Automated Verification Suite (23 Passing Tests)
Run the full test suite verifying pipeline integrity, whitelist security, scratch algorithm parity, and temporal alignment:
```bash
pytest tests/ -v
```

### 3. Execute Master Experiment Pipeline
Re-run all classification tuning, regression benchmarks, experiment freezing, and unsupervised clustering from scratch:
```bash
python -m src.run_experiments
```

### 4. Launch Interactive Streamlit Dashboard
Explore the data audit, model complexity curves, live threshold calibrator, and driver career lookup tool:
```bash
streamlit run app/streamlit_app.py
```

### 5. Compile Authoritative Scientific PDF Report
Rebuild the 4.5 MB multi-page academic report with all 11 high-resolution figures and master tables:
```bash
python -m reports.generate_report
```

---

## 📄 License & Academic Compliance
This project was designed and built as a definitive, pair-programmed machine learning engineering suite adhering to the highest pedagogical, mathematical, and software engineering standards. All raw telemetry is sourced from the historical Ergast / Kaggle Formula 1 World Championship dataset.
