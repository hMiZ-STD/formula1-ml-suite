"""Automated Jupyter Notebook Generator for Formula 1 ML Suite.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_cell(cell_type: str, source_text: str) -> dict:
    """Create a standard Jupyter cell dictionary."""
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def save_notebook(filename: str, cells: list[dict]) -> None:
    """Save cells to valid .ipynb file."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.7",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOKS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated notebook: {path}")


def create_all_notebooks() -> None:
    """Build all 7 authoritative assignment and appendix notebooks."""

    # =========================================================================
    # Notebook 1: Data Audit & Exploratory Data Analysis
    # =========================================================================
    nb1_cells = [
        make_cell("markdown", "# Formula 1 ML Suite — Notebook 01: Data Audit & Exploratory Data Analysis\n### Version 5.0 — Authoritative Table 1 Audit Pipeline"),
        make_cell("markdown", """## 1. Problem Formulation & Historical Overview
In Formula 1 history from 1950 to 2024, data anomalies like shared drives, non-starters, 
pit lane starts encoded as `grid = 0`, and lap time reporting errors can corrupt supervised models.
This notebook runs the 11-point data audit and produces Table 1."""),
        make_cell("code", """import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.loader import load_clean_starters, load_raw_csvs, load_audit_artifacts
from src.data.audit import run_table1_audit

sns.set_theme(style="whitegrid")
print("Libraries imported successfully!")"""),
        make_cell("markdown", "## 2. Execute Authoritative Table 1 Audit Block"),
        make_cell("code", """audit_summary = run_table1_audit()
table1_df = pd.read_csv("data/processed/audit/table1_audit.csv")
table1_df"""),
        make_cell("markdown", "## 3. Verify Non-Starter Status Filtering & Split Percentages"),
        make_cell("code", """df_starters = load_clean_starters()
print(f"Total clean starters (1994-2024): {len(df_starters):,} rows across {df_starters['raceId'].nunique()} races")
print(f"Base podium rate: {df_starters['y_podium'].mean():.2%}")
df_starters[['year', 'name', 'grid_effective', 'grid_zero_flag', 'y_podium', 'positionOrder']].head(10)"""),
    ]
    save_notebook("01_data_audit_and_eda.ipynb", nb1_cells)

    # =========================================================================
    # Notebook 2: Classification Benchmarks (Question 1)
    # =========================================================================
    nb2_cells = [
        make_cell("markdown", "# Formula 1 ML Suite — Notebook 02: Classification Benchmarks\n### Question 1: Predicting Podium Finishes (1994–2024)"),
        make_cell("markdown", """## 1. Methodology
We train and evaluate:
1. Majority Class Baseline
2. Grid Position Heuristic
3. K-Nearest Neighbors (KNN)
4. Decision Tree Classifier
5. Support Vector Machine (SVM with RBF Kernel)
6. Dual-Branch 1D-CNN (PyTorch sequence + static model)

Thresholds are tuned on the validation set (2017–2018) to maximize F1-score and frozen in `EXPERIMENT_FREEZE.yaml`."""),
        make_cell("code", """import pandas as pd
from IPython.display import Image, display

table2_df = pd.read_csv("reports/tables/table2_classification_benchmark.csv")
table2_df"""),
        make_cell("markdown", "## 2. Quad-Panel Test Confusion Matrices (Figure 1)"),
        make_cell("code", """display(Image("reports/figures/fig1_confusion_matrices.png"))"""),
        make_cell("markdown", "## 3. ROC Curves with 95% Bootstrap CIs & Precision-Recall Curves (Figures 2 & 3)"),
        make_cell("code", """display(Image("reports/figures/fig2_roc_curves.png"))
display(Image("reports/figures/fig3_pr_curves.png"))"""),
        make_cell("markdown", "## 4. Model Complexity & Overfitting Diagnostic Curves (Figure 4)"),
        make_cell("code", """display(Image("reports/figures/fig4_complexity_curves.png"))"""),
    ]
    save_notebook("02_classification_benchmarks.ipynb", nb2_cells)

    # =========================================================================
    # Notebook 3: Regression & Overfitting Diagnostics (Question 2)
    # =========================================================================
    nb3_cells = [
        make_cell("markdown", "# Formula 1 ML Suite — Notebook 03: Regression & Overfitting Diagnostics\n### Question 2: Finishing Position Regression & Regularization"),
        make_cell("markdown", """## 1. Regression Benchmark
Target: `positionOrder` for all starters (1994–2024; 100% complete).
We evaluate OLS, Scratch Gradient Descent with analytical stability derivation, Polynomial bias-variance sweep, and Ridge & Lasso regularization."""),
        make_cell("code", """import pandas as pd
from IPython.display import Image, display

table3a_df = pd.read_csv("reports/tables/table3a_regression_benchmark.csv")
table3a_df"""),
        make_cell("markdown", "## 2. Analytical Gradient Descent Stability Boundary (Figure 5)"),
        make_cell("code", """display(Image("reports/figures/fig5_gd_convergence.png"))"""),
        make_cell("markdown", "## 3. Polynomial Degree Bias-Variance Sweep (Table 3b & Figure 6)"),
        make_cell("code", """table3b_df = pd.read_csv("reports/tables/table3b_poly_demo.csv")
display(table3b_df)
display(Image("reports/figures/fig6_poly_bias_variance.png"))"""),
        make_cell("markdown", "## 4. Ridge & Lasso Sparsity Analysis (Table 4 & Figure 7)"),
        make_cell("code", """display(Image("reports/figures/fig7_regularization_cv_paths.png"))
table4_df = pd.read_csv("reports/tables/table4_regularization_sparsity.csv")
table4_df.head(15)"""),
    ]
    save_notebook("03_regression_overfitting.ipynb", nb3_cells)

    # =========================================================================
    # Notebook 4: Unsupervised Driver Career Discovery (Question 3)
    # =========================================================================
    nb4_cells = [
        make_cell("markdown", "# Formula 1 ML Suite — Notebook 04: Unsupervised Driver Career Discovery\n### Question 3: K-Means & PCA from First Principles"),
        make_cell("markdown", """## 1. Historical Scope & Era-Robust Feature Set
Analyzes all 252 Formula 1 drivers with >= 20 starts across 75 years (1950–2024).
Features: `win_share`, `podium_share`, `pts_share`, `norm_avg_finish`, `dnf_adj`, `teammate_delta`."""),
        make_cell("code", """import pandas as pd
from IPython.display import Image, display

table5_df = pd.read_csv("reports/tables/table5_kmeans_selection.csv")
table5_df"""),
        make_cell("markdown", "## 2. K-Means Elbow & Silhouette Optimization (Figure 8)"),
        make_cell("code", """display(Image("reports/figures/fig8_kmeans_elbow_silhouette.png"))"""),
        make_cell("markdown", "## 3. PCA Full Eigenvalue Spectrum & Loading Matrix (Tables 6 & 6b)"),
        make_cell("code", """table6_df = pd.read_csv("reports/tables/table6_pca_spectrum.csv")
table6b_df = pd.read_csv("reports/tables/table6b_pca_loadings.csv", index_col=0)
display(table6_df)
display(table6b_df)"""),
        make_cell("markdown", "## 4. PCA Biplot & 3D Career Space (Figures 9, 10, 11)"),
        make_cell("code", """display(Image("reports/figures/fig10_pca_scree.png"))
display(Image("reports/figures/fig9_pca_2d_clusters.png"))
display(Image("reports/figures/fig11_pca_3d.png"))"""),
    ]
    save_notebook("04_unsupervised_kmeans_pca.ipynb", nb4_cells)

    # =========================================================================
    # Appendix Notebook A: Digits Classification
    # =========================================================================
    nb_app_a_cells = [
        make_cell("markdown", "# Appendix A: Standard Dataset Classification Benchmark\n### Dataset: sklearn.datasets.load_digits (10 Classes, 64 Features)"),
        make_cell("markdown", "Demonstrates KNN, Decision Tree, SVM, and a PyTorch 2D-CNN on standard handwritten digits."),
        make_cell("code", """import pandas as pd
from src.evaluation.standard_datasets import run_appendix_a_digits

df_a = run_appendix_a_digits()
df_a"""),
    ]
    save_notebook("appendix_A_digits_classification.ipynb", nb_app_a_cells)

    # =========================================================================
    # Appendix Notebook B: Diabetes Regression
    # =========================================================================
    nb_app_b_cells = [
        make_cell("markdown", "# Appendix B: Standard Dataset Regression Benchmark\n### Dataset: sklearn.datasets.load_diabetes (442 Patients, 10 Features)"),
        make_cell("markdown", "Demonstrates OLS, Ridge, and Lasso regression on disease progression."),
        make_cell("code", """import pandas as pd
from src.evaluation.standard_datasets import run_appendix_b_diabetes

df_b = run_appendix_b_diabetes()
df_b"""),
    ]
    save_notebook("appendix_B_diabetes_regression.ipynb", nb_app_b_cells)

    # =========================================================================
    # Appendix Notebook C: Iris Clustering & PCA
    # =========================================================================
    nb_app_c_cells = [
        make_cell("markdown", "# Appendix C: Standard Dataset Unsupervised Benchmark\n### Dataset: sklearn.datasets.load_iris (150 Samples, 4 Features)"),
        make_cell("markdown", "Demonstrates Scratch K-Means (K=2, 3, 4) and Scratch PCA on Fisher's Iris dataset."),
        make_cell("code", """import pandas as pd
from src.evaluation.standard_datasets import run_appendix_c_iris

df_c_km, df_c_pca = run_appendix_c_iris()
print("--- Iris K-Means WCSS & Silhouette ---")
display(df_c_km)
print("\n--- Iris PCA Full Loading Matrix (4x4) ---")
display(df_c_pca)"""),
    ]
    save_notebook("appendix_C_iris_clustering_pca.ipynb", nb_app_c_cells)


if __name__ == "__main__":
    create_all_notebooks()
