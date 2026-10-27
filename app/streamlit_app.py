"""Formula 1 Machine Learning Engineering Suite — Interactive Streamlit Dashboard.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Formula 1 ML Engineering Suite (v5.0)",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("reports/tables")
FIGURES_DIR = Path("reports/figures")
AUDIT_DIR = Path("data/processed/audit")


@st.cache_data
def load_table(csv_name: str) -> pd.DataFrame:
    """Load a CSV table from reports/tables or audit."""
    p1 = TABLES_DIR / csv_name
    if p1.exists():
        return pd.read_csv(p1)
    p2 = AUDIT_DIR / csv_name
    if p2.exists():
        return pd.read_csv(p2)
    return pd.DataFrame()


@st.cache_data
def load_career_data() -> pd.DataFrame:
    """Load career clustering dataset."""
    p = PROCESSED_DIR / "career_clustering.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
st.sidebar.title("🏎️ F1 ML Suite v5.0")
st.sidebar.markdown(
    """
**Author:** Pair Programming Team  
**Dataset:** Ergast F1 (1950–2024)  
**Scope:** Modern Supervised Era (1994–2024)  
**Career History:** Full Era (1950–2024)  
"""
)
st.sidebar.markdown("---")
st.sidebar.info("All models sealed in `EXPERIMENT_FREEZE.yaml` prior to single test-set evaluation.")

# --- Header ---
st.title("Formula 1 End-to-End Machine Learning Engineering Suite")
st.subheader("Interactive Scientific Dashboard & Diagnostic Explorer (Version 5.0)")

tabs = st.tabs([
    "📊 Executive Overview & Data Audit",
    "🏆 Q1: Supervised Classification",
    "⏱️ Q2: Supervised Regression",
    "🔍 Q3: Driver Career Discovery (K-Means & PCA)",
    "📚 Standard Dataset Appendices",
])

# =============================================================================
# TAB 1: OVERVIEW & TABLE 1 AUDIT
# =============================================================================
with tabs[0]:
    st.markdown("### 1. Problem Formulation & Historical Data Ingestion")
    st.markdown(
        """
        Formula 1 racing presents a demanding predictive setting characterized by severe mechanical attrition,
        game-theoretic pit strategies, and dramatic technological rule shifts across decades.
        
        To prevent historical data corruption and ensure zero data leakage, our pipeline enforces an authoritative
        **Table 1 Data Audit** prior to freezing any feature code.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Supervised Scope", "1994–2024", "Shared drives eliminated")
    col2.metric("Total Clean Starters", "12,299 rows", "577 Grand Prix")
    col3.metric("Train/Val Pool", "79.19% (9,740)", "1994–2018 Seasons")
    col4.metric("Held-Out Test Set", "20.81% (2,559)", "2019–2024 Hybrid Era")

    st.markdown("---")
    st.markdown("#### Table 1: Empirical Data Audit & Partition Verification")
    df_t1 = load_table("table1_audit.csv")
    if not df_t1.empty:
        st.dataframe(df_t1, use_container_width=True, hide_index=True)

    with st.expander("🔍 View Frozen Audit Artifacts (JSON & Feature Registry)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Non-Starter Status IDs (Frozen):**")
            p_ns = AUDIT_DIR / "non_starter_status_ids.json"
            if p_ns.exists():
                st.json(json.load(open(p_ns)))
            st.markdown("**Sprint Race IDs (2021–2024, 18 Races):**")
            p_sp = AUDIT_DIR / "sprint_race_ids.json"
            if p_sp.exists():
                st.json(json.load(open(p_sp)))
        with c2:
            st.markdown("**Feature Registry Whitelist:**")
            df_feat = load_table("feature_registry.csv")
            if not df_feat.empty:
                st.dataframe(df_feat, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: CLASSIFICATION SUITE
# =============================================================================
with tabs[1]:
    st.markdown("### 🏆 Question 1: Supervised Podium Finish Classification")
    st.markdown(
        """
        **Target:** `y_podium = 1` if `positionOrder <= 3` and classified finisher, else 0 (podium rate = 15.01%).
        
        All models include `SimpleImputer(strategy='median', add_indicator=True)` and `StandardScaler()`
        inside Scikit-Learn Pipelines. Thresholds were calibrated on the validation set (2017–2018) to maximize F1-score.
        """
    )

    df_t2 = load_table("table2_classification_benchmark.csv")
    if not df_t2.empty:
        st.markdown("#### Table 2: Test Set Classification Benchmark (2019–2024)")
        st.dataframe(df_t2, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Diagnostic Visualization Suite")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Figure 1: Quad-Panel Test Confusion Matrices**")
        p_fig1 = FIGURES_DIR / "fig1_confusion_matrices.png"
        if p_fig1.exists():
            st.image(str(p_fig1), use_container_width=True)
    with c2:
        st.markdown("**Figure 4: Model Complexity & Overfitting Curves (Train vs Val Only)**")
        p_fig4 = FIGURES_DIR / "fig4_complexity_curves.png"
        if p_fig4.exists():
            st.image(str(p_fig4), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Figure 2: ROC Curves with 95% Bootstrap CIs**")
        p_fig2 = FIGURES_DIR / "fig2_roc_curves.png"
        if p_fig2.exists():
            st.image(str(p_fig2), use_container_width=True)
    with c4:
        st.markdown("**Figure 3: Precision-Recall Curves (15.01% Base Rate)**")
        p_fig3 = FIGURES_DIR / "fig3_pr_curves.png"
        if p_fig3.exists():
            st.image(str(p_fig3), use_container_width=True)

# =============================================================================
# TAB 3: REGRESSION SUITE
# =============================================================================
with tabs[2]:
    st.markdown("### ⏱️ Question 2: Supervised Finishing Position Regression")
    st.markdown(
        """
        **Definitive Target:** `positionOrder` for all starters (1994–2024; zero missing values).
        DNF results rank at the rear by Ergast convention. No sentinel values used.
        """
    )

    df_t3a = load_table("table3a_regression_benchmark.csv")
    if not df_t3a.empty:
        st.markdown("#### Table 3a: Test Set Regression Benchmark")
        st.dataframe(df_t3a, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Figure 5: Gradient Descent Stability Boundary")
        st.markdown(
            r"Theoretical stability requires $\alpha < \frac{2}{\lambda_{\max}}$. "
            r"With $\lambda_{\max} = 5.9366$, $\alpha_{\text{safe}} = 0.2698$ converges smoothly, "
            r"while $\alpha_{\text{unsafe}} = 0.7412$ diverges demonstrably."
        )
        p_fig5 = FIGURES_DIR / "fig5_gd_convergence.png"
        if p_fig5.exists():
            st.image(str(p_fig5), use_container_width=True)

    with col2:
        st.markdown("#### Figure 6: Polynomial Degree Bias-Variance U-Curve")
        st.markdown("Demonstration on controlled 60-row subsample (Degrees 1 to 12).")
        p_fig6 = FIGURES_DIR / "fig6_poly_bias_variance.png"
        if p_fig6.exists():
            st.image(str(p_fig6), use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Table 3b: Polynomial Degree Error Sweep")
        df_t3b = load_table("table3b_poly_demo.csv")
        if not df_t3b.empty:
            st.dataframe(df_t3b, use_container_width=True, hide_index=True)
    with col4:
        st.markdown("#### Figure 7: Ridge & Lasso Cross-Validation Error Paths")
        p_fig7 = FIGURES_DIR / "fig7_regularization_cv_paths.png"
        if p_fig7.exists():
            st.image(str(p_fig7), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Table 4: Regularization Sparsity Analysis (Degree-2 Feature Expansion)")
    df_t4 = load_table("table4_regularization_sparsity.csv")
    if not df_t4.empty:
        st.dataframe(df_t4, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: UNSUPERVISED DRIVER DISCOVERY
# =============================================================================
with tabs[3]:
    st.markdown("### 🔍 Question 3: Driver Career Clustering & PCA from First Principles")
    st.markdown(
        """
        **Scope:** Full 75-year championship history (1950–2024; 252 drivers with &ge; 20 career starts).
        **Era-Invariant Features (6):** `win_share`, `podium_share`, `pts_share`, `norm_avg_finish`, `dnf_adj`, `teammate_delta`.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Table 5: K-Means WCSS & Silhouette Selection")
        df_t5 = load_table("table5_kmeans_selection.csv")
        if not df_t5.empty:
            st.dataframe(df_t5, use_container_width=True, hide_index=True)
        p_fig8 = FIGURES_DIR / "fig8_kmeans_elbow_silhouette.png"
        if p_fig8.exists():
            st.image(str(p_fig8), use_container_width=True)

    with c2:
        st.markdown("#### Table 6: PCA Eigenvalue Spectrum")
        df_t6 = load_table("table6_pca_spectrum.csv")
        if not df_t6.empty:
            st.dataframe(df_t6, use_container_width=True, hide_index=True)
        p_fig10 = FIGURES_DIR / "fig10_pca_scree.png"
        if p_fig10.exists():
            st.image(str(p_fig10), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Table 6b: Full PCA Loading Matrix (6 Features &times; 6 Principal Components)")
    df_t6b = load_table("table6b_pca_loadings.csv")
    if not df_t6b.empty:
        st.dataframe(df_t6b, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Figure 9: 2D PCA Biplot with Driver Career Clusters")
        p_fig9 = FIGURES_DIR / "fig9_pca_2d_clusters.png"
        if p_fig9.exists():
            st.image(str(p_fig9), use_container_width=True)
    with c4:
        st.markdown("#### Figure 11: Supplementary 3D Driver Career Space")
        p_fig11 = FIGURES_DIR / "fig11_pca_3d.png"
        if p_fig11.exists():
            st.image(str(p_fig11), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🏎️ Interactive Driver Career Lookup Tool")
    df_career = load_career_data()
    if not df_career.empty:
        driver_search = st.selectbox(
            "Select Driver to View Career Statistics & Benchmark:",
            options=sorted(df_career["driver_name"].unique()),
            index=sorted(df_career["driver_name"].unique()).index("Lewis Hamilton") if "Lewis Hamilton" in df_career["driver_name"].values else 0,
        )
        driver_row = df_career[df_career["driver_name"] == driver_search].iloc[0]
        c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
        c_m1.metric("Career Starts", int(driver_row["total_starts"]))
        c_m2.metric("Total Wins", int(driver_row["total_wins"]), f"{driver_row['win_share']:.1%} win rate")
        c_m3.metric("Total Podiums", int(driver_row["total_podiums"]), f"{driver_row['podium_share']:.1%} podium rate")
        c_m4.metric("Avg Finish Index", f"{driver_row['norm_avg_finish']:.3f}", "0.0 = winner, 1.0 = last")
        c_m5.metric("Teammate Delta", f"{driver_row['teammate_delta']:+.3f}", "negative = beat teammate")

# =============================================================================
# TAB 5: APPENDICES
# =============================================================================
with tabs[4]:
    st.markdown("### 📚 Standard Dataset Compliance Appendices")
    st.markdown(
        """
        To satisfy formal assignment requirements ('use a standard dataset'), all algorithms
        were replicated on classic textbook machine learning datasets.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Appendix Table A: Digits Classification (10-Class, 8x8 Grayscale)")
        df_app_a = load_table("app_table_a_digits.csv")
        if not df_app_a.empty:
            st.dataframe(df_app_a, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Appendix Table B: Diabetes Progression Regression (10 Features)")
        df_app_b = load_table("app_table_b_diabetes.csv")
        if not df_app_b.empty:
            st.dataframe(df_app_b, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Appendix Table C: Iris Clustering & PCA")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Appendix Table C1: Iris K-Means WCSS & Silhouette**")
        df_app_c_km = load_table("app_table_c_iris_kmeans.csv")
        if not df_app_c_km.empty:
            st.dataframe(df_app_c_km, use_container_width=True, hide_index=True)
    with col_c2:
        st.markdown("**Appendix Table C2: Iris PCA Full Loading Matrix (4x4)**")
        df_app_c_pca = load_table("app_table_c_iris_pca.csv")
        if not df_app_c_pca.empty:
            st.dataframe(df_app_c_pca, use_container_width=True)
