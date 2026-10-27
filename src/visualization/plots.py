"""Publication-Quality Visualization and Diagnostic Plotting Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
A picture is worth a thousand words, especially in data science!
Even if our math is 100% correct, if our charts are blurry, confusing, or unreadable,
nobody will understand or trust our findings.

Every figure produced in this module adheres to strict publication standards:
- 300 DPI high-resolution export for crisp, razor-sharp rendering in scientific PDFs.
- Consistent color palettes and clear white-grid background styling.
- Meaningful annotations marking key inflection points (e.g. Elbow point, optimal K,
  early stopping epoch, safe vs unsafe learning rate divergence).
- Figure 4 complexity curves use training and validation curves ONLY, guaranteeing that
  the test set is never peeked at for hyperparameter tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

FIGURES_DIR = Path("reports/figures")


def setup_plotting_theme() -> None:
    """Set global matplotlib and seaborn styles."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })


def plot_fig1_confusion_matrices(
    y_true: np.ndarray,
    model_predictions: dict[str, tuple[np.ndarray, float, str]],
    output_path: Path = FIGURES_DIR / "fig1_confusion_matrices.png",
) -> None:
    """Figure 1: 2x2 quad-panel confusion matrices for KNN, Decision Tree, SVM, and CNN.

    Parameters
    ----------
    y_true : np.ndarray
        True binary test labels.
    model_predictions : dict[str, tuple[np.ndarray, float, str]]
        Dictionary mapping model name to tuple of (binary_preds, threshold, interpretation_str).
    output_path : Path
        Destination filepath.
    """
    setup_plotting_theme()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    models = ["KNN", "Decision Tree", "SVM (RBF)", "Dual-Branch 1D-CNN"]

    for i, name in enumerate(models):
        ax = axes[i]
        if name in model_predictions:
            preds, thresh, interp = model_predictions[name]
            cm = confusion_matrix(y_true, preds)
            # cm layout: [[TN, FP], [FN, TP]]
            tn, fp, fn, tp = cm.ravel()
            labels = np.array([
                [f"TN\n{tn:,}", f"FP\n{fp:,}"],
                [f"FN\n{fn:,}", f"TP\n{tp:,}"],
            ])
            sns.heatmap(
                cm,
                annot=labels,
                fmt="",
                cmap="Blues",
                cbar=False,
                ax=ax,
                annot_kws={"size": 13, "weight": "bold"},
            )
            ax.set_title(f"{name} (Threshold = {thresh:.3f})", weight="bold")
            ax.set_xlabel("Predicted Label (0: No Podium, 1: Podium)")
            ax.set_ylabel("True Label (0: No Podium, 1: Podium)")
            ax.text(
                0.5,
                -0.22,
                interp,
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=9.5,
                style="italic",
                wrap=True,
            )

    plt.suptitle("Figure 1: Quad-Panel Test Confusion Matrices with Val-Calibrated Thresholds", y=0.98, weight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig2_roc_curves(
    y_true: np.ndarray,
    model_scores: dict[str, tuple[np.ndarray, float, tuple[float, float]]],
    output_path: Path = FIGURES_DIR / "fig2_roc_curves.png",
) -> None:
    """Figure 2: ROC Curves for all classifiers and baselines with 95% CIs.

    Parameters
    ----------
    y_true : np.ndarray
        True binary test labels.
    model_scores : dict[str, tuple[np.ndarray, float, tuple[float, float]]]
        Dictionary mapping model name to tuple of (scores, auc, (ci_lower, ci_upper)).
    output_path : Path
        Destination filepath.
    """
    setup_plotting_theme()
    plt.figure(figsize=(9, 7))

    # Random guess diagonal
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.7)

    colors = {
        "Majority Class": "#7f7f7f",
        "Grid Heuristic": "#e377c2",
        "KNN": "#1f77b4",
        "Decision Tree": "#2ca02c",
        "SVM (RBF)": "#ff7f0e",
        "Dual-Branch 1D-CNN": "#d62728",
    }

    for name, (scores, auc_val, (ci_low, ci_high)) in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        label_text = f"{name}: AUC = {auc_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
        c = colors.get(name, None)
        plt.plot(fpr, tpr, label=label_text, linewidth=2.2, color=c)

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Figure 2: Test Set ROC Curves with 95% Race-Aligned Bootstrap CIs", weight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig3_pr_curves(
    y_true: np.ndarray,
    model_scores: dict[str, tuple[np.ndarray, float]],
    output_path: Path = FIGURES_DIR / "fig3_pr_curves.png",
) -> None:
    """Figure 3: Precision-Recall curves with empirical baseline rate."""
    setup_plotting_theme()
    plt.figure(figsize=(9, 7))

    base_rate = float(np.mean(y_true))
    plt.axhline(
        base_rate,
        color="k",
        linestyle="--",
        label=f"No-Skill Baseline (Podium Rate = {base_rate:.1%})",
        alpha=0.7,
    )

    colors = {
        "Majority Class": "#7f7f7f",
        "Grid Heuristic": "#e377c2",
        "KNN": "#1f77b4",
        "Decision Tree": "#2ca02c",
        "SVM (RBF)": "#ff7f0e",
        "Dual-Branch 1D-CNN": "#d62728",
    }

    for name, (scores, pr_auc_val) in model_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        label_text = f"{name}: PR-AUC = {pr_auc_val:.3f}"
        c = colors.get(name, None)
        plt.plot(rec, prec, label=label_text, linewidth=2.2, color=c)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Figure 3: Test Set Precision-Recall Curves (Class Imbalance ~14.1%)", weight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig4_complexity_curves(
    knn_k: list[int],
    knn_auc: list[float],
    dt_depths: list[int],
    dt_train_acc: list[float],
    dt_val_acc: list[float],
    cnn_epochs: list[int],
    cnn_train_loss: list[float],
    cnn_val_loss: list[float],
    cnn_stop_epoch: int,
    svm_c: list[float],
    svm_gamma: list[str | float],
    svm_heatmap_scores: np.ndarray,
    output_path: Path = FIGURES_DIR / "fig4_complexity_curves.png",
) -> None:
    """Figure 4: Four-panel complexity curves using Train vs Validation ONLY (never test)."""
    setup_plotting_theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: KNN Validation AUC vs K
    ax1 = axes[0, 0]
    ax1.plot(knn_k, knn_auc, "o-", color="#1f77b4", linewidth=2, markersize=5)
    best_k_idx = int(np.argmax(knn_auc))
    ax1.scatter([knn_k[best_k_idx]], [knn_auc[best_k_idx]], color="red", s=100, zorder=5, label=f"Optimal K={knn_k[best_k_idx]}")
    ax1.set_title("Panel A: KNN Cross-Validation AUC vs K", weight="bold")
    ax1.set_xlabel("Number of Neighbors (K)")
    ax1.set_ylabel("CV ROC-AUC")
    ax1.legend(loc="lower right")

    # Panel 2: Decision Tree Train vs Val Accuracy vs Depth
    ax2 = axes[0, 1]
    ax2.plot(dt_depths, dt_train_acc, "s--", color="#2ca02c", label="Train Accuracy", linewidth=1.8)
    ax2.plot(dt_depths, dt_val_acc, "o-", color="#d62728", label="Val Accuracy", linewidth=2.2)
    best_d_idx = int(np.argmax(dt_val_acc))
    ax2.scatter([dt_depths[best_d_idx]], [dt_val_acc[best_d_idx]], color="black", s=100, zorder=5, label=f"Peak Val Depth={dt_depths[best_d_idx]}")
    ax2.set_title("Panel B: Decision Tree Depth vs Train/Val Accuracy", weight="bold")
    ax2.set_xlabel("Max Tree Depth")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")

    # Panel 3: 1D-CNN Train vs Val BCE Loss
    ax3 = axes[1, 0]
    ax3.plot(cnn_epochs, cnn_train_loss, "b-", label="Train BCE Loss", linewidth=2)
    ax3.plot(cnn_epochs, cnn_val_loss, "r-", label="Val BCE Loss", linewidth=2)
    ax3.axvline(cnn_stop_epoch, color="k", linestyle=":", label=f"Early Stop Anchor (Epoch {cnn_stop_epoch})")
    ax3.set_title("Panel C: 1D-CNN Epoch vs Train/Val Loss (Early Stop on Val BCE)", weight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("BCE Loss")
    ax3.legend(loc="upper right")

    # Panel 4: SVM Validation Heatmap over C x gamma
    ax4 = axes[1, 1]
    sns.heatmap(
        svm_heatmap_scores,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[str(g) for g in svm_gamma],
        yticklabels=[str(c) for c in svm_c],
        ax=ax4,
    )
    ax4.set_title("Panel D: SVM RBF Validation AUC Heatmap (C x Gamma)", weight="bold")
    ax4.set_xlabel("Kernel Coefficient (Gamma)")
    ax4.set_ylabel("Penalty Parameter (C)")

    plt.suptitle("Figure 4: Model Complexity & Overfitting Diagnostic Panels (Train vs Validation)", y=0.99, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig5_gd_convergence(
    cost_safe: list[float],
    cost_unsafe: list[float],
    alpha_safe: float,
    alpha_unsafe: float,
    output_path: Path = FIGURES_DIR / "fig5_gd_convergence.png",
) -> None:
    """Figure 5: Gradient Descent Cost vs Iteration for safe and unsafe alpha."""
    setup_plotting_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Safe alpha (smooth convergence)
    iters_safe = range(1, len(cost_safe) + 1)
    ax1.plot(iters_safe[:500], cost_safe[:500], "b-", linewidth=2)
    ax1.set_title(f"Safe Alpha = {alpha_safe:.4f} (Smooth Convergence)", weight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cost J(theta)")
    ax1.grid(True)

    # Unsafe alpha (demonstrable divergence)
    iters_unsafe = range(1, len(cost_unsafe) + 1)
    ax2.plot(iters_unsafe, cost_unsafe, "r-o", linewidth=2, markersize=5)
    ax2.set_title(f"Unsafe Alpha = {alpha_unsafe:.4f} (Violent Divergence)", weight="bold")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cost J(theta)")
    ax2.set_yscale("log")
    ax2.grid(True)

    plt.suptitle("Figure 5: Empirical Verification of Analytical Gradient Descent Stability Boundary", y=0.98, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig6_poly_bias_variance(
    poly_summary_df: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "fig6_poly_bias_variance.png",
) -> None:
    """Figure 6: Polynomial Regression Train MSE and Held-Out MSE vs Degree."""
    setup_plotting_theme()
    plt.figure(figsize=(9, 6))

    degrees = poly_summary_df["Degree"].to_numpy()
    train_mse = poly_summary_df["Train MSE"].to_numpy()
    held_mse = poly_summary_df["Held-Out MSE"].to_numpy()

    plt.plot(degrees, train_mse, "s--", color="#2ca02c", label="Train MSE (60 rows)", linewidth=2)
    plt.plot(degrees, held_mse, "o-", color="#d62728", label="Held-Out Validation MSE (20 rows)", linewidth=2.2)

    plt.yscale("log")
    plt.xlabel("Polynomial Degree (on [-1, 1] Scaled Grid Position)")
    plt.ylabel("Mean Squared Error (Log Scale)")
    plt.title("Figure 6: Polynomial Degree Bias-Variance U-Curve Demonstration", weight="bold")
    plt.legend(loc="upper left")
    plt.xticks(degrees)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig7_regularization_cv_paths(
    ridge_alphas: np.ndarray,
    ridge_cv_mse: np.ndarray,
    ridge_best_alpha: float,
    lasso_alphas: np.ndarray,
    lasso_cv_mse: np.ndarray,
    lasso_best_alpha: float,
    output_path: Path = FIGURES_DIR / "fig7_regularization_cv_paths.png",
) -> None:
    """Figure 7: Ridge and Lasso Cross-Validation MSE vs Log(Alpha)."""
    setup_plotting_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Ridge path
    ax1.plot(ridge_alphas, ridge_cv_mse, "o-", color="#1f77b4", linewidth=2, markersize=4)
    ax1.axvline(ridge_best_alpha, color="r", linestyle="--", label=f"Optimal Alpha = {ridge_best_alpha:.4f}")
    ax1.set_xscale("log")
    ax1.set_title("Ridge Regression (L2) CV Error Path", weight="bold")
    ax1.set_xlabel("Regularization Strength (Alpha)")
    ax1.set_ylabel("Cross-Validation MSE")
    ax1.legend()

    # Lasso path
    ax2.plot(lasso_alphas, lasso_cv_mse, "s-", color="#ff7f0e", linewidth=2, markersize=4)
    ax2.axvline(lasso_best_alpha, color="r", linestyle="--", label=f"Optimal Alpha = {lasso_best_alpha:.4f}")
    ax2.set_xscale("log")
    ax2.set_title("Lasso Regression (L1) CV Error Path", weight="bold")
    ax2.set_xlabel("Regularization Strength (Alpha)")
    ax2.set_ylabel("Cross-Validation MSE")
    ax2.legend()

    plt.suptitle("Figure 7: Regularization Penalty Optimization via Race-Aligned Cross-Validation", y=0.98, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig8_kmeans_elbow_silhouette(
    k_range: list[int],
    wcss_values: list[float],
    silhouette_values: list[float],
    optimal_k: int,
    output_path: Path = FIGURES_DIR / "fig8_kmeans_elbow_silhouette.png",
) -> None:
    """Figure 8: Dual y-axis K-Means Elbow (WCSS) and Silhouette Score vs K."""
    setup_plotting_theme()
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_wcss = "#1f77b4"
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)", color=color_wcss, weight="bold")
    line1 = ax1.plot(k_range, wcss_values, "o-", color=color_wcss, linewidth=2.2, label="WCSS (Inertia)")
    ax1.tick_params(axis="y", labelcolor=color_wcss)

    ax2 = ax1.twinx()
    color_sil = "#2ca02c"
    ax2.set_ylabel("Silhouette Score", color=color_sil, weight="bold")
    line2 = ax2.plot(k_range, silhouette_values, "s--", color=color_sil, linewidth=2.2, label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_sil)

    ax1.axvline(optimal_k, color="red", linestyle=":", label=f"Optimal K = {optimal_k}")

    lines = line1 + line2 + [ax1.get_lines()[-1]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("Figure 8: K-Means Optimal K Selection (Dual Y-Axis Elbow & Silhouette)", weight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig9_pca_2d_clusters(
    z_2d: np.ndarray,
    cluster_labels: np.ndarray,
    feature_loadings: np.ndarray,
    feature_names: list[str],
    cluster_names: dict[int, str],
    driver_names: list[str] | None = None,
    output_path: Path = FIGURES_DIR / "fig9_pca_2d_clusters.png",
) -> None:
    """Figure 9: PC1 vs PC2 projection colored by K-Means clusters with biplot loading arrows."""
    setup_plotting_theme()
    plt.figure(figsize=(10, 8))

    unique_clusters = np.unique(cluster_labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))

    for k in unique_clusters:
        idx = cluster_labels == k
        name = cluster_names.get(k, f"Cluster {k}")
        plt.scatter(
            z_2d[idx, 0],
            z_2d[idx, 1],
            label=name,
            color=palette[k],
            alpha=0.75,
            s=50,
            edgecolors="none",
        )

    # Annotate famous driver outliers if provided
    if driver_names:
        famous_drivers = ["Lewis Hamilton", "Michael Schumacher", "Ayrton Senna", "Max Verstappen", "Fernando Alonso"]
        for name in famous_drivers:
            if name in driver_names:
                i = driver_names.index(name)
                plt.annotate(
                    name,
                    (z_2d[i, 0], z_2d[i, 1]),
                    fontsize=9,
                    weight="bold",
                    xytext=(5, 5),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6),
                )

    # Biplot arrows for feature loadings
    scale_factor = 4.0
    for j, feat in enumerate(feature_names):
        dx = feature_loadings[j, 0] * scale_factor
        dy = feature_loadings[j, 1] * scale_factor
        plt.arrow(
            0,
            0,
            dx,
            dy,
            color="black",
            alpha=0.6,
            width=0.03,
            head_width=0.15,
            length_includes_head=True,
        )
        plt.text(
            dx * 1.15,
            dy * 1.15,
            feat,
            color="black",
            ha="center",
            va="center",
            weight="bold",
            fontsize=9,
        )

    plt.xlabel("Principal Component 1 (Career Dominance & Win Share)")
    plt.ylabel("Principal Component 2 (Reliability & Completion Rate)")
    plt.title("Figure 9: Driver Career Clusters in 2D PCA Space with Biplot Feature Vectors", weight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.axhline(0, color="grey", linestyle=":", alpha=0.5)
    plt.axvline(0, color="grey", linestyle=":", alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig10_pca_scree(
    evr: np.ndarray,
    output_path: Path = FIGURES_DIR / "fig10_pca_scree.png",
) -> None:
    """Figure 10: Scree plot of individual and cumulative explained variance ratio."""
    setup_plotting_theme()
    fig, ax1 = plt.subplots(figsize=(9, 6))

    pcs = [f"PC{i+1}" for i in range(len(evr))]
    cum_evr = np.cumsum(evr)

    # Bar chart for individual EVR
    bars = ax1.bar(pcs, evr * 100.0, color="#1f77b4", alpha=0.7, label="Individual EVR (%)")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Individual Variance Explained (%)", color="#1f77b4", weight="bold")
    ax1.set_ylim([0, 105])

    # Line chart for cumulative EVR
    ax2 = ax1.twinx()
    line = ax2.plot(pcs, cum_evr * 100.0, "r-o", linewidth=2.5, label="Cumulative EVR (%)")
    ax2.set_ylabel("Cumulative Variance Explained (%)", color="red", weight="bold")
    ax2.set_ylim([0, 105])
    ax2.axhline(80, color="grey", linestyle="--", alpha=0.7, label="80% Threshold")

    # Annotate bars
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.title("Figure 10: PCA Scree Plot (Individual & Cumulative Explained Variance)", weight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fig11_pca_3d(
    z_3d: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: dict[int, str],
    output_path: Path = FIGURES_DIR / "fig11_pca_3d.png",
) -> None:
    """Figure 11: 3D scatter plot of PC1, PC2, and PC3 colored by cluster."""
    setup_plotting_theme()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    unique_clusters = np.unique(cluster_labels)
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))

    for k in unique_clusters:
        idx = cluster_labels == k
        name = cluster_names.get(k, f"Cluster {k}")
        ax.scatter(
            z_3d[idx, 0],
            z_3d[idx, 1],
            z_3d[idx, 2],
            label=name,
            color=palette[k],
            s=40,
            alpha=0.8,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title("Figure 11: Supplementary 3D Driver Career Space (PC1 x PC2 x PC3)", weight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
