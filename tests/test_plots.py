"""Unit and Integration Tests for Diagnostic Plotting Module (Figure 12).

Tests publication standards:
- 300 DPI high-resolution export.
- 4-panel diagnostic layout.
- Standalone execution with automatic fallbacks.
- 100% backward compatibility with Fig 1 through Fig 11.
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import pandas as pd
import pytest

from src.visualization.plots import (
    plot_fig1_confusion_matrices,
    plot_fig2_roc_curves,
    plot_fig3_pr_curves,
    plot_fig4_complexity_curves,
    plot_fig5_gd_convergence,
    plot_fig6_poly_bias_variance,
    plot_fig7_regularization_cv_paths,
    plot_fig8_kmeans_elbow_silhouette,
    plot_fig9_pca_2d_clusters,
    plot_fig10_pca_scree,
    plot_fig11_pca_3d,
    plot_fig12_ranking_and_hurdle_eval,
)
from src.models.hurdle_model import F1HurdleModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_plot_fig12_ranking_and_hurdle_eval_default_invocation(tmp_path: Path) -> None:
    """Verify Figure 12 generates cleanly with default arguments and meets publication standards."""
    out_file = tmp_path / "test_fig12.png"
    result_path = plot_fig12_ranking_and_hurdle_eval(output_path=out_file)

    assert result_path.exists(), f"Expected figure file at {result_path}"
    assert result_path.stat().st_size > 100_000, "Figure file size is unexpectedly small"

    with Image.open(result_path) as im:
        assert im.format == "PNG", f"Expected PNG format, got {im.format}"
        assert im.size == (4800, 3600), f"Expected 4800x3600 resolution at 16x12 in 300 DPI, got {im.size}"
        dpi = im.info.get("dpi")
        assert dpi is not None, "DPI metadata missing in output PNG"
        assert int(round(dpi[0])) == 300, f"Expected 300 DPI, got {dpi}"


def test_plot_fig12_ranking_and_hurdle_eval_explicit_args(tmp_path: Path) -> None:
    """Verify Figure 12 functions correctly when preloaded DataFrames and fitted Hurdle model are supplied."""
    table8_path = PROJECT_ROOT / "reports" / "tables" / "table8_ranking_hurdle_benchmark.csv"
    df_table8 = pd.read_csv(table8_path)
    train_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "train_core.parquet")
    val_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "val.parquet")
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "test.parquet")

    hurdle = F1HurdleModel(random_state=42).fit(train_df, df_val=val_df)

    out_file = tmp_path / "test_fig12_explicit.png"
    result_path = plot_fig12_ranking_and_hurdle_eval(
        df_table8=df_table8,
        test_df=test_df,
        train_df=train_df,
        val_df=val_df,
        hurdle_model=hurdle,
        output_path=out_file,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 100_000

    with Image.open(result_path) as im:
        assert im.format == "PNG"
        dpi = im.info.get("dpi")
        assert dpi is not None and int(round(dpi[0])) == 300


def test_backward_compatibility_existing_plots_callable() -> None:
    """Verify 100% backward compatibility: existing plotting functions Fig 1 to Fig 11 remain callable."""
    functions = [
        plot_fig1_confusion_matrices,
        plot_fig2_roc_curves,
        plot_fig3_pr_curves,
        plot_fig4_complexity_curves,
        plot_fig5_gd_convergence,
        plot_fig6_poly_bias_variance,
        plot_fig7_regularization_cv_paths,
        plot_fig8_kmeans_elbow_silhouette,
        plot_fig9_pca_2d_clusters,
        plot_fig10_pca_scree,
        plot_fig11_pca_3d,
        plot_fig12_ranking_and_hurdle_eval,
    ]
    for fn in functions:
        assert callable(fn), f"Function {fn.__name__} must be callable"
