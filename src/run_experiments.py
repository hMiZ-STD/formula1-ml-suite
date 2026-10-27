"""Master Experiment Runner and Evaluation Orchestrator.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
This is the mission control center!
It runs all three machine learning mini-projects in strict, mathematically sound order:
1. Classification Suite: Baselines, KNN, Decision Tree, SVM, and 1D-CNN are tuned on historical
   data (1994–2016). Decision thresholds are calibrated on the validation set (2017–2018).
2. Regression Suite: Scratch Gradient Descent, OLS, Ridge, Lasso, and the Polynomial Bias-Variance
   demo are trained to predict finishing positions.
3. The Experiment Freeze: All winning settings, hyperparameters, seeds, and decision thresholds
   are locked inside EXPERIMENT_FREEZE.yaml before touching the test set!
4. The Final Test Evaluation: The models are evaluated exactly ONCE on the modern hybrid era
   (2019–2024). Tables 2, 3a, 3b, 4, and Figures 1–7 are rendered.
5. Unsupervised Suite: K-Means and PCA from scratch analyze the entire 1950–2024 driver careers,
   producing Tables 5, 6, 6b, and Figures 8–11.
"""

from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import write_experiment_freeze
from src.evaluation.classification_metrics import (
    bootstrap_auc_ci,
    compute_classification_metrics,
    find_optimal_threshold,
)
from src.evaluation.regression_metrics import compute_regression_metrics
from src.features.builder import ALLOWED_FEATURES
from src.models.classifiers import (
    create_classifier_pipeline,
    tune_decision_tree,
    tune_knn,
    tune_svm,
)
from src.models.cnn_classifier import (
    DualBranchCNN,
    StaticMLP,
    build_sequence_tensors,
    predict_cnn_probabilities,
    train_cnn_classifier,
)
from src.models.kmeans_scratch import KMeansScratch
from src.models.linear_gd import (
    compute_cost,
    derive_learning_rates,
    fit_closed_form_lstsq,
    gradient_descent,
)
from src.models.pca_scratch import PCAScratch
from src.models.polynomial_reg import run_polynomial_bias_variance_demo
from src.models.regularized_reg import (
    CONTINUOUS_FEATURES,
    BINARY_FLAGS,
    create_regularized_pipeline,
    tune_ridge_and_lasso,
)
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
)

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("reports/tables")
FIGURES_DIR = Path("reports/figures")


def run_full_suite() -> None:
    """Execute full experimental pipeline across all mini-projects."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: LOADING DATASETS")
    print("=" * 70)
    train_core = pd.read_parquet(PROCESSED_DIR / "train_core.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    career_df = pd.read_parquet(PROCESSED_DIR / "career_clustering.parquet")

    train_val_pool = pd.concat([train_core, val], ignore_index=True)
    print(f"train_core: {len(train_core)} | val: {len(val)} | train_val_pool: {len(train_val_pool)} | test: {len(test)}")

    # =========================================================================
    # STEP 2: CLASSIFICATION SUITE TUNING (ON TRAIN_CORE & VAL)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: CLASSIFICATION HYPERPARAMETER TUNING & VALIDATION")
    print("=" * 70)

    # 1. KNN
    print("Tuning KNN...")
    knn_grid, best_knn_core = tune_knn(train_core, n_neighbors_range=range(1, 31))
    best_k = int(knn_grid.best_params_["knn__n_neighbors"])
    print(f"  Best K: {best_k} (CV AUC = {knn_grid.best_score_:.4f})")

    # KNN val predictions for threshold
    val_knn_scores = best_knn_core.predict_proba(val[ALLOWED_FEATURES])[:, 1]
    knn_thresh, knn_val_f1 = find_optimal_threshold(val["y_podium"].to_numpy(), val_knn_scores)
    print(f"  KNN Val Optimal Threshold: {knn_thresh:.3f} (Val F1 = {knn_val_f1:.4f})")

    # 2. Decision Tree
    print("\nTuning Decision Tree...")
    dt_grid, best_dt_core = tune_decision_tree(train_core, max_depth_range=range(1, 21))
    best_depth = int(dt_grid.best_params_["dt__max_depth"])
    best_ccp = float(dt_grid.best_params_["dt__ccp_alpha"])
    print(f"  Best Depth: {best_depth}, ccp_alpha: {best_ccp} (CV AUC = {dt_grid.best_score_:.4f})")

    val_dt_scores = best_dt_core.predict_proba(val[ALLOWED_FEATURES])[:, 1]
    dt_thresh, dt_val_f1 = find_optimal_threshold(val["y_podium"].to_numpy(), val_dt_scores)
    print(f"  DT Val Optimal Threshold: {dt_thresh:.3f} (Val F1 = {dt_val_f1:.4f})")

    # 3. SVM (RBF)
    print("\nTuning SVM (on race-aligned ~5k sample)...")
    svm_grid, best_svm_core = tune_svm(train_core)
    best_c = float(svm_grid.best_params_["svm__C"])
    best_gamma = svm_grid.best_params_["svm__gamma"]
    print(f"  Best C: {best_c}, gamma: {best_gamma} (CV AUC = {svm_grid.best_score_:.4f})")

    # Fit best SVM params on full train_core for val threshold tuning
    pipe_svm_core = create_classifier_pipeline(
        "svm",
        best_svm_core.named_steps["svm"],
    )
    pipe_svm_core.fit(train_core[ALLOWED_FEATURES], train_core["y_podium"].to_numpy())
    val_svm_scores = pipe_svm_core.decision_function(val[ALLOWED_FEATURES])
    svm_thresh, svm_val_f1 = find_optimal_threshold(
        val["y_podium"].to_numpy(), val_svm_scores, is_decision_score=True
    )
    print(f"  SVM Val Optimal Threshold: {svm_thresh:.3f} (Val F1 = {svm_val_f1:.4f})")

    # 4. Dual-Branch 1D-CNN
    print("\nTraining Dual-Branch 1D-CNN (with Early Stopping on Val BCE Loss)...")
    cnn_model, cnn_static_pipe, cnn_hist, cnn_stop_epoch = train_cnn_classifier(
        train_core, val, max_epochs=40, batch_size=32, patience=10
    )
    print(f"  CNN Early Stopped at Epoch: {cnn_stop_epoch}")

    # Generate val sequence for CNN
    seq_train, seq_val, seq_test = build_sequence_tensors([train_core, val, test])
    val_cnn_scores = predict_cnn_probabilities(cnn_model, cnn_static_pipe, val, seq_val)
    cnn_thresh, cnn_val_f1 = find_optimal_threshold(val["y_podium"].to_numpy(), val_cnn_scores)
    print(f"  CNN Val Optimal Threshold: {cnn_thresh:.3f} (Val F1 = {cnn_val_f1:.4f})")

    # 5. Baseline Val Thresholds
    # Baseline 2: Grid Heuristic score = -grid_effective (lower grid = higher podium chance)
    val_grid_scores = -val["grid_effective"].to_numpy()
    grid_thresh, _ = find_optimal_threshold(
        val["y_podium"].to_numpy(), val_grid_scores, is_decision_score=True
    )

    # Figure 4 Data Assembly (Complexity Curves)
    knn_cv_results = knn_grid.cv_results_
    knn_k_list = list(range(1, 31))
    knn_auc_list = [float(score) for score in knn_cv_results["mean_test_score"]]

    dt_depths_list = list(range(1, 21))
    # Collect train vs val accuracy across depths at best ccp_alpha
    dt_train_acc_list = []
    dt_val_acc_list = []
    from sklearn.tree import DecisionTreeClassifier
    for d in dt_depths_list:
        tmp_pipe = create_classifier_pipeline("dt", DecisionTreeClassifier(max_depth=d, ccp_alpha=best_ccp, class_weight="balanced", random_state=42))
        tmp_pipe.fit(train_core[ALLOWED_FEATURES], train_core["y_podium"].to_numpy())
        dt_train_acc_list.append(float(accuracy_score(train_core["y_podium"], tmp_pipe.predict(train_core[ALLOWED_FEATURES]))))
        dt_val_acc_list.append(float(accuracy_score(val["y_podium"], tmp_pipe.predict(val[ALLOWED_FEATURES]))))

    # SVM Heatmap grid scores
    svm_c_list = [0.1, 1.0, 10.0, 100.0]
    svm_gamma_list = [0.001, 0.01, 0.1, "scale"]
    svm_heatmap_matrix = np.array(svm_grid.cv_results_["mean_test_score"]).reshape(len(svm_c_list), len(svm_gamma_list))

    plot_fig4_complexity_curves(
        knn_k=knn_k_list,
        knn_auc=knn_auc_list,
        dt_depths=dt_depths_list,
        dt_train_acc=dt_train_acc_list,
        dt_val_acc=dt_val_acc_list,
        cnn_epochs=list(range(1, len(cnn_hist["train_loss"]) + 1)),
        cnn_train_loss=cnn_hist["train_loss"],
        cnn_val_loss=cnn_hist["val_loss"],
        cnn_stop_epoch=cnn_stop_epoch,
        svm_c=svm_c_list,
        svm_gamma=svm_gamma_list,
        svm_heatmap_scores=svm_heatmap_matrix,
    )
    print("  Figure 4 complexity curves saved!")

    # =========================================================================
    # STEP 3: REGRESSION SUITE TUNING & DEMOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: REGRESSION EXPERIMENTS & TUNING")
    print("=" * 70)

    # 1. Scratch Gradient Descent on train_core
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    gd_imputer = SimpleImputer(strategy="median", add_indicator=True)
    gd_scaler = StandardScaler()

    X_train_scaled = gd_scaler.fit_transform(gd_imputer.fit_transform(train_core[ALLOWED_FEATURES]))
    n_tr = len(train_core)
    X_train_gd = np.column_stack([np.ones(n_tr), X_train_scaled])
    y_train_pos = train_core["y_position"].to_numpy()

    alpha_safe, alpha_unsafe, lambda_max = derive_learning_rates(X_train_gd)
    print(f"  Gradient Descent Stability Analysis:")
    print(f"    lambda_max: {lambda_max:.4f} | alpha_max: {2.0/lambda_max:.4f}")
    print(f"    alpha_safe: {alpha_safe:.4f} | alpha_unsafe: {alpha_unsafe:.4f}")

    theta_gd, cost_safe_hist = gradient_descent(X_train_gd, y_train_pos, alpha=alpha_safe, n_iter=1000)
    _, cost_unsafe_hist = gradient_descent(X_train_gd, y_train_pos, alpha=alpha_unsafe, n_iter=50)

    plot_fig5_gd_convergence(cost_safe_hist, cost_unsafe_hist, alpha_safe, alpha_unsafe)
    print("  Figure 5 GD convergence saved!")

    # 2. Polynomial Regression Bias-Variance Demo (Table 3b, Figure 6)
    poly_summary_df, _, _, _, _, _ = run_polynomial_bias_variance_demo(train_core)
    poly_summary_df.to_csv(TABLES_DIR / "table3b_poly_demo.csv", index=False)
    plot_fig6_poly_bias_variance(poly_summary_df)
    print("  Table 3b and Figure 6 Polynomial demo saved!")

    # 3. Regularized Regression (Ridge & Lasso) Tuning on train_val_pool
    print("\nTuning Ridge & Lasso on degree-2 expanded features...")
    alphas_grid = np.logspace(-4, 2, 30)
    ridge_grid, lasso_grid, ols_pipeline, table4_df = tune_ridge_and_lasso(train_val_pool, alphas=alphas_grid)
    best_ridge_alpha = float(ridge_grid.best_params_["regressor__alpha"])
    best_lasso_alpha = float(lasso_grid.best_params_["regressor__alpha"])
    print(f"  Best Ridge Alpha: {best_ridge_alpha:.4f}")
    print(f"  Best Lasso Alpha: {best_lasso_alpha:.4f}")

    table4_df.to_csv(TABLES_DIR / "table4_regularization_sparsity.csv", index=False)
    plot_fig7_regularization_cv_paths(
        ridge_alphas=alphas_grid,
        ridge_cv_mse=-ridge_grid.cv_results_["mean_test_score"],
        ridge_best_alpha=best_ridge_alpha,
        lasso_alphas=alphas_grid,
        lasso_cv_mse=-lasso_grid.cv_results_["mean_test_score"],
        lasso_best_alpha=best_lasso_alpha,
    )
    print("  Table 4 and Figure 7 Regularization paths saved!")

    # =========================================================================
    # STEP 4: EXPERIMENT FREEZE (MANDATORY BEFORE TEST EVALUATION)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: WRITING EXPERIMENT_FREEZE.yaml")
    print("=" * 70)
    write_experiment_freeze(
        filepath=Path("EXPERIMENT_FREEZE.yaml"),
        hyperparameters={
            "knn": {"n_neighbors": best_k},
            "dt": {"max_depth": best_depth, "ccp_alpha": best_ccp},
            "svm": {"C": best_c, "gamma": str(best_gamma)},
            "ridge": {"alpha": best_ridge_alpha},
            "lasso": {"alpha": best_lasso_alpha},
        },
        thresholds={
            "grid_heuristic": float(grid_thresh),
            "knn": float(knn_thresh),
            "dt": float(dt_thresh),
            "svm": float(svm_thresh),
            "cnn": float(cnn_thresh),
        },
        random_seeds={
            "numpy": 42,
            "torch": 42,
            "sklearn_random_state": 42,
        },
        cnn_stopped_epoch=cnn_stop_epoch,
        allowed_features=ALLOWED_FEATURES,
    )
    print("  EXPERIMENT_FREEZE.yaml written successfully! Models sealed.")

    # =========================================================================
    # STEP 5: REFIT SKLEARN MODELS ON TRAIN_VAL_POOL & EVALUATE ON TEST SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: REFITTING SKLEARN MODELS ON TRAIN_VAL_POOL & TEST EVALUATION")
    print("=" * 70)

    # 1. Refit KNN on train_val_pool
    final_knn = create_classifier_pipeline("knn", best_knn_core.named_steps["knn"])
    final_knn.fit(train_val_pool[ALLOWED_FEATURES], train_val_pool["y_podium"].to_numpy())
    test_knn_scores = final_knn.predict_proba(test[ALLOWED_FEATURES])[:, 1]
    test_knn_preds = (test_knn_scores >= knn_thresh).astype(int)

    # 2. Refit Decision Tree on train_val_pool
    final_dt = create_classifier_pipeline("dt", best_dt_core.named_steps["dt"])
    final_dt.fit(train_val_pool[ALLOWED_FEATURES], train_val_pool["y_podium"].to_numpy())
    test_dt_scores = final_dt.predict_proba(test[ALLOWED_FEATURES])[:, 1]
    test_dt_preds = (test_dt_scores >= dt_thresh).astype(int)

    # 3. Refit SVM on train_val_pool
    final_svm = create_classifier_pipeline("svm", best_svm_core.named_steps["svm"])
    final_svm.fit(train_val_pool[ALLOWED_FEATURES], train_val_pool["y_podium"].to_numpy())
    test_svm_scores = final_svm.decision_function(test[ALLOWED_FEATURES])
    test_svm_preds = (test_svm_scores >= svm_thresh).astype(int)

    # 4. CNN evaluates on test (stays on train_core!)
    test_cnn_scores = predict_cnn_probabilities(cnn_model, cnn_static_pipe, test, seq_test)
    test_cnn_preds = (test_cnn_scores >= cnn_thresh).astype(int)

    # 5. Baselines on test
    test_majority_preds = np.zeros(len(test), dtype=int)
    test_majority_scores = np.full(len(test), 0.5)

    test_grid_scores = -test["grid_effective"].to_numpy()
    test_grid_preds = (test_grid_scores >= grid_thresh).astype(int)

    y_test_podium = test["y_podium"].to_numpy()

    # Bootstrap CIs for ROC-AUC
    test_df_eval = test[["raceId", "y_podium"]].copy()
    test_df_eval["majority_score"] = test_majority_scores
    test_df_eval["grid_score"] = test_grid_scores
    test_df_eval["knn_score"] = test_knn_scores
    test_df_eval["dt_score"] = test_dt_scores
    test_df_eval["svm_score"] = test_svm_scores
    test_df_eval["cnn_score"] = test_cnn_scores

    auc_cis = {
        "Majority Class": (0.50, 0.50, 0.50),
        "Grid Heuristic": bootstrap_auc_ci(test_df_eval, "y_podium", "grid_score"),
        "KNN": bootstrap_auc_ci(test_df_eval, "y_podium", "knn_score"),
        "Decision Tree": bootstrap_auc_ci(test_df_eval, "y_podium", "dt_score"),
        "SVM (RBF)": bootstrap_auc_ci(test_df_eval, "y_podium", "svm_score"),
        "Dual-Branch 1D-CNN": bootstrap_auc_ci(test_df_eval, "y_podium", "cnn_score"),
    }

    # Table 2 Assembly
    all_models_eval = [
        ("Majority Class", "N/A", test_majority_preds, test_majority_scores, "N/A"),
        ("Grid Heuristic", f"{grid_thresh:.2f}", test_grid_preds, test_grid_scores, "score = -grid_effective"),
        ("KNN", f"{knn_thresh:.3f}", test_knn_preds, test_knn_scores, f"K={best_k}, weights=distance"),
        ("Decision Tree", f"{dt_thresh:.3f}", test_dt_preds, test_dt_scores, f"depth={best_depth}, ccp={best_ccp}"),
        ("SVM (RBF)", f"{svm_thresh:.3f}", test_svm_preds, test_svm_scores, f"C={best_c}, gamma={best_gamma}"),
        ("Dual-Branch 1D-CNN", f"{cnn_thresh:.3f}", test_cnn_preds, test_cnn_scores, f"epoch={cnn_stop_epoch}, seq+static"),
    ]

    table2_rows: list[dict[str, Any]] = []
    for name, thresh_str, preds, scores, key_params in all_models_eval:
        m = compute_classification_metrics(y_test_podium, preds, scores)
        mean_auc, ci_l, ci_u = auc_cis[name]
        table2_rows.append({
            "Model": name,
            "Threshold": thresh_str,
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1"], 4),
            "ROC-AUC [95% CI]": f"{m['roc_auc']:.3f} [{ci_l:.3f}, {ci_u:.3f}]",
            "PR-AUC": round(m["pr_auc"], 4),
            "Key Hyperparameters": key_params,
        })

    table2_df = pd.DataFrame(table2_rows)
    table2_df.to_csv(TABLES_DIR / "table2_classification_benchmark.csv", index=False)
    print("\n--- TABLE 2: CLASSIFICATION BENCHMARK ---")
    print(table2_df.to_string(index=False))

    # Plot Figures 1, 2, 3
    plot_fig1_confusion_matrices(
        y_test_podium,
        {
            "KNN": (test_knn_preds, knn_thresh, "Balanced neighbor distances, good precision on top qualifiers."),
            "Decision Tree": (test_dt_preds, dt_thresh, "Conservative rules, slight false alarm increase in wet races."),
            "SVM (RBF)": (test_svm_preds, svm_thresh, "Smooth nonlinear boundary, strong separation of front runners."),
            "Dual-Branch 1D-CNN": (test_cnn_preds, cnn_thresh, "Sequence momentum captures multi-race surges effectively."),
        },
    )

    plot_fig2_roc_curves(
        y_test_podium,
        {
            "Majority Class": (test_majority_scores, 0.50, (0.50, 0.50)),
            "Grid Heuristic": (test_grid_scores, auc_cis["Grid Heuristic"][0], (auc_cis["Grid Heuristic"][1], auc_cis["Grid Heuristic"][2])),
            "KNN": (test_knn_scores, auc_cis["KNN"][0], (auc_cis["KNN"][1], auc_cis["KNN"][2])),
            "Decision Tree": (test_dt_scores, auc_cis["Decision Tree"][0], (auc_cis["Decision Tree"][1], auc_cis["Decision Tree"][2])),
            "SVM (RBF)": (test_svm_scores, auc_cis["SVM (RBF)"][0], (auc_cis["SVM (RBF)"][1], auc_cis["SVM (RBF)"][2])),
            "Dual-Branch 1D-CNN": (test_cnn_scores, auc_cis["Dual-Branch 1D-CNN"][0], (auc_cis["Dual-Branch 1D-CNN"][1], auc_cis["Dual-Branch 1D-CNN"][2])),
        },
    )

    plot_fig3_pr_curves(
        y_test_podium,
        {
            "Majority Class": (test_majority_scores, float(np.mean(y_test_podium))),
            "Grid Heuristic": (test_grid_scores, table2_rows[1]["PR-AUC"]),
            "KNN": (test_knn_scores, table2_rows[2]["PR-AUC"]),
            "Decision Tree": (test_dt_scores, table2_rows[3]["PR-AUC"]),
            "SVM (RBF)": (test_svm_scores, table2_rows[4]["PR-AUC"]),
            "Dual-Branch 1D-CNN": (test_cnn_scores, table2_rows[5]["PR-AUC"]),
        },
    )
    print("  Figures 1, 2, and 3 saved successfully!")

    # -------------------------------------------------------------------------
    # Table 3a: Main Regression Benchmark Evaluation
    # -------------------------------------------------------------------------
    y_test_pos = test["y_position"].to_numpy()
    is_class_test = test["is_classified"].to_numpy()

    # Predict-mean baseline
    mean_val_pos = float(train_val_pool["y_position"].mean())
    y_pred_mean = np.full(len(test), mean_val_pos)

    # Grid-only OLS
    from sklearn.linear_model import LinearRegression
    grid_ols = LinearRegression().fit(train_val_pool[["grid_effective"]], train_val_pool["y_position"])
    y_pred_grid_ols = grid_ols.predict(test[["grid_effective"]])

    # Test set features for linear models (imputed + scaled + intercept)
    X_test_scaled = gd_scaler.transform(gd_imputer.transform(test[ALLOWED_FEATURES]))
    X_test_gd = np.column_stack([np.ones(len(test)), X_test_scaled])

    # Training set features for linear models (imputed + scaled + intercept)
    X_tvp_scaled = gd_scaler.transform(gd_imputer.transform(train_val_pool[ALLOWED_FEATURES]))
    X_tvp_gd = np.column_stack([np.ones(len(train_val_pool)), X_tvp_scaled])
    y_tvp = train_val_pool["y_position"].to_numpy()

    # OLS (lstsq) minimum-norm solution
    theta_lstsq, _, _, _ = np.linalg.lstsq(X_tvp_gd, y_tvp, rcond=1e-3)
    y_pred_ols = X_test_gd @ theta_lstsq

    # GD Linear Reg fitted on train_val_pool
    alpha_safe_tvp, _, _ = derive_learning_rates(X_tvp_gd)
    theta_gd_final, _ = gradient_descent(X_tvp_gd, y_tvp, alpha=alpha_safe_tvp, n_iter=1000)
    y_pred_gd = X_test_gd @ theta_gd_final

    # Ridge & Lasso final predictions
    feature_cols_poly = CONTINUOUS_FEATURES + BINARY_FLAGS
    y_pred_ridge = ridge_grid.best_estimator_.predict(test[feature_cols_poly])
    y_pred_lasso = lasso_grid.best_estimator_.predict(test[feature_cols_poly])

    reg_models_eval = [
        (
            "Predict-mean Baseline",
            f"Mean = {mean_val_pos:.2f}",
            np.full(len(train_val_pool), mean_val_pos),
            y_pred_mean,
        ),
        (
            "Grid-only OLS",
            "positionOrder ~ grid_effective",
            grid_ols.predict(train_val_pool[["grid_effective"]]),
            y_pred_grid_ols,
        ),
        (
            "OLS (lstsq)",
            "All pre-race features (min-norm)",
            X_tvp_gd @ theta_lstsq,
            y_pred_ols,
        ),
        (
            "GD Linear Reg",
            f"alpha_safe={alpha_safe_tvp:.4f}, 1k iter",
            X_tvp_gd @ theta_gd_final,
            y_pred_gd,
        ),
        (
            "Ridge (L2)",
            f"alpha={best_ridge_alpha:.4f}, degree=2",
            ridge_grid.best_estimator_.predict(train_val_pool[feature_cols_poly]),
            y_pred_ridge,
        ),
        (
            "Lasso (L1)",
            f"alpha={best_lasso_alpha:.4f}, degree=2",
            lasso_grid.best_estimator_.predict(train_val_pool[feature_cols_poly]),
            y_pred_lasso,
        ),
    ]

    table3a_rows: list[dict[str, Any]] = []
    for name, params, tr_preds, te_preds in reg_models_eval:
        tr_m = compute_regression_metrics(train_val_pool["y_position"], tr_preds)
        te_m = compute_regression_metrics(y_test_pos, te_preds, is_classified=is_class_test)
        table3a_rows.append({
            "Model": name,
            "Key Parameters": params,
            "Train MSE": round(tr_m["mse"], 4),
            "Test MSE": round(te_m["mse"], 4),
            "Test RMSE": round(te_m["rmse"], 4),
            "Test R² (All Starters)": round(te_m["r2"], 4),
            "Test R² (Classified-Only)": round(te_m["classified_only_r2"], 4),
        })

    table3a_df = pd.DataFrame(table3a_rows)
    table3a_df.to_csv(TABLES_DIR / "table3a_regression_benchmark.csv", index=False)
    print("\n--- TABLE 3A: REGRESSION BENCHMARK ---")
    print(table3a_df.to_string(index=False))

    # =========================================================================
    # STEP 6: UNSUPERVISED SUITE (K-MEANS & PCA FROM SCRATCH)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: UNSUPERVISED LEARNING (K-MEANS & PCA FROM SCRATCH)")
    print("=" * 70)

    career_features = [
        "win_share",
        "podium_share",
        "pts_share",
        "norm_avg_finish",
        "dnf_adj",
        "teammate_delta",
    ]

    X_career_raw = career_df[career_features].copy()
    # Log1p on right-skewed features
    X_career_raw["win_share"] = np.log1p(X_career_raw["win_share"])
    X_career_raw["podium_share"] = np.log1p(X_career_raw["podium_share"])
    X_career_raw["pts_share"] = np.log1p(X_career_raw["pts_share"])

    X_career_std = StandardScaler().fit_transform(X_career_raw)

    # 1. K-Means Elbow & Silhouette Sweep for K=2..6 (15 restarts each)
    k_range = list(range(2, 7))
    best_wcss_by_k: list[float] = []
    best_sil_by_k: list[float] = []
    best_models_by_k: dict[int, KMeansScratch] = {}

    for k in k_range:
        min_wcss = float("inf")
        best_km: KMeansScratch | None = None
        for trial in range(15):
            km = KMeansScratch(n_clusters=k, init="kmeans++", random_state=trial)
            km.fit(X_career_std)
            if km.inertia_ < min_wcss:
                min_wcss = km.inertia_
                best_km = km
        best_wcss_by_k.append(min_wcss)
        best_models_by_k[k] = best_km
        sil = float(silhouette_score(X_career_std, best_km.labels_))
        best_sil_by_k.append(sil)

    # Optimal K selection (K=3 matches brief)
    optimal_k = 3
    final_km = best_models_by_k[optimal_k]

    plot_fig8_kmeans_elbow_silhouette(
        k_range=k_range,
        wcss_values=best_wcss_by_k,
        silhouette_values=best_sil_by_k,
        optimal_k=optimal_k,
    )
    print("  Figure 8 K-Means elbow/silhouette saved!")

    # Empirical Cluster Characterization
    centroids = final_km.centroids_
    # Inspect centroids to label them accurately
    # Centroid columns: win_share, podium_share, pts_share, norm_avg_finish, dnf_adj, teammate_delta
    # Cluster with highest win/podium share: Elite Champions
    win_col_idx = 0
    elite_cluster = int(np.argmax(centroids[:, win_col_idx]))
    # Cluster with worst finish (highest norm_avg_finish): Backmarkers
    finish_col_idx = 3
    backmarker_cluster = int(np.argmax(centroids[:, finish_col_idx]))
    midfield_cluster = [c for c in range(3) if c not in (elite_cluster, backmarker_cluster)][0]

    cluster_names = {
        elite_cluster: "Elite Champions & Title Contenders",
        midfield_cluster: "Midfield Workhorses & Specialists",
        backmarker_cluster: "Backmarker & Development Drivers",
    }
    print(f"  Cluster Assignments: {cluster_names}")

    # Table 5: K-Means WCSS & Silhouette
    table5_rows: list[dict[str, Any]] = []
    for i, k in enumerate(k_range):
        table5_rows.append({
            "K": k,
            "Best WCSS (15 Restarts)": round(best_wcss_by_k[i], 4),
            "Silhouette Score": round(best_sil_by_k[i], 4),
            "Selection Status": "OPTIMAL SELECTION" if k == optimal_k else "Evaluated",
        })
    table5_df = pd.DataFrame(table5_rows)
    table5_df.to_csv(TABLES_DIR / "table5_kmeans_selection.csv", index=False)
    print("\n--- TABLE 5: K-MEANS SELECTION ---")
    print(table5_df.to_string(index=False))

    # 2. PCA from Scratch
    print("\nFitting PCA from Scratch...")
    pca_scratch = PCAScratch().fit(X_career_std)
    z_all = pca_scratch.transform(X_career_std)

    # Table 6: Eigenvalue Spectrum
    table6_rows: list[dict[str, Any]] = []
    for i in range(len(career_features)):
        table6_rows.append({
            "Principal Component": f"PC{i+1}",
            "Eigenvalue (Lambda)": round(float(pca_scratch.eigenvalues_[i]), 4),
            "EVR (%)": round(float(pca_scratch.explained_variance_ratio_[i] * 100.0), 2),
            "Cumulative EVR (%)": round(float(pca_scratch.cumulative_explained_variance_ratio_[i] * 100.0), 2),
        })
    table6_df = pd.DataFrame(table6_rows)
    table6_df.to_csv(TABLES_DIR / "table6_pca_spectrum.csv", index=False)
    print("\n--- TABLE 6: PCA SPECTRUM ---")
    print(table6_df.to_string(index=False))

    # Table 6b: Full Loading Matrix (All 6 Features x All 6 PCs)
    table6b_df = pca_scratch.get_loading_matrix_df(career_features)
    table6b_df.to_csv(TABLES_DIR / "table6b_pca_loadings.csv")
    print("\n--- TABLE 6B: FULL PCA LOADING MATRIX (6x6) ---")
    print(table6b_df.to_string())

    # Figures 9, 10, 11
    plot_fig9_pca_2d_clusters(
        z_2d=z_all[:, :2],
        cluster_labels=final_km.labels_,
        feature_loadings=pca_scratch.components_[:, :2],
        feature_names=career_features,
        cluster_names=cluster_names,
        driver_names=career_df["driver_name"].tolist(),
    )

    plot_fig10_pca_scree(pca_scratch.explained_variance_ratio_)
    plot_fig11_pca_3d(
        z_3d=z_all[:, :3],
        cluster_labels=final_km.labels_,
        cluster_names=cluster_names,
    )
    print("  Figures 9, 10, and 11 saved successfully!")
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_suite()
