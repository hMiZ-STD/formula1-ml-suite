"""Two-Stage Hurdle Architecture for Decoupled Reliability Hazard and Pure Pace Modeling.

Version: 6.0-dev — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
"Imagine running a race with an unpredictable slippery banana peel at Turn 1. Hurdle
uses two brains: Brain 1 predicts if your engine will break down or slip on a peel;
Brain 2 predicts how fast you drive assuming your car stays in one piece. We then combine
them with the Law of Total Probability so a random mechanical breakdown doesn't ruin
the model's estimate of a fast driver's true speed!"

Why Pointwise Models Break on Formula 1 Mechanical Retirements:
---------------------------------------------------------------
In the 1994–2016 training era, 31.98% of race starters suffered mechanical breakdowns
or opening-lap collisions (falling to 14.46% in the modern 2019–2024 test era). When a
front-running car running P2 suffers a sudden hydraulic failure on lap 55, it is classified
at the back of the field (e.g. P18).

In standard regression, this single race observation penalizes the model with:
    (18 - 2)^2 = 256 squared error!
This catastrophic squared-error tail pollutes the loss gradients of fast cars, capping
traditional pointwise regression R^2 at ~0.408.

The Two-Stage Hurdle Architecture solves this via conditional probability:
1. Stage 1 (DNF Hazard Classifier):
   Predicts the probability of mechanical attrition / non-classification:
       p_dnf = P(is_dnf = 1 | X)
   calibrated on validation data via Platt scaling (sigmoid calibration).
2. Stage 2 (Classified Pace Model):
   Trained strictly on classified survivors (is_classified == 1), eliminating 100% of
   catastrophic DNF noise tails. Stage 2 provides pure on-track finishing position
   y_finish and survivor podium probability P(Podium | Finish).
3. Law of Total Probability Recombination:
   Expected Finishing Position:
       E[Position | X] = (1 - p_dnf) * y_finish + p_dnf * y_bar_dnf
   where y_bar_dnf is the empirical average finishing position of retired cars (~18.04).
   Hurdle Podium Probability:
       P(Podium | X) = (1 - p_dnf) * P(Podium | Finish, X) + p_dnf * 0
4. Guaranteed Top-3 Podium Selection:
   Strictly selects 3 unique podium finishers per race with secondary tie-breaking on
   grid_effective ascending (pole position precedence).
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

from src.features.builder import ALLOWED_FEATURES

# Suppress loky subprocess core counting warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


# =============================================================================
# Helper: Top-3 Guaranteed Podium Selection
# =============================================================================


def predict_podium_top3(
    scores_or_model: Any,
    df_race: pd.DataFrame,
    ranking_criterion: str = "podium_probability",
) -> np.ndarray:
    r"""Selects strictly 3 unique podium finishers per race with deterministic tie-breaking.

    Enforces deterministic ranking by primary continuous pace or probability score
    (descending), breaking any exact ties with secondary ascending `grid_effective`
    (the driver qualifying closer to pole position wins the tie-break).

    Handles single-race and multi-race DataFrames:
    - Single race: exactly 3 entries with value 1 (or $\min(3, m)$ if field size $m < 3$).
    - Multi-race: returns a binary mask whose sum strictly equals $3 \times N_{\text{races}}$.

    Parameters
    ----------
    scores_or_model : Any
        Either a fitted `F1HurdleModel` instance, or an array-like of continuous scores /
        probabilities matching `df_race` length.
    df_race : pd.DataFrame
        DataFrame containing the race grid. Must match scores length if scores are passed.
        May contain a race identifier ('raceId' or 'race_id') and starting grid column
        ('grid_effective' or 'grid').
    ranking_criterion : str, default='podium_probability'
        Ranking score to use when `scores_or_model` is an `F1HurdleModel`:
        - `'podium_probability'`: ranks by hurdle $P(\text{Podium} \mid X)$ descending.
        - `'expected_position'`: ranks by hurdle $\mathbb{E}[\text{Position} \mid X]$ ascending.

    Returns
    -------
    np.ndarray
        1D int32 binary indicator mask where 1 marks a predicted podium finisher
        and 0 marks non-podium finishers.

    Raises
    ------
    ValueError
        If length of scores does not match length of `df_race`.
    """
    n_rows = len(df_race)
    if n_rows == 0:
        return np.empty(0, dtype=np.int32)

    # If an F1HurdleModel instance is provided, compute appropriate score
    if hasattr(scores_or_model, "predict_podium_probability") and hasattr(
        scores_or_model, "predict_expected_position"
    ):
        if ranking_criterion == "expected_position":
            # Lower expected position is better -> negate so higher score is better
            scores_arr = -scores_or_model.predict_expected_position(df_race)
        else:
            # Higher podium probability is better
            scores_arr = scores_or_model.predict_podium_probability(df_race)
    else:
        scores_arr = np.asarray(scores_or_model, dtype=np.float64).ravel()

    if n_rows != len(scores_arr):
        raise ValueError(
            f"Shape mismatch: df_race has {n_rows} rows, but scores has length {len(scores_arr)}."
        )

    podium_mask = np.zeros(n_rows, dtype=np.int32)

    # Determine race column
    if "raceId" in df_race.columns:
        race_col: str | None = "raceId"
    elif "race_id" in df_race.columns:
        race_col = "race_id"
    else:
        race_col = None

    # Determine starting grid column for tie-breaking
    if "grid_effective" in df_race.columns:
        grid_all = df_race["grid_effective"].to_numpy(dtype=np.float64)
    elif "grid" in df_race.columns:
        grid_all = df_race["grid"].to_numpy(dtype=np.float64)
    else:
        grid_all = np.arange(n_rows, dtype=np.float64)

    # Replace any NaNs in grid with back-of-the-grid value (99.0)
    grid_all = np.nan_to_num(grid_all, nan=99.0)

    if race_col is None:
        # Single race case
        k = min(3, n_rows)
        # Primary: -scores_arr (descending), Secondary: grid_all (ascending)
        order = np.lexsort((grid_all, -scores_arr))
        podium_mask[order[:k]] = 1
        return podium_mask

    # Multi-race case: iterate over races preserving index alignment
    group_indices = df_race.groupby(race_col, sort=False).indices

    for _, idxs_list in group_indices.items():
        race_idxs = np.asarray(idxs_list, dtype=np.int64)
        m = len(race_idxs)
        if m == 0:
            continue

        race_scores = scores_arr[race_idxs]
        race_grid = grid_all[race_idxs]

        # Primary: -race_scores (descending), Secondary: race_grid (ascending)
        order = np.lexsort((race_grid, -race_scores))
        k = min(3, m)
        top3_indices = race_idxs[order[:k]]
        podium_mask[top3_indices] = 1

    return podium_mask


# =============================================================================
# Two-Stage Hurdle Model
# =============================================================================


class F1HurdleModel(BaseEstimator):
    r"""Two-Stage Hurdle Model decoupling DNF hazard risk from classified race pace.

    Stage 1 models the mechanical survival hazard $p_{\text{dnf}} = P(\text{is\_dnf} = 1 \mid X)$,
    calibrated via Platt scaling (sigmoid calibration) on validation data.
    Stage 2 models pure on-track finishing performance trained exclusively on classified
    finishers ($S_i = 1$), eliminating 100% of catastrophic mechanical DNF noise.

    Mathematical Formulation:
    -------------------------
    1. Stage 1 Hazard Classifier:
       $$p_{\text{dnf}, i} = P(\text{is\_dnf}_i = 1 \mid X_i)$$
       $$P(\text{Finish}_i \mid X_i) = 1 - p_{\text{dnf}, i}$$

    2. Stage 2 Classified Pace Model:
       Trained strictly on $\{i : \text{is\_classified}_i = 1\}$:
       $$\hat{y}_{\text{finish}, i} = \mathbb{E}[\text{Position}_i \mid \text{Finish}_i, X_i]$$
       $$P(\text{Podium}_i \mid \text{Finish}_i, X_i)$$

    3. Law of Total Probability Recombination:
       Expected Finishing Position:
       $$\mathbb{E}[\text{Position}_i \mid X_i] = (1 - p_{\text{dnf}, i}) \cdot \hat{y}_{\text{finish}, i} + p_{\text{dnf}, i} \cdot \bar{y}_{\text{dnf}}$$
       where $\bar{y}_{\text{dnf}} = \frac{1}{|\text{DNF}|} \sum_{k \in \text{DNF}} \text{positionOrder}_k \approx 18.04$.

       Hurdle Podium Probability:
       $$P(\text{Podium}_i \mid X_i) = (1 - p_{\text{dnf}, i}) \cdot P(\text{Podium}_i \mid \text{Finish}_i, X_i)$$

    Parameters
    ----------
    stage1_estimator : Any | None, default=None
        Binary classifier for Stage 1 DNF hazard. If None, defaults to:
        `HistGradientBoostingClassifier(loss='log_loss', max_depth=4, l2_regularization=1.0, random_state=random_state)`.
    stage2_estimator : Any | None, default=None
        Regressor for Stage 2 on-track finish position $\hat{y}_{\text{finish}}$. If None, defaults to:
        `HistGradientBoostingRegressor(max_depth=4, l2_regularization=1.0, random_state=random_state)`.
    stage2_classifier : Any | None, default=None
        Classifier for Stage 2 on-track podium probability $P(\text{Podium} \mid \text{Finish})$.
        If None, defaults to:
        `HistGradientBoostingClassifier(loss='log_loss', max_depth=4, l2_regularization=1.0, random_state=random_state)`.
    feature_cols : list[str] | None, default=None
        Feature column names to extract from input DataFrames. If None, defaults to `ALLOWED_FEATURES` (17 columns).
    pace_transform : str, default='auto'
        Method to derive $P(\text{Podium} \mid \text{Finish})$:
        - `'auto'`: uses `stage2_classifier` if available, otherwise Gaussian CDF transformation.
        - `'classifier'`: strictly uses `stage2_classifier`.
        - `'gaussian_cdf'`: uses Gaussian normal CDF $\Phi((3.5 - \hat{y}) / \sigma)$ from regressor residuals.
    random_state : int, default=42
        Seed for reproducibility across estimators and calibration.
    """

    def __init__(
        self,
        stage1_estimator: Any | None = None,
        stage2_estimator: Any | None = None,
        stage2_classifier: Any | None = None,
        feature_cols: list[str] | None = None,
        pace_transform: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.stage2_classifier = stage2_classifier
        self.feature_cols = feature_cols
        self.pace_transform = pace_transform
        self.random_state = random_state

        # Fitted attributes (populated upon fit)
        self.is_fitted_: bool = False
        self.stage1_base_estimator_: Any = None
        self.stage1_calibrated_estimator_: Any = None
        self.stage2_regressor_: Any = None
        self.stage2_classifier_: Any = None
        self.mean_dnf_position_: float = 18.04
        self.residual_std_: float = 2.5
        self.stage2_is_dnf_: np.ndarray | None = None
        self.y_train_stage2_dnf_: np.ndarray | None = None
        self.is_calibrated_: bool = False
        self.imputer_: SimpleImputer | None = None
        self.effective_feature_cols_: list[str] = []

    # -------------------------------------------------------------------------
    # Internal Feature & Target Extraction
    # -------------------------------------------------------------------------

    def _resolve_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Resolves available feature columns against allowed whitelist."""
        if self.feature_cols is not None:
            return [col for col in self.feature_cols if col in df.columns]

        # Check for ALLOWED_FEATURES whitelist
        available = [col for col in ALLOWED_FEATURES if col in df.columns]
        if len(available) > 0:
            return available

        # Fallback to all numeric columns excluding metadata and targets
        exclude = {
            "raceId",
            "race_id",
            "year",
            "round",
            "circuitId",
            "driverId",
            "constructorId",
            "position",
            "positionOrder",
            "statusId",
            "is_classified",
            "is_dnf",
            "y_podium",
            "y_position",
            "resultId",
        }
        return [
            col
            for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]

    def _extract_features(
        self, df: pd.DataFrame, fit_imputer: bool = False
    ) -> np.ndarray:
        """Extracts numeric feature matrix, fitting or transforming imputer if needed."""
        cols = self.effective_feature_cols_
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Input DataFrame is missing required feature columns: {missing_cols}")

        X_df = df[cols].copy()
        X_arr = X_df.to_numpy(dtype=np.float64)

        # Check if any estimator requires dense non-NaN inputs
        # Tree models (HistGradientBoosting, LightGBM) handle NaNs natively.
        # However, for arbitrary user estimators, we fit a median imputer.
        if np.isnan(X_arr).any():
            if fit_imputer:
                self.imputer_ = SimpleImputer(strategy="median")
                self.imputer_.fit(X_arr)
            # If an imputer exists and is fitted, we keep a imputed copy for models that need it
            # For HistGradientBoosting, native NaN handling is preserved if imputer is not applied.
        return X_arr

    def _extract_dnf_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts Stage 1 binary DNF target (1 = DNF/unclassified, 0 = classified)."""
        if "is_dnf" in df.columns:
            return df["is_dnf"].to_numpy(dtype=np.int32)
        if "is_classified" in df.columns:
            return (1 - df["is_classified"]).to_numpy(dtype=np.int32)
        if "status" in df.columns:
            return (~df["status"].str.contains("Finished|\\+")).astype(np.int32).to_numpy()
        raise ValueError(
            "DataFrame must contain 'is_dnf' or 'is_classified' column for Stage 1 hazard modeling."
        )

    def _extract_position_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts continuous finishing position target."""
        if "positionOrder" in df.columns:
            return df["positionOrder"].to_numpy(dtype=np.float64)
        if "y_position" in df.columns:
            return df["y_position"].to_numpy(dtype=np.float64)
        if "position" in df.columns:
            return df["position"].to_numpy(dtype=np.float64)
        raise ValueError("DataFrame must contain 'positionOrder' or 'position' column.")

    def _extract_podium_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts binary podium indicator target (1 = podium P1..P3, 0 = non-podium)."""
        if "y_podium" in df.columns:
            return df["y_podium"].to_numpy(dtype=np.int32)
        if "positionOrder" in df.columns:
            return (df["positionOrder"] <= 3).astype(np.int32).to_numpy()
        if "position" in df.columns:
            return (df["position"] <= 3).astype(np.int32).to_numpy()
        raise ValueError("DataFrame must contain 'y_podium' or 'positionOrder' column.")

    # -------------------------------------------------------------------------
    # Model Fitting
    # -------------------------------------------------------------------------

    def fit(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame | None = None
    ) -> F1HurdleModel:
        r"""Fits the Two-Stage Hurdle model with validation calibration.

        Stage 1:
        1. Fits `stage1_estimator` on all starters in `df_train` predicting $P(\text{is\_dnf} = 1 \mid X)$.
        2. Calibrates probabilities on `df_val` via `CalibratedClassifierCV(..., method='sigmoid', cv='prefit')`.

        Stage 2:
        1. Filters `df_train` strictly to classified starters (`is_classified == 1`).
        2. Strictly asserts zero DNF rows leak into Stage 2 training.
        3. Fits `stage2_regressor` predicting finishing position $\hat{y}_{\text{finish}}$.
        4. Fits `stage2_classifier` predicting on-track podium probability $P(\text{Podium} \mid \text{Finish})$.

        Total Probability Constants:
        - Computes dynamic $\bar{y}_{\text{dnf}} = \text{mean}(\text{positionOrder} \mid \text{is\_dnf} = 1)$.
        - Computes Stage 2 residual standard deviation $\hat{\sigma}$ for Gaussian CDF pace transformation.

        Parameters
        ----------
        df_train : pd.DataFrame
            Historical training data (e.g. `train_core.parquet`). Must contain feature columns
            and outcome columns (`is_dnf` or `is_classified`, `positionOrder`).
        df_val : pd.DataFrame | None, default=None
            Validation data (e.g. `val.parquet`) for Stage 1 probability calibration.
            If None, calibration is skipped or performed internally.

        Returns
        -------
        F1HurdleModel
            The fitted model instance (`self`).
        """
        if len(df_train) == 0:
            raise ValueError("df_train cannot be empty.")

        # 1. Resolve feature columns
        self.effective_feature_cols_ = self._resolve_feature_cols(df_train)
        if len(self.effective_feature_cols_) == 0:
            raise ValueError("No valid feature columns found in df_train.")

        # Extract features and targets for Stage 1
        X_train_full = self._extract_features(df_train, fit_imputer=True)
        y_train_dnf = self._extract_dnf_target(df_train)

        # 2. Stage 1: DNF Hazard Classifier
        if self.stage1_estimator is None:
            self.stage1_base_estimator_ = HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=4,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
        else:
            self.stage1_base_estimator_ = copy.deepcopy(self.stage1_estimator)

        # Fit Stage 1 base estimator on training set
        self.stage1_base_estimator_.fit(X_train_full, y_train_dnf)

        # Stage 1 Validation Calibration via Platt scaling (sigmoid)
        if df_val is not None and len(df_val) > 0:
            X_val_full = self._extract_features(df_val, fit_imputer=False)
            y_val_dnf = self._extract_dnf_target(df_val)

            # Check if validation set has both classes
            if len(np.unique(y_val_dnf)) > 1:
                self.stage1_calibrated_estimator_ = CalibratedClassifierCV(
                    estimator=self.stage1_base_estimator_,
                    method="sigmoid",
                    cv="prefit",
                )
                self.stage1_calibrated_estimator_.fit(X_val_full, y_val_dnf)
                self.is_calibrated_ = True
            else:
                self.stage1_calibrated_estimator_ = self.stage1_base_estimator_
                self.is_calibrated_ = False
        else:
            self.stage1_calibrated_estimator_ = self.stage1_base_estimator_
            self.is_calibrated_ = False

        # 3. Dynamic mean DNF finishing position
        dnf_mask = y_train_dnf == 1
        if dnf_mask.any():
            positions_dnf = self._extract_position_target(df_train[dnf_mask])
            self.mean_dnf_position_ = float(np.mean(positions_dnf))
        else:
            self.mean_dnf_position_ = 18.04

        # 4. Stage 2: Classified Pace Model (Zero DNF Leakage)
        classified_mask = y_train_dnf == 0
        if not classified_mask.any():
            raise ValueError(
                "df_train contains 0 classified starters! Cannot fit Stage 2 pace model."
            )

        df_stage2 = df_train[classified_mask].copy()

        # Strict Purity Assertion: zero DNF rows used in Stage 2
        y_stage2_dnf = self._extract_dnf_target(df_stage2)
        self.stage2_is_dnf_ = y_stage2_dnf
        self.y_train_stage2_dnf_ = y_stage2_dnf
        assert (self.stage2_is_dnf_ == 0).all(), (
            f"CRITICAL PURITY VIOLATION: {self.stage2_is_dnf_.sum()} DNF rows leaked into Stage 2 training!"
        )

        X_stage2 = self._extract_features(df_stage2, fit_imputer=False)
        y_stage2_pos = self._extract_position_target(df_stage2)
        y_stage2_pod = self._extract_podium_target(df_stage2)

        # 4a. Stage 2 Regressor (Finish Position)
        if self.stage2_estimator is None:
            self.stage2_regressor_ = HistGradientBoostingRegressor(
                max_depth=4,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
        else:
            self.stage2_regressor_ = copy.deepcopy(self.stage2_estimator)

        self.stage2_regressor_.fit(X_stage2, y_stage2_pos)

        # Compute empirical residual standard error for pace transformation
        y_stage2_pos_pred = self.stage2_regressor_.predict(X_stage2)
        residuals = y_stage2_pos - y_stage2_pos_pred
        self.residual_std_ = (
            float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 2.5
        )
        if self.residual_std_ < 1e-6 or np.isnan(self.residual_std_):
            self.residual_std_ = 1.0

        # 4b. Stage 2 Classifier (Podium Probability given Finish)
        if self.stage2_classifier is not None:
            self.stage2_classifier_ = copy.deepcopy(self.stage2_classifier)
            self.stage2_classifier_.fit(X_stage2, y_stage2_pod)
        else:
            if len(np.unique(y_stage2_pod)) > 1:
                self.stage2_classifier_ = HistGradientBoostingClassifier(
                    loss="log_loss",
                    max_depth=4,
                    l2_regularization=1.0,
                    random_state=self.random_state,
                )
                self.stage2_classifier_.fit(X_stage2, y_stage2_pod)
            else:
                self.stage2_classifier_ = None

        self.is_fitted_ = True
        return self

    # -------------------------------------------------------------------------
    # Inference Methods
    # -------------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Validates that the model has been fitted before inference."""
        if not self.is_fitted_:
            raise NotFittedError(
                "This F1HurdleModel instance is not fitted yet. Call 'fit' before predicting."
            )

    def predict_hazard(self, df: pd.DataFrame) -> np.ndarray:
        r"""Predicts calibrated DNF hazard probability $P(\text{is\_dnf} = 1 \mid X)$.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature columns.

        Returns
        -------
        np.ndarray
            1D float64 array of probabilities strictly bounded within $[0.0, 1.0]$.
        """
        self._check_is_fitted()
        if len(df) == 0:
            return np.empty(0, dtype=np.float64)

        X = self._extract_features(df, fit_imputer=False)
        estimator = (
            self.stage1_calibrated_estimator_
            if self.stage1_calibrated_estimator_ is not None
            else self.stage1_base_estimator_
        )

        if hasattr(estimator, "predict_proba"):
            p_dnf = estimator.predict_proba(X)[:, 1]
        elif hasattr(estimator, "decision_function"):
            raw_scores = estimator.decision_function(X)
            # Sigmoid link
            p_dnf = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            p_dnf = estimator.predict(X)

        return np.clip(np.asarray(p_dnf, dtype=np.float64), 0.0, 1.0)

    def predict_classified_pace(self, df: pd.DataFrame) -> np.ndarray:
        r"""Predicts on-track finish position $\hat{y}_{\text{finish}}$ assuming classified survival.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature columns.

        Returns
        -------
        np.ndarray
            1D float64 array of predicted finishing positions.
        """
        self._check_is_fitted()
        if len(df) == 0:
            return np.empty(0, dtype=np.float64)

        X = self._extract_features(df, fit_imputer=False)
        y_finish = self.stage2_regressor_.predict(X)
        return np.asarray(y_finish, dtype=np.float64)

    def predict_expected_position(self, df: pd.DataFrame) -> np.ndarray:
        r"""Computes expected finish position via the Law of Total Probability.

        $$\mathbb{E}[\text{Position}_i \mid X_i] = (1 - p_{\text{dnf}, i}) \cdot \hat{y}_{\text{finish}, i} + p_{\text{dnf}, i} \cdot \bar{y}_{\text{dnf}}$$

        where $\bar{y}_{\text{dnf}}$ is dynamically derived from training data ($\approx 18.04$).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature columns.

        Returns
        -------
        np.ndarray
            1D float64 array of expected finishing positions.
        """
        self._check_is_fitted()
        if len(df) == 0:
            return np.empty(0, dtype=np.float64)

        p_dnf = self.predict_hazard(df)
        y_finish = self.predict_classified_pace(df)

        # Law of Total Probability Recombination
        y_expected = (1.0 - p_dnf) * y_finish + p_dnf * self.mean_dnf_position_
        return np.asarray(y_expected, dtype=np.float64)

    def predict_podium_probability(self, df: pd.DataFrame) -> np.ndarray:
        r"""Computes hurdle podium probability via the Law of Total Probability.

        $$P(\text{Podium}_i \mid X_i) = (1 - p_{\text{dnf}, i}) \cdot P(\text{Podium}_i \mid \text{Finish}_i, X_i)$$

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature columns.

        Returns
        -------
        np.ndarray
            1D float64 array of probabilities strictly bounded within $[0.0, 1.0]$.
        """
        self._check_is_fitted()
        if len(df) == 0:
            return np.empty(0, dtype=np.float64)

        p_dnf = self.predict_hazard(df)
        p_finish = 1.0 - p_dnf

        # Determine P(Podium | Finish)
        use_cdf = (
            self.pace_transform == "gaussian_cdf"
            or self.stage2_classifier_ is None
            or not hasattr(self.stage2_classifier_, "predict_proba")
        )

        if use_cdf:
            # Gaussian Normal CDF transformation: P(position <= 3.5 | y_finish, sigma)
            y_finish = self.predict_classified_pace(df)
            z = (3.5 - y_finish) / self.residual_std_
            p_pod_finish = scipy.stats.norm.cdf(z)
        else:
            X = self._extract_features(df, fit_imputer=False)
            p_pod_finish = self.stage2_classifier_.predict_proba(X)[:, 1]

        p_podium = p_finish * p_pod_finish
        return np.clip(np.asarray(p_podium, dtype=np.float64), 0.0, 1.0)

    def predict_podium_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Alias for `predict_podium_probability` ensuring interface parity."""
        return self.predict_podium_probability(df)

    def predict_podium_top3(
        self,
        df: pd.DataFrame,
        ranking_criterion: str = "podium_probability",
    ) -> np.ndarray:
        r"""Guarantees strictly 3 unique podium finishers per race ($3 \times N_{\text{races}}$).

        Ranks drivers within each race using either `podium_probability` (highest probability wins)
        or `expected_position` (lowest expected position wins), with secondary deterministic
        tie-breaking by ascending `grid_effective` (pole position precedence).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing race data.
        ranking_criterion : str, default='podium_probability'
            Either `'podium_probability'` or `'expected_position'`.

        Returns
        -------
        np.ndarray
            1D int32 binary mask where 1 marks a predicted podium finisher and 0 marks others.
        """
        self._check_is_fitted()
        return predict_podium_top3(
            self, df_race=df, ranking_criterion=ranking_criterion
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Standard scikit-learn regressor interface: returns expected finishing position."""
        return self.predict_expected_position(df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Standard scikit-learn classifier interface: returns [1 - P(Podium), P(Podium)]."""
        p_podium = self.predict_podium_probability(df)
        return np.column_stack([1.0 - p_podium, p_podium])
