"""Learning-to-Rank (LTR) Architecture for Formula 1 Race Outcome Modeling.

Version: 6.0-dev — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Instead of asking "Is this runner good or bad?" in isolation, Learning-to-Rank holds
a mini-tournament between all runners in the race, giving 100x more coaching attention
to who gets the Gold, Silver, and Bronze medals rather than who finished 15th vs 16th!

In traditional machine learning (pointwise regression or classification), every driver
is evaluated in an imaginary vacuum, as if 20 drivers were doing separate time trials
on different days. But in Formula 1, cars share the track simultaneously: exactly one
car wins, exactly three step onto the podium, and driver positions are zero-sum.

Learning-to-Rank (LTR) directly models this intra-race competition:
1. LambdaMART: Uses gradient-boosted decision trees where tree splits are guided by
   pairwise swaps weighted by the delta in NDCG (Normalized Discounted Cumulative Gain).
   Swapping P1 and P2 carries massive gradient weight, while swapping P17 and P18 has
   almost zero effect.
2. Plackett-Luce ListMLE: A deep neural network optimizing the likelihood of the entire
   finishing permutation via sequential competitive elimination: car 1 is chosen first
   from all 20 cars, car 2 is chosen from the remaining 19, car 3 from 18, and so on.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from src.features.builder import ALLOWED_FEATURES


# -----------------------------------------------------------------------------
# Canonical Relevance Maps
# -----------------------------------------------------------------------------

# 2010-2024 FIA Championship Scale (classified finishers only)
F1_POINTS_MAP: dict[int, int] = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

# Podium Tier Scale (classified finishers only)
PODIUM_TIER_MAP: dict[int, int] = {
    1: 3,
    2: 2,
    3: 1,
}


# =============================================================================
# 1. Race-Aligned Dataset Assembly
# =============================================================================


class F1RankerDataset:
    r"""Assembles race-contiguous feature matrices and group query sizes for LTR.

    Extracts race features from `ALLOWED_FEATURES`, computes race group lengths
    $q_r = \text{size}(\text{race}_r)$, and maps finishing positions into discrete
    relevance targets ($y_i \in \mathbb{N}_0$).

    Supported Relevance Target Formulations:
    ----------------------------------------
    1. `'f1_points'`:
       Championship points: P1=25, P2=18, P3=15, P4=12, P5=10, P6=8, P7=6, P8=4, P9=2, P10=1,
       all others and unclassified (DNF) = 0.
    2. `'linear_rank'`:
       Inverted finishing rank:
       $$y_i = \max(0, m_r - \text{positionOrder}_i + 1)$$
       where $m_r$ is the field size of the race.
    3. `'podium_tier'`:
       Podium tiers: P1=3, P2=2, P3=1, all others and unclassified (DNF) = 0.

    Parameters
    ----------
    target_mapping : str, default='f1_points'
        Relevance formulation strategy ('f1_points', 'linear_rank', or 'podium_tier').
    feature_cols : list[str] | None, default=None
        List of feature column names. If None, defaults to `ALLOWED_FEATURES` (17 columns).
    impute_missing : bool, default=False
        If True, fit/transform a median imputer on features to cleanly fill NaNs
        (`grid_vs_quali_delta`, `q_gap_pct`).
    """

    def __init__(
        self,
        target_mapping: str = "f1_points",
        feature_cols: list[str] | None = None,
        impute_missing: bool = False,
    ) -> None:
        valid_mappings = {"f1_points", "linear_rank", "podium_tier"}
        if target_mapping not in valid_mappings:
            raise ValueError(
                f"Invalid target_mapping '{target_mapping}'. "
                f"Supported mappings are {sorted(valid_mappings)}."
            )
        self.target_mapping = target_mapping
        self.feature_cols = (
            list(feature_cols) if feature_cols is not None else list(ALLOWED_FEATURES)
        )
        self.impute_missing = impute_missing
        self.imputer_: SimpleImputer | None = (
            SimpleImputer(strategy="median") if impute_missing else None
        )
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> F1RankerDataset:
        """Fit feature preprocessors (e.g. median imputer) on the training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame containing `feature_cols`.

        Returns
        -------
        F1RankerDataset
            Fitted instance.
        """
        if self.imputer_ is not None:
            self.imputer_.fit(df[self.feature_cols].to_numpy(dtype=np.float64))
        self.is_fitted_ = True
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Transform a race DataFrame into contiguous feature matrix $X$, relevance $y$, and group lengths.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing race data. Must contain `feature_cols` and a race
            identifier (`raceId` or `race_id`). If `positionOrder` is present, it is
            mapped to relevance target $y$; otherwise $y$ defaults to zeros.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - $X$: 2D float64 array of shape `(N, n_features)`.
            - $y$: 1D int32 array of shape `(N,)` containing relevance values.
            - `group`: 1D int32 array of shape `(n_races,)` containing query group lengths.
        """
        if df.empty:
            return (
                np.empty((0, len(self.feature_cols)), dtype=np.float64),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        # Identify race column
        if "raceId" in df.columns:
            race_col = "raceId"
        elif "race_id" in df.columns:
            race_col = "race_id"
        else:
            raise KeyError("DataFrame must contain 'raceId' or 'race_id' column.")

        # Ensure race contiguity: if races are interleaved, sort stably by raceId
        is_contiguous = bool((df[race_col] != df[race_col].shift(1)).sum() == df[race_col].nunique())
        if not is_contiguous:
            sort_cols = [race_col, "positionOrder"] if "positionOrder" in df.columns else [race_col]
            df_work = df.sort_values(by=sort_cols, kind="mergesort")
        else:
            df_work = df

        # 1. Feature Matrix X
        X = df_work[self.feature_cols].to_numpy(dtype=np.float64)
        if self.impute_missing:
            if self.imputer_ is not None:
                if not self.is_fitted_:
                    X = self.imputer_.fit_transform(X)
                    self.is_fitted_ = True
                else:
                    X = self.imputer_.transform(X)
            else:
                col_medians = np.nanmedian(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_medians, inds[1])

        # 2. Group Query Lengths
        group = df_work.groupby(race_col, sort=False).size().to_numpy(dtype=np.int32)

        # 3. Relevance Targets y
        n_rows = len(df_work)
        if "positionOrder" not in df_work.columns:
            y = np.zeros(n_rows, dtype=np.int32)
            return X, y, group

        has_classified = "is_classified" in df_work.columns
        y_list: list[int] = []

        for _, df_race in df_work.groupby(race_col, sort=False):
            m = len(df_race)
            positions = df_race["positionOrder"].to_numpy(dtype=np.int64)
            classified = (
                df_race["is_classified"].to_numpy(dtype=np.int32) == 1
                if has_classified
                else np.ones(m, dtype=bool)
            )

            if self.target_mapping == "linear_rank":
                # y_i = max(0, m - positionOrder_i + 1)
                for p in positions:
                    val = max(0, int(m - p + 1))
                    y_list.append(val)
            elif self.target_mapping == "f1_points":
                for p, is_c in zip(positions, classified):
                    val = F1_POINTS_MAP.get(int(p), 0) if bool(is_c) else 0
                    y_list.append(val)
            elif self.target_mapping == "podium_tier":
                for p, is_c in zip(positions, classified):
                    val = PODIUM_TIER_MAP.get(int(p), 0) if bool(is_c) else 0
                    y_list.append(val)

        y = np.asarray(y_list, dtype=np.int32)
        return X, y, group

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit feature preprocessors and transform the DataFrame in one step.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to fit and transform.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (X, y, group).
        """
        return self.fit(df).transform(df)


# =============================================================================
# 2. LambdaMART GBDT Ranker
# =============================================================================


class LambdaMARTRanker:
    r"""LambdaMART listwise ranking model wrapping `lightgbm.LGBMRanker`.

    Mathematical Formulation:
    -------------------------
    LambdaMART optimizes Normalized Discounted Cumulative Gain (NDCG) directly
    by defining virtual gradients $\lambda_{ij}$ for each pair of items $(i, j)$
    in the same race where driver $i$ finished ahead of driver $j$ ($y_i > y_j$):

    $$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta \text{NDCG}_{ij}|$$

    $$\Delta \text{NDCG}_{ij} = \frac{1}{\text{IDCG}} \left( 2^{y_i} - 2^{y_j} \right) \left( \frac{1}{\log_2(1 + \text{rank}_i)} - \frac{1}{\log_2(1 + \text{rank}_j)} \right)$$

    The gradient naturally prioritizes podium positions (P1 vs P2 vs P3) where
    $|\Delta \text{NDCG}_{ij}|$ is largest, while giving negligible weight to lower
    midfield swaps.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting trees.
    learning_rate : float, default=0.05
        Boosting shrinkage rate.
    num_leaves : int, default=31
        Maximum tree leaves per base learner.
    min_child_samples : int, default=15
        Minimum number of samples required in a leaf node.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs : Any
        Additional keyword arguments forwarded to `lightgbm.LGBMRanker`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 15,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.kwargs = kwargs

        # Initialize underlying LightGBM ranker
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                eval_at=[1, 3, 10],
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
                **self.kwargs,
            )
        self.best_iteration_: int | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        group_val: np.ndarray | None = None,
        early_stopping_rounds: int = 15,
    ) -> LambdaMARTRanker:
        r"""Fit the LambdaMART ranker with validation-guided early stopping.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix of shape `(N_train, n_features)`.
        y_train : np.ndarray
            Training integer relevance vector of shape `(N_train,)`.
        group_train : np.ndarray
            Race query group sizes for training, summing to `N_train`.
        X_val : np.ndarray | None, default=None
            Validation feature matrix of shape `(N_val, n_features)`.
        y_val : np.ndarray | None, default=None
            Validation relevance vector of shape `(N_val,)`.
        group_val : np.ndarray | None, default=None
            Validation race query group sizes, summing to `N_val`.
        early_stopping_rounds : int, default=15
            Rounds of validation NDCG plateau allowed before early termination.

        Returns
        -------
        LambdaMARTRanker
            Fitted instance.
        """
        callbacks: list[Any] = []
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None
        eval_group: list[np.ndarray] | None = None

        if X_val is not None and y_val is not None and group_val is not None:
            eval_set = [(X_val, y_val)]
            eval_group = [group_val]
            if early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(
                X_train,
                y_train,
                group=group_train,
                eval_set=eval_set,
                eval_group=eval_group,
                callbacks=callbacks if len(callbacks) > 0 else None,
            )

        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous latent pace scores for candidates.

        Higher predicted scores correspond to higher predicted finishing rank
        (faster race pace / lower finishing positionOrder).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape `(N, n_features)`.

        Returns
        -------
        np.ndarray
            1D float64 array of latent pace scores of shape `(N,)`.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.model_.predict(X)
        return np.asarray(preds, dtype=np.float64).ravel()

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the fitted booster."""
        return self.model_.feature_importances_


# =============================================================================
# 3. Plackett-Luce ListMLE Neural Ranker
# =============================================================================


class _PlackettLuceMLP(nn.Module):
    """3-layer PyTorch MLP for continuous latent pace prediction.

    Architecture:
    Linear(input_dim, hidden_dim) -> ReLU() -> Dropout(0.2) ->
    Linear(hidden_dim, 32) -> ReLU() -> Linear(32, 1)
    """

    def __init__(self, input_dim: int = 17, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting 1D continuous scores."""
        return self.net(x).squeeze(-1)


def compute_list_mle_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    r"""Compute the Plackett-Luce ListMLE negative log-likelihood permutation loss.

    Mathematical Formulation:
    -------------------------
    $$\mathcal{L}_{\text{ListMLE}}(\boldsymbol{s}, \pi) = - \sum_{i=1}^{m-1} \left( s_{\pi_i} - \text{logsumexp}(s_{\pi_{i:m}}) \right)$$

    where:
    - $\pi$ is the permutation of candidates ordered by descending relevance / finish.
    - $s_{\pi_i}$ is the predicted latent pace of candidate placed at step $i$.
    - $\text{logsumexp}(s_{\pi_{i:m}})$ represents the normalization across the
      remaining candidate pool at step $i$.
    - Numerically stabilized via upper-triangular masking and `torch.logsumexp`.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted continuous scores of shape `(m,)`.
    relevance : torch.Tensor
        True relevance or inverted rank of shape `(m,)` (higher is better).

    Returns
    -------
    torch.Tensor
        Scalar loss tensor with attached gradient history.
    """
    m = len(scores)
    if m <= 1:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype, requires_grad=True)

    # Permutation: sort items by true relevance descending (higher relevance placed earlier)
    order = torch.argsort(relevance, descending=True, stable=True)
    s_perm = scores[order]

    # Upper triangular mask: row i attends to columns j >= i
    # mask[i, j] = 0.0 if j >= i else -inf
    mask = torch.triu(torch.ones(m, m, device=scores.device), diagonal=0)
    mask = torch.where(mask == 1.0, 0.0, float("-inf"))

    # s_matrix[i, j] = s_perm[j] if j >= i else -inf
    s_matrix = s_perm.unsqueeze(0) + mask

    # logsumexp across columns for rows 0 through m-2
    logsumexps = torch.logsumexp(s_matrix[:-1], dim=1)

    # ListMLE loss: - sum_{i=0}^{m-2} (s_perm[i] - logsumexps[i])
    loss = -torch.sum(s_perm[:-1] - logsumexps)
    return loss


class PlackettLuceRanker:
    r"""Deep Neural Learning-to-Rank architecture optimizing Plackett-Luce ListMLE loss.

    Architecture:
    -------------
    - Preprocessing: Median imputation and z-score standard scaling for numerical stability.
    - Architecture: 3-layer MLP: `Linear(17, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 32) -> ReLU -> Linear(32, 1)`.
    - Loss: Plackett-Luce ListMLE permutation likelihood via `compute_list_mle_loss`.
    - Optimization: Adam optimizer (`lr=1e-3`, `weight_decay=1e-4`).
    - Validation: Validation loss monitoring with early stopping and automatic weight rollback.

    Parameters
    ----------
    input_dim : int, default=17
        Number of input features.
    hidden_dim : int, default=64
        Hidden units in the first linear layer.
    lr : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=1e-4
        Adam L2 weight decay regularization.
    epochs : int, default=50
        Maximum training epochs.
    batch_size_races : int, default=16
        Number of races per optimizer gradient step.
    patience : int, default=10
        Validation plateau tolerance before early stopping triggers.
    random_state : int, default=42
        Random seed for torch and numpy.
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size_races: int = 16,
        patience: int = 10,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size_races = batch_size_races
        self.patience = patience
        self.random_state = random_state

        torch.manual_seed(self.random_state)
        self.net = _PlackettLuceMLP(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.best_epoch_: int | None = None
        self.best_val_loss_: float = float("inf")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        group_val: np.ndarray | None = None,
    ) -> PlackettLuceRanker:
        r"""Train the Plackett-Luce neural ranker using ListMLE loss.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix of shape `(N_train, input_dim)`.
        y_train : np.ndarray
            Training relevance vector of shape `(N_train,)` (higher is better).
        group_train : np.ndarray
            Query group sizes for training races, summing to `N_train`.
        X_val : np.ndarray | None, default=None
            Validation feature matrix of shape `(N_val, input_dim)`.
        y_val : np.ndarray | None, default=None
            Validation relevance vector of shape `(N_val,)`.
        group_val : np.ndarray | None, default=None
            Validation query group sizes, summing to `N_val`.

        Returns
        -------
        PlackettLuceRanker
            Fitted ranker instance.
        """
        torch.manual_seed(self.random_state)

        # 1. Clean and scale training features
        X_train_clean = self.scaler.fit_transform(self.imputer.fit_transform(X_train))

        # Prepare race slices
        train_slices: list[tuple[int, int]] = []
        cur = 0
        for g in group_train:
            train_slices.append((cur, cur + int(g)))
            cur += int(g)

        # Prepare validation data if provided
        has_val = X_val is not None and y_val is not None and group_val is not None
        val_slices: list[tuple[int, int]] = []
        X_val_clean: np.ndarray | None = None
        if has_val:
            assert X_val is not None and group_val is not None
            X_val_clean = self.scaler.transform(self.imputer.transform(X_val))
            cur_val = 0
            for g in group_val:
                val_slices.append((cur_val, cur_val + int(g)))
                cur_val += int(g)

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_weights: dict[str, torch.Tensor] = copy.deepcopy(self.net.state_dict())
        epochs_no_improve = 0
        self.best_val_loss_ = float("inf")

        for epoch in range(self.epochs):
            self.net.train()
            # Batch accumulator
            batch_loss = torch.tensor(0.0)
            n_in_batch = 0

            for i, (start, end) in enumerate(train_slices):
                X_race = torch.tensor(X_train_clean[start:end], dtype=torch.float32)
                y_race = torch.tensor(y_train[start:end], dtype=torch.float32)

                scores_race = self.net(X_race)
                loss_race = compute_list_mle_loss(scores_race, y_race)

                batch_loss = batch_loss + loss_race
                n_in_batch += 1

                if n_in_batch == self.batch_size_races or (i + 1) == len(train_slices):
                    optimizer.zero_grad()
                    (batch_loss / float(n_in_batch)).backward()
                    optimizer.step()
                    batch_loss = torch.tensor(0.0)
                    n_in_batch = 0

            # Validation evaluation
            if has_val and X_val_clean is not None and y_val is not None:
                self.net.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for start, end in val_slices:
                        X_r_val = torch.tensor(X_val_clean[start:end], dtype=torch.float32)
                        y_r_val = torch.tensor(y_val[start:end], dtype=torch.float32)
                        s_r_val = self.net(X_r_val)
                        val_loss_total += compute_list_mle_loss(s_r_val, y_r_val).item()

                mean_val_loss = val_loss_total / max(1, len(val_slices))
                if mean_val_loss < self.best_val_loss_:
                    self.best_val_loss_ = mean_val_loss
                    self.best_epoch_ = epoch
                    best_weights = copy.deepcopy(self.net.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break

        # Restore optimal weights if validation was used
        if has_val:
            self.net.load_state_dict(best_weights)

        self.net.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous latent pace scores for candidate drivers.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape `(N, input_dim)`.

        Returns
        -------
        np.ndarray
            1D float64 array of latent pace scores of shape `(N,)`.
        """
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)

        X_clean = self.scaler.transform(self.imputer.transform(X))
        self.net.eval()
        with torch.no_grad():
            scores_tensor = self.net(torch.tensor(X_clean, dtype=torch.float32))
        return scores_tensor.cpu().numpy().astype(np.float64).ravel()


# =============================================================================
# 4. Guaranteed Top-3 Podium Selection
# =============================================================================


def predict_podium_top3(
    scores: np.ndarray | pd.Series,
    df_race: pd.DataFrame,
) -> np.ndarray:
    r"""Select the top 3 podium finishers per race strictly guaranteeing 3 unique drivers.

    Enforces deterministic ranking by primary continuous pace score (descending),
    breaking any exact ties with secondary ascending `grid_effective` (the driver
    qualifying closer to pole position wins the tie-break).

    Handles single-race and multi-race DataFrames:
    - Single race: exactly 3 entries with value 1 (or $\min(3, m)$ if field size $m < 3$).
    - Multi-race: returns a binary mask whose sum strictly equals $3 \times N_{\text{races}}$.

    Parameters
    ----------
    scores : np.ndarray | pd.Series
        Predicted continuous scores or probabilities (higher score indicates higher
        predicted rank / faster pace).
    df_race : pd.DataFrame
        DataFrame containing the race grid. Must match `scores` length. May contain
        a race identifier ('raceId' or 'race_id') and starting grid column ('grid_effective').

    Returns
    -------
    np.ndarray
        1D int32 binary indicator mask where 1 marks a predicted podium finisher
        and 0 marks non-podium finishers.

    Raises
    ------
    ValueError
        If length of `scores` does not match length of `df_race`.
    """
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    n_rows = len(df_race)

    if n_rows != len(scores_arr):
        raise ValueError(
            f"Shape mismatch: df_race has {n_rows} rows, but scores has length {len(scores_arr)}."
        )

    if n_rows == 0:
        return np.empty(0, dtype=np.int32)

    podium_mask = np.zeros(n_rows, dtype=np.int32)

    # Determine race column
    if "raceId" in df_race.columns:
        race_col: str | None = "raceId"
    elif "race_id" in df_race.columns:
        race_col = "race_id"
    else:
        race_col = None

    has_grid = "grid_effective" in df_race.columns
    grid_all = (
        df_race["grid_effective"].to_numpy(dtype=np.float64)
        if has_grid
        else np.arange(n_rows, dtype=np.float64)
    )

    if race_col is None:
        # Single race case
        k = min(3, n_rows)
        # Primary: -scores_arr (descending), Secondary: grid_all (ascending)
        order = np.lexsort((grid_all, -scores_arr))
        podium_mask[order[:k]] = 1
        return podium_mask

    # Multi-race case: iterate over races preserving index alignment
    # Group indices dictionary preserves natural order
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
