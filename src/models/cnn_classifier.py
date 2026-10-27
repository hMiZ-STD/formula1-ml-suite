"""Dual-Branch 1D-CNN and Static-Only MLP Ablation Classification Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Imagine you are a Formula 1 team principal deciding whether to bet on a driver:
1. You have two kinds of information:
   - "Right Now" Clues (Static Features): Where are they starting on the grid today?
     How old are they? How fast is their car this weekend?
   - "Recent History" Film Reel (Temporal Sequence): What happened in their last 10 races?
     Did they finish on the podium 3 times in a row? Did they crash twice?
2. Our Dual-Branch Neural Network reads BOTH simultaneously:
   - Branch A (1D-CNN) is like watching the film reel of their last 10 races.
     It slides convolution filters across time to detect upward momentum or bad slumps.
   - Branch B (Static MLP) reads today's immediate starting conditions.
3. The model merges both branches together, combining historical form with today's grid spot
   to output a podium probability.
4. To prove whether the film reel actually helps, we also train a "Static-Only MLP" that only
   sees today's clues. Comparing the two proves the true value of sequence memory!
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.features.builder import ALLOWED_FEATURES


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for determinism.

    Parameters
    ----------
    seed : int, default=42
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DualBranchCNN(nn.Module):
    """Dual-branch neural network combining 1D temporal convolution with static MLP.

    Parameters
    ----------
    n_static : int
        Number of static input features after imputation and scaling.
    seq_channels : int, default=15
        Number of input channels in the sequence branch.
    dropout_rate : float, default=0.3
        Dropout probability in classification head.
    """

    def __init__(
        self,
        n_static: int,
        seq_channels: int = 15,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # Branch A: 1D Temporal Convolution over past races
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # Outputs [batch_size, 64]
        )

        # Branch B: Static MLP over today's race features
        self.static_branch = nn.Sequential(
            nn.Linear(n_static, 32),
            nn.ReLU(),
        )

        # Fusion Head: Concatenate 64 + 32 = 96
        self.fusion_head = nn.Sequential(
            nn.Linear(64 + 32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through dual branches and fusion head.

        Parameters
        ----------
        x_seq : torch.Tensor
            Sequence tensor of shape [batch_size, seq_channels, seq_len].
        x_static : torch.Tensor
            Static feature tensor of shape [batch_size, n_static].

        Returns
        -------
        torch.Tensor
            Raw output logits of shape [batch_size, 1].
        """
        out_temporal = self.temporal_branch(x_seq)
        out_static = self.static_branch(x_static)
        merged = torch.cat([out_temporal, out_static], dim=1)
        logits = self.fusion_head(merged)
        return logits


class StaticMLP(nn.Module):
    """Static-only Multi-Layer Perceptron ablation (no sequence branch).

    Parameters
    ----------
    n_static : int
        Number of static input features.
    dropout_rate : float, default=0.3
        Dropout probability.
    """

    def __init__(
        self,
        n_static: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_static, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """Forward pass through static MLP.

        Parameters
        ----------
        x_static : torch.Tensor
            Tensor of shape [batch_size, n_static].

        Returns
        -------
        torch.Tensor
            Logits of shape [batch_size, 1].
        """
        return self.network(x_static)


def build_sequence_tensors(
    all_dfs: list[pd.DataFrame],
    seq_len: int = 10,
) -> list[np.ndarray]:
    """Build sequence arrays of shape [N, 15, seq_len] for chronologically chained DataFrames.

    Parameters
    ----------
    all_dfs : list[pd.DataFrame]
        List of DataFrames in strict chronological order (e.g. [train_core, val, test]).
    seq_len : int, default=10
        Number of past races in sequence.

    Returns
    -------
    list[np.ndarray]
        List of sequence numpy arrays corresponding to each input DataFrame.
    """
    combined = pd.concat(all_dfs, ignore_index=True)
    seq_cols = [
        "norm_finish",
        "is_dnf",
        "grid_norm",
        "grid_zero_flag",
        "q_gap_pct",
        "q1_data_available",
        "quali_position_norm",
        "grid_vs_quali_delta",
        "driver_dnf_rate_p5",
        "constructor_pts_share",
        "constructor_season_rank",
        "rounds_completed",
        "driver_age_years",
        "is_rookie_window",
    ]

    mat = combined[seq_cols].fillna(0.0).to_numpy()
    driver_ids = combined["driverId"].to_numpy()
    n_total = len(combined)

    seq_array = np.zeros((n_total, 15, seq_len), dtype=np.float32)
    history: dict[int, list[np.ndarray]] = {}

    for i in range(n_total):
        did = int(driver_ids[i])
        past = history.get(did, [])
        k = len(past)

        if k > 0:
            actual_k = min(k, seq_len)
            recent = past[-actual_k:]
            start_slot = seq_len - actual_k
            for col_idx in range(14):
                seq_array[i, col_idx, start_slot:] = [r[col_idx] for r in recent]
            # Channel 14 is the mask channel (1 for real race, 0 for padding)
            seq_array[i, 14, start_slot:] = 1.0

        if did not in history:
            history[did] = []
        history[did].append(mat[i])

    # Slice back into individual splits
    split_arrays: list[np.ndarray] = []
    idx = 0
    for df_split in all_dfs:
        sz = len(df_split)
        split_arrays.append(seq_array[idx : idx + sz])
        idx += sz

    return split_arrays


def train_cnn_classifier(
    train_core: pd.DataFrame,
    val: pd.DataFrame,
    max_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    random_seed: int = 42,
) -> tuple[DualBranchCNN, Pipeline, dict[str, list[float]], int]:
    """Train DualBranchCNN with early stopping on validation BCE loss.

    Parameters
    ----------
    train_core : pd.DataFrame
        Core training set (1994–2016).
    val : pd.DataFrame
        Validation set (2017–2018).
    max_epochs : int, default=50
        Maximum training epochs.
    batch_size : int, default=32
        Mini-batch size.
    learning_rate : float, default=1e-3
        Adam optimizer learning rate.
    patience : int, default=10
        Early stopping patience epochs.
    random_seed : int, default=42
        Random seed.

    Returns
    -------
    tuple[DualBranchCNN, Pipeline, dict[str, list[float]], int]
        Fitted model, static preprocessor pipeline, training history, and best epoch.
    """
    set_seed(random_seed)

    # Step 1: Fit static preprocessor (imputer + scaler) on train_core only
    static_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    X_static_train = static_pipe.fit_transform(train_core[ALLOWED_FEATURES]).astype(np.float32)
    X_static_val = static_pipe.transform(val[ALLOWED_FEATURES]).astype(np.float32)

    # Step 2: Build sequence tensors
    seq_train, seq_val = build_sequence_tensors([train_core, val])

    y_train = train_core["y_podium"].to_numpy().astype(np.float32)
    y_val = val["y_podium"].to_numpy().astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(seq_train),
        torch.from_numpy(X_static_train),
        torch.from_numpy(y_train).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(seq_val),
        torch.from_numpy(X_static_val),
        torch.from_numpy(y_val).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class imbalance weighting
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([min(n_neg / max(n_pos, 1.0), 20.0)], dtype=torch.float32)

    n_static = X_static_train.shape[1]
    model = DualBranchCNN(n_static=n_static)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_weights: dict[str, Any] = {}
    best_epoch = 0
    patience_counter = 0

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_accum = 0.0
        n_train_batches = 0
        for b_seq, b_static, b_y in train_loader:
            optimizer.zero_grad()
            logits = model(b_seq, b_static)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss_accum / max(n_train_batches, 1)

        # Validation step
        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for b_seq, b_static, b_y in val_loader:
                logits = model(b_seq, b_static)
                loss = criterion(logits, b_y)
                val_loss_accum += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss_accum / max(n_val_batches, 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Early stopping check on validation BCE loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    return model, static_pipe, history, best_epoch


def predict_cnn_probabilities(
    model: DualBranchCNN,
    static_pipe: Pipeline,
    df: pd.DataFrame,
    seq_tensor: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate podium probabilities from trained DualBranchCNN.

    Parameters
    ----------
    model : DualBranchCNN
        Trained model.
    static_pipe : Pipeline
        Fitted imputer/scaler pipeline.
    df : pd.DataFrame
        DataFrame containing ALLOWED_FEATURES.
    seq_tensor : np.ndarray
        Sequence array of shape [len(df), 15, 10].
    batch_size : int, default=64
        Inference batch size.

    Returns
    -------
    np.ndarray
        1D array of predicted probabilities.
    """
    model.eval()
    X_static = static_pipe.transform(df[ALLOWED_FEATURES]).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(seq_tensor),
        torch.from_numpy(X_static),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs: list[float] = []
    with torch.no_grad():
        for b_seq, b_static in loader:
            logits = model(b_seq, b_static)
            p = torch.sigmoid(logits).squeeze(1).numpy()
            probs.extend(p.tolist())

    return np.array(probs, dtype=np.float64)
