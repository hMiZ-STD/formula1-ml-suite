"""Principal Component Analysis (PCA) from Scratch via Covariance Eigendecomposition.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Intuition:
Imagine you have a complex 3D Formula 1 car sculpture with 6 different measurements
(speed, points, wins, podiums, reliability, teammate dominance). It is hard for human eyes
to visualize 6 dimensions at the same time!
PCA is like shining a bright flashlight at that 6D sculpture to cast a flat 2D shadow onto the wall:
- If you shine the flashlight from a random bad angle, the shadow looks like a messy, squished blob.
- But if you rotate the flashlight to the exact magical angle that spreads the shadow out as wide as possible
  (maximizing variance), you capture almost all of the car's true shape and details in just 2 numbers!

1. How do we find that magical angle mathematically?
   We calculate the Covariance Matrix Sigma = (1 / (n - 1)) * X_std^T X_std, which measures how every
   feature varies together with every other feature.
2. The Eigenvalues (lambda) tell us how much variance (information) each shadow direction captures.
3. The Eigenvectors (V) are the directional compass headings of the flashlight beams.
4. Sign Convention:
   Eigenvectors have arbitrary +/- signs (pointing North vs South along the same line).
   To make them human-interpretable and deterministic, we enforce a strict rule: we flip the eigenvector
   so that its largest-magnitude feature loading is always POSITIVE!
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class PCAScratch:
    """Principal Component Analysis implemented from first mathematical principles.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep. If None, keeps all d components.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self.eigenvalues_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # Shape [d, n_components]
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_explained_variance_ratio_: np.ndarray | None = None
        self.covariance_matrix_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> PCAScratch:
        """Fit PCA to standardized feature matrix X.

        Parameters
        ----------
        X : np.ndarray
            Standardized data matrix of shape [n_samples, n_features] (mean=0, std=1).

        Returns
        -------
        PCAScratch
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        # Step 1: Compute sample covariance matrix Sigma = (1 / (n - 1)) * X^T @ X
        Sigma = (1.0 / (n - 1.0)) * (X.T @ X)
        self.covariance_matrix_ = Sigma

        # Step 2: Eigendecomposition of symmetric covariance matrix
        # np.linalg.eigh guarantees real eigenvalues and orthonormal eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

        # eigh returns eigenvalues in ASCENDING order -> reverse to DESCENDING
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 3: Numerical safety clamp on floating-point negative eigenvalues
        assert (eigenvalues[eigenvalues < 0] > -1e-10).all(), (
            f"Unexpected large negative eigenvalue detected: {eigenvalues[eigenvalues < 0]}"
        )
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        # Step 4: Sign convention for interpretability and determinism
        # Flip eigenvector so that the feature loading with largest magnitude is positive
        for j in range(d):
            max_mag_idx = int(np.argmax(np.abs(eigenvectors[:, j])))
            if eigenvectors[max_mag_idx, j] < 0:
                eigenvectors[:, j] *= -1.0

        # Step 5: Explained Variance Ratio (EVR)
        total_var = eigenvalues.sum()
        evr = eigenvalues / max(total_var, 1e-12)
        cum_evr = np.cumsum(evr)

        # Truncate to n_components if requested
        k = d if self.n_components is None else min(self.n_components, d)
        self.eigenvalues_ = eigenvalues[:k]
        self.components_ = eigenvectors[:, :k]  # Loadings: [d, k]
        self.explained_variance_ratio_ = evr[:k]
        self.cumulative_explained_variance_ratio_ = cum_evr[:k]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project standardized data X onto principal component subspace.

        Formula:
            Z = X @ V

        Parameters
        ----------
        X : np.ndarray
            Standardized data matrix of shape [n_samples, n_features].

        Returns
        -------
        np.ndarray
            Projected coordinates of shape [n_samples, n_components].
        """
        assert self.components_ is not None, "Model has not been fitted!"
        X = np.asarray(X, dtype=np.float64)
        return X @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return projected coordinates in one step."""
        return self.fit(X).transform(X)

    def get_loading_matrix_df(self, feature_names: list[str]) -> pd.DataFrame:
        """Produce full Table 6b loading matrix DataFrame formatted to 3 decimal places.

        Parameters
        ----------
        feature_names : list[str]
            List of feature names corresponding to rows.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape [n_features, n_components].
        """
        assert self.components_ is not None, "Model has not been fitted!"
        n_comp = self.components_.shape[1]
        col_names = [f"PC{i+1}" for i in range(n_comp)]
        df_loadings = pd.DataFrame(
            np.round(self.components_, 3),
            index=feature_names,
            columns=col_names,
        )
        return df_loadings
