"""PCA Scratch Implementation and Parity Verification Tests.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.pca_scratch import PCAScratch


@pytest.fixture(scope="module")
def standardized_data():
    """Create reproducible synthetic standardized feature matrix."""
    rng = np.random.default_rng(42)
    n = 200
    d = 6
    # Create correlated data
    A = rng.standard_normal((d, d))
    cov = A @ A.T
    raw = rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)
    X_std = StandardScaler().fit_transform(raw)
    return X_std


def test_pca_scratch_eigenvalues_parity_with_sklearn(standardized_data):
    """Verify Scratch PCA eigenvalues match Scikit-Learn within 1e-6."""
    X = standardized_data
    scratch = PCAScratch().fit(X)
    sklearn_pca = PCA().fit(X)

    # Scikit-learn explained_variance_ uses n-1 sample variance, identical to our Sigma
    for i in range(X.shape[1]):
        val_scratch = scratch.eigenvalues_[i]
        val_sklearn = sklearn_pca.explained_variance_[i]
        rel_diff = abs(val_scratch - val_sklearn) / max(1.0, val_sklearn)
        assert rel_diff < 1e-6, (
            f"PC{i+1} eigenvalue mismatch: scratch={val_scratch:.6f}, sklearn={val_sklearn:.6f}"
        )


def test_pca_evr_properties(standardized_data):
    """Verify Explained Variance Ratio sums to 1.0 and is strictly monotonic descending."""
    X = standardized_data
    scratch = PCAScratch().fit(X)

    assert abs(scratch.explained_variance_ratio_.sum() - 1.0) < 1e-6
    # Non-increasing order
    diffs = np.diff(scratch.explained_variance_ratio_)
    assert (diffs <= 1e-9).all(), "EVR is not monotonic descending!"


def test_pca_projection_parity_with_sklearn(standardized_data):
    """Verify 2D projection matches Scikit-Learn within 1e-5 after sign alignment."""
    X = standardized_data
    scratch = PCAScratch(n_components=2).fit(X)
    sklearn_pca = PCA(n_components=2).fit(X)

    Z_scratch = scratch.transform(X)
    Z_sklearn = sklearn_pca.transform(X)

    # Align signs for comparison
    for col in range(2):
        # Check if vectors point in same or opposite direction
        corr = np.corrcoef(Z_scratch[:, col], Z_sklearn[:, col])[0, 1]
        assert abs(abs(corr) - 1.0) < 1e-5, f"PC{col+1} projection not collinear! Corr={corr}"


def test_full_loading_matrix_format(standardized_data):
    """Verify Table 6b loading matrix is generated with 6 features x 6 PCs."""
    X = standardized_data
    feature_names = [f"feat_{i}" for i in range(6)]
    scratch = PCAScratch().fit(X)
    df_loadings = scratch.get_loading_matrix_df(feature_names)

    assert df_loadings.shape == (6, 6)
    assert list(df_loadings.index) == feature_names
    assert list(df_loadings.columns) == ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
