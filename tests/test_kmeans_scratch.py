"""K-Means Scratch Implementation and Parity Verification Tests.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from src.models.kmeans_scratch import KMeansScratch


@pytest.fixture(scope="module")
def cluster_data():
    """Create synthetic 2D clustering dataset with 3 well-separated blobs."""
    rng = np.random.default_rng(42)
    blob1 = rng.normal(loc=[-5.0, -5.0], scale=0.8, size=(100, 2))
    blob2 = rng.normal(loc=[0.0, 5.0], scale=0.8, size=(100, 2))
    blob3 = rng.normal(loc=[5.0, -5.0], scale=0.8, size=(100, 2))
    X = np.vstack([blob1, blob2, blob3])
    return X


def test_kmeans_wcss_parity_with_sklearn(cluster_data):
    """Verify that Scratch K-Means achieves WCSS parity within 1e-4 of Scikit-Learn with shared centroids."""
    X = cluster_data
    K = 3
    rng = np.random.default_rng(42)
    init_centroids = X[rng.choice(len(X), size=K, replace=False)]

    scratch = KMeansScratch(n_clusters=K, init=init_centroids, max_iter=300, tol=1e-6, random_state=42)
    scratch.fit(X)

    sklearn_km = KMeans(
        n_clusters=K,
        init=init_centroids,
        n_init=1,
        algorithm="lloyd",
        tol=1e-6,
        max_iter=300,
        random_state=42,
    )
    sklearn_km.fit(X)

    rel_diff = abs(scratch.inertia_ - sklearn_km.inertia_) / sklearn_km.inertia_
    assert rel_diff < 1e-4, (
        f"WCSS parity failed: scratch={scratch.inertia_:.4f}, sklearn={sklearn_km.inertia_:.4f}, rel_diff={rel_diff:.2e}"
    )


def test_kmeans_label_parity_up_to_permutation(cluster_data):
    """Verify cluster assignments match Scikit-Learn up to label permutation."""
    X = cluster_data
    K = 3
    rng = np.random.default_rng(42)
    init_centroids = X[rng.choice(len(X), size=K, replace=False)]

    scratch = KMeansScratch(n_clusters=K, init=init_centroids, max_iter=300, tol=1e-6, random_state=42)
    scratch.fit(X)

    sklearn_km = KMeans(
        n_clusters=K,
        init=init_centroids,
        n_init=1,
        algorithm="lloyd",
        tol=1e-6,
        max_iter=300,
        random_state=42,
    )
    sklearn_km.fit(X)

    # Contingency matrix between scratch and sklearn labels
    contingency = np.zeros((K, K), dtype=int)
    for s_lbl, sk_lbl in zip(scratch.labels_, sklearn_km.labels_):
        contingency[s_lbl, sk_lbl] += 1

    # Solve optimal assignment problem
    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched_count = contingency[row_ind, col_ind].sum()
    match_rate = matched_count / len(X)

    assert match_rate > 0.99, f"Label matching rate too low: {match_rate:.4f}"


def test_empty_cluster_recovery():
    """Verify that empty-cluster recovery triggers safely and reassigns points without crashing."""
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
    ])
    # Place 2 initial centroids near the cluster of 3 points, and 1 centroid near infinity
    init_centroids = np.array([
        [0.0, 0.0],
        [0.05, 0.05],
        [100.0, 100.0],  # Will get zero points
    ])
    km = KMeansScratch(n_clusters=3, init=init_centroids, max_iter=10)
    km.fit(X)
    assert km.empty_cluster_triggered_ is True
    assert len(np.unique(km.labels_)) == 3, "Failed to recover all 3 clusters!"
