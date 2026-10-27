"""K-Means Clustering from Scratch with K-Means++ and Empty-Cluster Recovery.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Intuition:
Imagine you have a large playground with 250 school children scattered all around.
You want to divide them into 3 game groups:
1. Step 1 (Pin Drop): You drop 3 colored flags (red, blue, green) onto the playground.
2. Step 2 (Assignment): You blow a whistle! Every child runs to whichever flag is closest to them.
3. Step 3 (Centroid Update): Each flag moves to the exact center (the average position)
   of its own crowd of children.
4. Step 4 (Repeat): You blow the whistle again. Some children near the edges switch flags!
   The flags adjust their positions again.
You repeat this until nobody changes groups and the flags stop moving.

- What is K-Means++?
  Instead of dropping the flags randomly (which might put all 3 flags right next to each other!),
  K-Means++ puts the first flag down, and then places each subsequent flag as far away as possible
  from existing flags. This spreads them out smartly before the game even begins!
- What if a flag is completely abandoned (Empty-Cluster Rule)?
  If all children run to the other flags leaving one flag completely alone, that flag immediately
  relocates to the child who is farthest away from everyone else, ensuring all K groups remain active.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class KMeansScratch:
    """K-Means clustering implemented from first mathematical principles.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K).
    init : str or np.ndarray, default='kmeans++'
        Method of initialization ('kmeans++', 'random', or explicit centroid array of shape [K, d]).
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance on centroid shift to declare convergence.
    random_state : int or None, default=42
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        init: str | np.ndarray = "kmeans++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = 42,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.init = init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0
        self.empty_cluster_triggered_: bool = False

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Initialize K centroids using K-Means++ or random selection."""
        n, d = X.shape
        K = self.n_clusters

        if isinstance(self.init, np.ndarray):
            assert self.init.shape == (K, d), (
                f"Explicit centroids shape {self.init.shape} does not match (K={K}, d={d})"
            )
            return self.init.copy().astype(np.float64)

        if self.init == "random":
            chosen_idx = rng.choice(n, size=K, replace=False)
            return X[chosen_idx].copy().astype(np.float64)

        # Default: K-Means++
        centroids = np.zeros((K, d), dtype=np.float64)
        first_idx = rng.choice(n)
        centroids[0] = X[first_idx]

        for k in range(1, K):
            # Compute distance from each point to closest existing centroid
            dist_sq = np.min(
                [np.sum((X - centroids[j]) ** 2, axis=1) for j in range(k)],
                axis=0,
            )
            probs = dist_sq / np.maximum(dist_sq.sum(), 1e-12)
            next_idx = rng.choice(n, p=probs)
            centroids[k] = X[next_idx]

        return centroids

    def fit(self, X: np.ndarray) -> KMeansScratch:
        """Fit K-Means clustering to data matrix X.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape [n_samples, n_features].

        Returns
        -------
        KMeansScratch
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        K = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        centroids = self._init_centroids(X, rng)
        labels = np.zeros(n, dtype=np.int64)

        for iteration in range(1, self.max_iter + 1):
            # Step 1: Assignment Step — compute squared Euclidean distances [n, K]
            # dist_sq[i, k] = sum((X[i] - centroids[k])^2)
            dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)

            # Step 2: Update Step — compute mean of each cluster
            new_centroids = np.zeros_like(centroids)
            for k in range(K):
                cluster_points = X[new_labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    # Empty-cluster recovery rule:
                    # Reassign centroid to the point farthest from its assigned centroid
                    self.empty_cluster_triggered_ = True
                    assigned_dists = dists[np.arange(n), new_labels]
                    farthest_idx = int(np.argmax(assigned_dists))
                    new_centroids[k] = X[farthest_idx]

            # Step 3: Check convergence
            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            labels = new_labels

            if shift <= self.tol:
                break

        # Compute final WCSS (inertia)
        final_dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        inertia = float(np.sum(np.min(final_dists, axis=1)))

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = iteration
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict closest cluster for each sample in X.

        Parameters
        ----------
        X : np.ndarray
            Matrix of shape [n_samples, n_features].

        Returns
        -------
        np.ndarray
            Cluster labels of shape [n_samples].
        """
        assert self.centroids_ is not None, "Model has not been fitted!"
        X = np.asarray(X, dtype=np.float64)
        dists = np.sum((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)
