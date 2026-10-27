"""Scratch Vectorized Gradient Descent and Stability Analysis Module.

Version: 5.0 — Authoritative Project Standards

ELI5 Explanation (Child-Friendly Analogy):
-----------------------------------------
Intuition:
Imagine you are standing at the top of a hilly foggy mountain bowl with a heavy bowling ball.
You cannot see the bottom, but you can feel which direction the ground slopes downward under your feet!
If you take a small step in the steepest downward direction (the negative gradient), and repeat
that step every second, you will eventually reach the very bottom of the bowl (the minimum error).

1. What is alpha (the learning rate)?
   It is your stride length!
   - If your stride is too small, it takes a million steps to reach the bottom.
   - If your stride is just right (alpha_safe), you jog smoothly down to the bottom.
   - If your stride is giant (alpha_unsafe), you leap clean over the valley and crash into the opposite
     mountain wall, bouncing higher and higher until you explode into infinity (divergence)!
2. How do we prove safety mathematically?
   The curvature of our bowl is given by the Hessian matrix H = (1/n) * X^T X.
   The steepest slope corresponds to the largest eigenvalue lambda_max.
   Linear algebra proves that gradient descent is stable if and only if alpha < 2 / lambda_max.
   We choose alpha_safe = 0.8 * (2 / lambda_max) for guaranteed smooth convergence,
   and demonstrate divergence empirically with alpha_unsafe = 2.2 * (2 / lambda_max).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Compute the Mean Squared Error cost J(theta).

    Formula:
        J(theta) = (1 / (2 * n)) * ||X @ theta - y||^2

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape [n, d] including intercept column.
    y : np.ndarray
        Target vector of shape [n].
    theta : np.ndarray
        Weight vector of shape [d].

    Returns
    -------
    float
        Scalar cost value.
    """
    n = len(y)
    residuals = X @ theta - y
    return float((1.0 / (2.0 * n)) * np.dot(residuals, residuals))


def compute_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute vectorized gradient of cost with respect to theta.

    Formula:
        grad = (1 / n) * X^T @ (X @ theta - y)

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape [n, d].
    y : np.ndarray
        Target vector of shape [n].
    theta : np.ndarray
        Weight vector of shape [d].

    Returns
    -------
    np.ndarray
        Gradient vector of shape [d].
    """
    n = len(y)
    return (1.0 / n) * (X.T @ (X @ theta - y))


def derive_learning_rates(X: np.ndarray) -> tuple[float, float, float]:
    """Derive analytically guaranteed safe and unsafe learning rates from eigenvalues of X^T X / n.

    Formula:
        lambda_max = max(eigvals(X^T X / n))
        alpha_max = 2.0 / lambda_max
        alpha_safe = 0.8 * alpha_max
        alpha_unsafe = 2.2 * alpha_max

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape [n, d] including intercept column.

    Returns
    -------
    tuple[float, float, float]
        Tuple of (alpha_safe, alpha_unsafe, lambda_max).
    """
    n = len(X)
    H = (1.0 / n) * (X.T @ X)
    # Guaranteed real eigenvalues for symmetric covariance matrix
    eigenvalues = np.linalg.eigvalsh(H)
    lambda_max = float(np.max(eigenvalues))

    alpha_max = 2.0 / lambda_max
    alpha_safe = 0.8 * alpha_max
    alpha_unsafe = 2.2 * alpha_max

    return alpha_safe, alpha_unsafe, lambda_max


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_iter: int = 10_000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, list[float]]:
    """Run Batch Gradient Descent optimization.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape [n, d] including intercept column.
    y : np.ndarray
        Target vector of shape [n].
    alpha : float
        Learning rate stride.
    n_iter : int, default=10000
        Maximum iterations.
    tol : float, default=1e-12
        Convergence tolerance on relative change in cost.

    Returns
    -------
    tuple[np.ndarray, list[float]]
        Tuple of (optimal_theta, cost_history).
    """
    d = X.shape[1]
    theta = np.zeros(d, dtype=np.float64)
    cost_history: list[float] = []

    for i in range(n_iter):
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if np.isnan(cost) or np.isinf(cost) or cost > 1e12:
            # Divergence detected (e.g. unsafe learning rate)
            break

        grad = compute_gradient(X, y, theta)
        theta_next = theta - alpha * grad

        if i > 0 and abs(cost_history[-2] - cost) < tol:
            break

        theta = theta_next

    return theta, cost_history


def fit_closed_form_lstsq(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit Ordinary Least Squares via closed-form minimum-norm solution.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape [n, d].
    y : np.ndarray
        Target vector of shape [n].

    Returns
    -------
    tuple[np.ndarray, float]
        Tuple of (theta_lstsq, lstsq_cost).
    """
    theta_lstsq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    cost_lstsq = compute_cost(X, y, theta_lstsq)
    return theta_lstsq, cost_lstsq
