"""Scratch Gradient Descent Convergence and Parity Verification Tests.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.linear_gd import (
    compute_cost,
    derive_learning_rates,
    fit_closed_form_lstsq,
    gradient_descent,
)


@pytest.fixture(scope="module")
def synthetic_regression_data():
    """Create reproducible synthetic linear regression dataset with intercept."""
    rng = np.random.default_rng(42)
    n = 500
    d = 5
    # Feature matrix with column 0 = 1s
    X_raw = rng.standard_normal((n, d - 1))
    X = np.column_stack([np.ones(n), X_raw])

    true_theta = np.array([3.5, -2.1, 0.8, 1.4, -0.5])
    noise = rng.normal(0, 0.2, size=n)
    y = X @ true_theta + noise
    return X, y


def test_linear_gd_lstsq_parity(synthetic_regression_data):
    """Verify that Gradient Descent achieves relative cost parity < 1e-6 with closed-form lstsq."""
    X, y = synthetic_regression_data
    alpha_safe, _, _ = derive_learning_rates(X)

    theta_gd, cost_hist = gradient_descent(X, y, alpha=alpha_safe, n_iter=10_000)
    theta_lstsq, cost_lstsq = fit_closed_form_lstsq(X, y)

    cost_gd = compute_cost(X, y, theta_gd)
    relative_excess = (cost_gd - cost_lstsq) / max(1.0, abs(cost_lstsq))

    assert relative_excess < 1e-6, (
        f"Parity failure: relative excess = {relative_excess:.2e} >= 1e-6 (J_gd={cost_gd}, J_lstsq={cost_lstsq})"
    )


def test_alpha_unsafe_divergence(synthetic_regression_data):
    """Verify that alpha_unsafe demonstrably diverges within 50 iterations."""
    X, y = synthetic_regression_data
    _, alpha_unsafe, _ = derive_learning_rates(X)

    _, cost_hist = gradient_descent(X, y, alpha=alpha_unsafe, n_iter=50)
    # Divergence manifests as cost exploding or increasing beyond starting cost J_0
    assert (
        np.isnan(cost_hist[-1])
        or np.isinf(cost_hist[-1])
        or cost_hist[-1] > cost_hist[0] * 10.0
    ), f"alpha_unsafe failed to diverge! Final cost: {cost_hist[-1]}, initial cost: {cost_hist[0]}"


def test_analytical_learning_rate_properties(synthetic_regression_data):
    """Verify analytical learning rates are strictly positive and derived from X only."""
    X, _ = synthetic_regression_data
    alpha_safe, alpha_unsafe, lambda_max = derive_learning_rates(X)

    assert lambda_max > 0.0
    assert 0.0 < alpha_safe < (2.0 / lambda_max)
    assert alpha_unsafe > (2.0 / lambda_max)
