"""Qualifying Parser and Lap Time Outlier Tests.

Version: 5.0 — Authoritative Project Standards
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.builder import build_pre_race_features, parse_q1_time


def test_parse_q1_time_formats():
    """Verify parse_q1_time correctly handles standard minute:second and second formats."""
    # Standard format: '1:24.350' -> 1*60 + 24.35 = 84.35
    assert abs(parse_q1_time("1:24.350") - 84.35) < 1e-4

    # Pure seconds format: '65.234' -> 65.234
    assert abs(parse_q1_time("65.234") - 65.234) < 1e-4

    # Two minute format: '2:01.002' -> 121.002
    assert abs(parse_q1_time("2:01.002") - 121.002) < 1e-4

    # None and missing inputs
    assert parse_q1_time(None) is None
    assert parse_q1_time(np.nan) is None
    assert parse_q1_time("DNF") is None


def test_q_gap_pct_non_negative():
    """Verify that qualifying gap percentage is strictly non-negative."""
    df = build_pre_race_features()
    valid_gaps = df["q_gap_pct"].dropna()
    assert (valid_gaps >= -1e-6).all(), "Found negative qualifying gaps!"
    assert (valid_gaps == 0.0).sum() > 0, "No pole position driver with 0.0% gap found!"
