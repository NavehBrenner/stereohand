"""Skew-decision predicate — the cv2-free core of capture (runs in CI)."""

from __future__ import annotations

from stereohand.capture import within_skew


def test_within_threshold_accepted():
    assert within_skew(100.000, 100.010, max_skew_s=0.02) is True


def test_over_threshold_rejected():
    assert within_skew(100.000, 100.050, max_skew_s=0.02) is False


def test_boundary_is_inclusive():
    assert within_skew(100.000, 100.020, max_skew_s=0.02) is True


def test_order_does_not_matter():
    assert within_skew(100.05, 100.0, max_skew_s=0.02) == within_skew(
        100.0, 100.05, max_skew_s=0.02
    )
