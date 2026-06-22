"""Recenter-gesture geometry test — pure numpy, no cv2 window (runs headless in CI)."""

from __future__ import annotations

import collections

import numpy as np

from stereohand.renderer import _PRESENCE_MIN_FRAC, HandRenderer, _palm_open_facing


def _pose(*, tip_y: float, mcp_axis: str) -> np.ndarray:
    """Build a 21x3 hand. ``mcp_axis`` spreads the knuckles along 'x' (palm faces camera)
    or 'z' (palm faces sideways); ``tip_y`` sets how far up the fingertips reach (open vs fist)."""
    pts = np.zeros((21, 3))
    pts[0] = [0.0, 0.0, 0.5]  # wrist, 50 cm out
    spread = [0.02, 0.0, -0.02, -0.04]  # index→pinky offset
    for mcp, tip, s in zip((5, 9, 13, 17), (8, 12, 16, 20), spread, strict=True):
        off = [s, 0.0, 0.0] if mcp_axis == "x" else [0.0, 0.0, s]
        pts[mcp] = pts[0] + np.array(off) + [0.0, -0.05, 0.0]
        pts[tip] = pts[0] + np.array(off) + [0.0, tip_y, 0.0]
    return pts


def test_open_palm_facing_camera_is_recenter_pose() -> None:
    assert _palm_open_facing(_pose(tip_y=-0.12, mcp_axis="x"))


def test_fist_is_not_recenter_pose() -> None:
    assert not _palm_open_facing(_pose(tip_y=-0.04, mcp_axis="x"))


def test_palm_facing_sideways_is_not_recenter_pose() -> None:
    assert not _palm_open_facing(_pose(tip_y=-0.12, mcp_axis="z"))


def _presence_gate(seen: list[bool]) -> bool:
    # Skip __init__ (it opens a cv2 window); we only exercise the presence math.
    renderer = HandRenderer.__new__(HandRenderer)
    renderer._presence = collections.deque((i * 0.01, s) for i, s in enumerate(seen))
    return renderer._recently_absent()


def test_single_frame_dropout_is_not_a_real_loss() -> None:
    assert not _presence_gate([True] * 9 + [False])  # 90% present → keep the hold


def test_sustained_loss_resets() -> None:
    # Well below the keep-alive threshold → real loss.
    n_present = max(1, int(_PRESENCE_MIN_FRAC * 10) - 1)
    assert _presence_gate([True] * n_present + [False] * (10 - n_present))


def test_empty_history_counts_as_absent() -> None:
    assert _presence_gate([])
