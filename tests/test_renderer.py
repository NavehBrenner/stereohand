"""Recenter-gesture geometry test — pure numpy, no cv2 window (runs headless in CI)."""

from __future__ import annotations

import numpy as np

from stereohand.renderer import (
    _DROPOUT_GRACE_S,
    _RECENTER_HOLD_S,
    HandRenderer,
    RenderConfig,
    _palm_open_facing,
    _project,
)


def test_project_origin_is_panel_center() -> None:
    assert _project(np.zeros(3), 800, 480, 0.5, 0.3, 1400.0, 1) == (400, 240)


def test_project_world_up_maps_to_screen_up() -> None:
    # World +Z (up) must land above the panel centre (smaller screen-y) at zero elevation.
    _, sy = _project(np.array([0.0, 0.0, 0.1]), 800, 480, 0.0, 0.0, 1400.0, 1)
    assert sy < 240


def test_project_mirror_flips_x() -> None:
    pt = np.array([0.05, 0.0, 0.0])
    sx_normal, _ = _project(pt, 800, 480, 0.0, 0.0, 1400.0, 1)
    sx_mirror, _ = _project(pt, 800, 480, 0.0, 0.0, 1400.0, -1)
    assert (sx_normal - 400) == -(sx_mirror - 400)


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


def _headless_renderer() -> HandRenderer:
    """A renderer with recenter on, bypassing __init__ (which opens a cv2 window)."""
    r = HandRenderer.__new__(HandRenderer)
    r._cfg = RenderConfig(recenter=True, smooth=0.5)
    r._smoothed = None
    r._origin = np.zeros(3)
    r._hold_start = None
    r._hold_anchor = None
    r._recentered = False
    r._calib_msg = None
    r._last_seen_t = None
    r._last_good_pose_t = None
    return r


def _drive(r: HandRenderer, fps: float, frames: list[bool], palm: np.ndarray) -> None:
    """Feed `frames` (present/absent) at `fps`, holding `palm` when present."""
    for i, present in enumerate(frames):
        r._advance_pose(i / fps, present, palm if present else None)


def test_brief_dropouts_dont_restart_recenter_at_low_fps() -> None:
    # 10 fps, two-frame dropout in every four — at the old 0.2 s window this looked like a
    # full hand loss and the 3 s countdown never completed. The wall-clock grace fixes it.
    r = _headless_renderer()
    palm = _pose(tip_y=-0.12, mcp_axis="x")
    fps = 10.0
    frames = [(i % 4) >= 2 for i in range(int(fps * (_RECENTER_HOLD_S + 1)))]
    _drive(r, fps, frames, palm)
    assert r._recentered


def test_brief_pose_flicker_within_grace_doesnt_restart() -> None:
    # Hand stays present but the smoothed pose flickers to a fist for a 0.1 s burst (< grace).
    # Like a dropout, a brief pose flicker must not restart the 3 s hold. With the old
    # instant-reset this burst left under 3 s of open palm and never completed.
    r = _headless_renderer()
    open_palm = _pose(tip_y=-0.12, mcp_axis="x")
    fist = _pose(tip_y=-0.04, mcp_axis="x")
    fps = 30.0
    burst = set(range(int(fps * 1.5), int(fps * 1.5) + 3))  # 3-frame (0.1 s) fist flicker
    for i in range(int(fps * (_RECENTER_HOLD_S + 1))):
        r._advance_pose(i / fps, True, fist if i in burst else open_palm)
    assert r._recentered


def test_sustained_loss_resets_the_hold() -> None:
    r = _headless_renderer()
    palm = _pose(tip_y=-0.12, mcp_axis="x")
    # Hold 1 s, then drop out for well beyond the grace period.
    _drive(r, 30.0, [True] * 30, palm)
    assert r._smoothed is not None
    n_gap = int(30.0 * (_DROPOUT_GRACE_S + 0.5))
    pts, _ = r._advance_pose(1.0 + n_gap / 30.0, False, None)
    assert pts is None
    assert r._smoothed is None and r._hold_start is None
