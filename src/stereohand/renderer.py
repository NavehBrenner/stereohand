"""Built-in visualisation for :class:`~stereohand.tracker.StereoHandTracker`.

All rendering is pure cv2 (no matplotlib) — typically ~1 ms per composite frame.  The
``mirror`` flag flips the view horizontally so the skeleton moves like a mirror image of
the user's hand (left/right inverted).

This module is imported lazily (only when ``render=True``) so headless deployments never
pull in cv2 at import time.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from stereohand.landmarker import _BLUE, _GREEN, _PEACH, _PURPLE, _YELLOW

if TYPE_CHECKING:
    from stereohand.landmarker import HandLandmarks2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RenderConfig:
    """Configuration knobs for the live visualisation window.

    Attributes:
        mirror:   Flip the view horizontally so it acts like a mirror.
        smooth:   EMA alpha for temporal smoothing (1 = no smoothing, 0.1 = very smooth).
        recenter: Enable the hold-palm-open gesture that re-zeros the world origin to the
                  current palm position (see :data:`_RECENTER_HOLD_S`).
    """

    mirror: bool = False
    smooth: float = 0.5
    recenter: bool = False


# ---------------------------------------------------------------------------
# 3D rendering constants — mirrors handpose3d (TemugeB/handpose3d).
# ---------------------------------------------------------------------------

# Map OpenCV camera coords (x=right, y=down, z=forward) to display (z=up).
#   display_x = cam_x   (left-right)
#   display_y = cam_z   (depth, into screen — dropped by the ortho front view)
#   display_z = -cam_y  (up: cam y is down, so negate)
_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)

# Finger groups — exact copy from handpose3d's show_3d_hands.py.
_FINGERS = [
    [[0, 17], [17, 18], [18, 19], [19, 20]],  # pinky
    [[0, 13], [13, 14], [14, 15], [15, 16]],  # ring
    [[0, 9], [9, 10], [10, 11], [11, 12]],  # middle
    [[0, 5], [5, 6], [6, 7], [7, 8]],  # index
    [[0, 1], [1, 2], [2, 3], [3, 4]],  # thumb
]
# Colors matched to MediaPipe's get_default_hand_connections_style() (reused from the 2D
# landmarker) so the 3D skeleton reads the same as the camera-feed overlay.  Order = _FINGERS.
_FINGER_COLORS_BGR = [
    _BLUE,  # pinky
    _GREEN,  # ring
    _YELLOW,  # middle
    _PURPLE,  # index
    _PEACH,  # thumb
]

# Palm-center landmark index: MediaPipe index 9 = middle-finger MCP, the geometric
# centre of the palm.  Index 0 is the wrist, which sits at the base.
_PALM_CENTER_IDX = 9

# Recenter gesture: hold an open palm, square to the camera and still, for this long to
# re-zero the world origin to the current palm position.
_RECENTER_HOLD_S = 3.0
_RECENTER_MOVE_TOL_M = 0.02  # palm may drift this much (2 cm) and still count as "still"
# A recenter hold survives brief MediaPipe dropouts: only a sustained loss (hand missing in
# >10% of frames over the last 100 ms) counts as the hand leaving. This also rejects the odd
# spurious single-frame detection during a real absence.
_PRESENCE_WINDOW_S = 0.2
_PRESENCE_MIN_FRAC = 0.2
_FINGER_TIPS = (8, 12, 16, 20)  # index, middle, ring, pinky tips (skip thumb)
_FINGER_MCPS = (5, 9, 13, 17)  # their knuckles


def _palm_open_facing(pts: np.ndarray) -> bool:
    """True when the hand is open and roughly square to the camera — the recenter pose.

    ``pts`` is the raw ``(21, 3)`` metric hand in the left-camera frame.
    """
    wrist = pts[0]
    extended = sum(
        np.linalg.norm(pts[tip] - wrist) > 1.4 * np.linalg.norm(pts[mcp] - wrist)
        for tip, mcp in zip(_FINGER_TIPS, _FINGER_MCPS, strict=True)
    )
    if extended < 3:
        return False
    normal = np.cross(pts[5] - wrist, pts[17] - wrist)
    norm = float(np.linalg.norm(normal))
    # ponytail: "square to camera" (palm-plane normal ≈ camera z-axis) — this can't tell
    # palm from back-of-hand. Add a handedness sign check if back-of-hand triggers it.
    return norm > 0 and abs(normal[2]) > 0.7 * norm


# Metres → pixels for the 3D panel.  Hand ≈15 cm; ~1400 px/m fills a 480 px panel.
_SCALE = 1400.0
_AXIS_LEN_M = 0.05  # 5 cm reference axes at the world origin

# HUD styling.
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.6
_HUD_COLOR = (200, 200, 200)
_HUD_THICK = 1

_WIN_NAME = "stereohand"


# ---------------------------------------------------------------------------
# 3D panel renderer
# ---------------------------------------------------------------------------


def _render_hand_3d(
    pts: np.ndarray | None,
    width: int,
    height: int = 480,
    *,
    mirror: bool = False,
    fps: float | None = None,
    palm_xyz: np.ndarray | None = None,
    calib_msg: str | None = None,
) -> np.ndarray:
    """Orthographic front view with fixed world axes; the hand moves in world space."""
    canvas = np.zeros((height, width, 3), np.uint8)
    cx, cy = width // 2, height // 2

    # Sign for horizontal direction: mirroring negates X on screen.
    xsign = -1 if mirror else 1

    # --- World-origin axes (always drawn, even when no hand) ---
    L = int(_SCALE * _AXIS_LEN_M)
    # X axis → red (right, or left when mirrored)
    cv2.arrowedLine(
        canvas, (cx, cy), (cx + xsign * L, cy), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15
    )
    cv2.putText(
        canvas,
        "X",
        (cx + xsign * L + xsign * 4, cy + 5),
        _HUD_FONT,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    # Z axis → blue (up, since we map cam -Y → display Z → screen up)
    cv2.arrowedLine(canvas, (cx, cy), (cx, cy - L), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.15)
    cv2.putText(canvas, "Z", (cx + 4, cy - L - 6), _HUD_FONT, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    # Y axis → green (into screen; show as a small dot / circle since ortho front view)
    cv2.circle(canvas, (cx, cy), 4, (0, 200, 0), -1, cv2.LINE_AA)
    cv2.putText(canvas, "Y", (cx - 16, cy - 8), _HUD_FONT, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

    if pts is None:
        cv2.putText(
            canvas,
            "no hand (must be visible in both views)",
            (20, cy + 60),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    else:
        # Front view: screen x = display-x (right), screen y = -display-z (up; cv2 y down).
        px = (cx + xsign * _SCALE * pts[:, 0]).astype(int)
        py = (cy - _SCALE * pts[:, 2]).astype(int)
        for finger, color in zip(_FINGERS, _FINGER_COLORS_BGR, strict=True):
            for a, b in finger:
                cv2.line(canvas, (px[a], py[a]), (px[b], py[b]), color, 4, cv2.LINE_AA)
        # Joint dots for visibility.
        for i in range(21):
            cv2.circle(canvas, (px[i], py[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # --- HUD: FPS (top-left) ---
    if fps is not None:
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f}",
            (12, 28),
            _HUD_FONT,
            _HUD_SCALE,
            _HUD_COLOR,
            _HUD_THICK,
            cv2.LINE_AA,
        )

    # --- HUD: palm-centre world XYZ (top-right) ---
    if palm_xyz is not None:
        x, y, z = palm_xyz
        txt = f"Palm  X:{x * 100:+6.1f}  Y:{y * 100:+6.1f}  Z:{z * 100:+6.1f}  cm"
        (tw, _), _ = cv2.getTextSize(txt, _HUD_FONT, _HUD_SCALE, _HUD_THICK)
        cv2.putText(
            canvas,
            txt,
            (width - tw - 12, 28),
            _HUD_FONT,
            _HUD_SCALE,
            _HUD_COLOR,
            _HUD_THICK,
            cv2.LINE_AA,
        )

    # --- HUD: recenter calibration prompt / countdown (top-centre, prominent) ---
    if calib_msg:
        (tw, _), _ = cv2.getTextSize(calib_msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.putText(
            canvas,
            calib_msg,
            ((width - tw) // 2, 56),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Renderer — manages the cv2 window and per-frame state
# ---------------------------------------------------------------------------


class HandRenderer:
    """Stateful renderer that owns the cv2 window and temporal smoothing.

    Instantiated internally by :class:`~stereohand.tracker.StereoHandTracker` when
    ``render=True``.  Call :meth:`step` each iteration; it returns ``False`` when the
    user closes the window / presses 'q'.
    """

    def __init__(self, config: RenderConfig) -> None:
        self._cfg = config
        self._smoothed: np.ndarray | None = None
        self._fps_ts: collections.deque[float] = collections.deque(maxlen=30)
        self._fps: float | None = None
        # Recenter gesture state.
        self._origin = np.zeros(3)
        self._hold_anchor: np.ndarray | None = None  # palm position the current hold started at
        self._hold_start: float | None = None  # monotonic time the hold began
        self._recentered = False  # latched after a successful recenter until the pose is released
        self._calib_msg: str | None = None
        self._presence: collections.deque[tuple[float, bool]] = collections.deque()
        cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)

    # -- public interface ---------------------------------------------------

    def step(
        self,
        *,
        frames: tuple[np.ndarray, np.ndarray] | None,
        landmarks_2d: tuple[HandLandmarks2D | None, HandLandmarks2D | None] | None,
        landmarks_3d: np.ndarray | None,
        present: bool,
    ) -> bool:
        """Render one composite frame.  Returns ``False`` when the window should close."""
        from stereohand.landmarker import draw_landmarks_on_frame

        if frames is None:
            return cv2.waitKey(10) & 0xFF not in (ord("q"), 27)

        fl, fr = frames
        if landmarks_2d is not None:
            lm_l, lm_r = landmarks_2d
            if lm_l is not None:
                fl = draw_landmarks_on_frame(fl, lm_l)
            if lm_r is not None:
                fr = draw_landmarks_on_frame(fr, lm_r)

        # Mirror: flip camera feeds horizontally.
        if self._cfg.mirror:
            fl = cv2.flip(fl, 1)
            fr = cv2.flip(fr, 1)
            # Swap left/right so the mirrored view feels natural.
            cam_panel = cv2.hconcat([fr, fl])
        else:
            cam_panel = cv2.hconcat([fl, fr])

        # FPS.
        now = time.monotonic()
        self._fps_ts.append(now)
        if len(self._fps_ts) >= 2:
            elapsed = self._fps_ts[-1] - self._fps_ts[0]
            self._fps = (len(self._fps_ts) - 1) / elapsed if elapsed > 0 else None

        # 3D skeleton.
        pts = None
        palm_xyz: np.ndarray | None = None
        has_hand = present and landmarks_3d is not None
        self._presence.append((now, has_hand))
        while self._presence and now - self._presence[0][0] > _PRESENCE_WINDOW_S:
            self._presence.popleft()
        if present and landmarks_3d is not None:
            alpha = self._cfg.smooth
            if self._smoothed is None:
                self._smoothed = landmarks_3d.copy()
            else:
                self._smoothed = alpha * landmarks_3d + (1 - alpha) * self._smoothed
            if self._cfg.recenter:
                self._update_recenter(now)
            centered = self._smoothed - self._origin
            pts = (_R @ centered.T).T
            palm_xyz = centered[_PALM_CENTER_IDX]
        elif self._cfg.recenter and self._smoothed is not None and not self._recently_absent():
            # Brief MediaPipe dropout — hold the last pose, countdown and origin alive so a
            # single missed frame doesn't restart the recenter timer.
            centered = self._smoothed - self._origin
            pts = (_R @ centered.T).T
            palm_xyz = centered[_PALM_CENTER_IDX]
        else:
            self._smoothed = None
            self._hold_start = None
            self._hold_anchor = None
            self._recentered = False
            self._calib_msg = None

        hand_panel = _render_hand_3d(
            pts,
            width=cam_panel.shape[1],
            mirror=self._cfg.mirror,
            fps=self._fps,
            palm_xyz=palm_xyz,
            calib_msg=self._calib_msg,
        )

        cv2.imshow(_WIN_NAME, cv2.vconcat([cam_panel, hand_panel]))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False
        if cv2.getWindowProperty(_WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def _recently_absent(self) -> bool:
        """True when the hand was missing for >10% of frames in the last 100 ms.

        Distinguishes a real hand loss (reset the recenter hold) from a single-frame
        MediaPipe miss (ignore). Call after the current frame's presence is recorded.
        """
        if not self._presence:
            return True
        present = sum(seen for _, seen in self._presence)
        return present / len(self._presence) < _PRESENCE_MIN_FRAC

    def _update_recenter(self, now: float) -> None:
        """Advance the hold-palm-open recenter gesture; sets ``self._origin`` on success.

        Assumes ``self._smoothed`` is set. Holding an open palm, square to the camera and
        still for :data:`_RECENTER_HOLD_S` seconds re-zeros the origin to the palm. The
        result latches until the pose is released, so it fires once per hold, not every
        frame.
        """
        assert self._smoothed is not None
        palm = self._smoothed[_PALM_CENTER_IDX]
        if not _palm_open_facing(self._smoothed):
            self._hold_start = None
            self._hold_anchor = None
            self._recentered = False  # pose released → re-arm for the next hold
            self._calib_msg = None
            return
        if self._recentered:
            self._calib_msg = "Recentered"
            return
        moved = (
            self._hold_anchor is not None
            and float(np.linalg.norm(palm - self._hold_anchor)) > _RECENTER_MOVE_TOL_M
        )
        if self._hold_start is None or moved:
            self._hold_anchor = palm.copy()
            self._hold_start = now
        remaining = _RECENTER_HOLD_S - (now - self._hold_start)
        if remaining <= 0:
            self._origin = palm.copy()
            self._recentered = True
            self._calib_msg = "Recentered"
        else:
            self._calib_msg = f"Calibrating... {math.ceil(remaining)}"

    def destroy(self) -> None:
        cv2.destroyAllWindows()
