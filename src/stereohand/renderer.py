"""Built-in visualisation for :class:`~stereohand.tracker.StereoHandTracker`.

All rendering is pure cv2 (no matplotlib) — typically ~1 ms per composite frame.  The
``mirror`` flag flips the view horizontally so the skeleton moves like a mirror image of
the user's hand (left/right inverted).

This module is imported lazily (only when ``render=True``) so headless deployments never
pull in cv2 at import time.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from stereohand.landmarker import HandLandmarks2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RenderConfig:
    """Configuration knobs for the live visualisation window.

    Attributes:
        mirror:  Flip the view horizontally so it acts like a mirror.
        smooth:  EMA alpha for temporal smoothing (1 = no smoothing, 0.1 = very smooth).
    """

    mirror: bool = False
    smooth: float = 0.5


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
# handpose3d colors in BGR; its "black" index finger → white so it shows on black.
_FINGER_COLORS_BGR = [
    (0, 0, 255),  # pinky  → red
    (255, 0, 0),  # ring   → blue
    (0, 200, 0),  # middle → green
    (240, 240, 240),  # index  → white (handpose3d uses black on a white bg)
    (0, 165, 255),  # thumb  → orange
]

# Palm-center landmark index: MediaPipe index 9 = middle-finger MCP, the geometric
# centre of the palm.  Index 0 is the wrist, which sits at the base.
_PALM_CENTER_IDX = 9

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
        if present and landmarks_3d is not None:
            alpha = self._cfg.smooth
            if self._smoothed is None:
                self._smoothed = landmarks_3d.copy()
            else:
                self._smoothed = alpha * landmarks_3d + (1 - alpha) * self._smoothed
            pts = (_R @ self._smoothed.T).T
            palm_xyz = self._smoothed[_PALM_CENTER_IDX]
        else:
            self._smoothed = None

        hand_panel = _render_hand_3d(
            pts,
            width=cam_panel.shape[1],
            mirror=self._cfg.mirror,
            fps=self._fps,
            palm_xyz=palm_xyz,
        )

        cv2.imshow(_WIN_NAME, cv2.vconcat([cam_panel, hand_panel]))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False
        if cv2.getWindowProperty(_WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        return True

    def destroy(self) -> None:
        cv2.destroyAllWindows()
