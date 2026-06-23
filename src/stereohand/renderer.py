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
# 3D rendering constants
# ---------------------------------------------------------------------------

# Map OpenCV camera coords (x=right, y=down, z=forward) to a Z-up world.
#   world_x = cam_x   (left-right)
#   world_y = cam_z   (depth, into screen)
#   world_z = -cam_y  (up: cam y is down, so negate)
# The orbiting projection below expects this Z-up world (azimuth turns around Z).
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
# A recenter hold survives brief MediaPipe dropouts: hold the last pose (and keep the
# countdown running) until the hand has been gone this long. Wall-clock based, so the
# tolerance is the same at 10 fps or 30 fps — a longer gap counts as the hand leaving.
_DROPOUT_GRACE_S = 0.4
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


# Light background, gray joints and red/blue/black axis gizmo — matches the reference
# viewer in project-wiki/raw/visualize_3d_pose_rt.py.
_BG_COLOR = (240, 240, 240)
_JOINT_COLOR = (80, 80, 80)
_AXIS_COLORS = [
    (0, 0, 255),  # X → red
    (255, 0, 0),  # Y → blue
    (0, 0, 0),  # Z → black
]
_AXIS_LEN_M = 0.12  # reference axes at the world origin (display units, tune to taste)
# Shrink the hand and pull it proportionally toward the origin so it reads as a small shape
# next to the (larger) axis gizmo instead of floating far away. Display-only — the Palm XYZ
# HUD still reports true metric position. Lower = smaller/closer.
_HAND_DISPLAY_SCALE = 0.4

# Orbit camera defaults. Zoom is px-per-metre; hand ≈15 cm, ~1400 px/m fills the panel.
_DEFAULT_AZIM = math.radians(45.0)
_DEFAULT_ELEV = math.radians(25.0)
_DEFAULT_ZOOM = 1400.0
_ZOOM_MIN, _ZOOM_MAX = 300.0, 8000.0

# HUD styling — dark text reads on the light background.
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.6
_HUD_COLOR = (60, 60, 60)
_HUD_THICK = 1


def _project(
    pt: np.ndarray, width: int, height: int, azim: float, elev: float, zoom: float, xsign: int
) -> tuple[int, int]:
    """Weak-perspective (orthographic + zoom) projection of a Z-up world point, with orbit.

    The camera orbits the world origin by ``azim`` (around Z) then ``elev`` (tilt around
    X'), exactly as in the reference viewer. ``xsign`` flips horizontally for mirror mode.
    """
    ca, sa = math.cos(azim), math.sin(azim)
    ce, se = math.cos(elev), math.sin(elev)
    # Azimuth around Z-up.
    x1 = ca * pt[0] + sa * pt[1]
    y1 = -sa * pt[0] + ca * pt[1]
    z1 = pt[2]
    # Elevation tilt around the new X axis (only z2 is needed for the screen-y).
    z2 = se * y1 + ce * z1
    sx = int(width / 2 + xsign * x1 * zoom)
    sy = int(height / 2 - z2 * zoom)
    return sx, sy


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
    azim: float = _DEFAULT_AZIM,
    elev: float = _DEFAULT_ELEV,
    zoom: float = _DEFAULT_ZOOM,
    fps: float | None = None,
    palm_xyz: np.ndarray | None = None,
    calib_msg: str | None = None,
) -> np.ndarray:
    """Orbiting weak-perspective view of the world; drag to orbit, scroll to zoom.

    Reproduces the reference viewer (Z-up world, orbit camera, light background, gray
    joints, red/blue/black axis gizmo). Finger colours are stereohand's, not the
    reference's.
    """
    canvas = np.full((height, width, 3), _BG_COLOR, np.uint8)
    cy = height // 2

    # Sign for horizontal direction: mirroring negates X on screen.
    xsign = -1 if mirror else 1

    def proj(world_pt: np.ndarray) -> tuple[int, int]:
        return _project(world_pt, width, height, azim, elev, zoom, xsign)

    # --- World-origin axes (always drawn, even when no hand) ---
    origin = proj(np.zeros(3))
    axis_ends = [
        np.array([_AXIS_LEN_M, 0.0, 0.0]),  # X
        np.array([0.0, _AXIS_LEN_M, 0.0]),  # Y
        np.array([0.0, 0.0, _AXIS_LEN_M]),  # Z
    ]
    for label, end, color in zip("XYZ", axis_ends, _AXIS_COLORS, strict=True):
        cv2.line(canvas, origin, proj(end), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, proj(end * 1.25), _HUD_FONT, 0.45, color, 1, cv2.LINE_AA)

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
        screen = [proj(pts[i] * _HAND_DISPLAY_SCALE) for i in range(21)]
        for finger, color in zip(_FINGERS, _FINGER_COLORS_BGR, strict=True):
            for a, b in finger:
                cv2.line(canvas, screen[a], screen[b], color, 3, cv2.LINE_AA)
        # Joint dots for visibility.
        for pt in screen:
            cv2.circle(canvas, pt, 4, _JOINT_COLOR, -1, cv2.LINE_AA)

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
            (0, 0, 200),
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
        self._last_seen_t: float | None = None  # monotonic time of the last good detection
        # Orbit-camera state (mouse-driven, see _mouse_cb).
        self._azim = _DEFAULT_AZIM
        self._elev = _DEFAULT_ELEV
        self._zoom = _DEFAULT_ZOOM
        self._dragging = False
        self._last_mx = 0
        self._last_my = 0
        cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN_NAME, self._mouse_cb)

    # -- public interface ---------------------------------------------------

    def step(
        self,
        *,
        frames: tuple[np.ndarray, np.ndarray] | None,
        landmarks_2d: tuple[HandLandmarks2D | None, HandLandmarks2D | None] | None,
        landmarks_3d: np.ndarray | None,
        present: bool,
    ) -> None:
        """Draw one composite frame. Call only on new data; :meth:`poll` keeps the window live."""
        from stereohand.landmarker import draw_landmarks_on_frame

        if frames is None:
            return

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

        # FPS — render rate, which run() syncs to the data-arrival rate.
        now = time.monotonic()
        self._fps_ts.append(now)
        if len(self._fps_ts) >= 2:
            elapsed = self._fps_ts[-1] - self._fps_ts[0]
            self._fps = (len(self._fps_ts) - 1) / elapsed if elapsed > 0 else None

        # 3D skeleton (cv2-free; testable headless).
        pts, palm_xyz = self._advance_pose(now, present, landmarks_3d)

        hand_panel = _render_hand_3d(
            pts,
            width=cam_panel.shape[1],
            mirror=self._cfg.mirror,
            azim=self._azim,
            elev=self._elev,
            zoom=self._zoom,
            fps=self._fps,
            palm_xyz=palm_xyz,
            calib_msg=self._calib_msg,
        )

        cv2.imshow(_WIN_NAME, cv2.vconcat([cam_panel, hand_panel]))

    def poll(self) -> bool:
        """Pump the cv2 GUI once (repaint, events). Returns ``False`` when the user closes
        the window / presses 'q' — call every loop iteration to stay responsive between draws.
        """
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False
        return cv2.getWindowProperty(_WIN_NAME, cv2.WND_PROP_VISIBLE) >= 1

    def _advance_pose(
        self, now: float, present: bool, landmarks_3d: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Update smoothing + recenter and return ``(world_pts, palm_xyz)`` for drawing.

        cv2-free so the recenter/dropout state machine can be exercised headlessly.
        """
        if present and landmarks_3d is not None:
            self._last_seen_t = now
            alpha = self._cfg.smooth
            if self._smoothed is None:
                self._smoothed = landmarks_3d.copy()
            else:
                self._smoothed = alpha * landmarks_3d + (1 - alpha) * self._smoothed
            if self._cfg.recenter:
                self._update_recenter(now)
        elif (
            self._cfg.recenter
            and self._smoothed is not None
            and self._last_seen_t is not None
            and now - self._last_seen_t <= _DROPOUT_GRACE_S
        ):
            # Brief MediaPipe dropout — hold the last pose and keep the recenter countdown
            # advancing so a few missed frames don't restart the timer.
            self._update_recenter(now)
        else:
            self._smoothed = None
            self._hold_start = None
            self._hold_anchor = None
            self._recentered = False
            self._calib_msg = None
            self._last_seen_t = None
            return None, None

        centered = self._smoothed - self._origin
        return (_R @ centered.T).T, centered[_PALM_CENTER_IDX]

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Left-drag orbits (azimuth/elevation); scroll zooms — same feel as the reference."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_mx, self._last_my = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._azim += (x - self._last_mx) * 0.005
            self._elev += (y - self._last_my) * 0.005
            self._elev = max(-math.pi / 2, min(math.pi / 2, self._elev))
            self._last_mx, self._last_my = x, y
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._zoom *= 1.1 if flags > 0 else 1 / 1.1
            self._zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self._zoom))

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

    def set_render_origin(self, new_origin: tuple[float, float, float]) -> None:
        self._origin = np.asarray(new_origin, dtype=float)

    def destroy(self) -> None:
        cv2.destroyAllWindows()
