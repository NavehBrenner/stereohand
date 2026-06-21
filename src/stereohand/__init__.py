"""stereohand — two webcams to metric 3D hand landmarks.

Pipeline: synchronized dual-webcam capture -> one-time ChArUco stereo calibration ->
per-view MediaPipe Tasks HandLandmarker -> linear-DLT triangulation -> (21, 3) metric
landmarks.

Typical use::

    from stereohand import StereoHandTracker, StereoCalibration

    calib = StereoCalibration.load("stereo_calib.json")
    with StereoHandTracker.open(calib, left=0, right=2) as tracker:
        reading = tracker.read()          # non-blocking
        if reading.present:
            print(reading.landmarks)      # (21, 3) metric xyz
"""

from __future__ import annotations

from stereohand.board import BOARD, CharucoBoardSpec, make_board, render_board
from stereohand.calibration import StereoCalibration, calibrate_from_charuco, live_calibrate
from stereohand.capture import StereoCapture
from stereohand.landmarker import HandLandmarker, HandLandmarks2D, draw_landmarks_on_frame
from stereohand.renderer import RenderConfig
from stereohand.tracker import StereoHandReading, StereoHandTracker
from stereohand.triangulation import triangulate_points

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "StereoHandTracker",
    "StereoHandReading",
    "StereoCalibration",
    "RenderConfig",
    "calibrate_from_charuco",
    "live_calibrate",
    "StereoCapture",
    "HandLandmarker",
    "HandLandmarks2D",
    "draw_landmarks_on_frame",
    "triangulate_points",
    "BOARD",
    "CharucoBoardSpec",
    "make_board",
    "render_board",
]
