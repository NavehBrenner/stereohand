"""stereohand — two webcams to metric 3D hand landmarks.

Pipeline: synchronized dual-webcam capture -> one-time ChArUco stereo calibration ->
per-view MediaPipe Tasks HandLandmarker -> linear-DLT triangulation -> (21, 3) metric
landmarks. The public API (``StereoHandTracker`` etc.) lands in later issues; for now this
package only declares its version.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
