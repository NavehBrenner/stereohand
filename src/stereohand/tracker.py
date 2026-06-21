"""``StereoHandTracker`` — the headline API: two webcams → metric 3D hand landmarks.

Ties the pieces together: a synced frame pair (:mod:`stereohand.capture`) is rectified
(:class:`stereohand.calibration.StereoCalibration`), landmarked in **both** views
(:mod:`stereohand.landmarker`), and the trivially-corresponded 21 landmarks are
triangulated (:mod:`stereohand.triangulation`) into ``(21, 3)`` metric coordinates.

The capture/landmarker dependencies are injected, so the whole capture→triangulate seam is
testable with fakes (no cameras, no model). :meth:`StereoHandTracker.open` is the live
factory that wires up the real components.

Output stays generic — just ``(21, 3)`` landmarks + presence + handedness. No robot, no
smoothing opinions: a consumer (e.g. a teleop layer) adds those on top.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from stereohand.calibration import StereoCalibration
from stereohand.landmarker import HandLandmarks2D
from stereohand.triangulation import triangulate_points

FloatArray = NDArray[np.float64]
Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class StereoHandReading:
    """One frame of stereo hand sensing.

    ``landmarks`` is ``(21, 3)`` metric xyz (same units as the calibration board, i.e.
    metres) in the rectified left camera's frame. When ``present`` is ``False`` the hand was
    missing/low-confidence in at least one view and ``landmarks`` is all-zero.
    """

    landmarks: FloatArray = field(default_factory=lambda: np.zeros((21, 3)))
    present: bool = False
    handedness: str = ""


_ABSENT = StereoHandReading()


class CaptureLike(Protocol):
    def read(self) -> tuple[Frame, Frame] | None: ...
    def close(self) -> None: ...


class LandmarkerLike(Protocol):
    def process(self, frame_bgr: Frame, timestamp_ms: int) -> HandLandmarks2D | None: ...
    def close(self) -> None: ...


class StereoHandTracker:
    """Two webcams → metric 3D hand. Inject components, or use :meth:`open` for the live rig.

    Call :meth:`step` for one synchronous cycle, or :meth:`read` for the latest reading off a
    background thread (non-blocking — for a fast consumer loop that must not stall on the
    camera). ``rectify=False`` skips rectification when fed already-rectified frames.
    """

    def __init__(
        self,
        calibration: StereoCalibration,
        capture: CaptureLike,
        landmarker_left: LandmarkerLike,
        landmarker_right: LandmarkerLike,
        *,
        rectify: bool = True,
    ) -> None:
        self._calib = calibration
        self._capture = capture
        self._lm_left = landmarker_left
        self._lm_right = landmarker_right
        self._rectify = rectify
        self._maps: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None
        self._t0 = time.monotonic()
        self._latest = _ABSENT
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def open(
        cls,
        calibration: StereoCalibration,
        *,
        left: int | str = 0,
        right: int | str = 2,
        max_skew_s: float = 0.02,
        **landmarker_kwargs: Any,
    ) -> StereoHandTracker:
        """Live factory: real :class:`StereoCapture` + two :class:`HandLandmarker`s."""
        from stereohand.capture import StereoCapture
        from stereohand.landmarker import HandLandmarker

        capture = StereoCapture(left, right, max_skew_s=max_skew_s)
        return cls(
            calibration,
            capture,
            HandLandmarker(**landmarker_kwargs),
            HandLandmarker(**landmarker_kwargs),
        )

    def step(self) -> StereoHandReading:
        """One synchronous cycle: capture → rectify → landmark both → triangulate."""
        pair = self._capture.read()
        if pair is None:
            return self._publish(_ABSENT)
        left, right = pair
        if self._rectify:
            if self._maps is None:
                self._maps = self._calib.rectification_maps()
            left, right = self._calib.rectify_pair(left, right, self._maps)

        timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        landmarks_left = self._lm_left.process(left, timestamp_ms)
        landmarks_right = self._lm_right.process(right, timestamp_ms)
        # Drop-out if the hand is missing in *either* view — can't triangulate from one.
        if landmarks_left is None or landmarks_right is None:
            return self._publish(_ABSENT)

        points_3d = triangulate_points(
            self._calib.P1, self._calib.P2, landmarks_left.landmarks, landmarks_right.landmarks
        )
        return self._publish(
            StereoHandReading(
                landmarks=points_3d, present=True, handedness=landmarks_left.handedness
            )
        )

    def _publish(self, reading: StereoHandReading) -> StereoHandReading:
        with self._lock:
            self._latest = reading
        return reading

    def read(self) -> StereoHandReading:
        """Latest reading (non-blocking). Lazily starts the background processing thread."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="stereo-hand", daemon=True)
            self._thread.start()
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            self.step()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._capture.close()
        self._lm_left.close()
        self._lm_right.close()

    def __enter__(self) -> StereoHandTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
