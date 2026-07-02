"""Synchronized dual-webcam capture.

The hand moves, so the two views must be grabbed near-simultaneously or triangulation
fuses two different poses. We do best-effort **software sync**: each camera is grabbed on
its own daemon thread with a capture timestamp, and :meth:`StereoCapture.read` only returns
a pair whose timestamps are within ``max_skew_s`` — otherwise the pair is dropped. Hardware
genlock would be overkill for approximate teleop.

The threading mirrors the ai-teleop ``hand_tracker`` pattern (daemon grabber + lock + stop
event); ``read`` is non-blocking so a fast consumer loop never stalls on a ~30 fps camera.
``cv2`` is imported lazily so the pure skew predicate (and its test) need no OpenCV.

The mismatched-camera reality (a laptop cam + a separate webcam) is fine for the geometry,
but rolling-shutter differences make sync the hard part — ``max_skew_s`` is the knob.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


def within_skew(timestamp_left: float, timestamp_right: float, max_skew_s: float) -> bool:
    """True if the two capture timestamps are close enough to treat as simultaneous."""
    return abs(timestamp_left - timestamp_right) <= max_skew_s


def open_capture(source: int | str) -> Any:
    """Open a camera. For integer indices (local USB cameras) request **MJPG**: two webcams on
    one USB controller overrun its bandwidth on raw YUYV (~10x the bytes) and one camera blacks
    out / stalls — MJPG is compressed and lets both stream. On Windows also force DirectShow
    (the default MSMF backend stalls ~20s per camera). URLs (the WSL bridge) and non-int sources
    use the default backend untouched. Returns a ``cv2.VideoCapture``."""
    import cv2

    if isinstance(source, int):
        capture = (
            cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if sys.platform == "win32"
            else cv2.VideoCapture(source)
        )
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
        return capture
    return cv2.VideoCapture(source)


class _CameraThread:
    """Background grabber for one camera: keeps only the latest (timestamp, frame)."""

    def __init__(self, source: int | str, name: str) -> None:
        self._capture = open_capture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera source {source!r}")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._timestamp = 0.0
        self._frame: Frame | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                continue
            with self._lock:
                self._timestamp = time.monotonic()
                self._frame = frame

    def latest(self) -> tuple[float, Frame | None]:
        with self._lock:
            return self._timestamp, self._frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._capture.release()


class StereoCapture:
    """Two webcams → time-aligned BGR frame pairs (software-synced, non-blocking).

    Parameters
    ----------
    left, right:
        OpenCV camera sources — an int device index, or a string (stream URL / device path;
        on WSL2, where host webcams have no UVC driver, a stream URL is required).
    max_skew_s:
        Maximum capture-time difference (seconds) for a pair to be delivered. Tune up for
        mismatched / rolling-shutter cameras, down for tighter sync.
    max_age_s:
        Reject a pair if either frame is older than this (a camera stalled).
    """

    def __init__(
        self,
        left: int | str,
        right: int | str,
        *,
        max_skew_s: float = 0.02,
        max_age_s: float = 0.5,
    ) -> None:
        self.max_skew_s = max_skew_s
        self.max_age_s = max_age_s
        self._left = _CameraThread(left, "stereo-capture-left")
        self._right = _CameraThread(right, "stereo-capture-right")
        self.last_skew_s: float | None = None

    def read(self) -> tuple[Frame, Frame] | None:
        """Latest synced BGR pair, or ``None`` if not ready / over-skew / stale.

        Non-blocking. Updates :attr:`last_skew_s` for monitoring even when it rejects.
        """
        ts_left, frame_left = self._left.latest()
        ts_right, frame_right = self._right.latest()
        if frame_left is None or frame_right is None:
            return None
        self.last_skew_s = abs(ts_left - ts_right)
        now = time.monotonic()
        if now - ts_left > self.max_age_s or now - ts_right > self.max_age_s:
            return None
        if not within_skew(ts_left, ts_right, self.max_skew_s):
            return None
        return frame_left, frame_right

    def latest_pair_timestamp(self) -> float:
        """Newest capture time across both cameras (non-blocking peek).

        Lets a consumer run *event-driven* — process only when a fresh frame has
        landed — instead of busy-spinning over the same stored pair. Returns 0.0
        before either camera has delivered its first frame.
        """
        ts_left, _ = self._left.latest()
        ts_right, _ = self._right.latest()
        return max(ts_left, ts_right)

    def close(self) -> None:
        self._left.close()
        self._right.close()

    def __enter__(self) -> StereoCapture:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
