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
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


def within_skew(timestamp_left: float, timestamp_right: float, max_skew_s: float) -> bool:
    """True if the two capture timestamps are close enough to treat as simultaneous."""
    return abs(timestamp_left - timestamp_right) <= max_skew_s


#: Why a pair was or wasn't delivered. These are **not** interchangeable to a consumer:
#: ``stale`` means a camera really has stopped producing frames, while ``over_skew`` means
#: only that the two free-running cameras are momentarily out of phase — it says nothing
#: about what is in front of them. Collapsing the two is what made a hand that never left
#: frame look like it vanished twice a second (see :func:`pair_status`).
ReadStatus = Literal["ok", "not_ready", "stale", "over_skew"]


def pair_status(
    timestamp_left: float,
    timestamp_right: float,
    now: float,
    *,
    max_skew_s: float,
    max_age_s: float,
) -> ReadStatus:
    """The single pairing decision, as a pure function of timestamps.

    Both :meth:`StereoCapture.read` and :meth:`StereoCapture.latest_pair_timestamp` route
    through this, and that is the point rather than an implementation detail. They used to
    apply *different* rules — ``read`` required both frames within ``max_skew_s`` while
    ``latest_pair_timestamp`` reported ``max(ts_left, ts_right)``, which advances whenever
    **either** camera ticks. A consumer that stepped on the latter and then found the former
    returning ``None`` could not tell "briefly out of phase" from "no hand", and reported the
    first as the second. One shared predicate makes that class of disagreement impossible.

    Callers pass ``now`` rather than reading the clock here so a caller can classify one
    instant consistently, and so the whole rule is testable without cameras or a clock.
    """
    # Staleness is tested first and deliberately outranks skew: a camera that stopped
    # delivering also drifts out of skew, and reporting *that* as ``over_skew`` would tell
    # the consumer to hold its last reading forever. ``stale`` is the backstop that makes
    # holding on ``over_skew`` safe, so it has to win whenever both are true.
    if now - timestamp_left > max_age_s or now - timestamp_right > max_age_s:
        return "stale"
    if not within_skew(timestamp_left, timestamp_right, max_skew_s):
        return "over_skew"
    return "ok"


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
        Maximum capture-time difference (seconds) for a pair to be delivered — an
        *alignment-quality* knob: how simultaneous the two views must be before triangulating
        them is meaningful. Tune up for mismatched / rolling-shutter cameras, down for
        tighter sync. It is **not** a drop-out remedy: a consumer seeing frequent absences
        should check that it distinguishes :data:`ReadStatus` values rather than widening
        this, which only trades triangulation accuracy for the appearance of a fix.
    max_age_s:
        Reject a pair if either frame is older than this (a camera stalled). This is the
        backstop that makes it safe for a consumer to hold its last reading through an
        ``over_skew`` miss, so it outranks skew in :func:`pair_status`.
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
        #: Why the most recent :meth:`read` returned what it did. Lets a consumer tell a
        #: transient pairing miss from a camera that has actually died — see
        #: :data:`ReadStatus`. Monitoring-style public attribute, like ``last_skew_s``.
        self.last_read_status: ReadStatus = "not_ready"

    def _status(self) -> tuple[ReadStatus, float, Frame | None, float, Frame | None]:
        """Classify the current pair once, and return it with the frames it classified."""
        ts_left, frame_left = self._left.latest()
        ts_right, frame_right = self._right.latest()
        if frame_left is None or frame_right is None:
            return "not_ready", ts_left, frame_left, ts_right, frame_right
        self.last_skew_s = abs(ts_left - ts_right)
        status = pair_status(
            ts_left,
            ts_right,
            time.monotonic(),
            max_skew_s=self.max_skew_s,
            max_age_s=self.max_age_s,
        )
        return status, ts_left, frame_left, ts_right, frame_right

    def read(self) -> tuple[Frame, Frame] | None:
        """Latest synced BGR pair, or ``None`` if not ready / over-skew / stale.

        Non-blocking. Updates :attr:`last_skew_s` for monitoring even when it rejects, and
        :attr:`last_read_status` with *why* — a caller that treats every ``None`` alike will
        report a momentary phase miss as a missing hand.
        """
        status, _, frame_left, _, frame_right = self._status()
        self.last_read_status = status
        if status != "ok" or frame_left is None or frame_right is None:
            return None
        return frame_left, frame_right

    def latest_pair_timestamp(self) -> float | None:
        """Capture time of a pair :meth:`read` would deliver, or ``None`` if there isn't one.

        Lets a consumer run *event-driven* — process only when a fresh, **usable** pair has
        landed — instead of busy-spinning over the same stored pair. The "usable" is what
        makes it safe: this used to return ``max(ts_left, ts_right)`` regardless, which
        advances whenever either camera ticks, so a consumer woke up on pairs that
        :meth:`read` then rejected. Both now route through :func:`pair_status`.

        Note this is still a separate sample from :meth:`read`'s — a camera can tick in
        between and flip the verdict. That race is rare and, by design, harmless: the
        consumer distinguishes the outcomes via :attr:`last_read_status` rather than relying
        on this to have predicted them.
        """
        status, ts_left, _, ts_right, _ = self._status()
        return max(ts_left, ts_right) if status == "ok" else None

    def close(self) -> None:
        # Each camera's close() can block for ~1s (thread join timeout) plus whatever
        # cv2.VideoCapture.release() takes on this backend/OS — DirectShow release is
        # slow on Windows. Closing sequentially doubles that wait; run them concurrently
        # so total shutdown time is the slower of the two, not the sum.
        threads = [threading.Thread(target=camera.close) for camera in (self._left, self._right)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def __enter__(self) -> StereoCapture:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
