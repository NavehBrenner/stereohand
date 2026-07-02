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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from stereohand.calibration import StereoCalibration
from stereohand.landmarker import HandLandmarks2D
from stereohand.triangulation import triangulate_points


def write_gif(
    path: str,
    frames: list[NDArray[np.uint8]],
    *,
    fps: float,
    max_width: int = 640,
    max_fps: float = 12.0,
) -> None:
    """Write BGR ``frames`` to an optimized GIF for README/markdown embeds.

    Subsamples to ``max_fps`` and downscales to ``max_width`` so the file stays
    small enough to commit and autoplay inline. ``frames`` are OpenCV BGR arrays.
    """
    import cv2
    from PIL import Image

    step = max(1, round(fps / max_fps))
    out_fps = fps / step
    images = []
    for frame in frames[::step]:
        height, width = frame.shape[:2]
        if width > max_width:
            frame = cv2.resize(frame, (max_width, round(height * max_width / width)))
        images.append(Image.fromarray(frame[:, :, ::-1]))  # BGR → RGB
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / out_fps),
        loop=0,
        optimize=True,
    )


FloatArray = NDArray[np.float64]

# Lazy import: only pulled in when render=True so headless stays cv2-free.
if TYPE_CHECKING:
    from stereohand.renderer import HandRenderer, RenderConfig
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
    def latest_pair_timestamp(self) -> float: ...


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
        max_fps: int | Literal["cam"] = "cam",
        *,
        rectify: bool = True,
        renderer: HandRenderer | None = None,
    ) -> None:
        self._calib = calibration
        self._capture = capture
        self._lm_left = landmarker_left
        self._lm_right = landmarker_right
        self._rectify = rectify
        self._maps: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None
        self._t0 = time.monotonic()
        self._last_timestamp_ms = -1  # strictly-increasing MediaPipe timestamp guard (see step())
        self._latest = _ABSENT
        self.last_frames: tuple[Frame, Frame] | None = None  # latest raw pair, for display
        self.last_processed_frames: tuple[Frame, Frame] | None = None  # post-rectify, to landmarker
        self.last_landmark_2d: tuple[HandLandmarks2D | None, HandLandmarks2D | None] | None = None
        self._renderer = renderer
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reading_ready = threading.Event()  # set on each publish; wakes the render loop
        self._thread: threading.Thread | None = None
        # Landmark both views concurrently: each view has its own detector, so the two
        # ~20 ms CPU inferences overlap instead of summing (≈25→40 fps on the step thread).
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stereo-lm")
        self._max_fps = max_fps

    @classmethod
    def open(
        cls,
        calibration: StereoCalibration,
        *,
        left: int | str = 0,
        right: int | str = 2,
        max_skew_s: float = 0.02,
        max_fps: int | Literal["cam"] = "cam",
        render: bool = False,
        render_config: RenderConfig | None = None,
        **landmarker_kwargs: Any,
    ) -> StereoHandTracker:
        """Live factory: real :class:`StereoCapture` + two :class:`HandLandmarker`s.

        Parameters
        ----------
        max_fps:
            Cap the background processing rate to this many frames/second; ``'cam'``
            (default) processes every new camera frame. A lower cap (e.g. 10) runs
            MediaPipe less often, freeing the GIL for a tight consumer loop.
        render:
            If ``True``, create a cv2 visualisation window.  The window is driven by
            :meth:`run` (blocking main-thread loop) or manually via :meth:`render_step`.
        render_config:
            Visualisation options (mirror, smoothing, …).  Ignored when *render* is
            ``False``.  Defaults to :class:`RenderConfig()` when *render* is ``True``.
        """
        from stereohand.capture import StereoCapture
        from stereohand.landmarker import HandLandmarker

        renderer: HandRenderer | None = None
        if render:
            from stereohand.renderer import HandRenderer as _HR
            from stereohand.renderer import RenderConfig as _RC

            renderer = _HR(render_config or _RC())

        capture = StereoCapture(left, right, max_skew_s=max_skew_s)
        return cls(
            calibration,
            capture,
            HandLandmarker(**landmarker_kwargs),
            HandLandmarker(**landmarker_kwargs),
            max_fps=max_fps,
            renderer=renderer,
        )

    def step(self) -> StereoHandReading:
        """One synchronous cycle: capture → rectify → landmark both → triangulate."""
        pair = self._capture.read()
        if pair is None:
            return self._publish(_ABSENT)
        left, right = pair
        self.last_frames = (left, right)
        if self._rectify:
            if self._maps is None:
                self._maps = self._calib.rectification_maps()
            left, right = self._calib.rectify_pair(left, right, self._maps)
        self.last_processed_frames = (left, right)

        # MediaPipe's detect_for_video requires *strictly* increasing timestamps. int(...*1000)
        # truncates to whole milliseconds, so two step()s in the same millisecond (bursty local
        # USB capture — seen on Windows) would submit a duplicate and raise "Input timestamp must
        # be monotonically increasing". Clamp to at least last+1 so the sequence never stalls.
        timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        fut_right = self._pool.submit(self._lm_right.process, right, timestamp_ms)
        landmarks_left = self._lm_left.process(left, timestamp_ms)
        landmarks_right = fut_right.result()
        self.last_landmark_2d = (landmarks_left, landmarks_right)
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
        self._reading_ready.set()
        return reading

    def read(self) -> StereoHandReading:
        """Latest reading (non-blocking). Lazily starts the background processing thread."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="stereo-hand", daemon=True)
            self._thread.start()
        with self._lock:
            return self._latest

    def _run(self) -> None:
        # Event-driven, with an optional rate cap. Only run the capture→landmark→triangulate
        # cycle when the cameras have delivered a *new* frame pair — without this the loop
        # spins MediaPipe over the same stored frames far faster than the ~30 fps cameras
        # produce them, wasted CPU that holds the GIL and starves a tight GIL-bound consumer
        # (the teleop control loop drops to ~0.56x real-time). When `max_fps` is an int, also
        # cap processing to that rate (e.g. 10 fps even if the cameras run 30) to shed still
        # more GIL pressure; 'cam' means no cap. A 1 ms poll sits well under the frame interval.
        min_interval = 0.0 if self._max_fps == "cam" else 1.0 / self._max_fps
        last_timestamp = -1.0
        last_processed = 0.0
        while not self._stop.is_set():
            timestamp = self._capture.latest_pair_timestamp()
            now = time.monotonic()
            if timestamp <= last_timestamp or now - last_processed < min_interval:
                time.sleep(0.001)
                continue
            last_timestamp = timestamp
            last_processed = now
            self.step()

    # -- Visualisation (main-thread) ----------------------------------------

    def render_step(self) -> np.ndarray | None:
        """Draw the latest state once. Pair with the renderer's ``poll()`` for responsiveness.

        Must be called from the **main thread** (cv2 GUI requirement).  The background
        tracker thread keeps running; this just visualises the latest state.

        Returns the composite BGR frame shown in the window, or ``None`` if no data yet.
        """
        if self._renderer is None:
            raise RuntimeError("render_step() requires render=True in StereoHandTracker.open()")
        reading = self.read()  # also starts the background thread on first call
        return self._renderer.step(
            frames=self.last_processed_frames,
            landmarks_2d=self.last_landmark_2d,
            landmarks_3d=reading.landmarks if reading.present else None,
            present=reading.present,
        )

    def poll(self) -> bool:
        """Pump the cv2 GUI once; returns ``False`` when the user closed the window / hit 'q'.

        The cheap counterpart to :meth:`render_step`: ``render_step`` draws (only worth doing
        on a new reading), ``poll`` flushes the imshow buffer to screen and services window
        events (so the actual paint happens here). An external main-thread loop that drives its
        own pacing — rather than calling :meth:`run` — should call this every iteration to keep
        the window painted and responsive. Must be called from the **main thread**.
        """
        if self._renderer is None:
            raise RuntimeError("poll() requires render=True in StereoHandTracker.open()")
        return self._renderer.poll()

    def set_renderer_origin(self, origin: tuple[float, float, float]) -> None:
        if self._renderer is None:
            raise RuntimeError(
                "set_renderer_origin() requires render=True in StereoHandTracker.open()"
            )
        self._renderer.set_render_origin(origin)

    def run(self, *, record_path: str | None = None) -> None:
        """Blocking main-thread loop: read + render until the user quits.

        Convenience wrapper around :meth:`render_step` — call this from ``main()`` and
        forget about the loop.

        Parameters
        ----------
        record_path:
            If given, write the composite window output to this file. A ``.gif``
            extension produces an optimized GIF (downscaled, ~12 fps) for inline
            markdown embeds; any other extension is a video — ``.mp4`` uses H.264
            (``avc1``), everything else falls back to MJPEG. Recording runs
            alongside the live display and stops when the user quits.
        """
        if self._renderer is None:
            raise RuntimeError("run() requires render=True in StereoHandTracker.open()")
        renderer = self._renderer
        import sys
        import time as _time

        import cv2  # noqa: E402 — lazy; already loaded because render=True

        _FPS_WARMUP = 10  # frames to buffer before measuring the real render rate

        def _open_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
            """Try several codecs in order; return the first that opens successfully."""
            codecs = ["mp4v", "avc1", "XVID"] if path.endswith(".mp4") else ["MJPG", "XVID"]
            for tag in codecs:
                fourcc = cv2.VideoWriter.fourcc(*tag)
                writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
                if writer.isOpened():
                    print(
                        f"Recording → {path}  (codec {tag}, {fps:.1f} fps)",
                        file=sys.stderr,
                    )
                    return writer
                writer.release()
            raise RuntimeError(f"Could not open VideoWriter for {path!r}: tried codecs {codecs}")

        is_gif = record_path is not None and record_path.lower().endswith(".gif")
        _GIF_FPS = 12.0  # gif target rate; subsampled live to keep RAM + file small
        gif_last_kept = 0.0

        writer: cv2.VideoWriter | None = None
        # Buffer early frames until we can measure the actual render rate; without this
        # the VideoWriter gets a hard-coded FPS that never matches reality, producing
        # slow-motion or fast-forward playback.
        frame_buffer: list[np.ndarray] = []
        frame_times: list[float] = []
        self.read()  # start the background thread so publishes (and the wake event) flow
        try:
            while True:
                # Redraw only on a new reading; poll() pumps the cv2 GUI every iteration so
                # the window stays responsive (repaint, close/quit) even when the feed
                # stalls. The wait timeout bounds that polling rate when no new frame arrives.
                if self._reading_ready.wait(timeout=0.1):
                    self._reading_ready.clear()
                    composite = self.render_step()
                    if composite is not None and is_gif:
                        # ponytail: time-gate to ~12fps so the in-RAM frame buffer
                        # stays small. Buffers in memory — fine for short demo
                        # clips; stream to disk if you ever record minutes.
                        now = _time.monotonic()
                        if now - gif_last_kept >= 1.0 / _GIF_FPS:
                            frame_buffer.append(composite)
                            gif_last_kept = now
                    elif composite is not None and record_path is not None:
                        if writer is None:
                            # Still warming up — buffer frames and timestamps.
                            frame_buffer.append(composite)
                            frame_times.append(_time.monotonic())
                            if len(frame_buffer) >= _FPS_WARMUP:
                                elapsed = frame_times[-1] - frame_times[0]
                                measured_fps = (
                                    (len(frame_times) - 1) / elapsed if elapsed > 0 else 20.0
                                )
                                height, width = composite.shape[:2]
                                writer = _open_writer(record_path, width, height, measured_fps)
                                for buffered_frame in frame_buffer:
                                    writer.write(buffered_frame)
                                frame_buffer.clear()
                                frame_times.clear()
                        else:
                            writer.write(composite)
                if not renderer.poll():
                    break
        finally:
            if is_gif and frame_buffer and record_path is not None:
                write_gif(record_path, frame_buffer, fps=_GIF_FPS)
                print(f"Saved recording → {record_path}", file=sys.stderr)
            elif writer is not None:
                writer.release()
                print(f"Saved recording → {record_path}", file=sys.stderr)
            elif frame_buffer and record_path is not None:
                # Quit before warmup finished — flush what we have at a default rate.
                height, width = frame_buffer[0].shape[:2]
                elapsed = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0
                fps = (len(frame_times) - 1) / elapsed if elapsed > 0 else 20.0
                writer = _open_writer(record_path, width, height, fps)
                for buffered_frame in frame_buffer:
                    writer.write(buffered_frame)
                writer.release()
                print(f"Saved recording → {record_path}", file=sys.stderr)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._pool.shutdown(wait=True)
        self._capture.close()
        self._lm_left.close()
        self._lm_right.close()
        if self._renderer is not None:
            self._renderer.destroy()

    def __enter__(self) -> StereoHandTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
