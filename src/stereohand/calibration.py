"""Stereo calibration: the persisted geometry that turns image points into metric 3D.

Two clearly separated halves:

- :class:`StereoCalibration` — a **cv2-free** value object: per-camera intrinsics + the
  stereo extrinsics and rectification outputs, with JSON ``save``/``load``. Being cv2-free,
  it (and its round-trip) unit-tests in CI without OpenCV. ``rectification_maps`` /
  ``rectify_pair`` lazily import cv2 — the heavy remap tables are recomputed on load, not
  persisted.
- :func:`calibrate_from_charuco` — the **cv2-gated** builder: detect the ChArUco board
  (``board.BOARD``) in collected frame pairs, solve per-camera intrinsics, then
  ``stereoCalibrate`` + ``stereoRectify`` to fill a :class:`StereoCalibration`.

The projection matrices ``P1``/``P2`` here are exactly what :mod:`stereohand.triangulation`
consumes. Calibration is valid only until the rig is physically disturbed.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from stereohand.board import BOARD, CharucoBoardSpec, make_board

FloatArray = NDArray[np.float64]


class StereoCalibrationDict(TypedDict):
    """On-disk JSON shape (arrays as nested lists)."""

    image_size: list[int]
    camera_matrix_left: list[list[float]]
    dist_left: list[float]
    camera_matrix_right: list[list[float]]
    dist_right: list[float]
    R: list[list[float]]
    T: list[float]
    R1: list[list[float]]
    R2: list[list[float]]
    P1: list[list[float]]
    P2: list[list[float]]
    Q: list[list[float]]
    rms: float


@dataclass(frozen=True)
class StereoCalibration:
    """Resolved stereo geometry. All matrices NumPy ``float64``; lengths in metres.

    ``R``/``T`` are the right camera's pose relative to the left (from
    ``stereoCalibrate``); ``R1``/``R2``/``P1``/``P2``/``Q`` come from ``stereoRectify``.
    ``P1``/``P2`` are the rectified projection matrices fed to triangulation.
    """

    image_size: tuple[int, int]  # (width, height)
    camera_matrix_left: FloatArray
    dist_left: FloatArray
    camera_matrix_right: FloatArray
    dist_right: FloatArray
    R: FloatArray
    T: FloatArray
    R1: FloatArray
    R2: FloatArray
    P1: FloatArray
    P2: FloatArray
    Q: FloatArray
    rms: float = 0.0

    @property
    def baseline(self) -> float:
        """Distance between the two camera centers (metres) — the depth-accuracy knob."""
        return float(np.linalg.norm(self.T))

    # --- persistence (cv2-free) ---------------------------------------------------

    def to_dict(self) -> StereoCalibrationDict:
        return {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "camera_matrix_left": self.camera_matrix_left.tolist(),
            "dist_left": self.dist_left.ravel().tolist(),
            "camera_matrix_right": self.camera_matrix_right.tolist(),
            "dist_right": self.dist_right.ravel().tolist(),
            "R": self.R.tolist(),
            "T": self.T.ravel().tolist(),
            "R1": self.R1.tolist(),
            "R2": self.R2.tolist(),
            "P1": self.P1.tolist(),
            "P2": self.P2.tolist(),
            "Q": self.Q.tolist(),
            "rms": float(self.rms),
        }

    @classmethod
    def from_dict(cls, data: StereoCalibrationDict) -> StereoCalibration:
        def arr(key: str) -> FloatArray:
            return np.asarray(data[key], dtype=np.float64)  # type: ignore[literal-required]

        w, h = data["image_size"]
        return cls(
            image_size=(int(w), int(h)),
            camera_matrix_left=arr("camera_matrix_left"),
            dist_left=arr("dist_left"),
            camera_matrix_right=arr("camera_matrix_right"),
            dist_right=arr("dist_right"),
            R=arr("R"),
            T=arr("T"),
            R1=arr("R1"),
            R2=arr("R2"),
            P1=arr("P1"),
            P2=arr("P2"),
            Q=arr("Q"),
            rms=float(data["rms"]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> StereoCalibration:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # --- rectification (cv2-gated) ------------------------------------------------

    def rectification_maps(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """``(map1x, map1y, map2x, map2y)`` for ``cv2.remap``, recomputed from stored params."""
        import cv2

        map1x, map1y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_left, self.R1, self.P1, self.image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right,
            self.dist_right,
            self.R2,
            self.P2,
            self.image_size,
            cv2.CV_32FC1,
        )
        return map1x, map1y, map2x, map2y

    def rectify_pair(
        self,
        left: NDArray[Any],
        right: NDArray[Any],
        maps: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Undistort + rectify a left/right image pair (pass cached ``maps`` to avoid rebuild)."""
        import cv2

        map1x, map1y, map2x, map2y = maps if maps is not None else self.rectification_maps()
        rect_left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right


def _charuco_object_image_points(detector: Any, board: Any, image: Any) -> tuple[Any, Any, Any]:
    """Detect the board in one image → (object_points, image_points, charuco_ids) or (None,…)."""
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    if charuco_ids is None or len(charuco_ids) < 4:
        return None, None, None
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return object_points, image_points, charuco_ids


def calibrate_from_charuco(
    images_left: list[Any],
    images_right: list[Any],
    *,
    spec: CharucoBoardSpec = BOARD,
    image_size: tuple[int, int] | None = None,
) -> StereoCalibration:
    """Build a :class:`StereoCalibration` from corresponding ChArUco frame pairs (cv2-gated).

    ``images_left[i]`` and ``images_right[i]`` are the two views of the board at the same
    instant (grayscale or BGR). Frames where the board isn't seen in both views are skipped.
    Returns the calibration; ``.rms`` is the stereo reprojection error.
    """
    import cv2

    if len(images_left) != len(images_right):
        raise ValueError("images_left and images_right must be the same length")

    board = make_board(spec)
    detector = cv2.aruco.CharucoDetector(board)

    obj_left: list[Any] = []
    img_left: list[Any] = []
    obj_right: list[Any] = []
    img_right: list[Any] = []
    stereo_obj: list[Any] = []
    stereo_l: list[Any] = []
    stereo_r: list[Any] = []

    for left, right in zip(images_left, images_right, strict=True):
        if image_size is None:
            image_size = (int(left.shape[1]), int(left.shape[0]))
        ol, il, ids_l = _charuco_object_image_points(detector, board, left)
        or_, ir, ids_r = _charuco_object_image_points(detector, board, right)
        if ol is not None:
            obj_left.append(ol)
            img_left.append(il)
        if or_ is not None:
            obj_right.append(or_)
            img_right.append(ir)
        # Stereo pair: keep only board corners seen in BOTH views (matched by ChArUco id).
        if ids_l is None or ids_r is None:
            continue
        common = np.intersect1d(ids_l.flatten(), ids_r.flatten())
        if len(common) < 6:
            continue
        all_corners = board.getChessboardCorners()
        mask_l = np.isin(ids_l.flatten(), common)
        mask_r = np.isin(ids_r.flatten(), common)
        stereo_obj.append(all_corners[common].astype(np.float32))
        stereo_l.append(il[mask_l].reshape(-1, 2).astype(np.float32))
        stereo_r.append(ir[mask_r].reshape(-1, 2).astype(np.float32))

    if image_size is None or len(stereo_obj) < 3:
        raise ValueError(
            "not enough usable board pairs to calibrate (need the board in both views)"
        )

    _, k_left, d_left, _, _ = cv2.calibrateCamera(obj_left, img_left, image_size, None, None)
    _, k_right, d_right, _, _ = cv2.calibrateCamera(obj_right, img_right, image_size, None, None)

    rms, k_left, d_left, k_right, d_right, R, T, _, _ = cv2.stereoCalibrate(
        stereo_obj,
        stereo_l,
        stereo_r,
        k_left,
        d_left,
        k_right,
        d_right,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(k_left, d_left, k_right, d_right, image_size, R, T)

    return StereoCalibration(
        image_size=image_size,
        camera_matrix_left=np.asarray(k_left, dtype=np.float64),
        dist_left=np.asarray(d_left, dtype=np.float64),
        camera_matrix_right=np.asarray(k_right, dtype=np.float64),
        dist_right=np.asarray(d_right, dtype=np.float64),
        R=np.asarray(R, dtype=np.float64),
        T=np.asarray(T, dtype=np.float64).ravel(),
        R1=np.asarray(R1, dtype=np.float64),
        R2=np.asarray(R2, dtype=np.float64),
        P1=np.asarray(P1, dtype=np.float64),
        P2=np.asarray(P2, dtype=np.float64),
        Q=np.asarray(Q, dtype=np.float64),
        rms=float(rms),
    )


def _board_detected(detector: Any, gray: Any, min_corners: int = 6) -> tuple[Any, Any]:
    """Detect the board in a grayscale frame → (charuco_corners, charuco_ids) or (None, None)."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < min_corners:
        return None, None
    return corners, ids


def _solve_with_spinner(
    cv2: Any,
    fn: Any,
    *args: Any,
    window: str,
    **kwargs: Any,
) -> Any:
    """Run fn(*args, **kwargs) in a daemon thread; animate a spinner in the cv2 window."""
    result: list[Any] = []
    exc: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(fn(*args, **kwargs))
        except BaseException as e:
            exc.append(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    spinner = itertools.cycle(r"|\-/")
    font = cv2.FONT_HERSHEY_SIMPLEX
    while thread.is_alive():
        panel = np.zeros((100, 520, 3), dtype=np.uint8)
        text = f"Computing calibration...  {next(spinner)}"
        cv2.putText(panel, text, (12, 58), font, 0.8, (255, 255, 255), 2)
        cv2.imshow(window, panel)
        cv2.waitKey(100)

    if exc:
        raise exc[0]
    return result[0]


def live_calibrate(
    left_source: int | str,
    right_source: int | str,
    *,
    spec: CharucoBoardSpec = BOARD,
    min_pairs: int = 15,
    auto_capture: bool = False,
    auto_capture_interval_s: float = 1.0,
    auto_accept: bool = False,
    max_rms: float = 1.0,
    save_path: str | Path | None = None,
    window: str = "stereohand calibration",
) -> StereoCalibration:
    """Interactively calibrate from a live two-camera feed and return the result (cv2-gated).

    Opens both cameras, previews them side by side with live ChArUco detection, and collects
    board pairs — on **SPACE** (manual) or automatically every ``auto_capture_interval_s`` in
    ``auto_capture`` mode, but only when the board is seen in *both* views. **ENTER** finishes
    (auto mode finishes at ``min_pairs``); **Q** aborts. Then it solves via
    :func:`calibrate_from_charuco` and validates: ``auto_accept`` accepts iff
    ``rms <= max_rms``; otherwise it shows the RMS and waits for **Y** (accept) / **R** (redo).

    Designed to be called inline right before a tracking loop::

        calib = live_calibrate(0, 1)
        with StereoHandTracker(calib, ...) as tracker:
            ...

    Pass ``save_path`` to also persist the calibration. Raises ``RuntimeError`` on abort.
    """
    import cv2

    board = make_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    capture_left = cv2.VideoCapture(int(left_source) if str(left_source).isdigit() else left_source)
    capture_right = cv2.VideoCapture(
        int(right_source) if str(right_source).isdigit() else right_source
    )
    if not (capture_left.isOpened() and capture_right.isOpened()):
        capture_left.release()
        capture_right.release()
        raise RuntimeError("could not open both cameras")

    try:
        while True:  # one pass = collect → solve → validate; redo loops back
            left_frames, right_frames = _collect_pairs(
                cv2,
                detector,
                capture_left,
                capture_right,
                min_pairs=min_pairs,
                auto_capture=auto_capture,
                auto_capture_interval_s=auto_capture_interval_s,
                window=window,
            )
            calibration: StereoCalibration = _solve_with_spinner(
                cv2, calibrate_from_charuco, left_frames, right_frames, spec=spec, window=window
            )
            if _confirm(cv2, calibration, auto_accept=auto_accept, max_rms=max_rms, window=window):
                break
    finally:
        capture_left.release()
        capture_right.release()
        cv2.destroyAllWindows()

    if save_path is not None:
        calibration.save(save_path)
    return calibration


def _collect_pairs(
    cv2: Any,
    detector: Any,
    capture_left: Any,
    capture_right: Any,
    *,
    min_pairs: int,
    auto_capture: bool,
    auto_capture_interval_s: float,
    window: str,
) -> tuple[list[Any], list[Any]]:
    left_frames: list[Any] = []
    right_frames: list[Any] = []
    last_auto = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok_left, frame_left = capture_left.read()
        ok_right, frame_right = capture_right.read()
        if not (ok_left and ok_right):
            continue
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        corners_left, _ = _board_detected(detector, gray_left)
        corners_right, _ = _board_detected(detector, gray_right)
        both = corners_left is not None and corners_right is not None

        for frame, corners in ((frame_left, corners_left), (frame_right, corners_right)):
            if corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(frame, corners)
        preview = cv2.hconcat([frame_left, frame_right])
        colour = (0, 230, 0) if both else (0, 200, 255)
        mode = "AUTO" if auto_capture else "SPACE=capture"
        cv2.putText(
            preview,
            f"pairs {len(left_frames)}/{min_pairs}  board:{'BOTH' if both else 'wait'}  "
            f"{mode}  ENTER=done Q=quit",
            (12, 28),
            font,
            0.7,
            colour,
            2,
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF

        now = time.monotonic()
        capture_now = both and (
            (auto_capture and now - last_auto >= auto_capture_interval_s) or key == ord(" ")
        )
        if capture_now:
            left_frames.append(gray_left)
            right_frames.append(gray_right)
            last_auto = now
        finished = key in (13, 10) or (auto_capture and len(left_frames) >= min_pairs)
        if key == ord("q"):
            raise RuntimeError("calibration aborted")
        if finished and len(left_frames) >= min_pairs:
            return left_frames, right_frames


def _confirm(
    cv2: Any, calibration: StereoCalibration, *, auto_accept: bool, max_rms: float, window: str
) -> bool:
    if auto_accept:
        if calibration.rms > max_rms:
            raise RuntimeError(f"calibration RMS {calibration.rms:.3f} px exceeds max {max_rms}")
        return True
    panel = np.zeros((140, 640, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        panel,
        f"RMS = {calibration.rms:.3f} px   baseline = {calibration.baseline * 100:.1f} cm",
        (12, 50),
        font,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(panel, "Y = accept    R = redo", (12, 100), font, 0.7, (0, 230, 0), 2)
    cv2.imshow(window, panel)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("y"):
            return True
        if key == ord("r"):
            return False
