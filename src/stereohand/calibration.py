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

import json
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
