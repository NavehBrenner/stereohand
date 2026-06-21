"""Linear-DLT stereo triangulation — pure NumPy, no OpenCV.

Given two camera projection matrices ``P1``, ``P2`` (each ``3x4``, mapping a
homogeneous world point to homogeneous image coordinates) and the image of the same
point in both views, recover the 3D point by the **direct linear transform (DLT)**:
stack the two per-view collinearity constraints into a ``4x4`` matrix ``A`` and take the
right singular vector of its smallest singular value (the homogeneous null-space).

Keeping this OpenCV-free (no ``cv2.triangulatePoints``) is deliberate: it's the
deterministic heart of the pipeline, so it unit-tests in CI with only ``numpy`` and adds
no heavy dependency. Linear DLT only — no bundle adjustment (over-refining the CV is the
failure mode here, not under-building it).

The single-point DLT is adapted from handpose3d (https://github.com/TemugeB/handpose3d)
by Temuge Batpurev, MIT-licensed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _dlt_rows(projection: FloatArray, image_point: FloatArray) -> FloatArray:
    """The two DLT constraint rows a single view contributes for one point.

    ``image_point`` is ``(2,)`` in the same coordinate convention as ``projection``
    (pixels for a pixel ``P``, normalized for a normalized ``P`` — the routine doesn't
    care, as long as both views match).
    """
    x, y = image_point
    return np.stack([y * projection[2] - projection[1], projection[0] - x * projection[2]])


def triangulate_point(
    projection1: FloatArray, projection2: FloatArray, point1: FloatArray, point2: FloatArray
) -> FloatArray:
    """Triangulate one 3D point from its image in two views.

    Parameters
    ----------
    projection1, projection2:
        ``(3, 4)`` camera projection matrices for view 1 and view 2.
    point1, point2:
        ``(2,)`` image coordinates of the point in each view.

    Returns
    -------
    ``(3,)`` world coordinates (dehomogenized).
    """
    constraints = np.vstack([_dlt_rows(projection1, point1), _dlt_rows(projection2, point2)])
    # Homogeneous solution = right singular vector of the smallest singular value.
    _, _, vh = np.linalg.svd(constraints)
    homogeneous = vh[-1]
    return np.asarray(homogeneous[:3] / homogeneous[3], dtype=np.float64)


def triangulate_points(
    projection1: FloatArray, projection2: FloatArray, points1: FloatArray, points2: FloatArray
) -> FloatArray:
    """Triangulate a batch of corresponded points (e.g. the 21 hand landmarks).

    Parameters
    ----------
    projection1, projection2:
        ``(3, 4)`` camera projection matrices.
    points1, points2:
        ``(N, 2)`` image coordinates in each view; row ``i`` of ``points1`` corresponds to
        row ``i`` of ``points2``.

    Returns
    -------
    ``(N, 3)`` world coordinates.
    """
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    if points1.shape != points2.shape or points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError(
            f"points1/points2 must be matching (N, 2) arrays, got {points1.shape} and "
            f"{points2.shape}"
        )

    n = points1.shape[0]
    # Stacked (N, 4, 4) constraint matrices; batched SVD solves all points at once.
    constraints = np.empty((n, 4, 4), dtype=np.float64)
    constraints[:, 0] = points1[:, 1, None] * projection1[2] - projection1[1]
    constraints[:, 1] = projection1[0] - points1[:, 0, None] * projection1[2]
    constraints[:, 2] = points2[:, 1, None] * projection2[2] - projection2[1]
    constraints[:, 3] = projection2[0] - points2[:, 0, None] * projection2[2]

    _, _, vh = np.linalg.svd(constraints)
    homogeneous = vh[:, -1, :]  # (N, 4)
    return np.asarray(homogeneous[:, :3] / homogeneous[:, 3, None], dtype=np.float64)
