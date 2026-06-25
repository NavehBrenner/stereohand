"""DLT triangulation round-trip on synthetic geometry — pure NumPy, runs in CI."""

from __future__ import annotations

import numpy as np

from stereohand.triangulation import triangulate_point, triangulate_points


def _intrinsics(fx: float = 800.0, fy: float = 800.0, cx: float = 320.0, cy: float = 240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _projection(K, R, t):
    """P = K [R | t], mapping world points (in cam-2's case, via R,t) to pixels."""
    return K @ np.hstack([R, t.reshape(3, 1)])


def _stereo_rig(baseline: float = 0.12):
    """Two cameras: cam1 at world origin, cam2 translated +baseline along x.

    World == cam1 frame, so P1 = K[I|0]. cam2 sees a world point at X - [baseline,0,0],
    i.e. R = I, t = [-baseline, 0, 0].
    """
    K = _intrinsics()
    P1 = _projection(K, np.eye(3), np.zeros(3))
    P2 = _projection(K, np.eye(3), np.array([-baseline, 0.0, 0.0]))
    return P1, P2


def _project(P, X):
    """Project world point(s) X (3,) or (N,3) to pixel coords (2,) or (N,2)."""
    X = np.atleast_2d(X)
    hom = np.hstack([X, np.ones((X.shape[0], 1))])
    px = (P @ hom.T).T
    px = px[:, :2] / px[:, 2, None]
    return px


def test_single_point_round_trip():
    P1, P2 = _stereo_rig()
    truth = np.array([0.05, -0.03, 0.6])  # 60 cm in front, off-center
    x1 = _project(P1, truth)[0]
    x2 = _project(P2, truth)[0]
    recovered = triangulate_point(P1, P2, x1, x2)
    assert np.allclose(recovered, truth, atol=1e-6)


def test_batch_round_trip():
    P1, P2 = _stereo_rig()
    rng = np.random.default_rng(0)
    # 21 points spread over a plausible hand volume ~0.5–0.7 m in front of the rig.
    truth = np.column_stack([
        rng.uniform(-0.1, 0.1, 21),
        rng.uniform(-0.1, 0.1, 21),
        rng.uniform(0.5, 0.7, 21),
    ])
    pts1 = _project(P1, truth)
    pts2 = _project(P2, truth)
    recovered = triangulate_points(P1, P2, pts1, pts2)
    assert recovered.shape == (21, 3)
    assert np.allclose(recovered, truth, atol=1e-6)


def test_small_pixel_noise_stays_bounded():
    P1, P2 = _stereo_rig()
    rng = np.random.default_rng(1)
    truth = np.array([0.0, 0.0, 0.6])
    x1 = _project(P1, truth)[0] + rng.normal(0, 0.5, 2)  # ~half-pixel noise
    x2 = _project(P2, truth)[0] + rng.normal(0, 0.5, 2)
    recovered = triangulate_point(P1, P2, x1, x2)
    # Half-pixel noise at this geometry should stay well under a centimeter.
    assert np.linalg.norm(recovered - truth) < 0.01


def test_mismatched_shapes_raise():
    P1, P2 = _stereo_rig()
    try:
        triangulate_points(P1, P2, np.zeros((21, 2)), np.zeros((20, 2)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched point counts")
