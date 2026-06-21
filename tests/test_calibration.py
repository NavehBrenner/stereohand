"""StereoCalibration round-trip (CI) + rectification sanity (cv2-gated)."""

from __future__ import annotations

import numpy as np
import pytest

from stereohand.calibration import StereoCalibration


def _synthetic_calib(baseline: float = 0.12) -> StereoCalibration:
    """An ideal rectified rig: identical pinholes, zero distortion, baseline along +x."""
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    dist = np.zeros(5)
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = np.hstack([K, np.array([[-800.0 * baseline], [0], [0]])])
    return StereoCalibration(
        image_size=(640, 480),
        camera_matrix_left=K.copy(),
        dist_left=dist.copy(),
        camera_matrix_right=K.copy(),
        dist_right=dist.copy(),
        R=np.eye(3),
        T=np.array([-baseline, 0.0, 0.0]),
        R1=np.eye(3),
        R2=np.eye(3),
        P1=P1,
        P2=P2,
        Q=np.eye(4),
        rms=0.21,
    )


def test_save_load_round_trip(tmp_path):
    calib = _synthetic_calib()
    path = calib.save(tmp_path / "stereo_calib.json")
    loaded = StereoCalibration.load(path)

    assert loaded.image_size == calib.image_size
    assert loaded.rms == pytest.approx(calib.rms)
    for field in ("camera_matrix_left", "dist_left", "R", "T", "P1", "P2", "Q"):
        np.testing.assert_allclose(getattr(loaded, field), getattr(calib, field))


def test_baseline_property():
    assert _synthetic_calib(baseline=0.12).baseline == pytest.approx(0.12)


def test_json_is_human_readable(tmp_path):
    path = _synthetic_calib().save(tmp_path / "c.json")
    text = path.read_text()
    assert '"baseline"' not in text  # derived, not stored
    assert '"P1"' in text and '"image_size"' in text


def test_rectification_maps_identity():
    pytest.importorskip("cv2")
    calib = _synthetic_calib()
    map1x, map1y, map2x, map2y = calib.rectification_maps()
    w, h = calib.image_size
    assert map1x.shape == (h, w)
    # Zero distortion + R1=I + P1=[K|0] => the left map is the identity sampling grid.
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    np.testing.assert_allclose(map1x, xs, atol=1e-3)
    np.testing.assert_allclose(map1y, ys, atol=1e-3)


def test_rectify_pair_identity_is_noop():
    pytest.importorskip("cv2")
    calib = _synthetic_calib()
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (480, 640), dtype=np.uint8)
    right = rng.integers(0, 255, (480, 640), dtype=np.uint8)
    rect_left, rect_right = calib.rectify_pair(left, right)
    # Identity rectification of the left camera leaves it essentially unchanged.
    assert np.mean(np.abs(rect_left.astype(int) - left.astype(int))) < 1.0
    assert rect_right.shape == right.shape
