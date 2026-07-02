"""StereoHandTracker seam test — fake capture + fake landmarkers, no cv2 (runs in CI)."""

from __future__ import annotations

import numpy as np

from stereohand.calibration import StereoCalibration
from stereohand.landmarker import HandLandmarks2D
from stereohand.tracker import StereoHandTracker, write_gif


def _calib(baseline: float = 0.12) -> StereoCalibration:
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = np.hstack([K, np.array([[-800.0 * baseline], [0], [0]])])
    return StereoCalibration(
        image_size=(640, 480),
        camera_matrix_left=K,
        dist_left=np.zeros(5),
        camera_matrix_right=K,
        dist_right=np.zeros(5),
        R=np.eye(3),
        T=np.array([-baseline, 0.0, 0.0]),
        R1=np.eye(3),
        R2=np.eye(3),
        P1=P1,
        P2=P2,
        Q=np.eye(4),
    )


def _project(P, X):
    hom = np.hstack([X, np.ones((X.shape[0], 1))])
    px = (P @ hom.T).T
    return px[:, :2] / px[:, 2, None]


class _FakeCapture:
    def __init__(self, pair):
        self._pair = pair

    def read(self):
        return self._pair

    def latest_pair_timestamp(self) -> float:
        return 0.0  # only the background _run thread reads this; step() tests don't

    def close(self):
        pass


class _FakeLandmarker:
    def __init__(self, points, handedness="Right"):
        self._points = points
        self._handedness = handedness

    def process(self, frame_bgr, timestamp_ms):
        if self._points is None:
            return None
        return HandLandmarks2D(landmarks=self._points, handedness=self._handedness)

    def close(self):
        pass


def _dummy_frames():
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    return f, f


def test_seam_triangulates_to_known_3d():
    calib = _calib()
    rng = np.random.default_rng(0)
    truth = np.column_stack([
        rng.uniform(-0.1, 0.1, 21),
        rng.uniform(-0.1, 0.1, 21),
        rng.uniform(0.5, 0.7, 21),
    ])
    pts1, pts2 = _project(calib.P1, truth), _project(calib.P2, truth)

    tracker = StereoHandTracker(
        calib,
        _FakeCapture(_dummy_frames()),
        _FakeLandmarker(pts1),
        _FakeLandmarker(pts2),
        rectify=False,
    )
    reading = tracker.step()

    assert reading.present
    assert reading.landmarks.shape == (21, 3)
    assert reading.handedness == "Right"
    np.testing.assert_allclose(reading.landmarks, truth, atol=1e-6)


def test_dropout_when_hand_missing_in_one_view():
    calib = _calib()
    tracker = StereoHandTracker(
        calib,
        _FakeCapture(_dummy_frames()),
        _FakeLandmarker(np.zeros((21, 2))),
        _FakeLandmarker(None),  # right view sees no hand
        rectify=False,
    )
    reading = tracker.step()
    assert not reading.present
    assert np.array_equal(reading.landmarks, np.zeros((21, 3)))


def test_absent_when_no_synced_pair():
    tracker = StereoHandTracker(
        _calib(),
        _FakeCapture(None),  # capture not ready / over-skew
        _FakeLandmarker(np.zeros((21, 2))),
        _FakeLandmarker(np.zeros((21, 2))),
        rectify=False,
    )
    assert not tracker.step().present


def test_timestamps_strictly_increase_within_one_millisecond(monkeypatch):
    """MediaPipe's detect_for_video demands *strictly* increasing timestamps; multiple step()s
    inside the same millisecond must not collide (would raise 'must be monotonically increasing').
    Freeze the clock so every step() derives the same raw ms, and assert the guard bumps them."""
    import stereohand.tracker as tracker_mod

    monkeypatch.setattr(tracker_mod.time, "monotonic", lambda: 100.0)

    class _RecordingLandmarker(_FakeLandmarker):
        def __init__(self, points):
            super().__init__(points)
            self.seen: list[int] = []

        def process(self, frame_bgr, timestamp_ms):
            self.seen.append(timestamp_ms)
            return super().process(frame_bgr, timestamp_ms)

    left = _RecordingLandmarker(np.zeros((21, 2)))
    tracker = StereoHandTracker(
        _calib(),
        _FakeCapture(_dummy_frames()),
        left,
        _RecordingLandmarker(np.zeros((21, 2))),
        rectify=False,
    )
    for _ in range(3):
        tracker.step()

    assert left.seen == [0, 1, 2]  # strictly increasing despite the frozen clock


def test_write_gif_subsamples_and_downscales(tmp_path):
    from PIL import Image

    # 30 BGR frames at 1000-wide; expect 30fps→10fps (every 3rd) and 1000→640 wide.
    frames = [np.full((100, 1000, 3), i, dtype=np.uint8) for i in range(30)]
    out = tmp_path / "out.gif"
    write_gif(str(out), frames, fps=30.0, max_width=640, max_fps=10.0)

    gif = Image.open(out)
    assert gif.size == (640, 64)
    assert gif.n_frames == 10
    assert gif.info.get("loop") == 0
