"""parse_result contract — pure, runs in CI against a stand-in MediaPipe result."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from stereohand.landmarker import HandLandmarks2D, parse_result


def _landmark(x: float, y: float):
    return SimpleNamespace(x=x, y=y, z=0.0)


def _result(n_landmarks: int = 21, handedness: str = "Right"):
    hand = [_landmark(0.5, 0.5) for _ in range(n_landmarks)]
    return SimpleNamespace(
        hand_landmarks=[hand],
        handedness=[[SimpleNamespace(category_name=handedness)]],
    )


def test_no_hand_returns_none():
    assert parse_result(SimpleNamespace(hand_landmarks=[], handedness=[]), 640, 480) is None


def test_landmarks_scaled_to_pixels():
    parsed = parse_result(_result(), width=640, height=480)
    assert isinstance(parsed, HandLandmarks2D)
    assert parsed.landmarks.shape == (21, 2)
    # x=0.5 * 640 = 320, y=0.5 * 480 = 240
    np.testing.assert_allclose(parsed.landmarks[0], [320.0, 240.0])
    assert parsed.handedness == "Right"


def test_wrong_landmark_count_returns_none():
    assert parse_result(_result(n_landmarks=20), 640, 480) is None
