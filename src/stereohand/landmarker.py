"""Per-view hand landmarks via the modern MediaPipe Tasks ``HandLandmarker``.

Deliberately the **Tasks** API (``mediapipe.tasks.python.vision``), not the legacy
``mediapipe.solutions.hands`` — the legacy API is what kept handpose3d from being reusable.
This wrapper is single-view only: it turns one frame into the 21 image landmarks +
handedness. The stereo tracker (SH7) runs one of these per camera and triangulates.

``num_hands=1`` on purpose: two hands, or left/right handedness flipping *between* the two
views, would silently break the trivial 1:1 landmark correspondence triangulation relies on.

The pure result parser (:func:`parse_result`) is unit-tested without a live model; the
``HandLandmarker`` class lazily imports mediapipe/cv2 and needs the downloaded model asset.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Float16 hand-landmarker bundle published by Google (MediaPipe model card).
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandLandmarks2D:
    """One hand in one view: 21 landmarks in **pixel** coordinates, plus handedness.

    ``landmarks`` is ``(21, 2)`` (image x, y in pixels — ready for triangulation against a
    pixel projection matrix). ``handedness`` is ``"Left"`` / ``"Right"`` as MediaPipe
    reports it (used by the tracker to guard the cross-view correspondence).
    """

    landmarks: FloatArray
    handedness: str


def parse_result(result: Any, width: int, height: int) -> HandLandmarks2D | None:
    """Convert a ``HandLandmarkerResult`` to :class:`HandLandmarks2D`, or ``None`` if no hand.

    Pure — no mediapipe import — so it tests against a stand-in result object. MediaPipe
    landmarks are image-normalized ``[0, 1]``; we scale to pixels by ``(width, height)``.
    """
    hands = getattr(result, "hand_landmarks", None)
    if not hands:
        return None
    points = np.array([[lm.x * width, lm.y * height] for lm in hands[0]], dtype=np.float64)
    if points.shape != (21, 2):
        return None
    handedness_groups = getattr(result, "handedness", None)
    handedness = ""
    if handedness_groups:
        handedness = handedness_groups[0][0].category_name
    return HandLandmarks2D(landmarks=points, handedness=handedness)


def default_model_path() -> Path:
    """Path to the cached model, downloading it on first use."""
    cache = Path.home() / ".cache" / "stereohand"
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / "hand_landmarker.task"
    if not model.exists():
        urllib.request.urlretrieve(_MODEL_URL, model)  # noqa: S310 (trusted Google URL)
    return model


class HandLandmarker:
    """Live single-view hand landmarking (VIDEO mode). Lazily imports mediapipe + cv2."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker as _MPHandLandmarker,
        )
        from mediapipe.tasks.python.vision import (
            HandLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        path = str(model_path) if model_path is not None else str(default_model_path())
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = _MPHandLandmarker.create_from_options(options)

    def process(self, frame_bgr: NDArray[np.uint8], timestamp_ms: int) -> HandLandmarks2D | None:
        """Landmark one BGR frame; ``timestamp_ms`` must be monotonically increasing."""
        import cv2

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        return parse_result(result, width, height)

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> HandLandmarker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
