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


# Exact colors from mediapipe/python/solutions/drawing_styles.py (BGR).
_WHITE  = (224, 224, 224)
_RED    = (48,  48,  255)
_PEACH  = (180, 229, 255)
_PURPLE = (128,  64, 128)
_YELLOW = (0,   204, 255)
_GREEN  = (48,  255,  48)
_BLUE   = (192, 101,  21)
_GRAY   = (128, 128, 128)

# Landmark color per index — matches get_default_hand_landmarks_style().
# Palm: 0, 1, 5, 9, 13, 17 → RED
# Thumb MCP/IP/TIP: 2, 3, 4 → PEACH
# Index PIP/DIP/TIP: 6, 7, 8 → PURPLE
# Middle PIP/DIP/TIP: 10, 11, 12 → YELLOW
# Ring PIP/DIP/TIP: 14, 15, 16 → GREEN
# Pinky PIP/DIP/TIP: 18, 19, 20 → BLUE
_LM_COLOR: list[tuple[int, int, int]] = [
    _RED, _RED, _PEACH, _PEACH, _PEACH,        # 0-4
    _RED, _PURPLE, _PURPLE, _PURPLE,            # 5-8
    _RED, _YELLOW, _YELLOW, _YELLOW,            # 9-12
    _RED, _GREEN, _GREEN, _GREEN,               # 13-16
    _RED, _BLUE, _BLUE, _BLUE,                  # 17-20
]
_LM_RADIUS = 5
_LM_BORDER_RADIUS = max(_LM_RADIUS + 1, int(_LM_RADIUS * 1.2))  # = 6

# Connection (a, b, color, thickness) — matches get_default_hand_connections_style().
_CONN: list[tuple[int, int, tuple[int, int, int], int]] = [
    # Palm — GRAY, thickness 3
    (0, 1, _GRAY, 3), (0, 5, _GRAY, 3), (9, 13, _GRAY, 3),
    (13, 17, _GRAY, 3), (5, 9, _GRAY, 3), (0, 17, _GRAY, 3),
    # Thumb — PEACH, thickness 2
    (1, 2, _PEACH, 2), (2, 3, _PEACH, 2), (3, 4, _PEACH, 2),
    # Index — PURPLE, thickness 2
    (5, 6, _PURPLE, 2), (6, 7, _PURPLE, 2), (7, 8, _PURPLE, 2),
    # Middle — YELLOW, thickness 2
    (9, 10, _YELLOW, 2), (10, 11, _YELLOW, 2), (11, 12, _YELLOW, 2),
    # Ring — GREEN, thickness 2
    (13, 14, _GREEN, 2), (14, 15, _GREEN, 2), (15, 16, _GREEN, 2),
    # Pinky — BLUE, thickness 2
    (17, 18, _BLUE, 2), (18, 19, _BLUE, 2), (19, 20, _BLUE, 2),
]
_HANDEDNESS_COLOR = (54, 205, 88)  # green (BGR) — matches MediaPipe notebook
_MARGIN = 10


def draw_landmarks_on_frame(
    frame_bgr: NDArray[np.uint8], landmarks: HandLandmarks2D
) -> NDArray[np.uint8]:
    """Draw hand landmarks replicating mediapipe.solutions drawing_utils exactly.

    mediapipe.solutions was removed in 0.10.x so we replicate it in pure cv2.
    Drawing order matches the source: connections first, then white-bordered dots on top.
    Colors and sizes are copied verbatim from drawing_styles.py.
    """
    import cv2

    annotated = frame_bgr.copy()
    pts = landmarks.landmarks.astype(int)  # (21, 2)

    # 1. Connections (lines, drawn first so dots appear on top).
    for a, b, color, thickness in _CONN:
        cv2.line(annotated, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)

    # 2. Landmark dots: white border circle, then colored fill (exact MediaPipe logic).
    for i, pt in enumerate(pts):
        cv2.circle(annotated, tuple(pt), _LM_BORDER_RADIUS, _WHITE, -1, cv2.LINE_AA)
        cv2.circle(annotated, tuple(pt), _LM_RADIUS, _LM_COLOR[i], -1, cv2.LINE_AA)

    # 3. Handedness label (MediaPipe notebook style).
    x_min = int(pts[:, 0].min()) - _MARGIN
    y_min = int(pts[:, 1].min()) - _MARGIN
    cv2.putText(
        annotated, landmarks.handedness, (x_min, y_min),
        cv2.FONT_HERSHEY_DUPLEX, 1, _HANDEDNESS_COLOR, 1, cv2.LINE_AA,
    )
    return annotated


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
