"""Path → camera-index parsing for the Windows MJPEG bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cv2")  # the bridge module imports cv2 at top level

_PATH = Path(__file__).parents[1] / "scripts" / "stream_webcams.py"
_spec = importlib.util.spec_from_file_location("stream_webcams", _PATH)
assert _spec and _spec.loader
stream_webcams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stream_webcams)
parse_camera_index = stream_webcams.parse_camera_index


def test_allowed_index_paths():
    allowed = {0, 1}
    assert parse_camera_index("/0", allowed) == 0
    assert parse_camera_index("/1/video", allowed) == 1


def test_disallowed_or_unknown_paths():
    allowed = {0, 1}
    assert parse_camera_index("/2", allowed) is None  # not whitelisted
    assert parse_camera_index("/video", allowed) is None  # not numeric
    assert parse_camera_index("/", allowed) is None
