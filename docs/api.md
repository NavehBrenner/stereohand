# stereohand API Reference

All configuration flows through `StereoHandTracker.open()` and `RenderConfig`.

## `StereoHandTracker.open()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calibration` | `StereoCalibration` | *(required)* | Stereo calibration data (load from JSON via `StereoCalibration.load()`, or obtain from `live_calibrate()`). |
| `left` | `int \| str` | `0` | Left camera source — an integer device index (e.g. `0`) or a string URL / path (e.g. `"http://host:8080/0"`). |
| `right` | `int \| str` | `2` | Right camera source — same format as `left`. |
| `max_skew_s` | `float` | `0.02` | Maximum capture-timestamp difference (seconds) for a frame pair to be accepted. Increase for mismatched or high-latency cameras; decrease for tighter sync. |
| `render` | `bool` | `False` | When `True`, create a cv2 visualisation window. Drive it with `tracker.run()` (blocking) or `tracker.render_step()` (single frame). |
| `render_config` | `RenderConfig \| None` | `None` | Visualisation options (see below). Ignored when `render=False`. Defaults to `RenderConfig()` when `render=True`. |
| `**landmarker_kwargs` | | | Forwarded to both `HandLandmarker` instances (see below). |

### Landmarker keyword arguments (forwarded via `**landmarker_kwargs`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str \| Path \| None` | `None` | Path to a custom MediaPipe `hand_landmarker.task` model. When `None`, the default Google-published float16 model is auto-downloaded and cached at `~/.cache/stereohand/`. |
| `min_detection_confidence` | `float` | `0.5` | Minimum confidence for the initial hand detection to succeed (0.0–1.0). Lower values detect more hands but with more false positives. |
| `min_tracking_confidence` | `float` | `0.5` | Minimum confidence for frame-to-frame landmark tracking (0.0–1.0). Below this threshold the detector re-runs instead of tracking, which is slower but more robust. |

## `RenderConfig` fields

`RenderConfig` is a dataclass controlling the built-in visualisation window.
Pass it to `open()` via the `render_config` parameter.

```python
from stereohand import RenderConfig

cfg = RenderConfig(mirror=True, smooth=0.3)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `mirror` | `bool` | `False` | Flip the view horizontally so it acts like a mirror — your right hand appears on the right side of the screen. Camera feeds are flipped and swapped; the 3D skeleton's X axis is negated. |
| `smooth` | `float` | `0.5` | EMA (exponential moving average) alpha for temporal smoothing of the 3D landmarks. `1.0` = no smoothing (raw values), `0.1` = very smooth (more lag). `0.5` is a good balance of low jitter and low latency. |

## Programmatic usage with rendering

```python
from stereohand import StereoHandTracker, StereoCalibration, RenderConfig

calib = StereoCalibration.load("stereo_calib.json")

# Headless (default) — no window, just data.
with StereoHandTracker.open(calib, left=0, right=2) as tracker:
    reading = tracker.read()

# With built-in visualisation.
cfg = RenderConfig(mirror=True, smooth=0.3)
with StereoHandTracker.open(calib, left=0, right=2, render=True, render_config=cfg) as tracker:
    tracker.run()  # blocks until the user presses Q / ESC
```
