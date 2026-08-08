# stereohand API Reference

All configuration flows through `StereoHandTracker.open()` and `RenderConfig`.

## `StereoHandTracker.open()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calibration` | `StereoCalibration` | *(required)* | Stereo calibration data (load from JSON via `StereoCalibration.load()`, or obtain from `live_calibrate()`). |
| `left` | `int \| str` | `0` | Left camera source — an integer device index (e.g. `0`) or a string URL / path (e.g. `"http://host:8080/0"`). |
| `right` | `int \| str` | `2` | Right camera source — same format as `left`. |
| `max_skew_s` | `float` | `0.02` | Maximum capture-timestamp difference (seconds) for a frame pair to be accepted — an **alignment-quality** knob: how simultaneous the two views must be for triangulating them to be meaningful. Increase for mismatched or high-latency cameras; decrease for tighter sync. It is *not* a drop-out remedy — a rejected pair no longer reports the hand as absent (see below), so widening this only trades triangulation accuracy away. |
| `render` | `bool` | `False` | When `True`, create a cv2 visualisation window. Drive it with `tracker.run()` (blocking) or `tracker.render_step()` (single frame). |
| `render_config` | `RenderConfig \| None` | `None` | Visualisation options (see below). Ignored when `render=False`. Defaults to `RenderConfig()` when `render=True`. |
| `**landmarker_kwargs` | | | Forwarded to both `HandLandmarker` instances (see below). |

### What `reading.present == False` means

`present=False` means **the hand was not detected**, in one or both views, or a camera has
stopped delivering frames (`max_age_s`, default 0.5 s). It does *not* fire merely because a
frame pair failed the `max_skew_s` test.

That distinction matters to any consumer that acts on presence rather than just drawing it.
The two cameras free-run, so their capture timestamps drift in and out of `max_skew_s` every
frame cycle; treating each of those misses as a lost hand makes a perfectly still hand appear
to vanish at camera rate. The tracker now holds its last reading through such a miss — it
carries no information about what is in front of the cameras — and `max_age_s` remains the
backstop for a camera that has genuinely died.

`StereoCapture.last_read_status` (`"ok" | "not_ready" | "stale" | "over_skew"`) exposes the
reason for the most recent `read()` if you need to distinguish these yourself.

### Depth sanity check (`tracker.depth_warning`)

A hand triangulating *behind* the cameras (wrist z < 0) is physically impossible; a
sustained streak of it (30 consecutive present frames, ~1 s) means the stereo geometry is
inverted — in practice the `left`/`right` sources are swapped relative to the calibration
(a USB replug can re-enumerate device indices). Everything else still looks healthy in
that state (per-view detection, fps, preview) — only depth-dependent output is garbage.

When tripped, the tracker logs one `logging` warning (logger `stereohand.tracker`), sets
`tracker.depth_warning` to the message (latched for the tracker's lifetime — sources are
fixed at construction), and the render window shows a red banner. Fix: swap the
`left`/`right` sources, or recalibrate.

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
| `recenter` | `bool` | `False` | Enable the hold-palm-open recenter gesture. Hold an open palm square to a camera and still for 3 s (an on-screen "Calibrating..." countdown runs); the current palm position then becomes the world origin `(0, 0, 0)` — both the HUD XYZ readout and where the hand renders (now centred). Re-arms after you drop the pose. |

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
