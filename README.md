# stereohand

**Two webcams → metric 3D hand landmarks, in real time.**

[![CI](https://github.com/NavehBrenner/stereohand/actions/workflows/ci.yml/badge.svg)](https://github.com/NavehBrenner/stereohand/actions/workflows/ci.yml)

Monocular [MediaPipe](https://ai.google.dev/edge/mediapipe) gives you 21 hand landmarks but
no real depth — its `z` is a relative guess. `stereohand` adds a second camera and recovers
**metric** 3D: synchronized dual-webcam capture → one-time **ChArUco** stereo calibration →
per-view **MediaPipe Tasks** `HandLandmarker` → **linear-DLT** triangulation → `(21, 3)`
landmarks in real-world units (metres).

<!-- TODO(demo): drop a recorded GIF of scripts/demo.py here — docs/demo.gif -->
<!-- ![demo](docs/demo.gif) -->

```python
from stereohand import StereoHandTracker, StereoCalibration

calib = StereoCalibration.load("stereo_calib.json")
with StereoHandTracker.open(calib, left=0, right=2) as tracker:
    while True:
        reading = tracker.read()          # non-blocking, latest reading
        if reading.present:
            print(reading.landmarks)      # (21, 3) metric xyz, in metres
            print(reading.handedness)     # "Left" / "Right"
```

## Install

```bash
pip install git+https://github.com/NavehBrenner/stereohand                       # library
pip install "stereohand[demo] @ git+https://github.com/NavehBrenner/stereohand"  # + 3D demo viewer
```

For development: `pip install -e ".[dev]"` (ruff + mypy + pytest).

> **System note:** MediaPipe's runtime needs OpenGL ES libraries present (`libGLESv2`,
> `libGL`). They're already there on a normal desktop; on a headless box install
> `libgl1 libglib2.0-0` (and run cameras from a graphical session). On WSL2, host webcams
> have no UVC driver — use a stream URL, or run on the Windows side.

## Hardware setup

- **Two webcams, rigidly co-mounted on one bar.** They need *not* be identical — the math
  uses each camera's own intrinsics + the stereo extrinsics, so a laptop cam + a separate
  webcam works. Accuracy just degrades to the weaker camera, and mismatched rolling
  shutters make *synchronization* the hard part (tune `max_skew_s`).
- **Baseline is the accuracy knob.** Too small → poor depth resolution; too wide → the
  views stop overlapping and MediaPipe loses the hand in one of them. Start ~10–15 cm.
- Calibration is valid **only until the rig is physically disturbed** — if a camera shifts,
  recalibrate.

## Calibration (one time)

1. **Print the board.** `stereohand-board -o board.pdf` (a ready-made [`board.pdf`](board.pdf)
   is in the repo). Print it on A4 **at 100% / actual size (no "fit to page")**, then
   measure a printed square — it must be **35 mm**. Mount it flat on something rigid.
2. **Run the guided session:**
   ```bash
   python scripts/calibrate.py --left 0 --right 2 --out stereo_calib.json
   ```
   Show the board to **both** cameras at once. Press **SPACE** to capture a pair whenever
   the board is visible in both previews; sweep it across the shared field of view at
   varied angles, distances, and positions. Collect ~15–25 pairs, then press **ENTER**.
3. It writes `stereo_calib.json` and prints the **RMS reprojection error** (lower is
   better; well under 1 px is good) and the measured baseline.

## Run the demo

```bash
python scripts/demo.py --calib stereo_calib.json --left 0 --right 2
```

A matplotlib 3D window draws the live hand skeleton and shows the wrist's **metric depth** —
the thing a single camera can't give you.

## How it works

| Stage | Module | What it does |
|---|---|---|
| Capture | `capture.py` | Two daemon-threaded grabbers; delivers only frame pairs within `max_skew_s` (software sync). |
| Calibration | `calibration.py` + `board.py` | ChArUco detection → per-camera intrinsics → `stereoCalibrate` → `stereoRectify` → `P1`/`P2`. Persisted as JSON. |
| Landmarking | `landmarker.py` | MediaPipe **Tasks** `HandLandmarker` (VIDEO mode), one per rectified view, `num_hands=1`. |
| Triangulation | `triangulation.py` | Linear DLT (SVD) on the 21 trivially-corresponded landmarks → `(21, 3)` metric. |
| Orchestration | `tracker.py` | `StereoHandTracker` ties it together; `read()` is the non-blocking public API. |

The 21 landmark indices are MediaPipe's standard hand model (0 = wrist, 4 = thumb tip,
8/12/16/20 = finger tips).

## Limitations

- **One hand** (`num_hands=1`) — two hands break the cross-view correspondence.
- **Teleop-grade, not metrology.** Linear DLT, best-effort software sync, no bundle
  adjustment. Good for "where is the hand," not sub-millimetre measurement.
- **Drop-out is all-or-nothing**: the hand must be visible in *both* views, or the reading
  is `present=False`.
- Smoothing/filtering is intentionally **not** built in — add your own (e.g. a one-euro
  filter) downstream.

## Acknowledgements & license

- Linear-DLT triangulation adapted from [handpose3d](https://github.com/TemugeB/handpose3d)
  by Temuge Batpurev (MIT).
- Hand landmarks via [MediaPipe](https://ai.google.dev/edge/mediapipe) (the
  `hand_landmarker` Tasks model, downloaded on first use).

MIT-licensed — see [`LICENSE`](LICENSE).
