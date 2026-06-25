# stereohand

**Two webcams → metric 3D hand landmarks, in real time.**

Monocular [MediaPipe](https://ai.google.dev/edge/mediapipe) gives you 21 hand landmarks but
no real depth — its `z` is a relative guess. `stereohand` adds a second camera and recovers
**metric** 3D: synchronized dual-webcam capture → one-time **ChArUco** stereo calibration →
per-view **MediaPipe Tasks** `HandLandmarker` → **linear-DLT** triangulation → `(21, 3)`
landmarks in real-world units (metres).

<p align="center">
  <video src="https://github.com/NavehBrenner/stereohand/raw/master/docs/demo.mp4" width="720" autoplay loop muted playsinline>
    <a href="docs/demo.mp4">Watch the demo video</a>
  </video>
</p>

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
   is in the repo). Print on A4 at **true 1:1 scale** — the squares are baked into the
   PDF at exactly 35 mm, so any scaling silently invalidates the calibration. Choose the
   setting that means "no scaling" in your print dialog:

   | App | Where | Pick |
   |---|---|---|
   | Adobe Acrobat / Reader | Page Sizing & Handling | **Actual size** |
   | Chrome / Edge / Firefox | More settings → Scale | type **100** |
   | macOS Preview | Scale field | **100%** |
   | Windows generic dialog | Page sizing | **Actual size** — or Custom → **100%** |

   Avoid **Fit to paper**, **Fit to printable area**, and **Default** — all rescale.
   After printing, measure one white square with a ruler: it must be **35 mm × 35 mm
   (3.5 cm × 3.5 cm / ~1⅜ in × 1⅜ in)**.
   If it's off, reprint with the correct setting. Mount the board flat on something rigid.
2. **Run the guided session:**
   ```bash
   python scripts/calibrate.py --left 0 --right 2 --out stereo_calib.json
   ```
   Show the board to **both** cameras at once. Press **SPACE** to capture a pair whenever
   the board is visible in both previews; sweep it across the shared field of view at
   varied angles, distances, and positions. Collect ~15–25 pairs, then press **ENTER**.
3. It writes `stereo_calib.json` and prints the **RMS reprojection error** (lower is
   better; well under 1 px is good) and the measured baseline.

Or calibrate **in code** with `live_calibrate` — same interactive session, returns a
`StereoCalibration` you can hand straight to the tracker (no JSON round-trip needed):

```python
from stereohand import StereoHandTracker, live_calibrate

calib = live_calibrate(0, 1)                 # SPACE to capture, ENTER, Y to accept
with StereoHandTracker(calib, ...) as tracker:
    ...                                       # tracking loop, calibration baked in
```

## Run the demo

```bash
python scripts/demo.py --calib stereo_calib.json --left 0 --right 2   # load a saved calib
python scripts/demo.py --calibrate --left 0 --right 1                 # calibrate inline, then track
python scripts/demo.py --mirror                                       # mirror mode (flipped view)
python scripts/demo.py --mirror --smooth 0.3                          # mirrored + extra smoothing
python scripts/demo.py --record docs/demo.mp4                          # record the window to a video file
```

The window shows live camera feeds (top) and a 3D hand skeleton (bottom) with FPS and
palm-centre XYZ overlaid. Press **Q** or **ESC** to quit.

## Configuration

All configuration flows through `StereoHandTracker.open()` and `RenderConfig`.
The demo script (`scripts/demo.py`) exposes the same options as CLI flags.

### `StereoHandTracker.open()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calibration` | `StereoCalibration` | *(required)* | Stereo calibration data (load from JSON via `StereoCalibration.load()`, or obtain from `live_calibrate()`). |
| `left` | `int \| str` | `0` | Left camera source — an integer device index (e.g. `0`) or a string URL / path (e.g. `"http://host:8080/0"`). |
| `right` | `int \| str` | `2` | Right camera source — same format as `left`. |
| `max_skew_s` | `float` | `0.02` | Maximum capture-timestamp difference (seconds) for a frame pair to be accepted. Increase for mismatched or high-latency cameras; decrease for tighter sync. |
| `render` | `bool` | `False` | When `True`, create a cv2 visualisation window. Drive it with `tracker.run()` (blocking) or `tracker.render_step()` (single frame). |
| `render_config` | `RenderConfig \| None` | `None` | Visualisation options (see below). Ignored when `render=False`. Defaults to `RenderConfig()` when `render=True`. |
| `**landmarker_kwargs` | | | Forwarded to both `HandLandmarker` instances (see below). |

#### Landmarker keyword arguments (forwarded via `**landmarker_kwargs`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str \| Path \| None` | `None` | Path to a custom MediaPipe `hand_landmarker.task` model. When `None`, the default Google-published float16 model is auto-downloaded and cached at `~/.cache/stereohand/`. |
| `min_detection_confidence` | `float` | `0.5` | Minimum confidence for the initial hand detection to succeed (0.0–1.0). Lower values detect more hands but with more false positives. |
| `min_tracking_confidence` | `float` | `0.5` | Minimum confidence for frame-to-frame landmark tracking (0.0–1.0). Below this threshold the detector re-runs instead of tracking, which is slower but more robust. |

### `RenderConfig` fields

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

### CLI flags (`scripts/demo.py`)

The demo script maps CLI flags to the parameters above:

| Flag | Maps to | Description |
|---|---|---|
| `--calib PATH` | `StereoCalibration.load(PATH)` | Path to the stereo calibration JSON file. Default: `stereo_calib.json`. |
| `--calibrate` | `live_calibrate(...)` | Run the interactive calibration session before starting the demo. |
| `--left SOURCE` | `left=` | Left camera index or URL. Default: `0`. |
| `--right SOURCE` | `right=` | Right camera index or URL. Default: `2`. |
| `--mirror` | `RenderConfig.mirror` | Enable mirror mode. |
| `--smooth ALPHA` | `RenderConfig.smooth` | EMA smoothing alpha. Default: `0.5`. |
| `--record PATH` | `tracker.run(record_path=PATH)` | Record the composite window output to a video file (e.g. `docs/demo.mp4`). Press Q to stop and save. |

### Programmatic usage with rendering

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

## Running from WSL (Windows camera bridge)

WSL2's kernel has no UVC driver, so your webcams never appear inside WSL (no `/dev/video*`).
The fix: run the cameras on **Windows** and have them serve MJPEG-over-HTTP streams that the
WSL side opens by URL — `stream_webcams.py` serves one stream per camera at `/<index>`.

**One-time WSL prerequisites** — install these once; they persist across shells and reboots:
```bash
sudo apt-get install -y libgles2 python3-tk
```
- `libgles2` — MediaPipe's C bindings need `libGLESv2.so.2` (not installed by default on WSL2).
- `python3-tk` — matplotlib's interactive window (`TkAgg` backend) needs Tkinter; without it the 3D skeleton viewer silently falls back to a non-interactive renderer and nothing appears.

Do the steps in order. Steps 1–2 run **in WSL**, steps 3–4 in a **Windows PowerShell** window,
steps 5–6 **back in WSL**. (Replace `0 1` with your camera indices — the built-in cam is
usually `0`, a USB webcam `1`; try `2` if not.)

1. **(WSL)** Install stereohand and get the Windows-accessible path to the bridge script.
   Windows can reach WSL files directly through the `\\wsl.localhost\` UNC share — no
   copying needed. From the cloned repo root:
   ```bash
   pip install -e ".[demo]"
   SCRIPT=$(wslpath -w "$(realpath scripts/stream_webcams.py)")
   printf '%s\n' "$SCRIPT"
   # e.g. \\wsl.localhost\Ubuntu\home\naveh\...\stream_webcams.py
   ```
   > **Don't use `echo` here** — zsh's `echo` interprets `\n`, `\a`, `\U` etc. inside
   > Windows paths as escape sequences and garbles the output. `printf '%s\n'` prints it
   > literally.

   To skip the copy-paste entirely, pipe straight to the Windows clipboard:
   ```bash
   printf '%s' "$SCRIPT" | clip.exe
   ```
   Then just `Ctrl-V` in PowerShell at step 4.

   **Alternative — copy to Desktop instead:** if you prefer not to use the UNC path, copy
   the script to your Windows Desktop and use `"$env:USERPROFILE\Desktop\stream_webcams.py"`
   in all the steps below:
   ```bash
   cp scripts/stream_webcams.py "$(wslpath "$(powershell.exe -NoProfile -Command \
       '[Environment]::GetFolderPath("Desktop")' | tr -d '\r')")/"
   ```

2. **(WSL)** Install Python on Windows if you don't have it:
   ```bash
   powershell.exe -NoProfile -Command "winget install Python.Python.3.12"
   ```
   Already have Windows Python? Skip this step.

3. **(Windows PowerShell)** Open PowerShell (Start → type "PowerShell" → Enter). Install
   [uv](https://docs.astral.sh/uv/) if you don't have it, create a small venv, and install
   OpenCV:
   ```powershell
   winget install astral-sh.uv        # skip if uv is already installed
   uv venv "$env:USERPROFILE\stereohand-env"
   uv pip install --python "$env:USERPROFILE\stereohand-env\Scripts\python.exe" opencv-python
   ```

4. **(Windows PowerShell)** Start the bridge. Paste the `\\wsl.localhost\...` path printed
   in step 1 (or `"$env:USERPROFILE\Desktop\stream_webcams.py"` if you copied it):
   ```powershell
   $SCRIPT = "\\wsl.localhost\Ubuntu\home\...\stream_webcams.py"  # paste your path here
   & "$env:USERPROFILE\stereohand-env\Scripts\python.exe" $SCRIPT --cameras 0 1
   ```
   You should see one line per camera, and the window stays open (it's serving — leave it
   running):
   ```
   camera 0: http://0.0.0.0:8080/0
   camera 1: http://0.0.0.0:8080/1
   ```
   The first run pops a **Windows Firewall** prompt — tick **Private networks** and *Allow
   access*, or WSL can't connect. (Camera light on? Good. Index wrong? Adjust `--cameras`.)

5. **(WSL)** Find the Windows host address and sanity-check **both** streams:
   ```bash
   WIN=$(ip route show default | awk '{print $3}')    # e.g. 172.20.16.1
   curl -sI "http://$WIN:8080/0" | head -1             # expect: HTTP/1.0 200 OK
   curl -sI "http://$WIN:8080/1" | head -1             # expect: HTTP/1.0 200 OK
   ```
   WSL is a VM with its own network, so Windows isn't `localhost` — it's the NAT gateway
   that `ip route` extracts. `200 OK` confirms the firewall is open and the streams are live.
   (Enabled *mirrored* networking — `networkingMode=mirrored` in `%UserProfile%\.wslconfig`?
   Then Windows *is* `localhost`; set `WIN=localhost`.)

6. **(WSL)** Preview both feeds side by side to confirm they look right before calibrating:
   ```bash
   python scripts/view_cameras.py \
       --left  "http://$WIN:8080/0" \
       --right "http://$WIN:8080/1"
   ```
   Press **Q** to close. Once both cameras look good, run the pipeline:
   ```bash
   python scripts/demo.py --calibrate \
       --left  "http://$WIN:8080/0" \
       --right "http://$WIN:8080/1"
   ```
   `calibrate.py`, `demo.py`, and `live_calibrate(...)` all accept these URLs anywhere they
   take a `--left`/`--right` (or `left=`/`right=`). In code:
   ```python
   from stereohand import StereoHandTracker, live_calibrate
   host = "172.20.16.1"  # the $WIN address from step 5 (or "localhost" with mirrored networking)
   left, right = f"http://{host}:8080/0", f"http://{host}:8080/1"
   calib = live_calibrate(left, right)
   tracker = StereoHandTracker.open(calib, left=left, right=right)
   ```

## How it works

| Stage | Module | What it does |
|---|---|---|
| Capture | `capture.py` | Two daemon-threaded grabbers; delivers only frame pairs within `max_skew_s` (software sync). |
| Calibration | `calibration.py` + `board.py` | ChArUco detection → per-camera intrinsics → `stereoCalibrate` → `stereoRectify` → `P1`/`P2`. Persisted as JSON. |
| Landmarking | `landmarker.py` | MediaPipe **Tasks** `HandLandmarker` (VIDEO mode), one per rectified view, `num_hands=1`. |
| Triangulation | `triangulation.py` | Linear DLT (SVD) on the 21 trivially-corresponded landmarks → `(21, 3)` metric. |
| Orchestration | `tracker.py` | `StereoHandTracker` ties it together; `read()` is the non-blocking public API. |
| Visualisation | `renderer.py` | Optional cv2-based live viewer (`RenderConfig` + `HandRenderer`). FPS, palm XYZ HUD, world axes, mirror mode. |

The 21 landmark indices are MediaPipe's standard hand model (0 = wrist, 4 = thumb tip,
8/12/16/20 = finger tips).

## Limitations

- **One hand** (`num_hands=1`) — two hands break the cross-view correspondence.
- **Teleop-grade, not metrology.** Linear DLT, best-effort software sync, no bundle
  adjustment. Good for "where is the hand," not sub-millimetre measurement.
- **Drop-out is all-or-nothing**: the hand must be visible in *both* views, or the reading
  is `present=False`.
- Smoothing is built in as a simple EMA (see `RenderConfig.smooth`); for advanced use
  cases consider a one-euro filter or Kalman filter downstream.

## Acknowledgements & license

- Linear-DLT triangulation adapted from [handpose3d](https://github.com/TemugeB/handpose3d)
  by Temuge Batpurev (MIT).
- Hand landmarks via [MediaPipe](https://ai.google.dev/edge/mediapipe) (the
  `hand_landmarker` Tasks model, downloaded on first use).

MIT-licensed — see [`LICENSE`](LICENSE).
