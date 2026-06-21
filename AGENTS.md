# stereohand Project Operations

## Overview
`stereohand` computes metric 3D hand landmarks in real-time from two webcams using MediaPipe. It outputs an array of `(21, 3)` coordinates in meters.

## Setup & Environment
- Environment: Python 3.12+ virtual environment.
- Installation: `pip install -e ".[dev,demo]"`
- Enable git hooks once per clone: `git config core.hooksPath .githooks`
- Tools: `ruff` (lint/format), `mypy` (types), `pytest` (tests).

## Git workflow & hooks
- `master` is the default branch and is **protected** — no direct pushes; all changes land via PR.
- Work on a feature branch, open a PR, let CI pass, merge in the GitHub UI.
- Local git hooks live in `.githooks/` (version-controlled). Enable them once per clone:
  ```bash
  git config core.hooksPath .githooks
  ```
  - **pre-commit**: `ruff format` on staged Python, re-staging what it touches.
  - **pre-push**: full CI gate — `ruff check` + `ruff format --check` + `mypy src` + `pytest`.
  Both no-op gracefully if `.venv` is missing.

## Key Operational Scripts
The project provides several scripts in the `scripts/` directory for core operations:

1. **Calibration (`scripts/calibrate.py`)**
   - **What it does**: Runs an interactive ChArUco board calibration session.
   - **Usage**: `python scripts/calibrate.py --left 0 --right 2 --out stereo_calib.json`
   - **Requirement**: Needs the printed board (`board.pdf` at 100% scale).

2. **Live Tracker Demo (`scripts/demo.py`)**
   - **What it does**: Opens the 3D landmark visualization window using the calibrated setup.
   - **Usage**: `python scripts/demo.py --calib stereo_calib.json --left 0 --right 2`
   - **Options**: Use `--mirror` to flip the view, or `--smooth 0.3` to apply EMA smoothing.

3. **WSL Webcam Bridge (`scripts/stream_webcams.py`)**
   - **What it does**: Since WSL cannot natively access USB webcams, this script is run on the Windows host to stream the webcams over HTTP to the WSL tracker.

## API Documentation
For programmatic integration, refer to the dedicated `docs/` folder.
- See **[docs/api.md](docs/api.md)** for the full documentation of the `StereoHandTracker` API, `RenderConfig`, and the `StereoCalibration` pipeline.
