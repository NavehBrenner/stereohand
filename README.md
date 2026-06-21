# stereohand

**Two webcams → metric 3D hand landmarks, in real time.**

Monocular MediaPipe gives you 21 hand landmarks but no real depth. `stereohand` adds a
second camera and recovers **metric** 3D: synchronized dual-webcam capture → one-time
**ChArUco** stereo calibration → per-view **MediaPipe Tasks** `HandLandmarker` →
**linear-DLT** triangulation → `(21, 3)` landmarks in real-world units.

It's a small, dependency-light library with a clean public API:

```python
from stereohand import StereoHandTracker, StereoCalibration

with StereoHandTracker(StereoCalibration.load("stereo_calib.json")) as tracker:
    reading = tracker.read()        # non-blocking
    if reading.present:
        print(reading.landmarks)    # (21, 3) metric xyz
```

> Status: early development. See the [project board] for the build order
> (scaffold → triangulation → ChArUco board → calibration → capture → landmarker →
> tracker → demo → docs). API above is the target shape.

## Install

```bash
pip install git+https://github.com/NavehBrenner/stereohand
```

## Acknowledgements

The linear-DLT triangulation is adapted from
[handpose3d](https://github.com/TemugeB/handpose3d) (MIT). Hand landmarks via
[MediaPipe](https://ai.google.dev/edge/mediapipe). MIT-licensed — see `LICENSE`.

[project board]: https://linear.app/naveh-brenner/project/stereohand-stereo-3d-hand-tracking-27b4f3c2af5d
