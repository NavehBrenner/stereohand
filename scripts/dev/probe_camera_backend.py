"""Time how long each OpenCV backend takes to open the cameras.

Run on the machine the cameras are attached to (Windows):
    python scripts/dev/probe_camera_backend.py 0 1

If MSMF takes ~10s and DSHOW takes <1s, the fix is to force CAP_DSHOW.
"""

import sys
import time

import cv2

BACKENDS = {
    "DEFAULT": cv2.CAP_ANY,
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
}


def main() -> None:
    sources = [int(s) for s in sys.argv[1:]] or [0]
    for source in sources:
        for name, backend in BACKENDS.items():
            start = time.monotonic()
            cap = cv2.VideoCapture(source, backend)
            opened = cap.isOpened()
            ok, _ = cap.read() if opened else (False, None)
            cap.release()
            elapsed = time.monotonic() - start
            print(f"cam {source} via {name:8s}: open={opened} read={ok} took {elapsed:5.2f}s")


if __name__ == "__main__":
    main()
