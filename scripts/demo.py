"""Live 3D demo: two webcams → metric hand skeleton, with a real depth readout.

Manual / hardware path (needs two cameras + a ``stereo_calib.json`` from ``calibrate.py``,
plus ``pip install stereohand[demo]``)::

    python scripts/demo.py --calib stereo_calib.json --left 0 --right 2

A matplotlib 3D window shows the 21 landmarks as a hand skeleton, and prints the wrist's
metric depth — the thing monocular MediaPipe can't give you. Close the window to quit.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from stereohand import StereoCalibration, StereoHandTracker

# MediaPipe's 21-landmark hand skeleton (bone connectivity), for drawing only.
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm base
]
_WRIST = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", required=True, help="stereo_calib.json from calibrate.py")
    parser.add_argument("--left", default="0", help="left camera index or stream URL")
    parser.add_argument("--right", default="2", help="right camera index or stream URL")
    args = parser.parse_args()

    calib = StereoCalibration.load(args.calib)
    left = int(args.left) if args.left.isdigit() else args.left
    right = int(args.right) if args.right.isdigit() else args.right

    plt.ion()
    figure = plt.figure("stereohand — metric 3D hand")
    axes = figure.add_subplot(111, projection="3d")

    with StereoHandTracker.open(calib, left=left, right=right) as tracker:
        while plt.fignum_exists(figure.number):
            reading = tracker.read()
            axes.cla()
            axes.set_xlabel("x (m)")
            axes.set_ylabel("y (m)")
            axes.set_zlabel("z (m)")
            if reading.present:
                points = reading.landmarks
                axes.scatter(points[:, 0], points[:, 1], points[:, 2], c="crimson", s=20)
                for a, b in HAND_CONNECTIONS:
                    axes.plot(*zip(points[a], points[b], strict=True), c="black", lw=1)
                axes.set_title(
                    f"{reading.handedness} hand | wrist depth = {points[_WRIST, 2]:.3f} m"
                )
            else:
                axes.set_title("no hand (present in both views required)")
            plt.pause(0.01)


if __name__ == "__main__":
    main()
