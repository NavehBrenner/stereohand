"""Live 3D demo: two webcams → metric hand skeleton.

    python scripts/demo.py --calib stereo_calib.json --left 0 --right 2

A cv2 window shows the live camera feeds (top) and the 3D hand skeleton (bottom),
rendered in the handpose3d style. Press 'q' or ESC (or close the window) to quit.

All rendering is built into ``StereoHandTracker`` — this script is just CLI glue.
"""

from __future__ import annotations

import argparse

from stereohand import RenderConfig, StereoCalibration, StereoHandTracker, live_calibrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", default="stereo_calib.json")
    parser.add_argument(
        "--calibrate", action="store_true", help="run calibration first then start demo"
    )
    parser.add_argument("--left", default="0")
    parser.add_argument("--right", default="2")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="EMA alpha: 1=no smoothing, 0.1=very smooth (default 0.5)",
    )
    parser.add_argument(
        "--mirror", action="store_true", help="flip the view horizontally (mirror mode)"
    )
    parser.add_argument(
        "--recenter",
        action="store_true",
        help="hold an open palm (square to a camera) still for 3 s to re-zero the origin",
    )
    args = parser.parse_args()

    left = int(args.left) if args.left.isdigit() else args.left
    right = int(args.right) if args.right.isdigit() else args.right
    calib = (
        live_calibrate(left, right, save_path=args.calib)
        if args.calibrate
        else StereoCalibration.load(args.calib)
    )

    render_cfg = RenderConfig(mirror=args.mirror, smooth=args.smooth, recenter=args.recenter)

    with StereoHandTracker.open(
        calib,
        left=left,
        right=right,
        render=True,
        render_config=render_cfg,
    ) as tracker:
        tracker.run()


if __name__ == "__main__":
    main()
