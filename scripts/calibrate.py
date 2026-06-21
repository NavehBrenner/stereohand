"""Guided stereo-calibration session: capture ChArUco board pairs → stereo_calib.json.

Manual / hardware path (needs two cameras + a printed board). Run::

    python scripts/calibrate.py --left 0 --right 2 --out stereo_calib.json

Show the printed board (``stereohand-board`` → board.pdf) to BOTH cameras at once. Press
SPACE to capture a pair when the board is visible in both previews; vary angle, distance,
and position across the shared field of view. Collect ~15-25 good pairs, then press ENTER
to calibrate. Press Q to abort.
"""

from __future__ import annotations

import argparse
import sys

import cv2

from stereohand.board import BOARD
from stereohand.calibration import calibrate_from_charuco


def _open(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"could not open camera {source!r}")
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="left camera index or stream URL")
    parser.add_argument("--right", required=True, help="right camera index or stream URL")
    parser.add_argument("--out", default="stereo_calib.json", help="output calibration file")
    parser.add_argument("--min-pairs", type=int, default=12, help="minimum board pairs to accept")
    args = parser.parse_args()

    left_cap, right_cap = _open(args.left), _open(args.right)
    left_frames: list = []
    right_frames: list = []
    print("SPACE = capture pair | ENTER = calibrate | Q = quit")
    try:
        while True:
            ok_l, frame_l = left_cap.read()
            ok_r, frame_r = right_cap.read()
            if not (ok_l and ok_r):
                continue
            preview = cv2.hconcat([frame_l, frame_r])
            cv2.putText(
                preview,
                f"pairs: {len(left_frames)} (need >= {args.min_pairs})",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 230, 0),
                2,
            )
            cv2.imshow("stereo calibration (left | right)", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                left_frames.append(cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY))
                right_frames.append(cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY))
                print(f"captured pair {len(left_frames)}")
            elif key in (13, 10):  # ENTER
                break
            elif key == ord("q"):
                sys.exit("aborted")
    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()

    if len(left_frames) < args.min_pairs:
        sys.exit(f"only {len(left_frames)} pairs captured; need >= {args.min_pairs}")

    print(
        f"calibrating from {len(left_frames)} pairs (board: {BOARD.squares_x}x{BOARD.squares_y})..."
    )
    calib = calibrate_from_charuco(left_frames, right_frames)
    calib.save(args.out)
    print(
        f"wrote {args.out} | RMS reprojection error = {calib.rms:.3f} px | baseline = "
        f"{calib.baseline * 100:.1f} cm"
    )


if __name__ == "__main__":
    main()
