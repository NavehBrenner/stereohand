"""Guided stereo-calibration session → stereo_calib.json (thin wrapper over live_calibrate).

Manual / hardware path (two cameras + a printed board from ``stereohand-board``). Run::

    python scripts/calibrate.py --left 0 --right 1 --out stereo_calib.json

Show the printed board to BOTH cameras; SPACE captures a pair when the board is seen in both
previews (or use --auto). Vary angle, distance, and position. ENTER finishes; then Y accepts
or R redoes. For the equivalent in code, call ``stereohand.live_calibrate`` directly.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_cv2_spec = importlib.util.find_spec("cv2")
if _cv2_spec and _cv2_spec.origin:
    Path(_cv2_spec.origin).parent.joinpath("qt", "fonts").mkdir(parents=True, exist_ok=True)

from stereohand import live_calibrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="left camera index or stream URL")
    parser.add_argument("--right", required=True, help="right camera index or stream URL")
    parser.add_argument("--out", default="stereo_calib.json", help="output calibration file")
    parser.add_argument("--min-pairs", type=int, default=15, help="board pairs to collect")
    parser.add_argument("--auto", action="store_true", help="auto-capture when board is in view")
    args = parser.parse_args()

    left = int(args.left) if args.left.isdigit() else args.left
    right = int(args.right) if args.right.isdigit() else args.right
    calib = live_calibrate(
        left, right, min_pairs=args.min_pairs, auto_capture=args.auto, save_path=args.out
    )
    print(f"wrote {args.out} | RMS = {calib.rms:.3f} px | baseline = {calib.baseline * 100:.1f} cm")


if __name__ == "__main__":
    main()
