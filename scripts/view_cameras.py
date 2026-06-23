"""Show two camera feeds side by side — sanity check before calibrating or running the demo.

Works with camera indices (native) or stream URLs (WSL via Windows bridge)::

    python scripts/view_cameras.py --left 0 --right 1
    python scripts/view_cameras.py --left "http://172.20.16.1:8080/0" --right "http://172.20.16.1:8080/1"

Press Q to quit.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_cv2_spec = importlib.util.find_spec("cv2")
if _cv2_spec and _cv2_spec.origin:
    Path(_cv2_spec.origin).parent.joinpath("qt", "fonts").mkdir(parents=True, exist_ok=True)

import cv2  # noqa: E402  (after cv2-Qt font path fix above)

from stereohand.capture import open_capture  # noqa: E402  (after cv2-Qt font path fix above)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", default="0", help="left camera index or stream URL")
    parser.add_argument("--right", default="2", help="right camera index or stream URL")
    args = parser.parse_args()

    left: int | str = int(args.left) if args.left.isdigit() else args.left
    right: int | str = int(args.right) if args.right.isdigit() else args.right

    cap_l = open_capture(left)
    cap_r = open_capture(right)

    try:
        while True:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()
            if not ok_l or not ok_r:
                print("camera read failed — check indices / URLs")
                break
            cv2.imshow("stereohand preview (Q to quit)", cv2.hconcat([frame_l, frame_r]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
