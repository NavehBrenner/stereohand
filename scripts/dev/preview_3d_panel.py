"""Render one 3D panel to a PNG so the axis/hand sizing can be eyeballed headlessly."""

from __future__ import annotations

import cv2
import numpy as np

from stereohand.renderer import _R, _render_hand_3d


def open_palm_world(palm_depth_m: float = 0.45) -> np.ndarray:
    """A 21x3 open hand sitting `palm_depth_m` out from the camera, in render-world coords."""
    pts = np.zeros((21, 3))
    pts[0] = [0.05, 0.05, palm_depth_m]  # wrist offset from origin, like a real hand
    spread = (0.02, 0.0, -0.02, -0.04)
    for mcp, tip, s in zip((5, 9, 13, 17), (8, 12, 16, 20), spread, strict=True):
        pts[mcp] = pts[0] + np.array([s, -0.03, 0.0])
        pts[tip] = pts[0] + np.array([s, -0.10, 0.0])
    return (_R @ pts.T).T  # caller normally applies _R before _render_hand_3d


if __name__ == "__main__":
    canvas = _render_hand_3d(open_palm_world(), width=960, palm_xyz=np.array([0.05, 0.05, 0.45]))
    out = "scripts/dev/preview_3d_panel.png"
    cv2.imwrite(out, canvas)
    print(f"wrote {out}")
