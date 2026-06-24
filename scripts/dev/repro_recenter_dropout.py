"""Repro: does the recenter countdown survive periodic MediaPipe dropouts?

Drives the cv2-free recenter state machine (HandRenderer._advance_pose) with a steady
open-palm pose at 30 fps, injecting a 1-frame dropout every N frames, and reports whether
the 3 s hold ever completes.
"""

from __future__ import annotations

import numpy as np

from stereohand.renderer import HandRenderer, RenderConfig


def open_palm() -> np.ndarray:
    """21x3 open hand, palm square to camera (the recenter pose)."""
    pts = np.zeros((21, 3))
    pts[0] = [0.0, 0.0, 0.5]
    spread = (0.02, 0.0, -0.02, -0.04)
    for mcp, tip, s in zip((5, 9, 13, 17), (8, 12, 16, 20), spread, strict=True):
        pts[mcp] = pts[0] + np.array([s, -0.05, 0.0])
        pts[tip] = pts[0] + np.array([s, -0.12, 0.0])
    return pts


def fresh_renderer() -> HandRenderer:
    r = HandRenderer.__new__(HandRenderer)
    r._cfg = RenderConfig(recenter=True, smooth=0.5)
    r._smoothed = None
    r._origin = np.zeros(3)
    r._hold_start = None
    r._hold_anchor = None
    r._recentered = False
    r._calib_msg = None
    r._last_seen_t = None
    r._last_good_pose_t = None
    return r


def run(dropout_every: int, fps: float = 30.0, dropout_len: int = 1) -> bool:
    r = fresh_renderer()
    palm = open_palm()
    fired = False
    n = int(fps * 7)
    for i in range(n):
        now = i / fps
        present = not (dropout_every and i > 0 and (i % dropout_every) < dropout_len)
        # tiny jitter on reacquisition, like real MediaPipe (< move tolerance)
        lm = palm + np.random.default_rng(i).normal(0, 0.003, palm.shape) if present else None
        r._advance_pose(now, present, lm)
        if r._recentered and not fired:
            print(f"    RECENTERED at t={now:.2f}s (frame {i})")
            fired = True
    if not fired:
        print(f"    never recentered (msg={r._calib_msg!r})")
    return fired


if __name__ == "__main__":
    print("Open palm held steady; recenter should fire ~3 s in despite dropouts:")
    for fps in (30.0, 10.0):
        print(f"  fps={fps:g}")
        for every, dlen in ((0, 1), (10, 1), (6, 2), (4, 2)):
            label = "baseline" if every == 0 else f"miss {dlen}/{every} frames"
            print(f"   {label}:")
            run(dropout_every=every, fps=fps, dropout_len=dlen)
