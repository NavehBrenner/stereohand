# Recentering Handoff — stereohand

Entry point for tuning the **recenter gesture** ("hold an open palm to re-zero the world
origin"). It currently "feels clunky" — hard to trigger and/or too easy to break mid-hold.

## Status (2026-06-23)
- Branch: `feat/tracker-event-gate-fps-cap`.
- Committed: event-gated tracker `_run` + `max_fps` cap (`ce48b44`); 3D-axis-label fix
  (`f1ac025`). These came out of a kevin teleop-latency debugging session.
- Sibling handoff: `../kevin/docs/recentering-handoff.md` (branch
  `feat/lab-74-stereo-teleop-debug`).

## Problem to solve
The recenter gesture — open palm, square to the camera, held **still** for
`_RECENTER_HOLD_S` (3 s) → re-zeros the world origin to the palm — feels clunky. Likely the
"held still" move-tolerance and/or the pose test are too strict, so it rarely completes or
resets partway.

## Where it lives — `src/stereohand/renderer.py` (line numbers approximate)
- `_RECENTER_HOLD_S = 3.0` (~L88) — hold duration.
- Pose test (~L100–120): fingers-extended ratio (~1.4×) + squareness (palm normal ≈ camera
  z, ~0.7). The squareness test is the finicky one.
- `_update_recenter()` (~L373): advances the hold; tracks `_hold_anchor` (start position)
  and a **move tolerance** — if the palm drifts past it, the hold resets. *The "still" knob.*
- Dropout handling (~L333–342): a brief MediaPipe dropout holds the countdown alive; a real
  hand loss resets the timer. `_recentered` latches so it fires once per hold.
- Enabled via `RenderConfig(recenter=True)`; countdown HUD draws top-centre (~L230).

## Note — kevin has a SEPARATE copy
kevin ports its own pose test (`kevin/src/ai_teleop/input/hand_tracker.py:_palm_open_facing`)
and its own recenter timer (`vision_input.py`) with the *same thresholds*. Tuning here does
**not** change kevin's path — keep the two in sync if you retune.

## Run the demo
See `README.md` (run section) for the exact entry command; it uses `stereo_calib.json` and
the two camera streams. Standalone, the tracking renders at ~60 fps.

## Tooling
- Own `.venv`; plain pip/uv — **no poe, no wiki**.
- `.venv/bin/python -m {ruff check, mypy, pytest}`.
- **Pre-commit hook runs `ruff format`** and auto-stages, so commits reformat staged files.
- Branch-per-feature → PR; `master` is PR-only.
