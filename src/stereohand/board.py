"""The ChArUco calibration board — one source of truth, plus a printable generator.

Both board *generation* (here) and board *detection* (during calibration) must agree on
the exact same parameters, or detection silently fails — so the canonical spec lives in
one place (``BOARD``) and the calibrator imports it.

ChArUco (a checkerboard with ArUco markers in the white squares) is preferred over a plain
checkerboard: the markers make the board identifiable even when partially occluded or seen
at a steep angle — which is exactly what happens with two cameras viewing it off-axis.
Needs ``opencv-contrib-python`` (``cv2.aruco``).

The spec itself (``CharucoBoardSpec``) is dependency-free; ``make_board`` / ``render_board``
lazily import cv2 + Pillow so the spec can be imported anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_INCH_M = 0.0254
A4_WIDTH_M = 0.210
A4_HEIGHT_M = 0.297


@dataclass(frozen=True)
class CharucoBoardSpec:
    """Canonical ChArUco board parameters. Lengths in metres (real-world scale).

    ``marker_length_m`` must be smaller than ``square_length_m`` (the marker sits inside a
    white square). Default 5×7 squares at 35 mm fits A4 with margins.
    """

    squares_x: int = 5
    squares_y: int = 7
    square_length_m: float = 0.035
    marker_length_m: float = 0.026
    dictionary: str = "DICT_5X5_100"

    @property
    def width_m(self) -> float:
        return self.squares_x * self.square_length_m

    @property
    def height_m(self) -> float:
        return self.squares_y * self.square_length_m

    @property
    def n_chessboard_corners(self) -> int:
        """Interior chessboard corners — the points calibration actually localizes."""
        return (self.squares_x - 1) * (self.squares_y - 1)


# The one board the whole package uses.
BOARD = CharucoBoardSpec()


def make_board(spec: CharucoBoardSpec = BOARD) -> Any:
    """Construct the ``cv2.aruco.CharucoBoard`` for ``spec`` (cv2-gated)."""
    import cv2

    if spec.marker_length_m >= spec.square_length_m:
        raise ValueError("marker_length_m must be smaller than square_length_m")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )


def render_board(path: str | Path, *, spec: CharucoBoardSpec = BOARD, dpi: int = 300) -> Path:
    """Render the board centered on an A4 page at true physical scale.

    Pixel dimensions are computed from the board's metric size and ``dpi``, so printing the
    output **at 100% (no fit-to-page)** reproduces ``square_length_m`` exactly — which is
    what makes the downstream calibration metric. Output format follows the file
    extension (``.pdf`` recommended; ``.png`` also supported, with DPI metadata).
    """
    from PIL import Image

    board = make_board(spec)
    px_per_m = dpi / _INCH_M
    board_w = round(spec.width_m * px_per_m)
    board_h = round(spec.height_m * px_per_m)
    board_img = board.generateImage((board_w, board_h))

    a4_w = round(A4_WIDTH_M * px_per_m)
    a4_h = round(A4_HEIGHT_M * px_per_m)
    canvas = np.full((a4_h, a4_w), 255, dtype=np.uint8)
    x0 = (a4_w - board_w) // 2
    y0 = (a4_h - board_h) // 2
    canvas[y0 : y0 + board_h, x0 : x0 + board_w] = board_img

    path = Path(path)
    image = Image.fromarray(canvas)
    if path.suffix.lower() == ".pdf":
        image.save(path, "PDF", resolution=float(dpi))
    else:
        image.save(path, dpi=(dpi, dpi))
    return path


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render the stereohand ChArUco board to print.")
    parser.add_argument("-o", "--out", default="board.pdf", help="output file (.pdf or .png)")
    parser.add_argument("--dpi", type=int, default=300, help="print resolution (default 300)")
    args = parser.parse_args()
    out = render_board(args.out, dpi=args.dpi)
    print(
        f"Wrote {out} — {BOARD.squares_x}x{BOARD.squares_y} squares @ "
        f"{BOARD.square_length_m * 1000:.0f} mm. Print at 100% (no fit-to-page); "
        f"verify a printed square measures {BOARD.square_length_m * 1000:.0f} mm."
    )


if __name__ == "__main__":
    _cli()
