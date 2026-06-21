"""Board spec + generator tests. Spec checks run in CI; generator checks are cv2-gated."""

from __future__ import annotations

import pytest

from stereohand.board import BOARD, CharucoBoardSpec, make_board, render_board


def test_marker_smaller_than_square():
    # Pure spec invariant — runs in CI without cv2.
    assert BOARD.marker_length_m < BOARD.square_length_m


def test_spec_derived_dimensions():
    spec = CharucoBoardSpec(squares_x=5, squares_y=7, square_length_m=0.035)
    assert spec.width_m == pytest.approx(0.175)
    assert spec.height_m == pytest.approx(0.245)
    assert spec.n_chessboard_corners == 4 * 6


def test_make_board_corner_count():
    pytest.importorskip("cv2")
    board = make_board()
    # cv2's interior-corner list must match our spec's count.
    assert board.getChessboardCorners().shape[0] == BOARD.n_chessboard_corners


def test_make_board_rejects_oversized_marker():
    pytest.importorskip("cv2")
    bad = CharucoBoardSpec(square_length_m=0.03, marker_length_m=0.03)
    with pytest.raises(ValueError):
        make_board(bad)


def test_render_board_writes_pdf(tmp_path):
    pytest.importorskip("cv2")
    out = render_board(tmp_path / "board.pdf", dpi=150)
    assert out.exists()
    assert out.stat().st_size > 0
