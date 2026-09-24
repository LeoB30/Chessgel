"""Test suite validating vision perception, contour detection, and move delta tracking."""

from __future__ import annotations
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import cv2
import numpy as np

from chess_tracker.vision.capture import detect_chessboard_contour
from chess_tracker.vision.detector import ChessBoardDetector


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(BASE_DIR, "reference_images")


def test_board_contour_detection():
    """Validates automatic board contour detection on full screenshot context."""
    screen_path = os.path.join(REF_DIR, "screen_context.png")
    assert os.path.isfile(screen_path), f"Missing {screen_path}"

    img = cv2.imread(screen_path)
    box = detect_chessboard_contour(img)
    assert box is not None, "Failed to detect chessboard contour on screen_context.png"

    bx, by, bw, bh = box
    # Should detect the ~597x597 board in the screenshot around x=58, y=129
    assert 50 <= bx <= 70, f"Unexpected bx: {bx}"
    assert 120 <= by <= 140, f"Unexpected by: {by}"
    assert 580 <= bw <= 610, f"Unexpected bw: {bw}"
    assert 580 <= bh <= 610, f"Unexpected bh: {bh}"


def test_move_sequence_detection_and_highlight_immunity():
    """Validates sequential move tracking across 1. e4 and 1... d5 with yellow highlights."""
    img_start = cv2.imread(os.path.join(REF_DIR, "initial_board.png"))
    img_e4 = cv2.imread(os.path.join(REF_DIR, "move1_e4.png"))
    img_d5 = cv2.imread(os.path.join(REF_DIR, "move2_d5.png"))

    box_start = detect_chessboard_contour(img_start)
    box_e4 = detect_chessboard_contour(img_e4)
    box_d5 = detect_chessboard_contour(img_d5)

    assert box_start and box_e4 and box_d5

    board_start = img_start[box_start[1]:box_start[1]+box_start[3], box_start[0]:box_start[0]+box_start[2]]
    board_e4 = img_e4[box_e4[1]:box_e4[1]+box_e4[3], box_e4[0]:box_e4[0]+box_e4[2]]
    board_d5 = img_d5[box_d5[1]:box_d5[1]+box_d5[3], box_d5[0]:box_d5[0]+box_d5[2]]

    detector = ChessBoardDetector(reference_image_path=os.path.join(REF_DIR, "initial_board.png"), debounce_required=1)
    game_board = chess.Board()

    # Step 1: Initialize features on starting board
    _, f0 = detector.process_frame(board_start, game_board)
    assert f0[chess.E2].is_occupied is True
    assert f0[chess.E4].is_occupied is False

    # Step 2: Feed board after 1. e4 (e2 emptied with yellow highlight, e4 occupied with yellow highlight)
    m1, f1 = detector.process_frame(board_e4, game_board)
    assert m1 is not None, "Failed to detect move 1. e4"
    assert m1.uci() == "e2e4", f"Expected e2e4, got {m1.uci()}"
    # Verify highlight immunity: e2 is detected as empty despite the bright yellow highlight
    assert f1[chess.E2].is_occupied is False, "e2 should be recognized as empty despite highlight"
    assert f1[chess.E4].is_occupied is True, "e4 should be recognized as occupied"

    game_board.push(m1)

    # Step 3: Feed board after 1... d5 (d7 emptied with yellow highlight, d5 occupied with yellow highlight)
    m2, f2 = detector.process_frame(board_d5, game_board)
    assert m2 is not None, "Failed to detect move 1... d5"
    assert m2.uci() == "d7d5", f"Expected d7d5, got {m2.uci()}"
    # Verify highlight immunity: d7 is recognized as empty despite the bright yellow highlight
    assert f2[chess.D7].is_occupied is False, "d7 should be recognized as empty despite highlight"
    assert f2[chess.D5].is_occupied is True, "d5 should be recognized as occupied"

    game_board.push(m2)
    assert game_board.fen() == "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


if __name__ == "__main__":
    test_board_contour_detection()
    print("test_board_contour_detection passed!")
    test_move_sequence_detection_and_highlight_immunity()
    print("test_move_sequence_detection_and_highlight_immunity passed!")
