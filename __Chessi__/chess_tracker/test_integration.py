"""Integration test verifying GUI, vision signal handling, and engine synchronization."""

from __future__ import annotations
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import chess
import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from chess_tracker.ui.main_window import MainWindow
from chess_tracker.vision.capture import detect_chessboard_contour


def test_full_application_flow():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = MainWindow()
    assert win.width() == 995 and win.height() == 960, f"Window size is {win.width()}x{win.height()}"

    # Load reference images
    ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")
    img_start = cv2.imread(os.path.join(ref_dir, "initial_board.png"))
    img_e4 = cv2.imread(os.path.join(ref_dir, "move1_e4.png"))
    img_d5 = cv2.imread(os.path.join(ref_dir, "move2_d5.png"))

    # Verify board mirror starts at starting position
    assert win.game_state.get_fen() == chess.STARTING_FEN

    # Simulate Move 1 (1. e4) detected by vision worker
    m1 = chess.Move.from_uci("e2e4")
    win._on_vision_move(m1)
    app.processEvents()

    assert win.game_state.last_move == m1
    assert win.board_mirror.last_move == m1
    assert win.move_table.rowCount() == 1
    assert win.move_table.item(0, 1).text() == "e4"
    assert "4P3" in win.txt_fen.text()

    # Simulate Move 2 (1... d5) detected by vision worker
    m2 = chess.Move.from_uci("d7d5")
    win._on_vision_move(m2)
    app.processEvents()

    assert win.game_state.last_move == m2
    assert win.board_mirror.last_move == m2
    assert win.move_table.rowCount() == 1
    assert win.move_table.item(0, 2).text() == "d5"

    # Verify PGN export
    pgn_str = win.game_state.get_pgn()
    assert "1. e4 d5" in pgn_str

    win.close()
    print("Full application integration flow test PASSED!")


if __name__ == "__main__":
    test_full_application_flow()
