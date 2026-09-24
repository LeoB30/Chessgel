"""Simulates live vision streaming frames through VisionWorker and verifies move signals."""

from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import chess
import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from chess_tracker.ui.main_window import MainWindow


def test_vision_stream():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = MainWindow()

    ref_dir = os.path.join(os.path.dirname(__file__), "reference_images")
    img_start = cv2.imread(os.path.join(ref_dir, "initial_board.png"))[15:15+597, 10:10+597]
    img_e4 = cv2.imread(os.path.join(ref_dir, "move1_e4.png"))[10:10+597, 8:8+597]
    img_d5 = cv2.imread(os.path.join(ref_dir, "move2_d5.png"))[6:6+597, 5:5+597]

    # Configure detector debounce = 1 for deterministic frame-by-frame feeding
    win.vision_worker.detector.debounce_required = 1

    # Feed Frame 0 (Starting position)
    m0, _ = win.vision_worker.detector.process_frame(img_start, win.game_state.board)
    assert m0 is None

    # Feed Frame 1 (1. e4 with yellow highlight)
    m1, _ = win.vision_worker.detector.process_frame(img_e4, win.game_state.board)
    assert m1 is not None and m1.uci() == "e2e4"
    win._on_vision_move(m1)
    app.processEvents()

    assert win.game_state.board.piece_at(chess.E4).symbol() == "P"
    assert win.game_state.board.piece_at(chess.E2) is None

    # Feed Frame 2 (1... d5 with yellow highlight)
    m2, _ = win.vision_worker.detector.process_frame(img_d5, win.game_state.board)
    assert m2 is not None and m2.uci() == "d7d5"
    win._on_vision_move(m2)
    app.processEvents()

    assert win.game_state.board.piece_at(chess.D5).symbol() == "p"
    assert win.game_state.board.piece_at(chess.D7) is None

    print("Live vision frame simulation test PASSED!")
    win.close()


if __name__ == "__main__":
    test_vision_stream()
