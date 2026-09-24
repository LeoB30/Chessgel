"""Application entry point configuring High-DPI scaling and launching MainWindow."""

from __future__ import annotations
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from chess_tracker.config import WINDOW_HEIGHT, WINDOW_WIDTH
from chess_tracker.ui.main_window import MainWindow


def main() -> None:
    # Explicit High-DPI scale factor rounding policy
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Antigravity Chess Tracker")
    app.setOrganizationName("Antigravity")

    window = MainWindow()
    window.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
