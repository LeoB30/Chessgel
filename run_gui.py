from __future__ import annotations
import sys
from PyQt6.QtWidgets import QApplication

# Enable Windows Per-Monitor DPI Awareness for pixel-perfect screen coordinate capture
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

from gui.app import ChessApp


def main() -> None:
    """Launches MiniChess PyQt6 Graphical User Interface."""
    qt_app: QApplication = QApplication(sys.argv)
    qt_app.setApplicationName("MiniChess AI")
    window: ChessApp = ChessApp()
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--engine", "--uci"):
        from engine.uci import UCIEngine
        engine: UCIEngine = UCIEngine()
        engine.run()
        sys.exit(0)
    main()
