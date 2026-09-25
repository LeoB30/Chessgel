from __future__ import annotations
import sys

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


def main() -> None:
    """Entry point for MiniChess application."""
    if len(sys.argv) > 1 and sys.argv[1] in ("--engine", "--uci"):
        from engine.uci import UCIEngine
        engine: UCIEngine = UCIEngine()
        engine.run()
        sys.exit(0)

    from PyQt6.QtWidgets import QApplication
    from gui.app import ChessApp

    app: QApplication = QApplication(sys.argv)
    app.setApplicationName("MiniChess AI")
    window: ChessApp = ChessApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
