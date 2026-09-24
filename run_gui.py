from __future__ import annotations
import sys
import tkinter as tk

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
    """Launches MiniChess Graphical User Interface."""
    root: tk.Tk = tk.Tk()
    app: ChessApp = ChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--engine":
        from engine.uci import UCIEngine
        engine: UCIEngine = UCIEngine()
        engine.run()
        sys.exit(0)
    main()
