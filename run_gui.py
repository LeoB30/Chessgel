from __future__ import annotations
import tkinter as tk
from gui.app import ChessApp


def main() -> None:
    """Launches MiniChess Graphical User Interface."""
    root: tk.Tk = tk.Tk()
    app: ChessApp = ChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
