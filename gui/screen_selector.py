from __future__ import annotations
import tkinter as tk
from typing import Callable


class ScreenRegionSelector:
    """Full-screen interactive overlay allowing the user to click and drag a bounding box."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        on_region_selected: Callable[[tuple[int, int, int, int]], None],
    ) -> None:
        self.parent: tk.Tk | tk.Toplevel = parent
        self.on_region_selected: Callable[[tuple[int, int, int, int]], None] = on_region_selected

        self.start_x: int = 0
        self.start_y: int = 0
        self.start_x_root: int = 0
        self.start_y_root: int = 0
        self.rect_id: int | None = None
        self.label_id: int | None = None

        self.overlay: tk.Toplevel = tk.Toplevel(parent)
        self.overlay.attributes("-fullscreen", True)
        self.overlay.attributes("-alpha", 0.35)
        self.overlay.attributes("-topmost", True)
        self.overlay.configure(bg="#11111B", cursor="crosshair")

        self.canvas: tk.Canvas = tk.Canvas(
            self.overlay,
            bg="#11111B",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)
        self.overlay.bind("<Escape>", self._on_cancel)

        # Instructions banner at top center
        sw: int = self.overlay.winfo_screenwidth()
        self.canvas.create_text(
            sw // 2,
            40,
            text="🎯 Drag a rectangle around the chessboard. Press ESC to cancel.",
            font=("Segoe UI", 14, "bold"),
            fill="#89B4FA",
        )

    def _on_button_press(self, event: tk.Event) -> None:
        self.start_x = event.x
        self.start_y = event.y
        self.start_x_root = event.x_root
        self.start_y_root = event.y_root
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        if self.label_id is not None:
            self.canvas.delete(self.label_id)

        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="#A6E3A1",
            width=3,
        )

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self.rect_id is None:
            return

        cur_x: int = event.x
        cur_y: int = event.y
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

        w: int = abs(cur_x - self.start_x)
        h: int = abs(cur_y - self.start_y)
        mid_x: int = min(self.start_x, cur_x) + w // 2
        mid_y: int = min(self.start_y, cur_y) + h // 2

        dim_text: str = f"{w} × {h} px"
        if self.label_id is not None:
            self.canvas.delete(self.label_id)

        self.label_id = self.canvas.create_text(
            mid_x,
            mid_y,
            text=dim_text,
            font=("Segoe UI", 12, "bold"),
            fill="#FFFFFF",
        )

    def _on_button_release(self, event: tk.Event) -> None:
        end_x_root: int = event.x_root
        end_y_root: int = event.y_root

        left: int = min(self.start_x_root, end_x_root)
        top: int = min(self.start_y_root, end_y_root)
        width: int = abs(end_x_root - self.start_x_root)
        height: int = abs(end_y_root - self.start_y_root)

        self._close()

        if width >= 50 and height >= 50:
            self.on_region_selected((left, top, width, height))

    def _on_cancel(self, event: tk.Event) -> None:
        self._close()

    def _close(self) -> None:
        try:
            self.overlay.destroy()
        except Exception:
            pass
