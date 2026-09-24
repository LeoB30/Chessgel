from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from PIL import Image, ImageTk


class GridAlignmentInspector:
    """Live visualizer displaying the 8x8 segmented grid overlay over the captured screen image."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        get_outer_margin: Callable[[], float],
        set_outer_margin: Callable[[float], None],
        get_inner_inset: Callable[[], float],
        set_inner_inset: Callable[[float], None],
        on_close_callback: Optional[Callable[[], None]] = None,
        on_toggle_overlay: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self.parent: tk.Tk | tk.Toplevel = parent
        self.get_outer_margin: Callable[[], float] = get_outer_margin
        self.set_outer_margin: Callable[[float], None] = set_outer_margin
        self.get_inner_inset: Callable[[], float] = get_inner_inset
        self.set_inner_inset: Callable[[float], None] = set_inner_inset
        self.on_close_callback: Optional[Callable[[], None]] = on_close_callback
        self.on_toggle_overlay: Optional[Callable[[bool], None]] = on_toggle_overlay

        self.window: tk.Toplevel = tk.Toplevel(parent)
        self.window.title("MiniChess - Vision Grid Alignment Inspector")
        self.window.geometry("560x640")
        self.window.minsize(440, 500)
        self.window.configure(bg="#181825")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._current_photo: Optional[ImageTk.PhotoImage] = None
        self._last_raw_image: Optional[Image.Image] = None
        self.debug_overlay_var: tk.BooleanVar = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        # Control Toolbar at Top
        ctrl_frame = tk.Frame(self.window, bg="#11111B", padx=10, pady=8)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        # Margin and Inset controls row
        row1 = tk.Frame(ctrl_frame, bg="#11111B")
        row1.pack(fill=tk.X, pady=(0, 4))

        tk.Label(row1, text="Outer Margin %:", font=("Segoe UI", 9, "bold"), bg="#11111B", fg="#89B4FA").pack(side=tk.LEFT, padx=(0, 4))
        self.outer_margin_var = tk.DoubleVar(value=self.get_outer_margin())
        self.outer_spin = ttk.Spinbox(
            row1,
            from_=0.0,
            to=15.0,
            increment=0.5,
            textvariable=self.outer_margin_var,
            width=5,
            command=self._on_margin_changed,
        )
        self.outer_spin.pack(side=tk.LEFT, padx=(0, 10))
        self.outer_spin.bind("<Return>", lambda e: self._on_margin_changed())

        tk.Label(row1, text="Inner Inset %:", font=("Segoe UI", 9, "bold"), bg="#11111B", fg="#A6E3A1").pack(side=tk.LEFT, padx=(0, 4))
        self.inner_inset_var = tk.DoubleVar(value=self.get_inner_inset() * 100.0)
        self.inner_spin = ttk.Spinbox(
            row1,
            from_=2.0,
            to=25.0,
            increment=1.0,
            textvariable=self.inner_inset_var,
            width=5,
            command=self._on_inset_changed,
        )
        self.inner_spin.pack(side=tk.LEFT, padx=(0, 10))
        self.inner_spin.bind("<Return>", lambda e: self._on_inset_changed())

        # Debug Overlay Toggle
        tk.Checkbutton(
            row1,
            text="CV Grid & Confidence Overlay",
            variable=self.debug_overlay_var,
            command=self._on_debug_toggle,
            bg="#11111B",
            fg="#F9E2AF",
            activebackground="#11111B",
            selectcolor="#181825",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Legend / Guide row
        legend_row = tk.Frame(ctrl_frame, bg="#11111B")
        legend_row.pack(fill=tk.X, pady=(2, 0))

        tk.Label(
            legend_row,
            text="Green: 8x8 Grid | Yellow: Inner ROI | Labels: Square:Piece & Match Confidence %",
            font=("Segoe UI", 8, "italic"),
            bg="#11111B",
            fg="#BAC2DE",
        ).pack(side=tk.LEFT)

        # Main Canvas Area
        canvas_container = tk.Frame(self.window, bg="#181825", padx=8, pady=8)
        canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas: tk.Canvas = tk.Canvas(
            canvas_container,
            bg="#11111B",
            highlightthickness=1,
            highlightbackground="#45475A",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def update_frame(self, overlay_image: Image.Image) -> None:
        """Renders the annotated overlay image scaled to fit the inspector canvas."""
        self._last_raw_image = overlay_image
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()

        if c_w <= 10 or c_h <= 10:
            c_w = 480
            c_h = 480

        target_size = max(100, min(c_w, c_h) - 10)
        resized = overlay_image.resize((target_size, target_size), Image.Resampling.BILINEAR)
        self._current_photo = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(c_w // 2, c_h // 2, image=self._current_photo)

    def _on_debug_toggle(self) -> None:
        if self.on_toggle_overlay is not None:
            self.on_toggle_overlay(self.debug_overlay_var.get())

    def _on_margin_changed(self) -> None:
        try:
            val = float(self.outer_margin_var.get())
            self.set_outer_margin(val)
        except Exception:
            pass

    def _on_inset_changed(self) -> None:
        try:
            val = float(self.inner_inset_var.get()) / 100.0
            self.set_inner_inset(val)
        except Exception:
            pass

    def is_alive(self) -> bool:
        """Returns True if the inspector window is currently open."""
        try:
            return self.window.winfo_exists()
        except Exception:
            return False

    def _on_close(self) -> None:
        if self.on_close_callback is not None:
            self.on_close_callback()
        try:
            self.window.destroy()
        except Exception:
            pass
