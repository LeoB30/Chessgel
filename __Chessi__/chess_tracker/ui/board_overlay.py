"""Click-through transparent overlay that draws engine arrows on the live screen board."""

from __future__ import annotations
from typing import List, Optional, Sequence
import chess

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import QWidget

from chess_tracker.ui.arrow_painter import (
    ARROW_COLORS_PRIMARY,
    ARROW_COLORS_SECONDARY,
    ARROW_COLORS_TERTIARY,
    ArrowSpec,
    MAX_ARROWS,
    build_arrow_specs,
    draw_arrow_specs,
)


class BoardOverlayWindow(QWidget):
    """Always-on-top, input-transparent window aligned to the captured 8x8 grid."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.is_flipped: bool = False
        self.white_to_move: bool = True
        self.enabled: bool = True
        self._primary: List[ArrowSpec] = []
        self._secondary: List[ArrowSpec] = []
        self._tertiary: List[ArrowSpec] = []
        self._geometry: Optional[tuple[int, int, int, int]] = None

    def set_overlay_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if enabled and self._geometry and (self._primary or self._secondary or self._tertiary):
            self.show_over_board(*self._geometry)
        elif not enabled:
            self.hide()

    def set_flipped(self, flipped: bool) -> None:
        self.is_flipped = flipped
        self.update()

    def set_white_to_move(self, white_to_move: bool) -> None:
        self.white_to_move = white_to_move

    def show_over_board(self, x: int, y: int, w: int, h: int) -> None:
        """Places the overlay on the physical/logical screen rectangle of the board."""
        self._geometry = (x, y, w, h)
        if not self.enabled:
            self.hide()
            return
        self.setGeometry(x, y, w, h)
        self.show()
        self.raise_()
        self.update()

    def hide_overlay(self) -> None:
        self.hide()

    def clear_arrows(self) -> None:
        self._primary = []
        self._secondary = []
        self._tertiary = []
        self.update()

    def set_primary_moves(
        self,
        moves: Sequence[chess.Move],
        scores: Sequence[int],
        white_to_move: bool,
        max_arrows: int = MAX_ARROWS,
    ) -> None:
        self.white_to_move = white_to_move
        self._primary = build_arrow_specs(
            moves, scores, ARROW_COLORS_PRIMARY, white_to_move, max_arrows=max_arrows
        )
        self.update()

    def set_secondary_moves(
        self,
        moves: Sequence[chess.Move],
        scores: Sequence[int],
        white_to_move: bool,
        max_arrows: int = MAX_ARROWS,
    ) -> None:
        self.white_to_move = white_to_move
        self._secondary = build_arrow_specs(
            moves,
            scores,
            ARROW_COLORS_SECONDARY,
            white_to_move,
            max_arrows=max_arrows,
            width_start=0.16,
        )
        self.update()

    def set_tertiary_moves(
        self,
        moves: Sequence[chess.Move],
        scores: Sequence[int],
        white_to_move: bool,
        max_arrows: int = MAX_ARROWS,
    ) -> None:
        self.white_to_move = white_to_move
        self._tertiary = build_arrow_specs(
            moves,
            scores,
            ARROW_COLORS_TERTIARY,
            white_to_move,
            max_arrows=max_arrows,
            width_start=0.14,
        )
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Subtle frame so the overlay is locatable without covering pieces.
        painter.setPen(QColor(0, 210, 211, 90))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        side = float(min(self.width(), self.height()))
        square_size = side / 8.0
        offset_x = (self.width() - side) / 2.0
        offset_y = (self.height() - side) / 2.0

        draw_arrow_specs(painter, self._tertiary, square_size, offset_x, offset_y, self.is_flipped)
        draw_arrow_specs(painter, self._secondary, square_size, offset_x, offset_y, self.is_flipped)
        draw_arrow_specs(painter, self._primary, square_size, offset_x, offset_y, self.is_flipped)
