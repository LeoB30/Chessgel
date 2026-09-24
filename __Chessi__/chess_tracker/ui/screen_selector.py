"""Full-screen interactive snipping overlay for selecting the chessboard area."""

from __future__ import annotations
from typing import Optional
from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPaintEvent, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget


class ScreenRegionSelector(QWidget):
    """Semi-transparent full-screen overlay for rubberband ROI selection."""

    sig_region_selected = pyqtSignal(int, int, int, int)  # x, y, width, height
    sig_cancelled = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._start_pos: Optional[QPoint] = None
        self._current_pos: Optional[QPoint] = None
        self._is_selecting: bool = False

    def start_selection(self) -> None:
        """Covers all screens and shows overlay."""
        # Calculate bounding box of all virtual screens
        total_rect = QRect()
        for screen in QApplication.screens():
            total_rect = total_rect.united(screen.geometry())

        self.setGeometry(total_rect)
        self._start_pos = None
        self._current_pos = None
        self._is_selecting = False
        self.show()
        self.raise_()
        self.activateWindow()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.globalPosition().toPoint()
            self._current_pos = self._start_pos
            self._is_selecting = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_selecting:
            self._current_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_selecting:
            self._is_selecting = False
            if self._start_pos and self._current_pos:
                rect = QRect(self._start_pos, self._current_pos).normalized()
                self.hide()
                if rect.width() >= 40 and rect.height() >= 40:
                    self.sig_region_selected.emit(
                        rect.x(), rect.y(), rect.width(), rect.height()
                    )
                else:
                    self.sig_cancelled.emit()
            else:
                self.hide()
                self.sig_cancelled.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.sig_cancelled.emit()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Dim entire screen
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        if self._start_pos and self._current_pos:
            # Map global coordinates to local widget coordinates
            p1 = self.mapFromGlobal(self._start_pos)
            p2 = self.mapFromGlobal(self._current_pos)
            sel_rect = QRect(p1, p2).normalized()

            # Clear the selected box
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(sel_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Draw highlight border
            pen = QPen(QColor("#00D2D3"), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(sel_rect)

            # Size readout badge
            badge_text = f"{sel_rect.width()} x {sel_rect.height()}"
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor("#00D2D3"))
            painter.drawText(sel_rect.x() + 4, max(18, sel_rect.y() - 6), badge_text)
