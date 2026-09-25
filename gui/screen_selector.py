from __future__ import annotations
from typing import Optional, Tuple

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPaintEvent, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget


class ScreenRegionSelector(QWidget):
    """Fullscreen drag-and-drop selector overlay to register screen bounding box coordinates."""

    sig_region_selected = pyqtSignal(tuple)  # (left, top, width, height)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._start_pos: Optional[QPoint] = None
        self._current_pos: Optional[QPoint] = None
        self._is_dragging: bool = False

        # Cover virtual geometry across all connected monitors
        screen_geo = QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(screen_geo)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Begins bounding box drag."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_pos = event.globalPosition().toPoint()
            self._current_pos = self._start_pos
            self._is_dragging = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Updates drag rectangle coordinates."""
        if self._is_dragging:
            self._current_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finalizes selection and emits registered coordinates."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging:
            self._is_dragging = False
            end_pos: QPoint = event.globalPosition().toPoint()

            if self._start_pos is not None:
                x1: int = min(self._start_pos.x(), end_pos.x())
                y1: int = min(self._start_pos.y(), end_pos.y())
                w: int = abs(end_pos.x() - self._start_pos.x())
                h: int = abs(end_pos.y() - self._start_pos.y())

                if w >= 64 and h >= 64:
                    self.sig_region_selected.emit((x1, y1, w, h))

            self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Cancels region selection when Escape is pressed."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paints dark screen mask and highlighted target rectangle."""
        painter: QPainter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Semi-transparent dark background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        # Instructions banner at top center
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.setPen(QColor("#89B4FA"))
        banner_rect = QRect(0, 40, self.width(), 35)
        painter.drawText(
            banner_rect,
            Qt.AlignmentFlag.AlignCenter,
            "🎯 Drag a rectangle around the target chessboard. Press ESC to cancel.",
        )

        # Draw selected bounding box
        if self._start_pos is not None and self._current_pos is not None:
            # Map global coordinates to local widget coordinates
            p1: QPoint = self.mapFromGlobal(self._start_pos)
            p2: QPoint = self.mapFromGlobal(self._current_pos)

            rect = QRect(
                min(p1.x(), p2.x()),
                min(p1.y(), p2.y()),
                abs(p2.x() - p1.x()),
                abs(p2.y() - p1.y()),
            )

            # Clear inner rectangle so user sees actual screen clearly
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Vibrant green dashed selection border
            pen = QPen(QColor("#A6E3A1"), 3, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            # Dimensions badge
            dim_str: str = f"{rect.width()} × {rect.height()} px"
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, dim_str)
