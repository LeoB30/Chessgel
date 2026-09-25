from __future__ import annotations
import math
from typing import Optional

from PyQt6.QtCore import QRectF, QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QPaintEvent, QPainter
from PyQt6.QtWidgets import QSizePolicy, QWidget


class EvalBar(QWidget):
    """Dynamic, smooth-interpolating centipawn evaluation bar for real-time advantage telemetry."""

    WHITE_COLOR: QColor = QColor("#F8FAFC")
    BLACK_COLOR: QColor = QColor("#1E293B")
    BORDER_COLOR: QColor = QColor("#334155")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(34)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.target_eval: int = 0
        self.current_eval: float = 0.0
        self.is_flipped: bool = False

        # 60 FPS animation timer for smooth gliding transitions
        self._timer: QTimer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate_step)
        self._timer.start()

    def set_evaluation(self, centipawns: int) -> None:
        """Sets target centipawn evaluation from White's absolute perspective."""
        self.target_eval = centipawns

    def set_flipped(self, flipped: bool) -> None:
        """Toggles White/Black vertical alignment."""
        self.is_flipped = flipped
        self.update()

    def _animate_step(self) -> None:
        """Interpolates current displayed evaluation towards target value at 60 FPS."""
        diff: float = float(self.target_eval) - self.current_eval
        if abs(diff) > 0.5:
            self.current_eval += diff * 0.12
            self.update()

    def _eval_to_white_ratio(self, cp: float) -> float:
        """Converts centipawn score to a winning probability / bar percentage using sigmoid."""
        if cp >= 80000:
            return 1.0
        if cp <= -80000:
            return 0.0
        # Standard Elo/logistic scaling: 400 cp ~= 90% winning probability
        ratio: float = 1.0 / (1.0 + math.pow(10.0, -cp / 400.0))
        return max(0.04, min(0.96, ratio))

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paints evaluation bar with rounded corners and formatted centipawn indicator."""
        painter: QPainter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w: float = float(self.width())
        h: float = float(self.height())
        radius: float = 6.0

        ratio: float = self._eval_to_white_ratio(self.current_eval)
        white_h: float = h * ratio

        # Determine top/bottom split based on perspective
        if not self.is_flipped:
            # White at bottom, Black at top
            black_rect = QRectF(0, 0, w, h - white_h)
            white_rect = QRectF(0, h - white_h, w, white_h)
        else:
            # White at top, Black at bottom
            white_rect = QRectF(0, 0, w, white_h)
            black_rect = QRectF(0, white_h, w, h - white_h)

        # Draw backgrounds
        painter.fillRect(black_rect, self.BLACK_COLOR)
        painter.fillRect(white_rect, self.WHITE_COLOR)

        # Center line marker (0.00)
        center_y: float = h * 0.5
        painter.setPen(QColor(148, 163, 184, 180))
        painter.drawLine(0, int(center_y), int(w), int(center_y))

        # Border outline
        painter.setPen(self.BORDER_COLOR)
        painter.drawRect(QRectF(0, 0, w - 1, h - 1))

        # Format label text
        text: str
        if abs(self.target_eval) >= 80000:
            moves_to_mate = max(1, (90000 - abs(self.target_eval)))
            text = f"M{moves_to_mate}"
        else:
            pawn_score: float = self.current_eval / 100.0
            sign = "+" if pawn_score > 0 else ""
            text = f"{sign}{pawn_score:.1f}"

        # Paint label inside dominating region
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        text_color: QColor
        if not self.is_flipped:
            if ratio >= 0.5:
                text_color = self.BLACK_COLOR
                text_rect = QRectF(0, h - 26, w, 22)
            else:
                text_color = self.WHITE_COLOR
                text_rect = QRectF(0, 6, w, 22)
        else:
            if ratio >= 0.5:
                text_color = self.BLACK_COLOR
                text_rect = QRectF(0, 6, w, 22)
            else:
                text_color = self.WHITE_COLOR
                text_rect = QRectF(0, h - 26, w, 22)

        painter.setPen(text_color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
