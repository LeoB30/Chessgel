"""Sleek vertical evaluation bar widget showing dynamic White/Black advantage."""

from __future__ import annotations
from typing import Optional
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPaintEvent, QPainter
from PyQt6.QtWidgets import QSizePolicy, QWidget

from chess_tracker.core.engine import EvaluationResult


class EvalBarWidget(QWidget):
    """Vertical advantage bar showing centipawn and mate evaluations."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(22)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.eval_result: EvaluationResult = EvaluationResult()
        self.target_white_ratio: float = 0.5
        self.current_white_ratio: float = 0.5

    def set_evaluation(self, result: EvaluationResult) -> None:
        """Updates evaluation readout and refreshes display."""
        self.eval_result = result
        self.target_white_ratio = max(0.02, min(0.98, result.win_probability_white))
        # Direct smooth update
        self.current_white_ratio = self.target_white_ratio
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paints black and white portions and centered evaluation badge."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())

        # Black section (top)
        black_height = h * (1.0 - self.current_white_ratio)
        painter.fillRect(QRectF(0, 0, w, black_height), QColor("#282832"))

        # White section (bottom)
        white_height = h * self.current_white_ratio
        painter.fillRect(QRectF(0, black_height, w, white_height), QColor("#F0F0F5"))

        # Outer border
        painter.setPen(QColor("#3A3A4C"))
        painter.drawRect(QRectF(0, 0, w - 1, h - 1))

        # Center evaluation text
        score_text = self.eval_result.display_score
        font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font)

        # Position text either at bottom if White ahead, or top if Black ahead
        if self.current_white_ratio >= 0.5:
            # White ahead: draw text near bottom in black
            painter.setPen(QColor("#11111B"))
            painter.drawText(QRectF(0, h - 30, w, 24), Qt.AlignmentFlag.AlignCenter, score_text)
        else:
            # Black ahead: draw text near top in white
            painter.setPen(QColor("#ECEFF4"))
            painter.drawText(QRectF(0, 6, w, 24), Qt.AlignmentFlag.AlignCenter, score_text)
