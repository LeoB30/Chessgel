"""Shared tactical-arrow drawing: eval labels in the shaft, blunders in red."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import chess

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen

BLUNDER_DROP_CP = 200
LOSING_STM_CP = 200
MAX_ARROWS = 10
COLOR_BLUNDER = QColor("#EF4444")

ARROW_COLORS_PRIMARY = [
    QColor("#00D2D3"),
    QColor("#A6E3A1"),
    QColor("#F9E2AF"),
    QColor("#89B4FA"),
    QColor("#94E2D5"),
]
ARROW_COLORS_SECONDARY = [
    QColor("#C084FC"),
    QColor("#A78BFA"),
    QColor("#DDD6FE"),
    QColor("#E879F9"),
    QColor("#F5D0FE"),
]
ARROW_COLORS_TERTIARY = [
    QColor("#F472B6"),
    QColor("#FBCFE8"),
    QColor("#FDF2F8"),
    QColor("#FB7185"),
    QColor("#FECDD3"),
]


@dataclass
class ArrowSpec:
    move: chess.Move
    score_cp: Optional[int]
    color: QColor
    alpha: int
    width_ratio: float
    is_blunder: bool


def format_eval_score(score: Optional[int]) -> str:
    """Formats a White-perspective centipawn score for arrow labels."""
    if score is None:
        return ""
    if score > 90000:
        return "M#"
    if score < -90000:
        return "-M#"
    return f"{score / 100.0:+.1f}"


def is_blunder_score(
    score_cp: int,
    best_score_cp: int,
    white_to_move: bool,
    drop_cp: int = BLUNDER_DROP_CP,
) -> bool:
    """True if this line is losing for the side to move, or far worse than the best line."""
    stm = 1 if white_to_move else -1
    stm_score = score_cp * stm
    stm_best = best_score_cp * stm
    if stm_score <= -LOSING_STM_CP:
        return True
    return (stm_best - stm_score) >= drop_cp


def build_arrow_specs(
    moves: Sequence[chess.Move],
    scores: Sequence[int],
    palette: Sequence[QColor],
    white_to_move: bool,
    max_arrows: int = MAX_ARROWS,
    width_start: float = 0.18,
    width_step: float = 0.02,
) -> List[ArrowSpec]:
    """Builds ranked MultiPV arrows with blunder flags and shrinking geometry."""
    specs: List[ArrowSpec] = []
    count = min(max_arrows, len(moves), max(len(scores), len(moves)))
    best = scores[0] if scores else 0
    for idx in range(count):
        move = moves[idx]
        score = scores[idx] if idx < len(scores) else best
        pal_idx = min(idx, len(palette) - 1)
        alpha = max(90, 210 - idx * 22)
        width_ratio = max(0.06, width_start - idx * width_step)
        blunder = is_blunder_score(score, best, white_to_move)
        color = QColor(COLOR_BLUNDER if blunder else palette[pal_idx])
        specs.append(
            ArrowSpec(
                move=move,
                score_cp=score,
                color=color,
                alpha=alpha,
                width_ratio=width_ratio,
                is_blunder=blunder,
            )
        )
    return specs


def square_rect(
    sq: chess.Square,
    square_size: float,
    offset_x: float,
    offset_y: float,
    is_flipped: bool,
) -> QRectF:
    file_idx = chess.square_file(sq)
    rank_idx = chess.square_rank(sq)
    col = (7 - file_idx) if is_flipped else file_idx
    row = rank_idx if is_flipped else (7 - rank_idx)
    return QRectF(
        offset_x + col * square_size,
        offset_y + row * square_size,
        square_size,
        square_size,
    )


def draw_eval_arrow(
    painter: QPainter,
    move: chess.Move,
    color: QColor,
    square_size: float,
    offset_x: float,
    offset_y: float,
    is_flipped: bool,
    width: float,
    text: str = "",
) -> None:
    """Draws a glowing arrow with the evaluation score in the middle of the shaft."""
    r_from = square_rect(move.from_square, square_size, offset_x, offset_y, is_flipped)
    r_to = square_rect(move.to_square, square_size, offset_x, offset_y, is_flipped)
    p1 = r_from.center()
    p2 = r_to.center()

    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return

    ux = dx / dist
    uy = dy / dist
    nx = -uy
    ny = ux

    head_len = min(square_size * 0.45, dist * 0.38)
    head_w = head_len * 0.75
    shaft_w = width

    arrow_tip = p2 - QPointF(ux * square_size * 0.15, uy * square_size * 0.15)
    arrow_base = p1 + QPointF(ux * square_size * 0.20, uy * square_size * 0.20)
    head_base = arrow_tip - QPointF(ux * head_len, uy * head_len)

    path = QPainterPath()
    path.moveTo(arrow_base + QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
    path.lineTo(head_base + QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
    path.lineTo(head_base + QPointF(nx * head_w, ny * head_w))
    path.lineTo(arrow_tip)
    path.lineTo(head_base - QPointF(nx * head_w, ny * head_w))
    path.lineTo(head_base - QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
    path.lineTo(arrow_base - QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
    path.closeSubpath()

    arrow_color = QColor(color)
    if arrow_color.alpha() < 40:
        arrow_color.setAlpha(200)
    painter.setPen(QPen(QColor(20, 20, 30, 220), 1.8))
    painter.setBrush(QBrush(arrow_color))
    painter.drawPath(path)

    if not text:
        return

    mid_x = (arrow_base.x() + head_base.x()) / 2.0
    mid_y = (arrow_base.y() + head_base.y()) / 2.0
    font = QFont("Segoe UI", max(8, int(square_size * 0.22)), QFont.Weight.Bold)
    painter.setFont(font)
    fm = painter.fontMetrics()
    rect = fm.boundingRect(text)
    pad = 3
    bg_rect = QRectF(
        mid_x - rect.width() / 2.0 - pad,
        mid_y - rect.height() / 2.0 - pad,
        rect.width() + pad * 2,
        rect.height() + pad * 2,
    )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 180))
    painter.drawRoundedRect(bg_rect, 4, 4)
    painter.setPen(QColor(255, 255, 255, 255))
    painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)


def draw_arrow_specs(
    painter: QPainter,
    specs: Sequence[ArrowSpec],
    square_size: float,
    offset_x: float,
    offset_y: float,
    is_flipped: bool,
) -> None:
    """Paints weaker lines first so the best move stays on top."""
    for spec in reversed(list(specs)):
        color = QColor(spec.color)
        color.setAlpha(spec.alpha)
        draw_eval_arrow(
            painter,
            spec.move,
            color,
            square_size,
            offset_x,
            offset_y,
            is_flipped,
            width=square_size * spec.width_ratio,
            text=format_eval_score(spec.score_cp),
        )
