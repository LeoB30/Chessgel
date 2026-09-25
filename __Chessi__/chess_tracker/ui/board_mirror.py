"""Mirrored chessboard widget replicating external board aesthetic with tactical arrows."""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import chess

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from chess_tracker.config import (
    COLOR_ARROW_VISION,
    COLOR_DARK_SQUARE,
    COLOR_LIGHT_SQUARE,
)
from chess_tracker.ui.arrow_painter import (
    ARROW_COLORS_PRIMARY,
    ARROW_COLORS_SECONDARY,
    ARROW_COLORS_TERTIARY,
    MAX_ARROWS,
    build_arrow_specs,
    draw_arrow_specs,
    draw_eval_arrow,
    format_eval_score,
    square_rect,
)


class BoardMirrorWidget(QWidget):
    """Accurate, real-time board mirror displaying live position, highlights, and arrows."""

    sig_move_requested = pyqtSignal(chess.Move)


    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self.board: chess.Board = chess.Board()
        self.is_flipped: bool = False
        self.last_move: Optional[chess.Move] = None
        self.engine_best_move: Optional[chess.Move] = None
        self.engine_top_moves: List[chess.Move] = []
        self.engine_top_scores: List[int] = []
        self.vision_move: Optional[chess.Move] = None
        self.white_to_move: bool = True
        self.max_primary_arrows: int = 3
        self.max_secondary_arrows: int = 2
        self.max_tertiary_arrows: int = 1

        self.nn_top_moves: List[chess.Move] = []
        self.nn_top_scores: List[int] = []
        self.tert_top_moves: List[chess.Move] = []
        self.tert_top_scores: List[int] = []

        self.selected_square: Optional[chess.Square] = None
        self.candidate_moves: List[chess.Move] = []

    def set_board(
        self,
        board: chess.Board,
        last_move: Optional[chess.Move] = None,
        engine_move: Optional[chess.Move] = None,
        vision_move: Optional[chess.Move] = None,
    ) -> None:
        """Updates internal board position and queues repaint."""
        self.board = board.copy()
        if last_move is not None:
            self.last_move = last_move
        if engine_move is not None:
            self.engine_best_move = engine_move
        if vision_move is not None:
            self.vision_move = vision_move
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        """Toggles White bottom vs Black bottom perspective."""
        self.is_flipped = flipped
        self.update()

    def set_engine_hint(self, move: Optional[chess.Move]) -> None:
        """Sets the recommended tactical move arrow (single best)."""
        self.engine_best_move = move
        self.update()

    def set_arrow_limits(self, primary: int, secondary: int, tertiary: int) -> None:
        """Caps how many arrows each engine may draw (Trio-Engine Mode)."""
        self.max_primary_arrows = max(1, min(MAX_ARROWS, primary))
        self.max_secondary_arrows = max(0, min(MAX_ARROWS, secondary))
        self.max_tertiary_arrows = max(0, min(MAX_ARROWS, tertiary))
        self.update()

    def set_engine_top_moves(
        self,
        moves: List[chess.Move],
        scores: Optional[List[int]] = None,
        white_to_move: Optional[bool] = None,
    ) -> None:
        """Sets primary engine candidate moves for multi-arrow display."""
        limit = self.max_primary_arrows
        self.engine_top_moves = list(moves[:limit])
        self.engine_top_scores = list((scores or [])[:limit])
        if white_to_move is not None:
            self.white_to_move = white_to_move
        if moves:
            self.engine_best_move = moves[0]
        self.update()

    def set_nn_top_moves(
        self,
        moves: List[chess.Move],
        scores: Optional[List[int]] = None,
        white_to_move: Optional[bool] = None,
    ) -> None:
        """Sets secondary engine candidate arrows."""
        limit = self.max_secondary_arrows
        self.nn_top_moves = list(moves[:limit])
        self.nn_top_scores = list((scores or [])[:limit])
        if white_to_move is not None:
            self.white_to_move = white_to_move
        self.update()

    def set_tert_top_moves(
        self,
        moves: List[chess.Move],
        scores: Optional[List[int]] = None,
        white_to_move: Optional[bool] = None,
    ) -> None:
        """Sets tertiary engine candidate arrows."""
        limit = self.max_tertiary_arrows
        self.tert_top_moves = list(moves[:limit])
        self.tert_top_scores = list((scores or [])[:limit])
        if white_to_move is not None:
            self.white_to_move = white_to_move
        self.update()

    def clear_nn_arrows(self) -> None:
        """Removes all secondary/tertiary engine arrows."""
        self.nn_top_moves = []
        self.nn_top_scores = []
        self.tert_top_moves = []
        self.tert_top_scores = []
        self.update()

    def set_vision_move(self, move: Optional[chess.Move]) -> None:
        """Sets the opponent/vision move arrow."""
        self.vision_move = move
        self.update()

    def _get_square_rect(
        self, sq: chess.Square, square_size: float, offset_x: float, offset_y: float
    ) -> QRectF:
        """Calculates square rectangle on canvas based on board orientation."""
        return square_rect(sq, square_size, offset_x, offset_y, self.is_flipped)

    def paintEvent(self, event: QPaintEvent) -> None:
        """Renders wood tiles, move highlights, coordinate markings, pieces, and arrows."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        side_len = float(min(self.width(), self.height()))
        square_size = side_len / 8.0
        offset_x = (self.width() - side_len) / 2.0
        offset_y = (self.height() - side_len) / 2.0

        # Background letterbox
        painter.fillRect(self.rect(), QColor("#14141E"))

        # 1. Paint 64 Wood-Themed Squares
        light_color = QColor(COLOR_LIGHT_SQUARE)
        dark_color = QColor(COLOR_DARK_SQUARE)

        for sq in chess.SQUARES:
            rect = self._get_square_rect(sq, square_size, offset_x, offset_y)
            file_idx = chess.square_file(sq)
            rank_idx = chess.square_rank(sq)
            is_light = (file_idx + rank_idx) % 2 == 1
            painter.fillRect(rect, light_color if is_light else dark_color)

        # 2. Paint Move Highlights (Semi-Transparent Yellow Overlays)
        highlight_color = QColor(247, 206, 50, 160)
        if self.last_move is not None:
            r_from = self._get_square_rect(self.last_move.from_square, square_size, offset_x, offset_y)
            r_to = self._get_square_rect(self.last_move.to_square, square_size, offset_x, offset_y)
            painter.fillRect(r_from, highlight_color)
            painter.fillRect(r_to, highlight_color)

        if self.selected_square is not None:
            r_sel = self._get_square_rect(self.selected_square, square_size, offset_x, offset_y)
            painter.fillRect(r_sel, QColor(100, 180, 246, 170))

        # Check Glow
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                r_k = self._get_square_rect(king_sq, square_size, offset_x, offset_y)
                grad = QRadialGradient(r_k.center(), square_size * 0.6)
                grad.setColorAt(0.0, QColor(235, 60, 60, 220))
                grad.setColorAt(1.0, QColor(235, 60, 60, 0))
                painter.fillRect(r_k, grad)

        # 3. Paint Rank & File Coordinates
        coord_font = QFont("Segoe UI", int(max(9, square_size * 0.16)), QFont.Weight.Bold)
        painter.setFont(coord_font)

        for i in range(8):
            file_char = chr(ord("h") - i) if self.is_flipped else chr(ord("a") + i)
            rank_char = str(i + 1) if self.is_flipped else str(8 - i)

            # File at bottom row
            file_sq_color = light_color if (i % 2 == 0) else dark_color
            text_color = QColor(70, 50, 30, 210) if file_sq_color == light_color else QColor(230, 210, 180, 210)
            painter.setPen(text_color)
            painter.drawText(
                int(offset_x + (i + 1) * square_size - square_size * 0.22),
                int(offset_y + 8 * square_size - 4),
                file_char,
            )

            # Rank at left column
            rank_sq_color = light_color if (i % 2 == 0) else dark_color
            text_color = QColor(70, 50, 30, 210) if rank_sq_color == light_color else QColor(230, 210, 180, 210)
            painter.setPen(text_color)
            painter.drawText(
                int(offset_x + 4),
                int(offset_y + i * square_size + square_size * 0.22),
                rank_char,
            )

        # 4. Paint Chess Pieces
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is not None:
                rect = self._get_square_rect(sq, square_size, offset_x, offset_y)
                self._draw_piece(painter, piece, rect)

        # 5. Paint Interactive Move Candidate Indicators
        for move in self.candidate_moves:
            dst_rect = self._get_square_rect(move.to_square, square_size, offset_x, offset_y)
            center = dst_rect.center()
            if self.board.piece_at(move.to_square) is not None:
                # Capture Ring
                pen = QPen(QColor(50, 50, 50, 120), square_size * 0.08)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, square_size * 0.38, square_size * 0.38)
            else:
                # Move Dot
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(50, 50, 50, 90)))
                painter.drawEllipse(center, square_size * 0.15, square_size * 0.15)

        # 6. Paint Strategic Tactical Arrows (multi-PV) with eval in the shaft
        if self.vision_move is not None:
            draw_eval_arrow(
                painter,
                self.vision_move,
                QColor(COLOR_ARROW_VISION),
                square_size,
                offset_x,
                offset_y,
                self.is_flipped,
                width=square_size * 0.14,
            )

        tert_specs = build_arrow_specs(
            self.tert_top_moves,
            self.tert_top_scores,
            ARROW_COLORS_TERTIARY,
            self.white_to_move,
            max_arrows=self.max_tertiary_arrows,
            width_start=0.14,
        )
        nn_specs = build_arrow_specs(
            self.nn_top_moves,
            self.nn_top_scores,
            ARROW_COLORS_SECONDARY,
            self.white_to_move,
            max_arrows=self.max_secondary_arrows,
            width_start=0.16,
        )
        arrows_to_draw = self.engine_top_moves if self.engine_top_moves else (
            [self.engine_best_move] if self.engine_best_move else []
        )
        primary_specs = build_arrow_specs(
            arrows_to_draw,
            self.engine_top_scores,
            ARROW_COLORS_PRIMARY,
            self.white_to_move,
            max_arrows=self.max_primary_arrows,
        )
        draw_arrow_specs(painter, tert_specs, square_size, offset_x, offset_y, self.is_flipped)
        draw_arrow_specs(painter, nn_specs, square_size, offset_x, offset_y, self.is_flipped)
        draw_arrow_specs(painter, primary_specs, square_size, offset_x, offset_y, self.is_flipped)

    def _draw_piece(self, painter: QPainter, piece: chess.Piece, rect: QRectF) -> None:
        """Renders crisp vector chess piece matching Neo/Classic aesthetic."""
        is_white = piece.color == chess.WHITE
        symbol = piece.symbol().upper()

        cx = rect.center().x()
        cy = rect.center().y()
        scale = rect.width() * 0.44

        # Colors matching external browser board piece appearance
        if is_white:
            fill_brush = QBrush(QColor("#FFFFFF"))
            border_pen = QPen(QColor("#242628"), max(2.0, scale * 0.075))
            inner_line_pen = QPen(QColor("#404448"), max(1.2, scale * 0.04))
        else:
            fill_brush = QBrush(QColor("#383B40"))
            border_pen = QPen(QColor("#1A1C1E"), max(2.2, scale * 0.08))
            inner_line_pen = QPen(QColor("#5A5E66"), max(1.2, scale * 0.04))

        border_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        border_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(border_pen)
        painter.setBrush(fill_brush)

        path = QPainterPath()

        if symbol == "P":  # Pawn
            path.moveTo(cx - scale * 0.52, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.52, cy + scale * 0.85)
            path.quadTo(cx + scale * 0.42, cy + scale * 0.65, cx + scale * 0.22, cy + scale * 0.52)
            path.lineTo(cx + scale * 0.18, cy + scale * 0.12)
            path.quadTo(cx + scale * 0.32, cy + scale * 0.02, cx + scale * 0.25, cy - scale * 0.06)
            path.lineTo(cx - scale * 0.25, cy - scale * 0.06)
            path.quadTo(cx - scale * 0.32, cy + scale * 0.02, cx - scale * 0.18, cy + scale * 0.12)
            path.lineTo(cx - scale * 0.22, cy + scale * 0.52)
            path.quadTo(cx - scale * 0.42, cy + scale * 0.65, cx - scale * 0.52, cy + scale * 0.85)
            path.closeSubpath()
            painter.drawPath(path)
            # Head sphere
            painter.drawEllipse(QPointF(cx, cy - scale * 0.38), scale * 0.33, scale * 0.33)

        elif symbol == "R":  # Rook
            path.moveTo(cx - scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.50, cy + scale * 0.60)
            path.lineTo(cx + scale * 0.36, cy - scale * 0.25)
            # Battlements
            path.lineTo(cx + scale * 0.52, cy - scale * 0.38)
            path.lineTo(cx + scale * 0.52, cy - scale * 0.78)
            path.lineTo(cx + scale * 0.28, cy - scale * 0.78)
            path.lineTo(cx + scale * 0.28, cy - scale * 0.55)
            path.lineTo(cx + scale * 0.10, cy - scale * 0.55)
            path.lineTo(cx + scale * 0.10, cy - scale * 0.78)
            path.lineTo(cx - scale * 0.10, cy - scale * 0.78)
            path.lineTo(cx - scale * 0.10, cy - scale * 0.55)
            path.lineTo(cx - scale * 0.28, cy - scale * 0.55)
            path.lineTo(cx - scale * 0.28, cy - scale * 0.78)
            path.lineTo(cx - scale * 0.52, cy - scale * 0.78)
            path.lineTo(cx - scale * 0.52, cy - scale * 0.38)
            path.lineTo(cx - scale * 0.36, cy - scale * 0.25)
            path.lineTo(cx - scale * 0.50, cy + scale * 0.60)
            path.closeSubpath()
            painter.drawPath(path)

        elif symbol == "N":  # Knight
            path.moveTo(cx - scale * 0.60, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.60, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.45, cy + scale * 0.58)
            path.quadTo(cx + scale * 0.65, cy + scale * 0.12, cx + scale * 0.36, cy - scale * 0.28)
            # Ears
            path.lineTo(cx + scale * 0.20, cy - scale * 0.82)
            path.lineTo(cx + scale * 0.05, cy - scale * 0.68)
            # Forehead and snout
            path.lineTo(cx - scale * 0.32, cy - scale * 0.52)
            path.lineTo(cx - scale * 0.65, cy - scale * 0.18)
            path.lineTo(cx - scale * 0.50, cy - scale * 0.02)
            path.lineTo(cx - scale * 0.25, cy - scale * 0.08)
            path.quadTo(cx - scale * 0.30, cy + scale * 0.35, cx - scale * 0.50, cy + scale * 0.58)
            path.closeSubpath()
            painter.drawPath(path)
            # Eye dot
            painter.setBrush(QBrush(border_pen.color()))
            painter.drawEllipse(QPointF(cx - scale * 0.15, cy - scale * 0.36), scale * 0.07, scale * 0.07)

        elif symbol == "B":  # Bishop
            path.moveTo(cx - scale * 0.58, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.58, cy + scale * 0.85)
            path.quadTo(cx + scale * 0.45, cy + scale * 0.60, cx + scale * 0.22, cy + scale * 0.45)
            path.lineTo(cx + scale * 0.22, cy + scale * 0.20)
            path.quadTo(cx + scale * 0.42, cy - scale * 0.10, cx, cy - scale * 0.70)
            path.quadTo(cx - scale * 0.42, cy - scale * 0.10, cx - scale * 0.22, cy + scale * 0.20)
            path.lineTo(cx - scale * 0.22, cy + scale * 0.45)
            path.quadTo(cx - scale * 0.45, cy + scale * 0.60, cx - scale * 0.58, cy + scale * 0.85)
            path.closeSubpath()
            painter.drawPath(path)
            # Cross ball on top
            painter.drawEllipse(QPointF(cx, cy - scale * 0.78), scale * 0.10, scale * 0.10)
            # Slit line
            painter.setPen(inner_line_pen)
            painter.drawLine(
                int(cx - scale * 0.15), int(cy - scale * 0.40), int(cx + scale * 0.10), int(cy - scale * 0.15)
            )

        elif symbol == "Q":  # Queen
            path.moveTo(cx - scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.65, cy + scale * 0.85)
            path.quadTo(cx + scale * 0.50, cy + scale * 0.60, cx + scale * 0.28, cy + scale * 0.45)
            path.lineTo(cx + scale * 0.28, cy + scale * 0.15)
            # Five Crown Points
            path.lineTo(cx + scale * 0.72, cy - scale * 0.45)
            path.lineTo(cx + scale * 0.38, cy - scale * 0.15)
            path.lineTo(cx + scale * 0.38, cy - scale * 0.72)
            path.lineTo(cx + scale * 0.18, cy - scale * 0.20)
            path.lineTo(cx, cy - scale * 0.82)
            path.lineTo(cx - scale * 0.18, cy - scale * 0.20)
            path.lineTo(cx - scale * 0.38, cy - scale * 0.72)
            path.lineTo(cx - scale * 0.38, cy - scale * 0.15)
            path.lineTo(cx - scale * 0.72, cy - scale * 0.45)
            path.lineTo(cx - scale * 0.28, cy + scale * 0.15)
            path.lineTo(cx - scale * 0.28, cy + scale * 0.45)
            path.quadTo(cx - scale * 0.50, cy + scale * 0.60, cx - scale * 0.65, cy + scale * 0.85)
            path.closeSubpath()
            painter.drawPath(path)
            # Pearls on crown tips
            for px_offset, py_offset in [(-0.72, -0.45), (-0.38, -0.72), (0.0, -0.82), (0.38, -0.72), (0.72, -0.45)]:
                painter.drawEllipse(QPointF(cx + scale * px_offset, cy + scale * py_offset), scale * 0.08, scale * 0.08)

        elif symbol == "K":  # King
            path.moveTo(cx - scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.65, cy + scale * 0.85)
            path.quadTo(cx + scale * 0.48, cy + scale * 0.60, cx + scale * 0.26, cy + scale * 0.45)
            path.lineTo(cx + scale * 0.26, cy + scale * 0.18)
            path.quadTo(cx + scale * 0.55, cy - scale * 0.25, cx + scale * 0.42, cy - scale * 0.55)
            path.lineTo(cx - scale * 0.42, cy - scale * 0.55)
            path.quadTo(cx - scale * 0.55, cy - scale * 0.25, cx - scale * 0.26, cy + scale * 0.18)
            path.lineTo(cx - scale * 0.26, cy + scale * 0.45)
            path.quadTo(cx - scale * 0.48, cy + scale * 0.60, cx - scale * 0.65, cy + scale * 0.85)
            path.closeSubpath()
            painter.drawPath(path)
            # Royal Cross
            painter.setPen(border_pen)
            cross_size = scale * 0.18
            cross_y = cy - scale * 0.72
            painter.drawLine(int(cx), int(cross_y - cross_size), int(cx), int(cross_y + cross_size * 0.6))
            painter.drawLine(int(cx - cross_size * 0.8), int(cross_y - cross_size * 0.3), int(cx + cross_size * 0.8), int(cross_y - cross_size * 0.3))

    def _format_score(self, score: int) -> str:
        return format_eval_score(score)
