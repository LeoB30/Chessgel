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
    COLOR_ARROW_ENGINE,
    COLOR_ARROW_VISION,
    COLOR_DARK_SQUARE,
    COLOR_LIGHT_SQUARE,
)


class BoardMirrorWidget(QWidget):
    """Accurate, real-time board mirror displaying live position, highlights, and arrows."""

    sig_move_requested = pyqtSignal(chess.Move)

    # Arrow colors for top 3 primary engine lines (decreasing confidence)
    ARROW_COLORS_TOP3 = [
        QColor("#00D2D3"),  # #1 Best: Cyan neon
        QColor("#A6E3A1"),  # #2 Second: Mint green
        QColor("#F9E2AF"),  # #3 Third: Warm amber
    ]
    ARROW_ALPHAS_TOP3 = [210, 150, 110]  # Decreasing opacity
    ARROW_WIDTHS_TOP3 = [0.18, 0.14, 0.10]  # Decreasing thickness (ratio of square_size)

    # Arrow colors for secondary (neural net) engine lines
    ARROW_COLORS_NN = [
        QColor("#C084FC"),  # #1 NN Best: Bright violet
        QColor("#A78BFA"),  # #2 NN Second: Soft purple
        QColor("#DDD6FE"),  # #3 NN Third: Pale purple
    ]
    ARROW_ALPHAS_NN = [200, 140, 100]
    ARROW_WIDTHS_NN = [0.16, 0.12, 0.08]

    # Arrow colors for tertiary engine lines
    ARROW_COLORS_TERT = [
        QColor("#F472B6"),  # Pink
        QColor("#FBCFE8"),  # Light Pink
        QColor("#FDF2F8"),  # Pale Pink
    ]
    ARROW_ALPHAS_TERT = [200, 140, 100]
    ARROW_WIDTHS_TERT = [0.14, 0.10, 0.06]

    # Losing move override color
    COLOR_LOSING = QColor("#EF4444")  # Red for evaluated losing moves

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self.board: chess.Board = chess.Board()
        self.is_flipped: bool = False
        self.last_move: Optional[chess.Move] = None
        self.engine_best_move: Optional[chess.Move] = None
        self.engine_top_moves: List[chess.Move] = []  # Up to 3 ranked moves
        self.engine_top_scores: List[int] = []  # Centipawn scores for each top move
        self.vision_move: Optional[chess.Move] = None

        # Secondary (neural net) engine arrows for hybrid mode
        self.nn_top_moves: List[chess.Move] = []  # Up to 2 NN moves
        self.nn_top_scores: List[int] = []  # Centipawn scores for NN moves

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

    def set_engine_top_moves(self, moves: List[chess.Move], scores: Optional[List[int]] = None) -> None:
        """Sets up to 3 top engine candidate moves for multi-arrow display."""
        self.engine_top_moves = moves[:3]
        self.engine_top_scores = (scores or [])[:3]
        if moves:
            self.engine_best_move = moves[0]
        self.update()

    def set_nn_top_moves(self, moves: List[chess.Move], scores: Optional[List[int]] = None) -> None:
        """Sets up to 3 secondary engine candidate arrows."""
        self.nn_top_moves = moves[:3]
        self.nn_top_scores = (scores or [])[:3]
        self.update()

    def set_tert_top_moves(self, moves: List[chess.Move], scores: Optional[List[int]] = None) -> None:
        """Sets up to 3 tertiary engine candidate arrows."""
        self.tert_top_moves = moves[:3]
        self.tert_top_scores = (scores or [])[:3]
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
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)

        col = (7 - file_idx) if self.is_flipped else file_idx
        row = rank_idx if self.is_flipped else (7 - rank_idx)

        x = offset_x + col * square_size
        y = offset_y + row * square_size
        return QRectF(x, y, square_size, square_size)

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

        # 6. Paint Strategic Tactical Arrows (multi-PV)
        if self.vision_move is not None:
            self._draw_arrow(
                painter, self.vision_move, QColor(COLOR_ARROW_VISION), square_size, offset_x, offset_y, width=square_size * 0.14
            )

        # Draw tertiary arrows (bottom layer)
        for idx in range(min(3, len(self.tert_top_moves)) - 1, -1, -1):
            move = self.tert_top_moves[idx]
            is_losing = (idx < len(self.tert_top_scores) and self.tert_top_scores[idx] <= -100)
            if is_losing:
                color = QColor(self.COLOR_LOSING)
                color.setAlpha(200)
            else:
                color = QColor(self.ARROW_COLORS_TERT[idx])
                color.setAlpha(self.ARROW_ALPHAS_TERT[idx])
            score_text = self._format_score(self.tert_top_scores[idx]) if idx < len(self.tert_top_scores) else ""
            self._draw_arrow(painter, move, color, square_size, offset_x, offset_y, width=square_size * self.ARROW_WIDTHS_TERT[idx], text=score_text)

        # Draw secondary arrows
        for idx in range(min(3, len(self.nn_top_moves)) - 1, -1, -1):
            move = self.nn_top_moves[idx]
            is_losing = (idx < len(self.nn_top_scores) and self.nn_top_scores[idx] <= -100)
            if is_losing:
                color = QColor(self.COLOR_LOSING)
                color.setAlpha(200)
            else:
                color = QColor(self.ARROW_COLORS_NN[idx])
                color.setAlpha(self.ARROW_ALPHAS_NN[idx])
            score_text = self._format_score(self.nn_top_scores[idx]) if idx < len(self.nn_top_scores) else ""
            self._draw_arrow(painter, move, color, square_size, offset_x, offset_y, width=square_size * self.ARROW_WIDTHS_NN[idx], text=score_text)

        # Draw primary engine arrows (top layer)
        arrows_to_draw = self.engine_top_moves if self.engine_top_moves else (
            [self.engine_best_move] if self.engine_best_move else []
        )
        scores_to_check = self.engine_top_scores if self.engine_top_scores else []
        for idx in range(min(3, len(arrows_to_draw)) - 1, -1, -1):
            move = arrows_to_draw[idx]
            is_losing = (idx < len(scores_to_check) and scores_to_check[idx] <= -100)
            if is_losing:
                color = QColor(self.COLOR_LOSING)
                color.setAlpha(200)
            else:
                color = QColor(self.ARROW_COLORS_TOP3[idx])
                color.setAlpha(self.ARROW_ALPHAS_TOP3[idx])
            score_text = self._format_score(scores_to_check[idx]) if idx < len(scores_to_check) else ""
            self._draw_arrow(painter, move, color, square_size, offset_x, offset_y, width=square_size * self.ARROW_WIDTHS_TOP3[idx], text=score_text)

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
        """Formats an internal centipawn score into a string for the arrow text."""
        # Check if mate
        if score > 90000:
            return f"M{99999 - score}"
        elif score < -90000:
            return f"-M{score + 99999}"
        else:
            val = score / 100.0
            return f"{val:+.1f}"

    def _draw_arrow(
        self,
        painter: QPainter,
        move: chess.Move,
        color: QColor,
        square_size: float,
        offset_x: float,
        offset_y: float,
        width: float = 8.0,
        text: str = "",
    ) -> None:
        """Renders glowing tactical arrow between source and target square centers."""
        r_from = self._get_square_rect(move.from_square, square_size, offset_x, offset_y)
        r_to = self._get_square_rect(move.to_square, square_size, offset_x, offset_y)

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
        # Shaft
        path.moveTo(arrow_base + QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
        path.lineTo(head_base + QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
        # Arrowhead wings
        path.lineTo(head_base + QPointF(nx * head_w, ny * head_w))
        path.lineTo(arrow_tip)
        path.lineTo(head_base - QPointF(nx * head_w, ny * head_w))
        path.lineTo(head_base - QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
        path.lineTo(arrow_base - QPointF(nx * shaft_w / 2.0, ny * shaft_w / 2.0))
        path.closeSubpath()

        arrow_color = QColor(color)
        arrow_color.setAlpha(200)
        pen = QPen(QColor(20, 20, 30, 220), 1.8)
        painter.setPen(pen)
        painter.setBrush(QBrush(arrow_color))
        painter.drawPath(path)

        # Draw the text in the middle of the arrow
        if text:
            mid_x = (arrow_base.x() + head_base.x()) / 2
            mid_y = (arrow_base.y() + head_base.y()) / 2
            
            # Setup text font and bounding box
            painter.setPen(QColor(255, 255, 255, 240))  # White text
            font = QFont("Segoe UI", int(square_size * 0.22), QFont.Weight.Bold)
            painter.setFont(font)
            
            # Draw a subtle background for text readability
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            pad = 2
            bg_rect = QRectF(mid_x - rect.width() / 2 - pad, mid_y - rect.height() / 2 - pad, rect.width() + pad*2, rect.height() + pad*2)
            
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bg_rect, 4, 4)
            
            painter.setPen(QColor(255, 255, 255, 255))
            painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)
