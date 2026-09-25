from __future__ import annotations
import math
from typing import List, Optional, Tuple
import chess

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget


class BoardView(QWidget):
    """High-contrast, 60 FPS interactive chessboard with solid opaque vector piece rendering."""

    sig_move_made = pyqtSignal(object)  # chess.Move

    LIGHT_SQUARE: QColor = QColor("#EFE4C8")
    DARK_SQUARE: QColor = QColor("#A36E41")
    SELECTED_COLOR: QColor = QColor(246, 224, 94, 180)
    LAST_MOVE_COLOR: QColor = QColor(184, 211, 68, 160)
    CHECK_COLOR: QColor = QColor(239, 68, 68, 200)
    CANDIDATE_DOT_COLOR: QColor = QColor(60, 60, 60, 90)
    ARROW_ENGINE_COLOR: QColor = QColor("#00D2D3")
    ARROW_VISION_COLOR: QColor = QColor("#F59E0B")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self.board: chess.Board = chess.Board()
        self.is_flipped: bool = False
        self.selected_square: Optional[chess.Square] = None
        self.candidate_moves: List[chess.Move] = []
        self.last_move: Optional[chess.Move] = None
        self.hint_move: Optional[chess.Move] = None
        self.vision_move: Optional[chess.Move] = None

    def set_board(
        self,
        board: chess.Board,
        last_move: Optional[chess.Move] = None,
        hint_move: Optional[chess.Move] = None,
    ) -> None:
        """Updates internal board state and triggers repaint."""
        self.board = board.copy()
        self.last_move = last_move
        self.hint_move = hint_move
        self.selected_square = None
        self.candidate_moves.clear()
        self.update()

    def set_flipped(self, flipped: bool) -> None:
        """Toggles White/Black perspective."""
        self.is_flipped = flipped
        self.update()

    def set_engine_hint(self, move: Optional[chess.Move]) -> None:
        """Sets the engine suggested move arrow."""
        self.hint_move = move
        self.update()

    def set_vision_move(self, move: Optional[chess.Move]) -> None:
        """Sets the vision detected opponent move arrow."""
        self.vision_move = move
        self.update()

    def _get_square_rect(self, sq: chess.Square, square_size: float, offset_x: float, offset_y: float) -> QRectF:
        """Calculates square QRectF based on orientation."""
        file_idx: int = chess.square_file(sq)
        rank_idx: int = chess.square_rank(sq)

        col: int = (7 - file_idx) if self.is_flipped else file_idx
        row: int = rank_idx if self.is_flipped else (7 - rank_idx)

        x: float = offset_x + col * square_size
        y: float = offset_y + row * square_size
        return QRectF(x, y, square_size, square_size)

    def _get_square_at_pos(self, pos_x: float, pos_y: float, square_size: float, offset_x: float, offset_y: float) -> Optional[chess.Square]:
        """Maps canvas mouse coordinates to chess.Square."""
        if pos_x < offset_x or pos_y < offset_y:
            return None
        col: int = int((pos_x - offset_x) // square_size)
        row: int = int((pos_y - offset_y) // square_size)

        if 0 <= col < 8 and 0 <= row < 8:
            file_idx: int = (7 - col) if self.is_flipped else col
            rank_idx: int = row if self.is_flipped else (7 - row)
            return chess.square(file_idx, rank_idx)
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handles square selection and move execution for standalone mode."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        side_len: float = min(self.width(), self.height())
        square_size: float = side_len / 8.0
        offset_x: float = (self.width() - side_len) / 2.0
        offset_y: float = (self.height() - side_len) / 2.0

        clicked_sq = self._get_square_at_pos(
            event.position().x(), event.position().y(), square_size, offset_x, offset_y
        )
        if clicked_sq is None:
            return

        # Check if clicking on legal candidate destination
        if self.selected_square is not None:
            for move in self.candidate_moves:
                if move.to_square == clicked_sq:
                    self.board.push(move)
                    self.last_move = move
                    self.selected_square = None
                    self.candidate_moves.clear()
                    self.sig_move_made.emit(move)
                    self.update()
                    return

        piece: Optional[chess.Piece] = self.board.piece_at(clicked_sq)
        if piece is not None and piece.color == self.board.turn:
            self.selected_square = clicked_sq
            self.candidate_moves = [
                m for m in self.board.legal_moves if m.from_square == clicked_sq
            ]
        else:
            self.selected_square = None
            self.candidate_moves.clear()

        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paints chessboard, highlights, solid opaque pieces, and tactical arrows."""
        painter: QPainter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        side_len: float = min(self.width(), self.height())
        square_size: float = side_len / 8.0
        offset_x: float = (self.width() - side_len) / 2.0
        offset_y: float = (self.height() - side_len) / 2.0

        # Background letterbox fill
        painter.fillRect(self.rect(), QColor("#11111B"))

        # 1. Paint 64 Squares
        for sq in chess.SQUARES:
            rect: QRectF = self._get_square_rect(sq, square_size, offset_x, offset_y)
            file_idx: int = chess.square_file(sq)
            rank_idx: int = chess.square_rank(sq)
            is_light: bool = (file_idx + rank_idx) % 2 == 1
            painter.fillRect(rect, self.LIGHT_SQUARE if is_light else self.DARK_SQUARE)

        # 2. Paint Move Highlights
        if self.last_move is not None:
            r1 = self._get_square_rect(self.last_move.from_square, square_size, offset_x, offset_y)
            r2 = self._get_square_rect(self.last_move.to_square, square_size, offset_x, offset_y)
            painter.fillRect(r1, self.LAST_MOVE_COLOR)
            painter.fillRect(r2, self.LAST_MOVE_COLOR)

        if self.selected_square is not None:
            r_sel = self._get_square_rect(self.selected_square, square_size, offset_x, offset_y)
            painter.fillRect(r_sel, self.SELECTED_COLOR)

        # King in check highlight
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                r_k = self._get_square_rect(king_sq, square_size, offset_x, offset_y)
                painter.fillRect(r_k, self.CHECK_COLOR)

        # 3. Paint Rank & File Coordinates
        font: QFont = QFont("Segoe UI", int(max(9, square_size * 0.16)), QFont.Weight.Bold)
        painter.setFont(font)
        for i in range(8):
            file_name = chr(ord('h') - i) if self.is_flipped else chr(ord('a') + i)
            rank_name = str(i + 1) if self.is_flipped else str(8 - i)

            # Files at bottom edge
            painter.setPen(QColor(40, 40, 40, 160))
            painter.drawText(
                int(offset_x + i * square_size + 4),
                int(offset_y + 8 * square_size - 4),
                file_name,
            )
            # Ranks at left edge
            painter.drawText(
                int(offset_x + 4),
                int(offset_y + i * square_size + int(square_size * 0.22)),
                rank_name,
            )

        # 4. Paint Pieces using Solid Opaque Vector Rendering
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is not None:
                rect = self._get_square_rect(sq, square_size, offset_x, offset_y)
                self._draw_solid_piece(painter, piece, rect)

        # 5. Paint Candidate Move Markers
        for move in self.candidate_moves:
            dst_rect = self._get_square_rect(move.to_square, square_size, offset_x, offset_y)
            center = dst_rect.center()
            radius = square_size * 0.14
            if self.board.piece_at(move.to_square) is not None:
                # Capture ring
                painter.setPen(QPen(self.CANDIDATE_DOT_COLOR, square_size * 0.08))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, square_size * 0.38, square_size * 0.38)
            else:
                # Move dot
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(self.CANDIDATE_DOT_COLOR))
                painter.drawEllipse(center, radius, radius)

        # 6. Paint Strategic & Vision Arrows
        if self.vision_move is not None:
            self._draw_arrow(painter, self.vision_move, self.ARROW_VISION_COLOR, square_size, offset_x, offset_y)
        if self.hint_move is not None:
            self._draw_arrow(painter, self.hint_move, self.ARROW_ENGINE_COLOR, square_size, offset_x, offset_y)

    def _draw_solid_piece(self, painter: QPainter, piece: chess.Piece, rect: QRectF) -> None:
        """Renders solid, 100% opaque vector piece graphics with zero edge-bleed."""
        is_white: bool = (piece.color == chess.WHITE)
        symbol: str = piece.symbol().upper()

        cx: float = rect.center().x()
        cy: float = rect.center().y()
        scale: float = rect.width() * 0.42

        fill_color: QColor = QColor("#FFFFFF") if is_white else QColor("#22252A")
        outline_color: QColor = QColor("#141416") if is_white else QColor("#ECEFF4")
        accent_color: QColor = QColor("#333333") if is_white else QColor("#CBD5E1")

        pen = QPen(outline_color, max(1.8, scale * 0.08))
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill_color))

        path = QPainterPath()

        if symbol == "P":  # Pawn
            path.moveTo(cx - scale * 0.55, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.55, cy + scale * 0.85)
            path.quadTo(cx + scale * 0.45, cy + scale * 0.65, cx + scale * 0.25, cy + scale * 0.55)
            path.lineTo(cx + scale * 0.18, cy + scale * 0.15)
            path.quadTo(cx + scale * 0.35, cy + scale * 0.05, cx + scale * 0.28, cy - scale * 0.05)
            path.lineTo(cx - scale * 0.28, cy - scale * 0.05)
            path.quadTo(cx - scale * 0.35, cy + scale * 0.05, cx - scale * 0.18, cy + scale * 0.15)
            path.lineTo(cx - scale * 0.25, cy + scale * 0.55)
            path.quadTo(cx - scale * 0.45, cy + scale * 0.65, cx - scale * 0.55, cy + scale * 0.85)
            path.closeSubpath()
            painter.drawPath(path)
            # Head circle
            painter.drawEllipse(QPointF(cx, cy - scale * 0.38), scale * 0.34, scale * 0.34)

        elif symbol == "R":  # Rook
            path.moveTo(cx - scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.50, cy + scale * 0.60)
            path.lineTo(cx + scale * 0.38, cy - scale * 0.25)
            # Battlements
            path.lineTo(cx + scale * 0.55, cy - scale * 0.40)
            path.lineTo(cx + scale * 0.55, cy - scale * 0.80)
            path.lineTo(cx + scale * 0.30, cy - scale * 0.80)
            path.lineTo(cx + scale * 0.30, cy - scale * 0.55)
            path.lineTo(cx + scale * 0.10, cy - scale * 0.55)
            path.lineTo(cx + scale * 0.10, cy - scale * 0.80)
            path.lineTo(cx - scale * 0.10, cy - scale * 0.80)
            path.lineTo(cx - scale * 0.10, cy - scale * 0.55)
            path.lineTo(cx - scale * 0.30, cy - scale * 0.55)
            path.lineTo(cx - scale * 0.30, cy - scale * 0.80)
            path.lineTo(cx - scale * 0.55, cy - scale * 0.80)
            path.lineTo(cx - scale * 0.55, cy - scale * 0.40)
            path.lineTo(cx - scale * 0.38, cy - scale * 0.25)
            path.lineTo(cx - scale * 0.50, cy + scale * 0.60)
            path.closeSubpath()
            painter.drawPath(path)

        elif symbol == "N":  # Knight
            path.moveTo(cx - scale * 0.60, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.60, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.45, cy + scale * 0.60)
            path.quadTo(cx + scale * 0.65, cy + scale * 0.10, cx + scale * 0.35, cy - scale * 0.30)
            # Ear
            path.lineTo(cx + scale * 0.20, cy - scale * 0.80)
            path.lineTo(cx + scale * 0.05, cy - scale * 0.65)
            # Muzzle
            path.lineTo(cx - scale * 0.45, cy - scale * 0.40)
            path.lineTo(cx - scale * 0.60, cy - scale * 0.05)
            path.lineTo(cx - scale * 0.30, cy + scale * 0.10)
            path.quadTo(cx - scale * 0.10, cy + scale * 0.45, cx - scale * 0.45, cy + scale * 0.60)
            path.closeSubpath()
            painter.drawPath(path)
            # Eye dot
            painter.setBrush(QBrush(accent_color))
            painter.drawEllipse(QPointF(cx - scale * 0.18, cy - scale * 0.35), scale * 0.08, scale * 0.08)

        elif symbol == "B":  # Bishop
            path.moveTo(cx - scale * 0.60, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.60, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.40, cy + scale * 0.60)
            path.quadTo(cx + scale * 0.55, cy + scale * 0.05, cx + scale * 0.40, cy - scale * 0.35)
            path.quadTo(cx + scale * 0.15, cy - scale * 0.75, cx, cy - scale * 0.78)
            path.quadTo(cx - scale * 0.15, cy - scale * 0.75, cx - scale * 0.40, cy - scale * 0.35)
            path.quadTo(cx - scale * 0.55, cy + scale * 0.05, cx - scale * 0.40, cy + scale * 0.60)
            path.closeSubpath()
            painter.drawPath(path)
            # Top bead
            painter.drawEllipse(QPointF(cx, cy - scale * 0.85), scale * 0.12, scale * 0.12)
            # Slash cut
            painter.setPen(QPen(accent_color, max(1.5, scale * 0.08)))
            painter.drawLine(
                QPointF(cx - scale * 0.15, cy - scale * 0.45),
                QPointF(cx + scale * 0.18, cy - scale * 0.20),
            )

        elif symbol == "Q":  # Queen
            path.moveTo(cx - scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.45, cy + scale * 0.60)
            # Crown spikes
            path.lineTo(cx + scale * 0.68, cy - scale * 0.45)
            path.lineTo(cx + scale * 0.38, cy - scale * 0.15)
            path.lineTo(cx + scale * 0.35, cy - scale * 0.65)
            path.lineTo(cx + scale * 0.15, cy - scale * 0.20)
            path.lineTo(cx, cy - scale * 0.75)
            path.lineTo(cx - scale * 0.15, cy - scale * 0.20)
            path.lineTo(cx - scale * 0.35, cy - scale * 0.65)
            path.lineTo(cx - scale * 0.38, cy - scale * 0.15)
            path.lineTo(cx - scale * 0.68, cy - scale * 0.45)
            path.lineTo(cx - scale * 0.45, cy + scale * 0.60)
            path.closeSubpath()
            painter.drawPath(path)
            # Crown pearls
            painter.drawEllipse(QPointF(cx - scale * 0.68, cy - scale * 0.48), scale * 0.09, scale * 0.09)
            painter.drawEllipse(QPointF(cx - scale * 0.35, cy - scale * 0.68), scale * 0.09, scale * 0.09)
            painter.drawEllipse(QPointF(cx, cy - scale * 0.78), scale * 0.09, scale * 0.09)
            painter.drawEllipse(QPointF(cx + scale * 0.35, cy - scale * 0.68), scale * 0.09, scale * 0.09)
            painter.drawEllipse(QPointF(cx + scale * 0.68, cy - scale * 0.48), scale * 0.09, scale * 0.09)

        elif symbol == "K":  # King
            path.moveTo(cx - scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.65, cy + scale * 0.85)
            path.lineTo(cx + scale * 0.45, cy + scale * 0.60)
            path.quadTo(cx + scale * 0.60, cy + scale * 0.15, cx + scale * 0.45, cy - scale * 0.35)
            path.lineTo(cx + scale * 0.20, cy - scale * 0.50)
            path.lineTo(cx - scale * 0.20, cy - scale * 0.50)
            path.lineTo(cx - scale * 0.45, cy - scale * 0.35)
            path.quadTo(cx - scale * 0.60, cy + scale * 0.15, cx - scale * 0.45, cy + scale * 0.60)
            path.closeSubpath()
            painter.drawPath(path)

            # King's Latin Cross
            cross_pen = QPen(outline_color, max(2.0, scale * 0.10))
            painter.setPen(cross_pen)
            painter.drawLine(QPointF(cx, cy - scale * 0.50), QPointF(cx, cy - scale * 0.88))
            painter.drawLine(QPointF(cx - scale * 0.18, cy - scale * 0.72), QPointF(cx + scale * 0.18, cy - scale * 0.72))

    def _draw_arrow(
        self,
        painter: QPainter,
        move: chess.Move,
        color: QColor,
        square_size: float,
        offset_x: float,
        offset_y: float,
    ) -> None:
        """Paints smooth, anti-aliased directional vector arrows."""
        r1 = self._get_square_rect(move.from_square, square_size, offset_x, offset_y)
        r2 = self._get_square_rect(move.to_square, square_size, offset_x, offset_y)

        p1: QPointF = r1.center()
        p2: QPointF = r2.center()

        dx: float = p2.x() - p1.x()
        dy: float = p2.y() - p1.y()
        angle: float = math.atan2(dy, dx)
        dist: float = math.hypot(dx, dy)

        if dist < 1.0:
            return

        arrow_head_len: float = square_size * 0.38
        line_thickness: float = square_size * 0.12

        pen = QPen(color, line_thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QBrush(color))

        # Line shaft ending before arrow point
        shaft_end_x: float = p2.x() - arrow_head_len * 0.7 * math.cos(angle)
        shaft_end_y: float = p2.y() - arrow_head_len * 0.7 * math.sin(angle)
        painter.drawLine(p1, QPointF(shaft_end_x, shaft_end_y))

        # Arrow head triangle
        head_angle: float = math.pi / 6.0
        p_left = QPointF(
            p2.x() - arrow_head_len * math.cos(angle - head_angle),
            p2.y() - arrow_head_len * math.sin(angle - head_angle),
        )
        p_right = QPointF(
            p2.x() - arrow_head_len * math.cos(angle + head_angle),
            p2.y() - arrow_head_len * math.sin(angle + head_angle),
        )

        arrow_poly = QPolygonF([p2, p_left, p_right])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(arrow_poly)
