from __future__ import annotations
import tkinter as tk
from typing import Callable
import chess

# Solid opaque chess glyphs (using filled characters for both White and Black)
SOLID_PIECE_GLYPHS: dict[str, str] = {
    "P": "♟",
    "N": "♞",
    "B": "♝",
    "R": "♜",
    "Q": "♛",
    "K": "♚",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


class BoardCanvas(tk.Canvas):
    """Interactive desktop canvas for rendering chess board, pieces, arrows, and highlights."""

    LIGHT_SQUARE: str = "#EEEED2"
    DARK_SQUARE: str = "#769656"
    SELECTED_COLOR: str = "#F6F669"
    LAST_MOVE_COLOR: str = "#BACA44"
    CHECK_COLOR: str = "#E84118"
    MOVE_HINT_COLOR: str = "#557A3C"
    ARROW_COLOR: str = "#00D2D3"
    OPPONENT_ARROW_COLOR: str = "#FFA500"

    def __init__(
        self,
        master: tk.Misc,
        board_size: int = 560,
        on_move: Callable[[chess.Move], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            master,
            width=board_size,
            height=board_size,
            highlightthickness=0,
            bg="#181825",
            **kwargs,
        )
        self.board_size: int = board_size
        self.square_size: int = board_size // 8
        self.board: chess.Board = chess.Board()
        self.is_flipped: bool = False
        self.interactive: bool = True
        self.show_arrow: bool = True
        self.on_move: Callable[[chess.Move], None] | None = on_move

        self.selected_square: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.hint_move: chess.Move | None = None
        self.detected_opponent_move: chess.Move | None = None
        self.candidate_moves: list[chess.Move] = []

        self.bind("<Button-1>", self._on_click)

    def set_board(
        self,
        board: chess.Board,
        last_move: chess.Move | None = None,
        hint_move: chess.Move | None = None,
    ) -> None:
        """Updates internal board state and redraws."""
        self.board = board
        self.last_move = last_move
        self.hint_move = hint_move
        self.selected_square = None
        self.candidate_moves.clear()
        self.redraw()

    def set_hint_move(self, hint_move: chess.Move | None) -> None:
        """Sets or clears the suggested engine move arrow."""
        self.hint_move = hint_move
        self.redraw()

    def set_vision_moves(
        self,
        opponent_move: chess.Move | None,
        engine_move: chess.Move | None,
    ) -> None:
        """Sets both detected opponent move arrow and engine counter-move arrow."""
        self.detected_opponent_move = opponent_move
        self.hint_move = engine_move
        self.redraw()

    def set_flipped(self, flipped: bool) -> None:
        """Toggles White/Black perspective."""
        self.is_flipped = flipped
        self.redraw()

    def redraw(self) -> None:
        """Performs a complete redraw of squares, highlights, hints, pieces, and arrows."""
        self.delete("all")
        sq_size: int = self.square_size

        # 1. Draw Squares & Highlights
        for rank in range(8):
            for file in range(8):
                sq: chess.Square = self._coords_to_square(file, rank)
                x1: int = file * sq_size
                y1: int = rank * sq_size
                x2: int = x1 + sq_size
                y2: int = y1 + sq_size

                is_light: bool = (file + rank) % 2 == 0
                bg_color: str = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE

                # Highlight last move squares
                if self.last_move and (sq == self.last_move.from_square or sq == self.last_move.to_square):
                    bg_color = self.LAST_MOVE_COLOR

                # Highlight in-check King square
                piece: chess.Piece | None = self.board.piece_at(sq)
                if piece and piece.piece_type == chess.KING and piece.color == self.board.turn and self.board.is_check():
                    bg_color = self.CHECK_COLOR

                # Highlight user-selected square
                if self.selected_square == sq:
                    bg_color = self.SELECTED_COLOR

                self.create_rectangle(x1, y1, x2, y2, fill=bg_color, outline="")

                # Coordinates indicator on border squares
                coord_color: str = self.DARK_SQUARE if is_light else self.LIGHT_SQUARE
                if (not self.is_flipped and rank == 7) or (self.is_flipped and rank == 0):
                    file_char: str = chr(ord("a") + (7 - file if self.is_flipped else file))
                    self.create_text(
                        x1 + sq_size - 10,
                        y2 - 10,
                        text=file_char,
                        font=("Segoe UI", 9, "bold"),
                        fill=coord_color,
                    )
                if (not self.is_flipped and file == 0) or (self.is_flipped and file == 7):
                    rank_num: str = str(rank + 1 if self.is_flipped else 8 - rank)
                    self.create_text(
                        x1 + 10,
                        y1 + 10,
                        text=rank_num,
                        font=("Segoe UI", 9, "bold"),
                        fill=coord_color,
                    )

        # 2. Draw Legal Move Destination Hints
        for move in self.candidate_moves:
            dst_file, dst_rank = self._square_to_coords(move.to_square)
            cx: int = dst_file * sq_size + sq_size // 2
            cy: int = dst_rank * sq_size + sq_size // 2

            if self.board.piece_at(move.to_square):
                r: int = sq_size // 2 - 4
                self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=self.MOVE_HINT_COLOR, width=4)
            else:
                r = sq_size // 6
                self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=self.MOVE_HINT_COLOR, outline="")

        # 3. Draw Move Arrows (Opponent Detected Move & Engine Best Move)
        if self.show_arrow:
            if self.detected_opponent_move is not None:
                self._draw_arrow(
                    self.detected_opponent_move.from_square,
                    self.detected_opponent_move.to_square,
                    color=self.OPPONENT_ARROW_COLOR,
                )
            if self.hint_move is not None:
                self._draw_arrow(
                    self.hint_move.from_square,
                    self.hint_move.to_square,
                    color=self.ARROW_COLOR,
                )

        # 4. Draw Pieces (100% Solid Opaque Rendering)
        font_size: int = int(sq_size * 0.68)
        for sq in chess.SQUARES:
            p: chess.Piece | None = self.board.piece_at(sq)
            if p is not None:
                f, r = self._square_to_coords(sq)
                cx = f * sq_size + sq_size // 2
                cy = r * sq_size + sq_size // 2
                glyph: str = SOLID_PIECE_GLYPHS.get(p.symbol(), "")
                is_white: bool = p.color == chess.WHITE

                body_color: str = "#FFFFFF" if is_white else "#11111B"
                halo_color: str = "#11111B" if is_white else "#F8F9FA"

                # Multi-offset opaque boundary halo (eliminates all square transparency)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1), (-2, 0), (2, 0), (0, -2), (0, 2)):
                    self.create_text(
                        cx + dx,
                        cy + dy,
                        text=glyph,
                        font=("Segoe UI Symbol", font_size),
                        fill=halo_color,
                    )

                # Solid opaque body fill
                self.create_text(
                    cx,
                    cy,
                    text=glyph,
                    font=("Segoe UI Symbol", font_size),
                    fill=body_color,
                )

    def _draw_arrow(self, from_sq: chess.Square, to_sq: chess.Square, color: str = "#00D2D3") -> None:
        """Renders an arrow indicating an engine recommendation or tactical threat."""
        fx, fy = self._square_to_coords(from_sq)
        tx, ty = self._square_to_coords(to_sq)
        sq_size: int = self.square_size
        x1: int = fx * sq_size + sq_size // 2
        y1: int = fy * sq_size + sq_size // 2
        x2: int = tx * sq_size + sq_size // 2
        y2: int = ty * sq_size + sq_size // 2

        self.create_line(
            x1,
            y1,
            x2,
            y2,
            arrow=tk.LAST,
            arrowshape=(16, 20, 7),
            width=5,
            fill=color,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )

    def _on_click(self, event: tk.Event) -> None:
        """Handles mouse clicks for piece selection and legal moves."""
        if not self.interactive:
            return

        file: int = event.x // self.square_size
        rank: int = event.y // self.square_size
        if not (0 <= file < 8 and 0 <= rank < 8):
            return

        clicked_square: chess.Square = self._coords_to_square(file, rank)

        # If a square was already selected, check if user clicked a valid destination
        if self.selected_square is not None:
            chosen_move: chess.Move | None = None
            for move in self.candidate_moves:
                if move.to_square == clicked_square:
                    # Auto-promote to Queen if pawn reaches final rank
                    if move.promotion and move.promotion != chess.QUEEN:
                        continue
                    chosen_move = move
                    break

            if chosen_move is not None:
                self.selected_square = None
                self.candidate_moves.clear()
                self.redraw()
                if self.on_move:
                    self.on_move(chosen_move)
                return

        # Otherwise, select the clicked piece if it matches the side to move
        piece: chess.Piece | None = self.board.piece_at(clicked_square)
        if piece is not None and piece.color == self.board.turn:
            self.selected_square = clicked_square
            self.candidate_moves = [
                m for m in self.board.legal_moves if m.from_square == clicked_square
            ]
        else:
            self.selected_square = None
            self.candidate_moves.clear()

        self.redraw()

    def _coords_to_square(self, file: int, rank: int) -> chess.Square:
        """Converts canvas file/rank grid coordinates to chess.Square."""
        if self.is_flipped:
            f: int = 7 - file
            r: int = rank
        else:
            f = file
            r = 7 - rank
        return chess.square(f, r)

    def _square_to_coords(self, sq: chess.Square) -> tuple[int, int]:
        """Converts chess.Square to canvas file/rank grid coordinates."""
        f: int = chess.square_file(sq)
        r: int = chess.square_rank(sq)
        if self.is_flipped:
            return 7 - f, r
        return f, 7 - r
