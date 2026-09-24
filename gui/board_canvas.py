from __future__ import annotations
import tkinter as tk
from typing import Callable
import chess

# Unicode chess glyphs
PIECE_UNICODE: dict[str, str] = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


class BoardCanvas(tk.Canvas):
    """Tkinter Canvas widget that renders an interactive chess board."""

    LIGHT_SQUARE: str = "#EEEED2"
    DARK_SQUARE: str = "#769656"
    SELECTED_COLOR: str = "#F6F669"
    LAST_MOVE_COLOR: str = "#BACA44"
    CHECK_COLOR: str = "#E84118"
    MOVE_HINT_COLOR: str = "#557A3C"

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
            bg="#2B2D42",
            **kwargs,
        )
        self.board_size: int = board_size
        self.square_size: int = board_size // 8
        self.board: chess.Board = chess.Board()
        self.is_flipped: bool = False
        self.interactive: bool = True
        self.on_move: Callable[[chess.Move], None] | None = on_move

        self.selected_square: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.candidate_moves: list[chess.Move] = []

        self.bind("<Button-1>", self._on_click)

    def set_board(self, board: chess.Board, last_move: chess.Move | None = None) -> None:
        """Updates internal board state and redraws."""
        self.board = board
        self.last_move = last_move
        self.selected_square = None
        self.candidate_moves.clear()
        self.redraw()

    def set_flipped(self, flipped: bool) -> None:
        """Toggles White/Black perspective."""
        self.is_flipped = flipped
        self.redraw()

    def redraw(self) -> None:
        """Performs a complete redraw of squares, highlights, pieces, and coordinates."""
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

                # Base square color
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

                # Coordinates indicator
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
                # Ring for captures
                r: int = sq_size // 2 - 4
                self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=self.MOVE_HINT_COLOR, width=4)
            else:
                # Small circle for empty squares
                r = sq_size // 6
                self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=self.MOVE_HINT_COLOR, outline="")

        # 3. Draw Pieces
        font_size: int = int(sq_size * 0.65)
        for sq in chess.SQUARES:
            p: chess.Piece | None = self.board.piece_at(sq)
            if p is not None:
                f, r = self._square_to_coords(sq)
                cx = f * sq_size + sq_size // 2
                cy = r * sq_size + sq_size // 2
                glyph: str = PIECE_UNICODE.get(p.symbol(), "")
                color: str = "#FFFFFF" if p.color == chess.WHITE else "#111111"

                # Subtle shadow outline for contrast
                shadow_color: str = "#222222" if p.color == chess.WHITE else "#EEEEEE"
                self.create_text(
                    cx + 1,
                    cy + 1,
                    text=glyph,
                    font=("Segoe UI Symbol", font_size),
                    fill=shadow_color,
                )
                self.create_text(
                    cx,
                    cy,
                    text=glyph,
                    font=("Segoe UI Symbol", font_size),
                    fill=color,
                )

    def _on_click(self, event: tk.Event) -> None:
        """Handles mouse clicks for piece selection and moves."""
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
            # Collect legal moves originating from clicked square
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
