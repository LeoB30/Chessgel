"""Game state manager maintaining board logic, legal moves, and PGN history."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import chess
import chess.pgn
import io

@dataclass
class MoveRecord:
    move: chess.Move
    san: str
    fen_after: str
    move_number: int
    is_white: bool

class GameState:
    """Encapsulates the chess game state, history, and legality rules."""

    def __init__(self, initial_fen: Optional[str] = None) -> None:
        self.board: chess.Board = chess.Board(initial_fen) if initial_fen else chess.Board()
        self.move_history: List[MoveRecord] = []
        self.is_flipped: bool = False
        self.last_move: Optional[chess.Move] = None

    def reset(self, fen: Optional[str] = None) -> None:
        """Resets the board state to starting position or designated FEN."""
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history.clear()
        self.last_move = None

    def apply_move(self, move: chess.Move) -> bool:
        """Executes a move if legal according to python-chess."""
        if move in self.board.legal_moves:
            san = self.board.san(move)
            move_number = self.board.fullmove_number
            is_white = self.board.turn == chess.WHITE
            self.board.push(move)
            self.last_move = move
            self.move_history.append(
                MoveRecord(
                    move=move,
                    san=san,
                    fen_after=self.board.fen(),
                    move_number=move_number,
                    is_white=is_white,
                )
            )
            return True
        return False

    def undo_last_move(self) -> bool:
        """Pops the last move, reverting the board and history.

        Used by the self-correction system when a false positive is detected.
        Returns True if a move was undone, False if the stack was empty.
        """
        if self.board.move_stack and self.move_history:
            self.board.pop()
            self.move_history.pop()
            self.last_move = self.move_history[-1].move if self.move_history else None
            return True
        return False

    def is_legal(self, move: chess.Move) -> bool:
        """Checks if a candidate move is legal in current position."""
        return move in self.board.legal_moves

    def get_fen(self) -> str:
        """Returns the current FEN string."""
        return self.board.fen()

    def get_pgn(self) -> str:
        """Exports the move history to standard PGN format."""
        game = chess.pgn.Game.from_board(self.board)
        # Or construct from move records
        root = chess.pgn.Game()
        root.headers["Event"] = "External Screen Live Analysis"
        root.headers["Site"] = "Antigravity Chess Tracker"
        node = root
        temp_board = chess.Board()
        for rec in self.move_history:
            node = node.add_variation(rec.move)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return root.accept(exporter)

    def get_formatted_move_table(self) -> List[Tuple[int, str, str]]:
        """Returns move history formatted as rows: [(move_num, white_san, black_san), ...]."""
        table: List[Tuple[int, str, str]] = []
        current_num = 1
        white_san = ""
        black_san = ""

        for rec in self.move_history:
            if rec.is_white:
                if white_san:  # Flush previous unfinished row if any
                    table.append((current_num, white_san, ""))
                current_num = rec.move_number
                white_san = rec.san
                black_san = ""
            else:
                black_san = rec.san
                table.append((rec.move_number, white_san if white_san else "...", black_san))
                white_san = ""
                black_san = ""

        if white_san:
            table.append((current_num, white_san, ""))

        return table

    def get_status_text(self) -> str:
        """Returns current game condition (check, checkmate, turn)."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"Checkmate! {winner} wins."
        if self.board.is_stalemate():
            return "Draw by stalemate."
        if self.board.is_insufficient_material():
            return "Draw: Insufficient material."
        if self.board.is_seventyfive_moves():
            return "Draw: 75-move rule."
        if self.board.is_fivefold_repetition():
            return "Draw: Fivefold repetition."
        if self.board.is_check():
            turn_str = "White" if self.board.turn == chess.WHITE else "Black"
            return f"Check! ({turn_str} to move)"
        
        turn_str = "White" if self.board.turn == chess.WHITE else "Black"
        return f"{turn_str} to move"

    def get_material_balance(self) -> Tuple[int, Dict[str, int], Dict[str, int]]:
        """Calculates material score (White - Black) and captured piece counts."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        white_pieces = {p: len(self.board.pieces(p, chess.WHITE)) for p in piece_values}
        black_pieces = {p: len(self.board.pieces(p, chess.BLACK)) for p in piece_values}

        white_score = sum(white_pieces[p] * piece_values[p] for p in piece_values)
        black_score = sum(black_pieces[p] * piece_values[p] for p in piece_values)
        diff = white_score - black_score

        initial_counts = {
            chess.PAWN: 8,
            chess.KNIGHT: 2,
            chess.BISHOP: 2,
            chess.ROOK: 2,
            chess.QUEEN: 1,
        }
        # Captured pieces:
        # Pieces White lost = initial - white_pieces (captured by Black)
        # Pieces Black lost = initial - black_pieces (captured by White)
        white_lost = {chess.piece_symbol(p).upper(): max(0, initial_counts[p] - white_pieces[p]) for p in piece_values}
        black_lost = {chess.piece_symbol(p).lower(): max(0, initial_counts[p] - black_pieces[p]) for p in piece_values}

        return diff, white_lost, black_lost
