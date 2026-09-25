from __future__ import annotations
import logging
from typing import Callable, Dict, Optional, Tuple
import chess

logger: logging.Logger = logging.getLogger("MiniChess.CoreSync")


class GameStateSynchronizer:
    """Manages board state synchronization, sanity validation, noise filtering, and legal move inference."""

    def __init__(
        self,
        board: Optional[chess.Board] = None,
        consecutive_frames_required: int = 2,
        min_move_agreement: float = 0.93,
    ) -> None:
        self.board: chess.Board = board.copy() if board is not None else chess.Board()
        self.consecutive_frames_required: int = max(1, consecutive_frames_required)
        self.min_move_agreement: float = min_move_agreement

        self._pending_move: Optional[chess.Move] = None
        self._pending_count: int = 0
        self._last_confirmed_fen: str = self.board.fen()

    def reset(self, new_board: Optional[chess.Board] = None) -> None:
        """Resets the synchronizer with an initial or designated board state."""
        self.board = new_board.copy() if new_board is not None else chess.Board()
        self._pending_move = None
        self._pending_count = 0
        self._last_confirmed_fen = self.board.fen()

    @staticmethod
    def validate_detection_sanity(
        piece_grid: Dict[chess.Square, Optional[chess.Piece]],
    ) -> Tuple[bool, Optional[str]]:
        """Verifies fundamental chess legality: kings count, piece caps, rank bounds."""
        white_kings: int = sum(
            1 for p in piece_grid.values() if p is not None and p.symbol() == "K"
        )
        black_kings: int = sum(
            1 for p in piece_grid.values() if p is not None and p.symbol() == "k"
        )

        if white_kings != 1:
            return False, f"King count violation: Found {white_kings} White kings (must be 1)"
        if black_kings != 1:
            return False, f"King count violation: Found {black_kings} Black kings (must be 1)"

        white_pieces: int = sum(
            1 for p in piece_grid.values() if p is not None and p.color == chess.WHITE
        )
        black_pieces: int = sum(
            1 for p in piece_grid.values() if p is not None and p.color == chess.BLACK
        )

        if white_pieces > 16:
            return False, f"Piece count violation: White has {white_pieces} pieces (max 16)"
        if black_pieces > 16:
            return False, f"Piece count violation: Black has {black_pieces} pieces (max 16)"

        # Pawns on 1st or 8th rank are impossible in legal chess
        for file_idx in range(8):
            sq_rank1: chess.Square = chess.square(file_idx, 0)
            p1: Optional[chess.Piece] = piece_grid.get(sq_rank1)
            if p1 is not None and p1.piece_type == chess.PAWN:
                return False, f"Pawn rule violation: Pawn on rank 1 at file {file_idx}"

            sq_rank8: chess.Square = chess.square(file_idx, 7)
            p8: Optional[chess.Piece] = piece_grid.get(sq_rank8)
            if p8 is not None and p8.piece_type == chess.PAWN:
                return False, f"Pawn rule violation: Pawn on rank 8 at file {file_idx}"

        return True, None

    def infer_legal_move(
        self,
        detected_grid: Dict[chess.Square, Optional[chess.Piece]],
    ) -> Tuple[Optional[chess.Move], float]:
        """Compares detected piece placement against all legal moves from current board state.

        Returns (best_legal_move, agreement_score 0.0-1.0).
        """
        # Check agreement with current board state (no move made yet)
        current_matches: int = 0
        for sq in chess.SQUARES:
            exp_p: Optional[chess.Piece] = self.board.piece_at(sq)
            det_p: Optional[chess.Piece] = detected_grid.get(sq)
            if (exp_p is None and det_p is None) or (
                exp_p is not None and det_p is not None and exp_p.symbol() == det_p.symbol()
            ):
                current_matches += 1

        if current_matches >= 63:
            # Board matches current state, no new move
            return None, 0.0

        best_move: Optional[chess.Move] = None
        best_score: float = -1.0

        for move in self.board.legal_moves:
            test_board: chess.Board = self.board.copy(stack=False)
            test_board.push(move)

            matches: int = 0
            for sq in chess.SQUARES:
                exp_p = test_board.piece_at(sq)
                det_p = detected_grid.get(sq)
                if (exp_p is None and det_p is None) or (
                    exp_p is not None and det_p is not None and exp_p.symbol() == det_p.symbol()
                ):
                    matches += 1

            score: float = matches / 64.0
            if score > best_score:
                best_score = score
                best_move = move

        if best_score >= self.min_move_agreement:
            return best_move, best_score

        return None, 0.0

    def process_detected_frame(
        self,
        piece_grid: Dict[chess.Square, Optional[chess.Piece]],
    ) -> Tuple[Optional[chess.Move], bool]:
        """Processes a detected board frame through sanity verification, move inference,

        and consecutive-frame debounce noise filtering.
        Returns:
            (confirmed_move | None, is_valid_frame)
        """
        # 1. Sanity Verification
        is_sane, error_msg = self.validate_detection_sanity(piece_grid)
        if not is_sane:
            logger.debug("Discarding garbled/illegal frame: %s", error_msg)
            # Fail-safe: discard frame silently without resetting state
            return None, False

        if self.board.is_game_over():
            return None, True

        # 2. Legal Move Inference
        inferred_move, confidence = self.infer_legal_move(piece_grid)
        if inferred_move is None or confidence < self.min_move_agreement:
            # No legal move candidate matched or transient noise
            self._pending_move = None
            self._pending_count = 0
            return None, True

        # 3. Two-Consecutive-Frame Debounce (filters piece-dragging visual noise)
        if self._pending_move == inferred_move:
            self._pending_count += 1
        else:
            self._pending_move = inferred_move
            self._pending_count = 1

        if self._pending_count >= self.consecutive_frames_required:
            confirmed: chess.Move = inferred_move
            self.board.push(confirmed)
            self._last_confirmed_fen = self.board.fen()
            self._pending_move = None
            self._pending_count = 0
            logger.info("Confirmed legal move via screen synchronization: %s", confirmed.uci())
            return confirmed, True

        return None, True
