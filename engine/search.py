from __future__ import annotations
from typing import Callable
import chess
from engine.evaluator import PIECE_VALUES, evaluate_board
from engine.opening_book import get_book_move

# Infinity constants for search bounds
INFINITY: int = 1_000_000
MATE_SCORE: int = 90_000


def find_best_move(
    board: chess.Board,
    depth: int = 3,
    use_book: bool = True,
    info_callback: Callable[[int, int, int, chess.Move], None] | None = None,
) -> chess.Move | None:
    """Finds the best legal move using Opening Book lookup, Negamax with Alpha-Beta pruning,

    MVV-LVA move ordering, and Quiescence search.
    """
    # 1. Opening Book Lookup: Instant return if enabled and position matches known repertoire
    if use_book:
        book_move: chess.Move | None = get_book_move(board)
        if book_move is not None:
            if info_callback is not None:
                info_callback(1, 0, 1, book_move)
            return book_move

    # 2. Fallback to Alpha-Beta Search
    legal_moves: list[chess.Move] = _order_moves(board)
    if not legal_moves:
        return None

    best_move: chess.Move = legal_moves[0]
    alpha: int = -INFINITY
    beta: int = INFINITY
    nodes_evaluated: int = 0

    for move in legal_moves:
        board.push(move)
        score: int = -_alpha_beta(board, depth - 1, -beta, -alpha)
        board.pop()
        nodes_evaluated += 1

        if score > alpha:
            alpha = score
            best_move = move

            if info_callback is not None:
                info_callback(depth, alpha, nodes_evaluated, best_move)

    return best_move


def _alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """Negamax search with Alpha-Beta pruning."""
    if board.is_checkmate():
        # Prefer faster mates: penalize depth to reward shallower mates
        return -MATE_SCORE - depth

    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0

    if depth <= 0:
        return _quiescence(board, alpha, beta)

    ordered_moves: list[chess.Move] = _order_moves(board)
    for move in ordered_moves:
        board.push(move)
        score: int = -_alpha_beta(board, depth - 1, -beta, -alpha)
        board.pop()

        if score >= beta:
            # Beta cutoff (prune branch)
            return beta
        if score > alpha:
            alpha = score

    return alpha


def _quiescence(board: chess.Board, alpha: int, beta: int) -> int:
    """Quiescence search: evaluates captures only to prevent the horizon effect."""
    stand_pat: int = _relative_eval(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Only examine legal captures
    capture_moves: list[chess.Move] = _order_moves(board, captures_only=True)
    for move in capture_moves:
        board.push(move)
        score: int = -_quiescence(board, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def _relative_eval(board: chess.Board) -> int:
    """Returns static evaluation from the perspective of the side to move."""
    white_score: int = evaluate_board(board)
    return white_score if board.turn == chess.WHITE else -white_score


def _order_moves(board: chess.Board, captures_only: bool = False) -> list[chess.Move]:
    """Orders moves prioritizing Most Valuable Victim - Least Valuable Attacker (MVV-LVA)."""
    moves: list[chess.Move]
    if captures_only:
        moves = [m for m in board.legal_moves if board.is_capture(m)]
    else:
        moves = list(board.legal_moves)

    def move_priority(m: chess.Move) -> int:
        score: int = 0
        if board.is_capture(m):
            victim: chess.Piece | None = board.piece_at(m.to_square)
            attacker: chess.Piece | None = board.piece_at(m.from_square)
            v_val: int = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            a_val: int = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            # Higher score for high-value victim taken by lower-value attacker
            score = 10_000 + (v_val * 10 - a_val)

        if m.promotion:
            score += 8_000

        return score

    moves.sort(key=move_priority, reverse=True)
    return moves
