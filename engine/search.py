from __future__ import annotations
import time
from typing import Callable
import chess
from engine.evaluator import PIECE_VALUES, evaluate_board
from engine.opening_book import get_book_move

# Infinity constants for search bounds
INFINITY: int = 1_000_000
MATE_SCORE: int = 90_000


class SearchTimeout(Exception):
    """Raised when search exceeds the allocated movetime budget."""
    pass


def find_best_move(
    board: chess.Board,
    depth: int = 3,
    movetime_ms: int | None = None,
    use_book: bool = True,
    book_repertoire: str = "catalan_carokann",
    book_selection: str = "best",
    book_max_ply: int = 24,
    info_callback: Callable[[int, int, int, chess.Move, str], None] | None = None,
) -> chess.Move | None:
    """Finds the best legal move using opening book lookup, iterative deepening,

    Alpha-Beta pruning, MVV-LVA move ordering, and Quiescence search.
    """
    # 1. Opening Book Lookup: Instant return if enabled and within book ply limit
    if use_book and board.ply() < book_max_ply:
        book_move: chess.Move | None = get_book_move(
            board,
            repertoire=book_repertoire,
            selection_mode=book_selection,
        )
        if book_move is not None:
            if info_callback is not None:
                info_callback(1, 0, 1, book_move, "book")
            return book_move

    # 2. Prepare Search
    legal_moves: list[chess.Move] = _order_moves(board)
    if not legal_moves:
        return None

    if len(legal_moves) == 1:
        # Only one legal move available
        if info_callback is not None:
            info_callback(1, 0, 1, legal_moves[0], "forced")
        return legal_moves[0]

    start_time: float = time.perf_counter()
    deadline: float | None = None
    if movetime_ms is not None and movetime_ms > 0:
        # Leave a safety margin (e.g. 50ms) to ensure timely transmission
        budget_sec: float = max(0.02, (movetime_ms - 20) / 1000.0)
        deadline = start_time + budget_sec

    best_overall_move: chess.Move = legal_moves[0]
    nodes_evaluated: int = 0

    # Iterative deepening from depth 1 up to target depth
    max_target_depth: int = max(1, depth)

    for current_depth in range(1, max_target_depth + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            break

        alpha: int = -INFINITY
        beta: int = INFINITY
        iteration_best_move: chess.Move | None = None
        iteration_nodes: int = 0

        # Place the previous iteration's best move first for optimal pruning
        current_moves = list(legal_moves)
        if best_overall_move in current_moves:
            current_moves.remove(best_overall_move)
            current_moves.insert(0, best_overall_move)

        interrupted: bool = False

        for move in current_moves:
            if deadline is not None and time.perf_counter() >= deadline:
                interrupted = True
                break

            board.push(move)
            try:
                score: int = -_alpha_beta(board, current_depth - 1, -beta, -alpha, deadline)
            except SearchTimeout:
                board.pop()
                interrupted = True
                break

            board.pop()
            iteration_nodes += 1
            nodes_evaluated += 1

            if score > alpha:
                alpha = score
                iteration_best_move = move

                if info_callback is not None:
                    info_callback(current_depth, alpha, nodes_evaluated, iteration_best_move, "search")

        if not interrupted and iteration_best_move is not None:
            best_overall_move = iteration_best_move
            # If mate was found, stop searching deeper
            if alpha >= MATE_SCORE - 100 or alpha <= -MATE_SCORE + 100:
                break

    return best_overall_move


def _alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    deadline: float | None = None,
) -> int:
    """Negamax search with Alpha-Beta pruning and optional deadline checks."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout()

    if board.is_checkmate():
        return -MATE_SCORE - depth

    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0

    if depth <= 0:
        return _quiescence(board, alpha, beta, deadline)

    ordered_moves: list[chess.Move] = _order_moves(board)
    for move in ordered_moves:
        board.push(move)
        try:
            score: int = -_alpha_beta(board, depth - 1, -beta, -alpha, deadline)
        finally:
            board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def _quiescence(
    board: chess.Board,
    alpha: int,
    beta: int,
    deadline: float | None = None,
) -> int:
    """Quiescence search: evaluates captures to resolve tactical tactical tension."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout()

    stand_pat: int = _relative_eval(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    capture_moves: list[chess.Move] = _order_moves(board, captures_only=True)
    for move in capture_moves:
        board.push(move)
        try:
            score: int = -_quiescence(board, -beta, -alpha, deadline)
        finally:
            board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def _relative_eval(board: chess.Board) -> int:
    """Returns static evaluation from perspective of player to move."""
    white_score: int = evaluate_board(board)
    return white_score if board.turn == chess.WHITE else -white_score


def _order_moves(board: chess.Board, captures_only: bool = False) -> list[chess.Move]:
    """Orders moves prioritizing MVV-LVA (Most Valuable Victim - Least Valuable Attacker)."""
    if captures_only:
        moves: list[chess.Move] = [m for m in board.legal_moves if board.is_capture(m)]
    else:
        moves = list(board.legal_moves)

    def move_priority(m: chess.Move) -> int:
        score: int = 0
        if board.is_capture(m):
            victim: chess.Piece | None = board.piece_at(m.to_square)
            attacker: chess.Piece | None = board.piece_at(m.from_square)
            v_val: int = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            a_val: int = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            score = 10_000 + (v_val * 10 - a_val)

        if m.promotion:
            score += 8_000

        return score

    moves.sort(key=move_priority, reverse=True)
    return moves
