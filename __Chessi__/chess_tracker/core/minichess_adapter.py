"""Adapter for the AAA III MiniChess engine (opening book + alpha-beta + quiescence).

Wraps the old engine's search module into the EvaluationResult interface used by the
tracker so it can be selected alongside Stockfish per-side.
"""

from __future__ import annotations
import logging
import os
import sys
import threading
from typing import Callable, List, Optional, Tuple
import chess

from chess_tracker.core.engine import EvaluationResult

logger = logging.getLogger("ChessTracker.MiniChessAdapter")

# Ensure the parent AAA III directory is importable
_AAA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _AAA_ROOT not in sys.path:
    sys.path.insert(0, _AAA_ROOT)

from engine.search import find_best_move  # type: ignore
from engine.evaluator import evaluate_board  # type: ignore


class MiniChessEngine:
    """Non-blocking wrapper around the AAA III MiniChess engine.

    Exposes the same callback-based interface as UCIEngineManager so the
    EngineWorker can use it interchangeably with Stockfish.
    """

    ENGINE_NAME = "MiniChess (Custom)"

    def __init__(self, depth: int = 5, movetime_ms: int = 2000) -> None:
        self.depth = depth
        self.movetime_ms = movetime_ms
        self.is_running = True
        self.current_fen: str = chess.STARTING_FEN
        self.latest_result: EvaluationResult = EvaluationResult()
        self.on_evaluation_callback: Optional[Callable[[EvaluationResult], None]] = None
        logger.info("MiniChess engine adapter initialized (depth=%d, movetime=%dms)", depth, movetime_ms)

    def evaluate_position_async(self, fen: str, depth: int = 5) -> None:
        """Launches analysis in a background thread (non-blocking)."""
        self.current_fen = fen
        threading.Thread(
            target=self._run_analysis,
            args=(fen, depth),
            daemon=True,
        ).start()

    def _run_analysis(self, fen: str, depth: int) -> None:
        """Executes the full MiniChess search and emits result."""
        board = chess.Board(fen)
        if board.is_game_over():
            return

        collected_info: List[dict] = []

        def info_cb(d: int, score: int, nodes: int, move: chess.Move, source: str) -> None:
            collected_info.append({
                "depth": d, "score": score, "nodes": nodes, "move": move, "source": source,
            })

        try:
            best_move = find_best_move(
                board,
                depth=depth,
                movetime_ms=self.movetime_ms,
                use_book=True,
                info_callback=info_cb,
            )
        except Exception as e:
            logger.error("MiniChess search error: %s", e)
            return

        # Compute evaluation from White's perspective
        score_cp = evaluate_board(board)

        best_san = None
        if best_move and best_move in board.legal_moves:
            best_san = board.san(best_move)

        # Build PV from the collected info (last iteration's best move path)
        pv_uci: List[str] = [best_move.uci()] if best_move else []
        pv_san: List[str] = [best_san] if best_san else []

        final_depth = collected_info[-1]["depth"] if collected_info else depth
        total_nodes = collected_info[-1]["nodes"] if collected_info else 0

        result = EvaluationResult(
            score_cp=score_cp,
            depth=final_depth,
            nodes=total_nodes,
            nps=total_nodes * 5 if total_nodes else 0,  # rough estimate
            pv_uci=pv_uci,
            pv_san=pv_san,
            best_move=best_move,
            best_move_san=best_san,
            engine_name=self.ENGINE_NAME,
        )
        self.latest_result = result
        if self.on_evaluation_callback:
            self.on_evaluation_callback(result)

    def stop(self) -> None:
        """Cleanup — nothing to do for the in-process engine."""
        self.is_running = False
