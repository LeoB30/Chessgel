"""Asynchronous UCI chess engine interface with Stockfish and fallback evaluation."""

from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
import queue
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple
import chess

from chess_tracker.config import STOCKFISH_PATH, ENGINE_ANALYSIS_DEPTH

logger = logging.getLogger("ChessTracker.Engine")

@dataclass
class EvaluationResult:
    score_cp: Optional[int] = 0        # From White's perspective (positive = White ahead)
    mate_in: Optional[int] = None      # Turns to mate (positive = White wins, negative = Black wins)
    depth: int = 0
    nodes: int = 0
    nps: int = 0
    pv_uci: List[str] = field(default_factory=list)
    pv_san: List[str] = field(default_factory=list)
    best_move: Optional[chess.Move] = None
    best_move_san: Optional[str] = None
    engine_name: str = "Stockfish"
    # Multi-PV: top N candidate moves with their scores and principal variations
    # Each entry: (move: chess.Move, score_cp: int, pv_san: List[str])
    top_moves: List[Tuple[chess.Move, int, List[str]]] = field(default_factory=list)

    @property
    def display_score(self) -> str:
        if self.mate_in is not None:
            sign = "+" if self.mate_in > 0 else "-"
            return f"M{abs(self.mate_in)}" if self.mate_in > 0 else f"-M{abs(self.mate_in)}"
        if self.score_cp is not None:
            pawns = self.score_cp / 100.0
            return f"{pawns:+.2f}"
        return "0.00"

    @property
    def win_probability_white(self) -> float:
        """Converts centipawns to 0.0 - 1.0 winning probability for White."""
        if self.mate_in is not None:
            return 1.0 if self.mate_in > 0 else 0.0
        cp = self.score_cp or 0
        # Standard Lichess winning percentage formula: 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
        import math
        try:
            return 1.0 / (1.0 + math.exp(-0.00368208 * cp))
        except OverflowError:
            return 1.0 if cp > 0 else 0.0


# Piece-Square Tables for Fallback Evaluator
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]
KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

class FallbackEngine:
    """Lightweight built-in engine evaluator when external UCI binary is not available."""

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    @classmethod
    def evaluate_board(cls, board: chess.Board) -> int:
        """Static material + PST evaluation in centipawns (from White perspective)."""
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        val = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            p_val = cls.PIECE_VALUES[piece.piece_type]
            pst = 0
            if piece.piece_type == chess.PAWN:
                pst = PAWN_TABLE[sq if piece.color == chess.WHITE else 63 - sq]
            elif piece.piece_type == chess.KNIGHT:
                pst = KNIGHT_TABLE[sq if piece.color == chess.WHITE else 63 - sq]
            elif piece.piece_type == chess.BISHOP:
                pst = BISHOP_TABLE[sq if piece.color == chess.WHITE else 63 - sq]

            total = p_val + pst
            if piece.color == chess.WHITE:
                val += total
            else:
                val -= total
        return val

    @classmethod
    def search(cls, board: chess.Board, depth: int = 3) -> Tuple[Optional[chess.Move], int]:
        """Simple minimax search with alpha-beta pruning."""
        best_move: Optional[chess.Move] = None
        is_white = board.turn == chess.WHITE

        def alpha_beta(b: chess.Board, d: int, alpha: int, beta: int) -> int:
            if d == 0 or b.is_game_over():
                return cls.evaluate_board(b)

            if b.turn == chess.WHITE:
                max_eval = -999999
                for m in b.legal_moves:
                    b.push(m)
                    ev = alpha_beta(b, d - 1, alpha, beta)
                    b.pop()
                    if ev > max_eval:
                        max_eval = ev
                    alpha = max(alpha, ev)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = 999999
                for m in b.legal_moves:
                    b.push(m)
                    ev = alpha_beta(b, d - 1, alpha, beta)
                    b.pop()
                    if ev < min_eval:
                        min_eval = ev
                    beta = min(beta, ev)
                    if beta <= alpha:
                        break
                return min_eval

        alpha = -999999
        beta = 999999
        if is_white:
            best_eval = -999999
            for m in board.legal_moves:
                board.push(m)
                ev = alpha_beta(board, depth - 1, alpha, beta)
                board.pop()
                if ev > best_eval:
                    best_eval = ev
                    best_move = m
                alpha = max(alpha, ev)
        else:
            best_eval = 999999
            for m in board.legal_moves:
                board.push(m)
                ev = alpha_beta(board, depth - 1, alpha, beta)
                board.pop()
                if ev < best_eval:
                    best_eval = ev
                    best_move = m
                beta = min(beta, ev)

        return best_move, best_eval


class UCIEngineManager:
    """Non-blocking UCI engine controller communicating over pipes."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        name: Optional[str] = None,
        extra_options: Optional[dict] = None,
    ) -> None:
        self.binary_path = binary_path or STOCKFISH_PATH
        self.engine_name = name or ("Stockfish 15.1" if "Stockfish" in str(self.binary_path) else "UCI Engine")
        self.extra_options = extra_options or {}
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self._lock = threading.Lock()
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self.current_fen: str = chess.STARTING_FEN
        self.latest_result: EvaluationResult = EvaluationResult(engine_name=self.engine_name)
        self.on_evaluation_callback: Optional[Callable[[EvaluationResult], None]] = None
        self._current_multipv = 3

        self._start_engine()

    def set_multipv(self, lines: int) -> None:
        """Dynamically changes the number of analysis lines (MultiPV)."""
        if self.is_running and lines != self._current_multipv:
            self._current_multipv = max(1, lines)
            self._send_command(f"setoption name MultiPV value {self._current_multipv}")
            logger.info("Set %s MultiPV to %d", self.engine_name, self._current_multipv)

    def set_aggressiveness(self, contempt_val: int) -> None:
        """Attempts to set aggressiveness via UCI Contempt (or similar options).
        
        Values > 0 usually mean playing more aggressively against weaker players
        and avoiding draws.
        """
        if self.is_running:
            self._send_command(f"setoption name Contempt value {contempt_val}")
            # Some engines like Patricia use different names, we can throw a few common ones
            self._send_command(f"setoption name Aggressiveness value {contempt_val}")
            logger.info("Set %s Aggressiveness/Contempt to %d", self.engine_name, contempt_val)

    def _start_engine(self) -> None:
        """Launches the UCI process or falls back to internal engine."""
        if self.binary_path and os.path.isfile(self.binary_path):
            try:
                working_dir = os.path.dirname(os.path.abspath(self.binary_path))
                self.process = subprocess.Popen(
                    [self.binary_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    cwd=working_dir,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                self._multipv_results: dict[int, dict] = {}
                self._read_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._read_thread.start()

                self._send_command("uci")
                self._send_command("setoption name Threads value 2")
                self._send_command("setoption name Hash value 32")
                self._send_command("setoption name MultiPV value 3")
                for opt_k, opt_v in self.extra_options.items():
                    self._send_command(f"setoption name {opt_k} value {opt_v}")
                self._send_command("isready")

                # Wait up to 2 seconds for engine to complete handshake
                self._ready_event.wait(timeout=2.0)
                self.is_running = True
                logger.info("%s started successfully at: %s", self.engine_name, self.binary_path)
                return
            except Exception as e:
                logger.error("Failed to start %s process: %s. Using fallback evaluator.", self.engine_name, e)
                self.process = None

        logger.info("Operating in Built-in Fallback Evaluator mode.")
        self.is_running = True

    def _send_command(self, cmd: str) -> None:
        """Sends a command to the UCI engine process."""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
            except Exception as e:
                logger.error("Failed to write to engine stdin: %s", e)

    def _reader_loop(self) -> None:
        """Reads engine stdout in background thread."""
        while not self._stop_event.is_set() and self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                self._parse_uci_line(line)
            except Exception as e:
                logger.error("Error reading engine stdout: %s", e)
                break

    def _parse_uci_line(self, line: str) -> None:
        """Parses info and bestmove UCI tokens, collecting multi-PV data."""
        if line == "readyok":
            self._ready_event.set()
            return

        if line.startswith("id name "):
            detected = line[8:].strip()
            if detected and self.engine_name in ("UCI Engine", "Stockfish"):
                self.engine_name = detected
            return
        elif line.startswith("id ") and not line.startswith("id author"):
            detected = line[3:].strip()
            if detected and self.engine_name in ("UCI Engine", "Stockfish"):
                self.engine_name = detected
            return

        if line.startswith("info ") and "score " in line:
            tokens = line.split()
            depth = 0
            nodes = 0
            nps = 0
            score_cp: Optional[int] = None
            mate_in: Optional[int] = None
            pv_tokens: List[str] = []
            multipv_index: int = 1  # default PV line 1

            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t == "depth" and i + 1 < len(tokens):
                    try:
                        depth = int(tokens[i + 1])
                    except ValueError:
                        pass
                    i += 1
                elif t == "nodes" and i + 1 < len(tokens):
                    try:
                        nodes = int(tokens[i + 1])
                    except ValueError:
                        pass
                    i += 1
                elif t == "nps" and i + 1 < len(tokens):
                    try:
                        nps = int(tokens[i + 1])
                    except ValueError:
                        pass
                    i += 1
                elif t == "multipv" and i + 1 < len(tokens):
                    try:
                        multipv_index = int(tokens[i + 1])
                    except ValueError:
                        pass
                    i += 1
                elif t == "score":
                    if i + 2 < len(tokens):
                        score_type = tokens[i + 1]
                        try:
                            score_val = int(tokens[i + 2])
                            # Adjust score to always be from White's perspective
                            temp_board = chess.Board(self.current_fen)
                            turn_multiplier = 1 if temp_board.turn == chess.WHITE else -1
                            if score_type == "cp":
                                score_cp = score_val * turn_multiplier
                            elif score_type == "mate":
                                mate_in = score_val * turn_multiplier
                        except ValueError:
                            pass
                        i += 2
                elif t == "pv":
                    pv_tokens = tokens[i + 1:]
                    break
                i += 1

            best_move: Optional[chess.Move] = None
            best_move_san: Optional[str] = None
            pv_san: List[str] = []
            if pv_tokens:
                temp_b = chess.Board(self.current_fen)
                try:
                    best_move = chess.Move.from_uci(pv_tokens[0])
                    if best_move in temp_b.legal_moves:
                        best_move_san = temp_b.san(best_move)
                    for uci_m in pv_tokens[:8]:
                        mv = chess.Move.from_uci(uci_m)
                        if mv in temp_b.legal_moves:
                            pv_san.append(temp_b.san(mv))
                            temp_b.push(mv)
                        else:
                            break
                except Exception:
                    pass

            # Store per-PV line data
            if hasattr(self, '_multipv_results'):
                self._multipv_results[multipv_index] = {
                    "move": best_move,
                    "score_cp": score_cp if score_cp is not None else (99999 if mate_in and mate_in > 0 else -99999 if mate_in else 0),
                    "pv_san": pv_san,
                    "depth": depth,
                    "nodes": nodes,
                    "nps": nps,
                    "mate_in": mate_in,
                }

            # Only emit full evaluation updates for PV line 1 (the best line)
            if multipv_index == 1:
                # Build top_moves from all collected PV lines
                top_moves: List[Tuple[chess.Move, int, List[str]]] = []
                if hasattr(self, '_multipv_results'):
                    for pv_idx in sorted(self._multipv_results.keys()):
                        pv_data = self._multipv_results[pv_idx]
                        if pv_data["move"] is not None:
                            top_moves.append((
                                pv_data["move"],
                                pv_data["score_cp"],
                                pv_data["pv_san"],
                            ))

                res = EvaluationResult(
                    score_cp=score_cp,
                    mate_in=mate_in,
                    depth=depth,
                    nodes=nodes,
                    nps=nps,
                    pv_uci=pv_tokens[:8],
                    pv_san=pv_san,
                    best_move=best_move,
                    best_move_san=best_move_san,
                    engine_name=self.engine_name,
                    top_moves=top_moves,
                )
                self.latest_result = res
                if self.on_evaluation_callback:
                    self.on_evaluation_callback(res)

        elif line.startswith("bestmove "):
            tokens = line.split()
            if len(tokens) >= 2 and tokens[1] != "(none)":
                try:
                    m = chess.Move.from_uci(tokens[1])
                    temp_b = chess.Board(self.current_fen)
                    if m in temp_b.legal_moves:
                        san = temp_b.san(m)
                        self.latest_result.best_move = m
                        self.latest_result.best_move_san = san
                        if not self.latest_result.top_moves:
                            self.latest_result.top_moves = [(m, self.latest_result.score_cp or 0, [san])]
                        self.latest_result.engine_name = self.engine_name
                        if self.on_evaluation_callback:
                            self.on_evaluation_callback(self.latest_result)
                except Exception:
                    pass
            # Reset multi-PV collection for next position
            if hasattr(self, '_multipv_results'):
                self._multipv_results.clear()

    def evaluate_position_async(self, fen: str, depth: int = ENGINE_ANALYSIS_DEPTH) -> None:
        """Sends new position to engine for asynchronous analysis."""
        self.current_fen = fen
        if self.process and self.is_running:
            self._send_command("stop")
            self._send_command(f"position fen {fen}")
            self._send_command(f"go depth {depth}")
        else:
            # Run fallback in background thread
            threading.Thread(target=self._run_fallback_analysis, args=(fen, min(3, depth)), daemon=True).start()

    def _run_fallback_analysis(self, fen: str, depth: int = 3) -> None:
        """Executes fallback evaluation in worker thread without GUI block."""
        board = chess.Board(fen)
        best_m, score = FallbackEngine.search(board, depth=depth)
        san = board.san(best_m) if best_m else None
        res = EvaluationResult(
            score_cp=score,
            depth=depth,
            nodes=1000,
            nps=50000,
            pv_uci=[best_m.uci()] if best_m else [],
            pv_san=[san] if san else [],
            best_move=best_m,
            best_move_san=san,
            engine_name="Internal Mini-Engine",
        )
        self.latest_result = res
        if self.on_evaluation_callback:
            self.on_evaluation_callback(res)

    def stop(self) -> None:
        """Stops the engine and cleans up process handles."""
        self._stop_event.set()
        if self.process:
            try:
                self._send_command("quit")
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                pass
            self.process = None
