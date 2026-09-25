from __future__ import annotations
import logging
import time
from typing import Dict, Optional, Tuple
import chess
import cv2
import numpy as np
from PIL import Image

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from core.sync import GameStateSynchronizer
from engine.opening_book import (
    MoveExplanation,
    get_book_move,
    get_candidate_explanation,
    get_move_explanation_for_position,
    identify_opening,
)
from engine.search import find_best_move
from vision.capture import ScreenCapture, slice_board_grid
from vision.classifier import BoardClassifier, ClassificationResult
from vision.templates import TemplateStore, load_or_extract_templates

logger: logging.Logger = logging.getLogger("MiniChess.GUIWorkers")


class VisionWorker(QThread):
    """Dedicated background thread for low-latency screen capture and CV board classification."""

    sig_status = pyqtSignal(str, bool, bool)  # message, is_active, is_error
    sig_move_detected = pyqtSignal(object)  # chess.Move
    sig_fen_detected = pyqtSignal(str, dict, dict)  # fen, piece_grid, confidences
    sig_debug_image = pyqtSignal(QImage)

    def __init__(
        self,
        get_board_func: Optional[callable] = None,
        poll_interval: float = 0.20,
        consecutive_frames_required: int = 2,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.get_board_func = get_board_func
        self.poll_interval: float = poll_interval
        self.consecutive_frames_required: int = consecutive_frames_required

        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.is_flipped: bool = False
        self.outer_margin_pct: float = 0.0
        self.inner_inset_pct: float = 0.12
        self.show_debug: bool = True

        self._template_store: Optional[TemplateStore] = None
        self._classifier: Optional[BoardClassifier] = None
        self._synchronizer: GameStateSynchronizer = GameStateSynchronizer(
            consecutive_frames_required=self.consecutive_frames_required
        )
        self._last_error_time: float = 0.0

    def set_region(self, bbox: Tuple[int, int, int, int]) -> None:
        """Sets the screen bounding box coordinates."""
        self.bbox = bbox

    def set_flipped(self, flipped: bool) -> None:
        """Sets board perspective."""
        self.is_flipped = flipped

    def sync_board_state(self, board: chess.Board) -> None:
        """Syncs the internal synchronizer board with the GUI board."""
        self._synchronizer.reset(board)

    def run(self) -> None:
        """Worker loop executing continuous screen capture and recognition."""
        if self._template_store is None:
            try:
                self._template_store = load_or_extract_templates()
            except Exception as e:
                logger.error("Failed to initialize vision templates: %s", e)
                self.sig_status.emit(f"Template load error: {e}", False, True)
                return

        self._classifier = BoardClassifier(template_store=self._template_store)
        capture: ScreenCapture = ScreenCapture()

        self.sig_status.emit("Vision Active (Observing Board)", True, False)

        try:
            while not self.isInterruptionRequested():
                t0: float = time.perf_counter()

                if self.bbox is not None:
                    frame_bgr: Optional[np.ndarray] = None
                    try:
                        frame_bgr = capture.capture_region(self.bbox)
                    except Exception as ex:
                        now: float = time.time()
                        if now - self._last_error_time > 3.0:
                            self._last_error_time = now
                            self.sig_status.emit(f"Capture error: {ex}", True, True)

                    if frame_bgr is not None:
                        self._process_frame(frame_bgr)

                elapsed: float = time.perf_counter() - t0
                sleep_sec: float = max(0.01, self.poll_interval - elapsed)
                self.msleep(int(sleep_sec * 1000))

        except Exception as e:
            logger.error("VisionWorker loop exception: %s", e, exc_info=True)
            self.sig_status.emit(f"Vision exception: {e}", False, True)
        finally:
            capture.close()
            self.sig_status.emit("Vision Stopped", False, False)

    def _process_frame(self, frame_bgr: np.ndarray) -> None:
        """Slices board, classifies squares, emits debug telemetry, and checks move transitions."""
        assert self._classifier is not None

        grid_slices = slice_board_grid(
            frame_bgr,
            is_flipped=self.is_flipped,
            outer_margin_pct=self.outer_margin_pct,
            inner_inset_pct=self.inner_inset_pct,
        )

        result: ClassificationResult = self._classifier.classify_board(grid_slices)

        # Discard invalid or garbled frames
        if not result.is_legal:
            return

        # Emit detected FEN and piece grid
        self.sig_fen_detected.emit(result.placement_fen, result.piece_grid, result.confidences)

        # Emit debug visualization image if enabled
        if self.show_debug:
            qimg: QImage = self._create_debug_qimage(frame_bgr, grid_slices, result)
            self.sig_debug_image.emit(qimg)

        # Synchronize board from GUI callback if provided
        if self.get_board_func is not None:
            current_gui_board = self.get_board_func()
            if self._synchronizer.board.fen() != current_gui_board.fen():
                self._synchronizer.reset(current_gui_board)

        confirmed_move, _ = self._synchronizer.process_detected_frame(result.piece_grid)
        if confirmed_move is not None:
            self.sig_move_detected.emit(confirmed_move)

    def _create_debug_qimage(
        self,
        frame_bgr: np.ndarray,
        grid_slices: Dict[chess.Square, Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]],
        result: ClassificationResult,
    ) -> QImage:
        """Builds annotated overlay QImage for the CV debug dashboard."""
        overlay: np.ndarray = frame_bgr.copy()
        bh, bw = overlay.shape[:2]

        for sq, ((x1, y1, x2, y2), _, _) in grid_slices.items():
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (76, 175, 80), 1)

            pad_x: int = int(round((x2 - x1) * self.inner_inset_pct))
            pad_y: int = int(round((y2 - y1) * self.inner_inset_pct))
            cv2.rectangle(overlay, (x1 + pad_x, y1 + pad_y), (x2 - pad_x, y2 - pad_y), (255, 235, 59), 1)

            sq_name: str = chess.square_name(sq)
            piece: Optional[chess.Piece] = result.piece_grid.get(sq)
            p_sym: str = piece.symbol() if piece else "."
            conf: float = result.confidences.get(sq, 0.0)

            label_text: str = f"{sq_name}:{p_sym}"
            conf_text: str = f"{int(conf * 100)}%"

            cell_w: int = x2 - x1
            font_scale: float = max(0.30, min(0.65, cell_w / 140.0))

            cv2.putText(
                overlay,
                label_text,
                (x1 + 4, y1 + int(cell_w * 0.3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                conf_text,
                (x1 + 4, y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.85,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        fen_preview: str = f"FEN: {result.placement_fen[:32]}... Legal={result.is_legal}"
        cv2.rectangle(overlay, (0, 0), (bw, 22), (30, 30, 30), -1)
        cv2.putText(
            overlay,
            fen_preview,
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (100, 255, 100) if result.is_legal else (100, 100, 255),
            1,
            cv2.LINE_AA,
        )

        rgb: np.ndarray = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line: int = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class EngineWorker(QThread):
    """Dedicated background thread for chess engine search and opening repertoire evaluation."""

    # score_cp, best_move, source, commentary_title, commentary_body, pv_san
    sig_eval_ready = pyqtSignal(int, object, str, str, str, str)
    sig_engine_status = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._current_board: Optional[chess.Board] = None
        self._pending_board: Optional[chess.Board] = None
        self._search_depth: int = 4
        self._movetime_ms: int = 800

    def evaluate_board(self, board: chess.Board, depth: int = 4, movetime_ms: int = 800) -> None:
        """Submits a board state for asynchronous engine analysis."""
        self._pending_board = board.copy()
        self._search_depth = depth
        self._movetime_ms = movetime_ms

        if not self.isRunning():
            self.start()

    def run(self) -> None:
        """Worker loop processing board evaluation requests."""
        while not self.isInterruptionRequested():
            if self._pending_board is None:
                self.msleep(50)
                continue

            board_to_eval: chess.Board = self._pending_board.copy()
            self._pending_board = None
            self._current_board = board_to_eval

            self._analyze_position(board_to_eval)

    def _analyze_position(self, board: chess.Board) -> None:
        """Executes opening repertoire lookup with instant handoff to Alpha-Beta search."""
        if board.is_game_over():
            score: int = 0
            if board.is_checkmate():
                score = -99999 if board.turn == chess.WHITE else 99999
            self.sig_eval_ready.emit(score, None, "terminal", "Game Over", "Position is terminal.", "")
            return

        opening_name: str = identify_opening(board)

        # 1. Opening Book Prioritization
        book_move: Optional[chess.Move] = get_book_move(
            board, repertoire="catalan_carokann", selection_mode="best"
        )

        if book_move is not None:
            self.sig_engine_status.emit(f"Book: {opening_name}")
            explanation: MoveExplanation = get_candidate_explanation(board, book_move)
            title: str = f"{opening_name} ({explanation.variation_name})"
            body: str = (
                f"Strategic Intent: {explanation.strategic_intent}\n\n"
                f"Key Ideas: {explanation.key_ideas}\n\n"
                f"Positional Goals: {explanation.positional_goals}"
            )
            # Standard opening eval is neutral (~0.15 for White)
            base_eval: int = 20 if board.turn == chess.WHITE else -20
            self.sig_eval_ready.emit(base_eval, book_move, "book", title, body, book_move.uci())
            return

        # 2. Alpha-Beta Minimax Search (Smooth Engine Handoff)
        self.sig_engine_status.emit(f"Thinking: Depth {self._search_depth}...")

        best_move_found: Optional[chess.Move] = None
        eval_score: int = 0

        def info_cb(curr_d: int, curr_score: int, nodes: int, move: chess.Move, src: str) -> None:
            nonlocal eval_score, best_move_found
            eval_score = curr_score
            best_move_found = move

        best_move: Optional[chess.Move] = find_best_move(
            board,
            depth=self._search_depth,
            movetime_ms=self._movetime_ms,
            use_book=False,
            info_callback=info_cb,
        )

        if self.isInterruptionRequested() or self._pending_board is not None:
            return

        chosen_move = best_move or best_move_found
        title = f"Engine Analysis: {opening_name}"
        # Convert eval to White's absolute perspective for evaluation bar
        abs_eval: int = eval_score if board.turn == chess.WHITE else -eval_score

        # Move explanation or tactical annotation
        pos_exp: Optional[MoveExplanation] = get_move_explanation_for_position(board)
        if pos_exp is not None:
            body = (
                f"Out of primary book lines. Active structure derived from {pos_exp.variation_name}.\n\n"
                f"Positional context: {pos_exp.key_ideas}"
            )
        else:
            body = (
                f"Engine recommends {board.san(chosen_move) if chosen_move else 'None'}.\n"
                f"Computed via Alpha-Beta minimax with Quiescence search and MVV-LVA move ordering."
            )

        pv_str: str = board.san(chosen_move) if chosen_move else ""
        self.sig_engine_status.emit("Analysis Complete")
        self.sig_eval_ready.emit(abs_eval, chosen_move, "search", title, body, pv_str)
