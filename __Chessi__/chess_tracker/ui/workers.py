"""Thread workers for asynchronous non-blocking vision perception and engine analysis."""

from __future__ import annotations
import logging
import time
from typing import Callable, Dict, Optional, Tuple
import chess
import cv2
import numpy as np

from PyQt6.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal
from PyQt6.QtGui import QImage

from chess_tracker.config import POLL_INTERVAL_MS
from chess_tracker.core.engine import EvaluationResult, UCIEngineManager
from chess_tracker.vision.capture import ScreenCapture, detect_chessboard_contour, crop_board_from_roi, refine_board_grid
from chess_tracker.vision.detector import ChessBoardDetector

logger = logging.getLogger("ChessTracker.Workers")


class VisionWorker(QThread):
    """Background screen vision thread running zero-DOM polling and move delta detection."""

    sig_move_detected = pyqtSignal(chess.Move)
    sig_status = pyqtSignal(str, bool)  # message, is_active
    sig_preview_updated = pyqtSignal(QImage, str)  # preview_image, status_text
    sig_auto_detected_box = pyqtSignal(int, int, int, int)
    sig_undo_move = pyqtSignal()  # Self-correction: request main window to undo last move
    sig_overlay_geometry = pyqtSignal(int, int, int, int)  # inner 8x8 grid in screen coords

    def __init__(
        self,
        get_board_func: Callable[[], chess.Board],
        poll_interval_ms: int = POLL_INTERVAL_MS,
    ) -> None:
        super().__init__()
        self.get_board_func = get_board_func
        self.poll_interval_ms = poll_interval_ms

        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.is_flipped: bool = False
        self.is_tracking: bool = False
        self._running: bool = True
        self._mutex = QMutex()

        self.capture = ScreenCapture()
        self.detector = ChessBoardDetector()

        # Flag to invalidate cached inner-board offset when ROI changes
        self._inner_offset_dirty: bool = True

        # Mock frame provider for offline simulation / testing
        self.mock_frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._last_overlay_geom: Optional[Tuple[int, int, int, int]] = None

    def set_region(self, x: int, y: int, width: int, height: int) -> None:
        """Sets the captured screen coordinates."""
        with QMutexLocker(self._mutex):
            self.bbox = (x, y, width, height)
            self.detector.reset_delta()
            self._inner_offset_dirty = True
        self.sig_status.emit(f"Locked on: {width}x{height} at ({x}, {y})", True)

    def set_flipped(self, flipped: bool) -> None:
        """Sets board perspective."""
        with QMutexLocker(self._mutex):
            self.is_flipped = flipped
            self.detector.reset_delta()
            self._inner_offset_dirty = True

    def set_tracking(self, enabled: bool) -> None:
        """Enables or pauses live screen polling."""
        with QMutexLocker(self._mutex):
            self.is_tracking = enabled
            self.detector.reset_delta()
            self._inner_offset_dirty = True
        status_text = "Tracking Active" if enabled else "Tracking Paused"
        self.sig_status.emit(status_text, enabled)

    def reset_vision(self) -> None:
        """Resets detector internal delta state."""
        with QMutexLocker(self._mutex):
            self.detector.reset_delta()
            self._inner_offset_dirty = True

    def trigger_auto_detect(self) -> None:
        """Captures full screen and seeks chessboard contour."""
        try:
            full_frame = self.capture.capture_full_screen()
            box = detect_chessboard_contour(full_frame)
            if box:
                bx, by, bw, bh = box
                self.set_region(bx, by, bw, bh)
                self.sig_auto_detected_box.emit(bx, by, bw, bh)
                self.sig_status.emit(f"Auto-detected board: {bw}x{bh} at ({bx}, {by})", True)
            else:
                self.sig_status.emit("Could not detect chessboard. Select area manually.", False)
        except Exception as e:
            logger.error("Auto detect failed: %s", e)
            self.sig_status.emit(f"Auto-detect error: {e}", False)

    def run(self) -> None:
        """Continuous polling loop in worker thread.

        Each iteration:
        1. Captures the outer ROI (may include chess.com chrome).
        2. Crops to the actual 8×8 tile grid via wood-HSV refinement.
        3. Feeds the cropped board into the delta detector.
        4. Checks occupancy mismatch for self-correction.
        """
        # Cache the inner board offset to avoid running HSV every frame.
        inner_offset: Optional[Tuple[int, int, int, int]] = None
        recalc_counter: int = 0
        RECALC_INTERVAL: int = 30  # re-check inner offset every ~4.5s at 6.7 FPS

        # Self-correction: track consecutive frames where visual occupancy
        # disagrees with internal board state.
        mismatch_streak: int = 0
        MISMATCH_SQUARE_THRESHOLD: int = 4   # ≥4 squares wrong
        MISMATCH_FRAME_THRESHOLD: int = 15   # sustained for ~2.2s at 6.7 FPS

        while self._running:
            try:
                frame_raw: Optional[np.ndarray] = None

                # Check if mock frame is provided (for tests)
                if self.mock_frame_provider:
                    frame_raw = self.mock_frame_provider()
                elif self.is_tracking and self.bbox:
                    x, y, w, h = self.bbox
                    frame_raw = self.capture.capture_region(x, y, w, h)

                if frame_raw is not None and frame_raw.size > 0:
                    # --- Pass 2: crop to actual board grid within the ROI ---
                    recalc_counter += 1
                    if inner_offset is None or self._inner_offset_dirty or recalc_counter >= RECALC_INTERVAL:
                        inner_offset = refine_board_grid(frame_raw)
                        recalc_counter = 0
                        self._inner_offset_dirty = False
                        logger.debug("Inner board offset: %s from ROI %s", inner_offset, frame_raw.shape[:2])

                    ix, iy, iw, ih = inner_offset
                    board_frame = frame_raw[iy : iy + ih, ix : ix + iw]
                    if self.bbox:
                        ox, oy, _, _ = self.bbox
                        geom = (ox + ix, oy + iy, iw, ih)
                        if geom != self._last_overlay_geom:
                            self._last_overlay_geom = geom
                            self.sig_overlay_geometry.emit(*geom)

                    if board_frame.size == 0:
                        board_frame = frame_raw  # safety fallback

                    current_board = self.get_board_func()
                    confirmed_move, features = self.detector.process_frame(
                        board_frame, current_board, is_flipped=self.is_flipped
                    )

                    if confirmed_move is not None:
                        self.sig_move_detected.emit(confirmed_move)
                        mismatch_streak = 0  # reset after a confirmed move
                    else:
                        # --- Self-correction: detect board divergence ---
                        mismatch = self.detector.compute_occupancy_mismatch(features, current_board)
                        if mismatch >= MISMATCH_SQUARE_THRESHOLD:
                            mismatch_streak += 1
                            if mismatch_streak >= MISMATCH_FRAME_THRESHOLD:
                                logger.warning(
                                    "Board divergence detected (%d squares mismatch for %d frames). "
                                    "Requesting undo.",
                                    mismatch, mismatch_streak,
                                )
                                self.sig_undo_move.emit()
                                mismatch_streak = 0
                        else:
                            mismatch_streak = 0

                    # Generate mini QImage preview for UI — show the CROPPED board
                    preview_thumb = cv2.resize(board_frame, (120, 120), interpolation=cv2.INTER_AREA)
                    preview_thumb_rgb = cv2.cvtColor(preview_thumb, cv2.COLOR_BGR2RGB)
                    ph, pw, ch = preview_thumb_rgb.shape
                    qimg = QImage(preview_thumb_rgb.data, pw, ph, ch * pw, QImage.Format.Format_RGB888).copy()

                    grid_size = f"{iw}x{ih}" if inner_offset else ""
                    status = f"6.7 FPS | {grid_size}" if self.bbox else "Online"
                    self.sig_preview_updated.emit(qimg, status)

            except Exception as e:
                logger.error("Vision worker loop exception: %s", e)

            self.msleep(self.poll_interval_ms)

    def stop(self) -> None:
        """Terminates thread safely."""
        self._running = False
        self.wait(1000)
        self.capture.close()


class EngineWorker(QThread):
    """Background engine thread managing per-side engine assignment without locking GUI.

    Supports dynamically registered UCI engines (Stockfish, Patricia, Berserk, OpenTal,
    custom user executables), MiniChess, or None (disabled).

    Dual-Engine / Trio-Engine Hybrid Mode:
    When enabled, runs up to three engines simultaneously. Results are emitted on separate signals
    so the board mirror can render multiple arrays of arrows with color-coded losing moves.
    """

    sig_evaluation_ready = pyqtSignal(object)  # EvaluationResult (primary engine)
    sig_secondary_eval_ready = pyqtSignal(object)  # EvaluationResult (secondary engine)
    sig_tertiary_eval_ready = pyqtSignal(object)   # EvaluationResult (tertiary engine)

    def __init__(self) -> None:
        super().__init__()
        # Per-side engine selection (defaults: both Stockfish)
        self._white_engine_name: str = "stockfish"
        self._black_engine_name: str = "stockfish"

        # Hybrid mode state
        self._hybrid_mode: bool = False
        self._hybrid_primary_id: str = "stockfish"
        self._hybrid_secondary_id: str = "patricia"
        self._hybrid_tertiary_id: str = "none"

        # Engine instances
        self._engines: dict[str, object] = {}
        self._secondary_engines: dict[str, object] = {}
        self._tertiary_engines: dict[str, object] = {}
        self._init_default_engines()

    def _init_default_engines(self) -> None:
        """Preloads default Stockfish engine instance for instant evaluation."""
        try:
            sf = UCIEngineManager(name="Stockfish 15.1")
            sf.on_evaluation_callback = self._on_eval_callback
            self._engines["stockfish"] = sf
        except Exception as e:
            logger.error("Failed to init Stockfish: %s", e)

    def _on_eval_callback(self, result: EvaluationResult) -> None:
        """Bridges primary engine thread callback to Qt signal."""
        self.sig_evaluation_ready.emit(result)

    def _on_secondary_eval_callback(self, result: EvaluationResult) -> None:
        """Bridges secondary engine thread callback to Qt signal."""
        self.sig_secondary_eval_ready.emit(result)

    def _on_tertiary_eval_callback(self, result: EvaluationResult) -> None:
        """Bridges tertiary engine thread callback to Qt signal."""
        self.sig_tertiary_eval_ready.emit(result)

    def _get_or_create_engine(self, engine_id: str, pool_type: int = 0) -> Optional[object]:
        """Retrieves existing engine instance or creates it from registry definition.

        Args:
            engine_id: The engine identifier.
            pool_type: 0 for primary, 1 for secondary, 2 for tertiary
        """
        if pool_type == 1:
            pool = self._secondary_engines
            callback = self._on_secondary_eval_callback
            role = "secondary"
        elif pool_type == 2:
            pool = self._tertiary_engines
            callback = self._on_tertiary_eval_callback
            role = "tertiary"
        else:
            pool = self._engines
            callback = self._on_eval_callback
            role = "primary"

        if engine_id in pool:
            return pool[engine_id]

        if engine_id == "none":
            return None

        from chess_tracker.core.engine_registry import get_all_available_engines
        avail = {e.id: e for e in get_all_available_engines()}
        engine_def = avail.get(engine_id)

        if engine_def is None:
            logger.warning("Engine ID '%s' not found in registry.", engine_id)
            return None

        if engine_def.engine_type == "uci" and engine_def.binary_path:
            try:
                engine = UCIEngineManager(
                    binary_path=engine_def.binary_path,
                    name=engine_def.name,
                    extra_options=engine_def.extra_options,
                )
                engine.on_evaluation_callback = callback
                pool[engine_id] = engine
                logger.info("Instantiated UCI engine '%s' (%s) as %s", engine_def.name, engine_id, role)
                return engine
            except Exception as e:
                logger.error("Failed to instantiate UCI engine '%s': %s", engine_def.name, e)
                return None

        elif engine_def.engine_type == "minichess":
            try:
                from chess_tracker.core.minichess_adapter import MiniChessEngine
                mc = MiniChessEngine(depth=5, movetime_ms=2000)
                mc.on_evaluation_callback = callback
                pool[engine_id] = mc
                logger.info("Instantiated MiniChess engine")
                return mc
            except Exception as e:
                logger.warning("MiniChess engine not available: %s", e)
                return None

        return None

    def set_engine_for_side(self, side: str, engine_name: str) -> None:
        """Assigns an engine to a side ('white' or 'black')."""
        engine_id = engine_name.lower().strip()
        # Pre-instantiate if not yet loaded
        if engine_id != "none":
            self._get_or_create_engine(engine_id)

        if side.lower() == "white":
            self._white_engine_name = engine_id
            logger.info("White engine set to: %s", engine_id)
        elif side.lower() == "black":
            self._black_engine_name = engine_id
            logger.info("Black engine set to: %s", engine_id)

    def get_engine_for_side(self, side: str) -> str:
        """Returns the engine name assigned to a side."""
        if side.lower() == "white":
            return self._white_engine_name
        return self._black_engine_name

    # --- Hybrid Mode API ---

    def set_hybrid_mode(self, enabled: bool) -> None:
        """Enables or disables multi-engine hybrid mode."""
        self._hybrid_mode = enabled
        logger.info("Hybrid mode %s", "enabled" if enabled else "disabled")

    def is_hybrid_mode(self) -> bool:
        """Returns True if hybrid mode is active."""
        return self._hybrid_mode

    def set_trio_engines(self, primary_id: str, secondary_id: str, tertiary_id: str) -> None:
        """Sets the engines for trio-engine mode."""
        self._hybrid_primary_id = primary_id.lower().strip()
        self._hybrid_secondary_id = secondary_id.lower().strip()
        self._hybrid_tertiary_id = tertiary_id.lower().strip()
        # Pre-instantiate
        if self._hybrid_primary_id != "none":
            self._get_or_create_engine(self._hybrid_primary_id, pool_type=0)
        if self._hybrid_secondary_id != "none":
            self._get_or_create_engine(self._hybrid_secondary_id, pool_type=1)
        if self._hybrid_tertiary_id != "none":
            self._get_or_create_engine(self._hybrid_tertiary_id, pool_type=2)

    def set_engine_multipv(self, engine_id: str, pool_type: int, lines: int) -> None:
        """Updates the MultiPV setting for a specific engine instance."""
        engine = self._get_or_create_engine(engine_id, pool_type=pool_type)
        if engine and hasattr(engine, "set_multipv"):
            engine.set_multipv(lines)

    def set_engine_aggressiveness(self, contempt_val: int) -> None:
        """Updates the aggressiveness/contempt for all active engines."""
        for pool in [self._engines, self._secondary_engines, self._tertiary_engines]:
            for eng in pool.values():
                if hasattr(eng, "set_aggressiveness"):
                    eng.set_aggressiveness(contempt_val)

    def request_analysis(self, fen: str, depth: int = 14) -> None:
        """Routes analysis request to the correct engine(s).

        In hybrid mode, all active engines in the trio analyze the position simultaneously.
        In normal mode, routes to the per-side engine.
        """
        if self._hybrid_mode:
            self._request_hybrid_analysis(fen, depth)
            return

        board = chess.Board(fen)
        engine_id = self._white_engine_name if board.turn == chess.WHITE else self._black_engine_name

        if engine_id == "none":
            res = EvaluationResult(engine_name="None (Disabled)")
            self.sig_evaluation_ready.emit(res)
            return

        engine = self._get_or_create_engine(engine_id)
        if engine is None:
            logger.warning("Engine '%s' could not be loaded, using default.", engine_id)
            engine = self._engines.get("stockfish")

        if engine is not None:
            engine.evaluate_position_async(fen, depth=depth)

    def _request_hybrid_analysis(self, fen: str, depth: int = 14) -> None:
        """Sends analysis requests to all hybrid engines simultaneously."""
        # Primary engine
        primary = self._get_or_create_engine(self._hybrid_primary_id, pool_type=0)
        if primary is None:
            primary = self._engines.get("stockfish")
        if primary is not None:
            primary.evaluate_position_async(fen, depth=depth)

        # Secondary engine
        secondary = self._get_or_create_engine(self._hybrid_secondary_id, pool_type=1)
        if secondary is not None:
            secondary.evaluate_position_async(fen, depth=max(8, depth - 2))

        # Tertiary engine
        tertiary = self._get_or_create_engine(self._hybrid_tertiary_id, pool_type=2)
        if tertiary is not None:
            tertiary.evaluate_position_async(fen, depth=max(8, depth - 2))

    def stop(self) -> None:
        """Stops all engine instances."""
        for engine in self._engines.values():
            try:
                engine.stop()
            except Exception:
                pass
        for engine in self._secondary_engines.values():
            try:
                engine.stop()
            except Exception:
                pass
        for engine in self._tertiary_engines.values():
            try:
                engine.stop()
            except Exception:
                pass
        self.wait(500)
