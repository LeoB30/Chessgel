from __future__ import annotations
import logging
import threading
import time
from typing import Callable, Dict, Optional, Tuple
import chess
import cv2
import numpy as np
from PIL import Image

from core.sync import GameStateSynchronizer
from vision.capture import ScreenCapture, detect_chessboard_contour, slice_board_grid
from vision.classifier import BoardClassifier, ClassificationResult
from vision.templates import TemplateStore, load_or_extract_templates

logger: logging.Logger = logging.getLogger("MiniChess.VisionSync")


class BoardVisionSync:
    """Non-blocking background thread polling for real-time board state synchronization."""

    def __init__(
        self,
        get_board_callback: Callable[[], chess.Board],
        on_move_detected: Callable[[chess.Move], None],
        on_status_update: Callable[[str, bool, bool], None],
        on_fen_detected: Optional[
            Callable[[str, Dict[chess.Square, Optional[chess.Piece]], Dict[chess.Square, float]], None]
        ] = None,
        on_debug_frame: Optional[Callable[[Image.Image], None]] = None,
        poll_interval: float = 0.25,
        consecutive_frames_required: int = 2,
        outer_margin_pct: float = 0.0,
        inner_inset_pct: float = 0.12,
        template_store: Optional[TemplateStore] = None,
    ) -> None:
        self.get_board_callback: Callable[[], chess.Board] = get_board_callback
        self.on_move_detected: Callable[[chess.Move], None] = on_move_detected
        self.on_status_update: Callable[[str, bool, bool], None] = on_status_update
        self.on_fen_detected: Optional[
            Callable[[str, Dict[chess.Square, Optional[chess.Piece]], Dict[chess.Square, float]], None]
        ] = on_fen_detected
        self.on_debug_frame: Optional[Callable[[Image.Image], None]] = on_debug_frame

        self.poll_interval: float = poll_interval
        self.consecutive_frames_required: int = consecutive_frames_required
        self.outer_margin_pct: float = outer_margin_pct
        self.inner_inset_pct: float = inner_inset_pct

        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.is_flipped: bool = False
        self.is_running: bool = False
        self.show_debug_overlay: bool = True

        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._last_error_time: float = 0.0

        self._template_store: Optional[TemplateStore] = template_store
        self._classifier: Optional[BoardClassifier] = None
        self._capture: Optional[ScreenCapture] = None
        self._synchronizer: GameStateSynchronizer = GameStateSynchronizer(
            board=self.get_board_callback(),
            consecutive_frames_required=self.consecutive_frames_required,
        )

        # Mock frame provider for unit tests
        self.mock_image_provider: Optional[Callable[[], Optional[Image.Image]]] = None

    @property
    def classifier(self) -> BoardClassifier:
        if self._classifier is None:
            if self._template_store is None:
                self._template_store = load_or_extract_templates()
            self._classifier = BoardClassifier(template_store=self._template_store)
        return self._classifier

    def set_region(self, bbox: Tuple[int, int, int, int]) -> None:
        """Sets target screen bounding box (left, top, width, height)."""
        self.bbox = bbox

    def set_flipped(self, flipped: bool) -> None:
        """Sets board perspective (False = White at bottom, True = Black at bottom)."""
        self.is_flipped = flipped

    def set_outer_margin(self, pct: float) -> None:
        """Sets margin percentage to exclude coordinates."""
        self.outer_margin_pct = max(0.0, min(15.0, pct))

    def set_inner_inset(self, pct: float) -> None:
        """Sets cell inset percentage to eliminate boundary lines."""
        self.inner_inset_pct = max(0.02, min(0.25, pct))

    def auto_detect_board(self) -> Optional[Tuple[int, int, int, int]]:
        """Captures display, detects chessboard contour, and updates region if found."""
        capture = ScreenCapture()
        try:
            full_frame = capture.capture_full_screen()
            best_box = detect_chessboard_contour(full_frame)
            if best_box is not None:
                self.set_region(best_box)
                return best_box
            return None
        finally:
            capture.close()

    def start(self) -> bool:
        """Starts background vision capture loop."""
        if self.is_running:
            return True

        if self.bbox is None and self.mock_image_provider is None:
            self.on_status_update("No screen region selected", False, True)
            return False

        if self.bbox is not None:
            _, _, w, h = self.bbox
            if w < 64 or h < 64:
                self.on_status_update(f"Region too small ({w}x{h} px, min 64x64)", False, True)
                return False

        self.is_running = True
        self._stop_event.clear()
        self._synchronizer.reset(self.get_board_callback())

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="BoardVisionThread")
        self._thread.start()
        self.on_status_update("Vision Active (Polling Screen)", True, False)
        return True

    def stop(self) -> None:
        """Stops background vision capture loop cleanly."""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self.on_status_update("Vision Stopped", False, False)

    def _run_loop(self) -> None:
        """Main polling loop running on background thread at ~2-5 Hz."""
        self._capture = ScreenCapture()
        try:
            while not self._stop_event.is_set():
                t0: float = time.perf_counter()

                frame_bgr: Optional[np.ndarray] = self._grab_frame()
                if frame_bgr is not None:
                    self._process_frame(frame_bgr)

                elapsed: float = time.perf_counter() - t0
                sleep_sec: float = max(0.01, self.poll_interval - elapsed)
                self._stop_event.wait(timeout=sleep_sec)
        except Exception as e:
            logger.error("Unhandled exception in BoardVisionSync loop: %s", e, exc_info=True)
            self.on_status_update(f"Vision error: {type(e).__name__}", False, True)
        finally:
            if self._capture is not None:
                self._capture.close()
                self._capture = None

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Captures BGR numpy frame from screen or mock provider."""
        if self.mock_image_provider is not None:
            pil_img = self.mock_image_provider()
            if pil_img is None:
                return None
            rgb: np.ndarray = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.bbox is None or self._capture is None:
            return None

        try:
            return self._capture.capture_region(self.bbox)
        except Exception as e:
            now: float = time.time()
            if now - self._last_error_time > 3.0:
                self._last_error_time = now
                self.on_status_update(f"Screen grab failed: {type(e).__name__}", True, True)
            return None

    def _process_frame(self, frame_bgr: np.ndarray) -> None:
        """Slices grid, classifies pieces, validates legality, updates GUI, and infers legal moves."""
        grid_slices = slice_board_grid(
            frame_bgr,
            is_flipped=self.is_flipped,
            outer_margin_pct=self.outer_margin_pct,
            inner_inset_pct=self.inner_inset_pct,
        )

        result: ClassificationResult = self.classifier.classify_board(grid_slices)

        # Fail-safe defensive check: If captured position is structurally illegal, discard frame
        if not result.is_legal:
            logger.debug("Discarding illegal board frame: %s", result.validation_error)
            return

        # Dispatch live FEN and confidence map to GUI
        if self.on_fen_detected is not None:
            try:
                self.on_fen_detected(result.placement_fen, result.piece_grid, result.confidences)
            except TypeError:
                self.on_fen_detected(result.placement_fen, result.piece_grid)

        # Dispatch debug visualizer overlay if enabled
        if self.show_debug_overlay and self.on_debug_frame is not None:
            overlay_pil = self.generate_debug_overlay(frame_bgr, grid_slices, result)
            self.on_debug_frame(overlay_pil)

        # Sync internal board state
        current_board = self.get_board_callback()
        if self._synchronizer.board.fen() != current_board.fen():
            self._synchronizer.reset(current_board)

        confirmed_move, _ = self._synchronizer.process_detected_frame(result.piece_grid)
        if confirmed_move is not None:
            self.on_move_detected(confirmed_move)

    def infer_legal_move(
        self,
        board: chess.Board,
        detected_grid: Dict[chess.Square, Optional[chess.Piece]],
    ) -> Tuple[Optional[chess.Move], float]:
        """Delegates legal move inference to core GameStateSynchronizer."""
        sync = GameStateSynchronizer(board=board, min_move_agreement=0.93)
        return sync.infer_legal_move(detected_grid)

    def generate_debug_overlay(
        self,
        frame_bgr: np.ndarray,
        grid_slices: Dict[chess.Square, Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]],
        result: ClassificationResult,
    ) -> Image.Image:
        """Generates an annotated visual overlay image displaying the 8x8 grid and confidence scores."""
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
        return Image.fromarray(rgb)
