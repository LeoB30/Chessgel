from __future__ import annotations
import logging
from typing import Optional, Tuple
import chess

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QCloseEvent, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gui.board_view import BoardView
from gui.cv_overlay import CVOverlayWindow
from gui.eval_bar import EvalBar
from gui.screen_selector import ScreenRegionSelector
from gui.telemetry_view import TelemetryView
from gui.workers import EngineWorker, VisionWorker
from vision.capture import ScreenCapture, detect_chessboard_contour

logger: logging.Logger = logging.getLogger("MiniChess.App")


class ChessApp(QMainWindow):
    """Main Application Window integrating OpenCV Screen Vision, Alpha-Beta Engine, and PyQt6 GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MiniChess AI - Vision Tracking & Strategic Analysis")
        self.resize(1120, 720)
        self.setMinimumSize(900, 600)

        # Global Dark Mode Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #11111B;
            }
            QToolBar {
                background-color: #181825;
                border-bottom: 1px solid #313244;
                spacing: 8px;
                padding: 6px;
            }
            QToolButton {
                background-color: #313244;
                color: #CDD6F4;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #45475A;
                color: #89B4FA;
            }
            QToolButton:pressed {
                background-color: #585B70;
            }
            QStatusBar {
                background-color: #181825;
                color: #BAC2DE;
                border-top: 1px solid #313244;
                font-size: 11px;
            }
        """)

        # Core Game State
        self.board: chess.Board = chess.Board()
        self.is_flipped: bool = False
        self.captured_bbox: Optional[Tuple[int, int, int, int]] = None

        # Instantiate Sub-Widgets
        self.eval_bar: EvalBar = EvalBar()
        self.board_view: BoardView = BoardView()
        self.telemetry_view: TelemetryView = TelemetryView()
        self.cv_overlay: CVOverlayWindow = CVOverlayWindow()
        self.screen_selector: Optional[ScreenRegionSelector] = None

        # Build UI Layout
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()

        # Instantiate Thread Workers
        self.vision_worker: VisionWorker = VisionWorker(get_board_func=lambda: self.board)
        self.engine_worker: EngineWorker = EngineWorker()

        # Connect Signals & Slots
        self._connect_signals()

        # Trigger Initial Analysis for Starting Position
        self._request_engine_analysis()

    def _setup_ui(self) -> None:
        """Constructs responsive three-column dashboard layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # Column 1: Centipawn Evaluation Bar
        root_layout.addWidget(self.eval_bar)

        # Column 2: Interactive High-Contrast Chessboard
        board_container = QFrame()
        board_container.setStyleSheet("background-color: #181825; border-radius: 8px; border: 1px solid #313244;")
        board_layout = QVBoxLayout(board_container)
        board_layout.setContentsMargins(8, 8, 8, 8)
        board_layout.addWidget(self.board_view)
        root_layout.addWidget(board_container, stretch=3)

        # Column 3: Real-Time Telemetry Dashboard
        root_layout.addWidget(self.telemetry_view, stretch=2)

    def _setup_toolbar(self) -> None:
        """Constructs top toolbar actions."""
        toolbar: QToolBar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.act_select_region = QAction("🎯 Select Screen Region", self)
        self.act_select_region.triggered.connect(self._on_select_screen_region)
        toolbar.addAction(self.act_select_region)

        self.act_auto_detect = QAction("🔍 Auto Detect Board", self)
        self.act_auto_detect.triggered.connect(self._on_auto_detect_board)
        toolbar.addAction(self.act_auto_detect)

        toolbar.addSeparator()

        self.act_toggle_vision = QAction("▶ Start Vision Tracking", self)
        self.act_toggle_vision.triggered.connect(self._on_toggle_vision)
        toolbar.addAction(self.act_toggle_vision)

        toolbar.addSeparator()

        self.act_flip_board = QAction("🔄 Flip Perspective", self)
        self.act_flip_board.triggered.connect(self._on_flip_board)
        toolbar.addAction(self.act_flip_board)

        self.act_new_game = QAction("⏮ Reset Game", self)
        self.act_new_game.triggered.connect(self._on_reset_game)
        toolbar.addAction(self.act_new_game)

        toolbar.addSeparator()

        self.act_toggle_debug = QAction("👁 Toggle CV Debug Overlay", self)
        self.act_toggle_debug.triggered.connect(self._on_toggle_cv_debug)
        toolbar.addAction(self.act_toggle_debug)

    def _setup_statusbar(self) -> None:
        """Initializes bottom telemetry status bar."""
        self.status_bar: QStatusBar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready | Standalone mode active (Move pieces on board or start vision tracking)")

    def _connect_signals(self) -> None:
        """Wires thread-safe Qt Signals and Slots across Workers and GUI widgets."""
        # Board User Moves (Standalone Mode)
        self.board_view.sig_move_made.connect(self._on_user_board_move)

        # Vision Worker Signals
        self.vision_worker.sig_move_detected.connect(self._on_vision_move_detected)
        self.vision_worker.sig_status.connect(self._on_vision_status_update)
        self.vision_worker.sig_debug_image.connect(self.cv_overlay.update_debug_frame)

        # Engine Worker Signals
        self.engine_worker.sig_eval_ready.connect(self._on_engine_eval_ready)
        self.engine_worker.sig_engine_status.connect(self._on_engine_status_update)

    def _on_select_screen_region(self) -> None:
        """Opens fullscreen drag-to-select overlay for target board coordinates."""
        self.screen_selector = ScreenRegionSelector()
        self.screen_selector.sig_region_selected.connect(self._on_region_chosen)
        self.screen_selector.show()

    def _on_region_chosen(self, bbox: Tuple[int, int, int, int]) -> None:
        """Registers user-selected bounding box and initializes vision worker target."""
        self.captured_bbox = bbox
        self.vision_worker.set_region(bbox)
        x, y, w, h = bbox
        self.status_bar.showMessage(f"Selected Screen Region: ({x}, {y}) {w}x{h} px")

        if not self.vision_worker.isRunning():
            self._start_vision_worker()

    def _on_auto_detect_board(self) -> None:
        """Executes full screen contour localization to locate active chessboard."""
        capture = ScreenCapture()
        try:
            full_frame = capture.capture_full_screen()
            best_box = detect_chessboard_contour(full_frame)
            if best_box is not None:
                self._on_region_chosen(best_box)
                QMessageBox.information(
                    self,
                    "Board Detected",
                    f"Chessboard contour successfully localized at:\n{best_box[0]}, {best_box[1]} ({best_box[2]}x{best_box[3]} px)",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Auto Detect Failed",
                    "No square chessboard contour detected. Please use 'Select Screen Region' to manually drag target area.",
                )
        finally:
            capture.close()

    def _on_toggle_vision(self) -> None:
        """Starts or stops the background vision thread."""
        if self.vision_worker.isRunning():
            self._stop_vision_worker()
        else:
            if self.captured_bbox is None:
                QMessageBox.warning(
                    self,
                    "No Region Selected",
                    "Please click 'Select Screen Region' or 'Auto Detect Board' first.",
                )
                return
            self._start_vision_worker()

    def _start_vision_worker(self) -> None:
        """Starts background vision capture thread."""
        self.vision_worker.set_flipped(self.is_flipped)
        self.vision_worker.sync_board_state(self.board)
        self.vision_worker.start()
        self.act_toggle_vision.setText("⏹ Stop Vision Tracking")

    def _stop_vision_worker(self) -> None:
        """Stops background vision capture thread cleanly."""
        self.vision_worker.requestInterruption()
        self.vision_worker.wait(1000)
        self.act_toggle_vision.setText("▶ Start Vision Tracking")

    def _on_flip_board(self) -> None:
        """Flips board perspective between White and Black."""
        self.is_flipped = not self.is_flipped
        self.board_view.set_flipped(self.is_flipped)
        self.eval_bar.set_flipped(self.is_flipped)
        self.vision_worker.set_flipped(self.is_flipped)

    def _on_reset_game(self) -> None:
        """Resets game state to starting position."""
        self.board.reset()
        self.board_view.set_board(self.board)
        self.board_view.set_vision_move(None)
        self.board_view.set_engine_hint(None)
        self.eval_bar.set_evaluation(0)
        self.telemetry_view.update_move_history(self.board)
        self.vision_worker.sync_board_state(self.board)
        self._request_engine_analysis()
        self.status_bar.showMessage("Game reset to starting position.")

    def _on_toggle_cv_debug(self) -> None:
        """Toggles the CV debug overlay window visibility."""
        if self.cv_overlay.isVisible():
            self.cv_overlay.hide()
        else:
            self.cv_overlay.show()

    def _on_user_board_move(self, move: chess.Move) -> None:
        """Handles manual move made directly on board view (standalone mode)."""
        self.telemetry_view.update_move_history(self.board)
        self.vision_worker.sync_board_state(self.board)
        self._request_engine_analysis()

    def _on_vision_move_detected(self, move: chess.Move) -> None:
        """Handles legal move detected on live screen by CV pipeline."""
        if move in self.board.legal_moves:
            self.board.push(move)
            self.board_view.set_board(self.board, last_move=move)
            self.board_view.set_vision_move(move)
            self.telemetry_view.update_move_history(self.board)
            self.status_bar.showMessage(f"Vision synchronized live move: {move.uci()}")
            self._request_engine_analysis()

    def _on_vision_status_update(self, message: str, is_active: bool, is_error: bool) -> None:
        """Updates status bar with vision capture telemetry."""
        prefix = "🔴 " if is_error else ("🟢 " if is_active else "⚪ ")
        self.status_bar.showMessage(f"{prefix}{message}")

    def _on_engine_status_update(self, status: str) -> None:
        """Displays engine status update."""
        logger.debug("Engine status: %s", status)

    def _on_engine_eval_ready(
        self,
        score_cp: int,
        best_move: Optional[chess.Move],
        source: str,
        title: str,
        body: str,
        pv_san: str,
    ) -> None:
        """Receives engine evaluation and updates evaluation bar and telemetry."""
        self.eval_bar.set_evaluation(score_cp)
        self.board_view.set_engine_hint(best_move)
        self.telemetry_view.update_engine_eval(
            score_cp=score_cp,
            best_move=best_move,
            source=source,
            title=title,
            body=body,
            pv_san=pv_san,
        )

    def _request_engine_analysis(self) -> None:
        """Requests asynchronous evaluation from the EngineWorker."""
        self.engine_worker.evaluate_board(self.board, depth=4, movetime_ms=750)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Performs graceful shutdown of worker threads and releases OS resources."""
        self._stop_vision_worker()
        self.engine_worker.requestInterruption()
        self.engine_worker.wait(500)
        self.cv_overlay.close()
        event.accept()
