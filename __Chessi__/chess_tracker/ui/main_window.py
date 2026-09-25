"""Main application window strictly budgeted to 995 x 960 pixels."""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import chess

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from chess_tracker.config import (
    BOARD_DISPLAY_SIZE,
    SIDEBAR_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from chess_tracker.core.engine import EvaluationResult
from chess_tracker.core.engine_registry import (
    EngineDefinition,
    get_all_available_engines,
    probe_uci_engine,
    save_custom_engine,
)
from chess_tracker.core.game_state import GameState
from chess_tracker.ui.board_mirror import BoardMirrorWidget
from chess_tracker.ui.board_overlay import BoardOverlayWindow
from chess_tracker.ui.eval_bar import EvalBarWidget
from chess_tracker.ui.screen_selector import ScreenRegionSelector
from chess_tracker.ui.workers import EngineWorker, VisionWorker

logger = logging.getLogger("ChessTracker.MainWindow")


class MainWindow(QMainWindow):
    """Zero-DOM passive chess tracking and engine synchronization desktop application."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Antigravity Chess Tracker & Engine Mirror")
        # Strict 995 x 960 Window Budget
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMaximumSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Core Game State
        self.game_state = GameState()

        # Build UI Elements
        self._apply_stylesheet()
        self._init_subwidgets()
        self._build_layout()

        # Screen Region Selector Overlay
        self.screen_selector = ScreenRegionSelector()
        self.screen_selector.sig_region_selected.connect(self._on_region_selected)

        # Transparent click-through arrows on the live board
        self.board_overlay = BoardOverlayWindow()
        self.chk_screen_overlay = QCheckBox("Screen overlay")
        self.chk_screen_overlay.setChecked(True)
        self.chk_screen_overlay.setToolTip("Draw analysis arrows on top of the captured chessboard.")
        self.chk_screen_overlay.setStyleSheet(
            "QCheckBox { color: #89B4FA; font-weight: bold; font-size: 11px; spacing: 4px; }"
        )

        # Worker Threads
        self.vision_worker = VisionWorker(get_board_func=lambda: self.game_state.board)
        self.engine_worker = EngineWorker()

        # Connect Signals
        self._connect_signals()

        # Start Workers
        self.vision_worker.start()
        self.engine_worker.start()

        # Initial Engine Evaluation
        self._request_analysis()

    def _apply_stylesheet(self) -> None:
        """Sets cohesive dark modern theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #11111B;
            }
            QWidget {
                color: #CDD6F4;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QFrame.card {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel.title {
                font-size: 13px;
                font-weight: bold;
                color: #89B4FA;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            QLabel.badge {
                background-color: #313244;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
                color: #A6ADC8;
            }
            QPushButton {
                background-color: #313244;
                color: #CDD6F4;
                border: 1px solid #45475A;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45475A;
                color: #FFFFFF;
                border-color: #89B4FA;
            }
            QPushButton:pressed {
                background-color: #585B70;
            }
            QPushButton.primary {
                background-color: #1E66F5;
                color: #FFFFFF;
                border: 1px solid #7287FD;
            }
            QPushButton.primary:hover {
                background-color: #2A72FF;
            }
            QPushButton.success {
                background-color: #2E7D32;
                color: #FFFFFF;
                border: 1px solid #4CAF50;
            }
            QPushButton.success:hover {
                background-color: #388E3C;
            }
            QPushButton.danger {
                background-color: #C62828;
                color: #FFFFFF;
                border: 1px solid #EF5350;
            }
            QPushButton.danger:hover {
                background-color: #D32F2F;
            }
            QTableWidget {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
                gridline-color: #313244;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #1E1E2E;
                color: #89B4FA;
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #313244;
                padding: 4px;
            }
            QLineEdit {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 5px;
                color: #BAC2DE;
                padding: 5px 8px;
                font-size: 11px;
                font-family: 'Consolas', monospace;
            }
        """)

    def _init_subwidgets(self) -> None:
        """Instantiates all board, readout, and control sub-widgets."""
        # Board & Eval Bar
        self.eval_bar = EvalBarWidget()
        self.eval_bar.setFixedHeight(BOARD_DISPLAY_SIZE)

        self.board_mirror = BoardMirrorWidget()
        self.board_mirror.setFixedSize(BOARD_DISPLAY_SIZE, BOARD_DISPLAY_SIZE)

        # Player Badges
        self.lbl_black_player = QLabel("Black (External)")
        self.lbl_black_player.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.lbl_black_material = QLabel("+0")
        self.lbl_black_material.setStyleSheet("color: #BAC2DE; font-size: 11px;")

        self.lbl_white_player = QLabel("White (External)")
        self.lbl_white_player.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.lbl_white_material = QLabel("+0")
        self.lbl_white_material.setStyleSheet("color: #BAC2DE; font-size: 11px;")

        # Board Controls
        self.btn_select_roi = QPushButton("Select Screen Area")
        self.btn_auto_detect = QPushButton("Auto-Detect Board")
        self.btn_toggle_tracking = QPushButton("Start Tracking")
        self.btn_toggle_tracking.setCheckable(True)
        self.btn_flip_board = QPushButton("Flip Board")
        self.btn_reset_game = QPushButton("Reset Position")

        # Undo & Sync Board Buttons
        self.btn_undo_move = QPushButton("⟲ Undo Move")
        self.btn_undo_move.setToolTip("Undo the last detected move")
        self.btn_undo_move.setStyleSheet(
            "QPushButton { background-color: #F59E0B; color: #11111B; font-weight: bold; border-radius: 6px; padding: 6px 10px; } "
            "QPushButton:hover { background-color: #FBBF24; }"
        )

        self.btn_sync_board = QPushButton("📷 Sync Board")
        self.btn_sync_board.setToolTip("Force-capture and re-read the external board state")
        self.btn_sync_board.setStyleSheet(
            "QPushButton { background-color: #6366F1; color: #FFFFFF; font-weight: bold; border-radius: 6px; padding: 6px 10px; } "
            "QPushButton:hover { background-color: #818CF8; }"
        )

        # Hybrid Trio-Engine Mode Toggle
        self.chk_hybrid_mode = QCheckBox("🔀 Trio-Engine Mode")
        self.chk_hybrid_mode.setToolTip(
            "Run up to three engines simultaneously with individual arrow control.\n"
            "Losing moves shown in red."
        )
        self.chk_hybrid_mode.setStyleSheet(
            "QCheckBox { color: #C084FC; font-weight: bold; font-size: 11px; spacing: 4px; } "
            "QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #6C7086; background-color: #181825; } "
            "QCheckBox::indicator:checked { background-color: #C084FC; border-color: #C084FC; }"
        )

        # Engine Readout Elements
        self.lbl_eval_score = QLabel("+0.00")
        self.lbl_eval_score.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.lbl_eval_score.setStyleSheet("color: #A6E3A1;")

        self.lbl_best_move = QLabel("Best: --")
        self.lbl_best_move.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.lbl_best_move.setStyleSheet("color: #00D2D3;")

        self.lbl_engine_depth = QLabel("Depth: 0")
        self.lbl_engine_nps = QLabel("NPS: 0")
        self.lbl_engine_pv = QLabel("PV: --")
        self.lbl_engine_pv.setStyleSheet("color: #94E2D5; font-size: 11px;")
        self.lbl_engine_pv.setWordWrap(True)

        # Per-Side Engine Assignment Dropdowns
        combo_style = (
            "QComboBox { background-color: #181825; color: #CDD6F4; border: 1px solid #45475A; "
            "border-radius: 4px; padding: 3px 6px; font-size: 11px; } "
            "QComboBox QAbstractItemView { background-color: #1E1E2E; color: #CDD6F4; selection-background-color: #45475A; }"
        )
        self.cmb_white_engine = QComboBox()
        self.cmb_white_engine.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cmb_white_engine.setStyleSheet(combo_style)

        self.cmb_black_engine = QComboBox()
        self.cmb_black_engine.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cmb_black_engine.setStyleSheet(combo_style)
        
        self.cmb_tertiary_engine = QComboBox()
        self.cmb_tertiary_engine.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cmb_tertiary_engine.setStyleSheet(combo_style)

        # Engine Spinboxes for Lines (MultiPV)
        spin_style = "QSpinBox { background-color: #181825; color: #CDD6F4; border: 1px solid #45475A; border-radius: 3px; font-size: 10px; }"
        self.spin_lines_1 = QSpinBox()
        self.spin_lines_1.setRange(1, 10)
        self.spin_lines_1.setValue(3)
        self.spin_lines_1.setStyleSheet(spin_style)
        
        self.spin_lines_2 = QSpinBox()
        self.spin_lines_2.setRange(1, 10)
        self.spin_lines_2.setValue(2)
        self.spin_lines_2.setStyleSheet(spin_style)
        
        self.spin_lines_3 = QSpinBox()
        self.spin_lines_1.setToolTip("How many arrows Engine 1 draws")
        self.spin_lines_2.setToolTip("How many arrows Engine 2 draws")
        self.spin_lines_3.setToolTip("How many arrows Engine 3 draws")
        
        # Engine Aggressiveness / Contempt
        self.spin_aggressiveness = QSpinBox()
        self.spin_aggressiveness.setRange(-100, 100)
        self.spin_aggressiveness.setValue(0)
        self.spin_aggressiveness.setStyleSheet(spin_style)
        self.spin_aggressiveness.setToolTip("Aggressiveness/Contempt score. >0 avoids draws and prefers complex lines.")

        self.btn_add_engine = QPushButton("+ Add Custom Engine...")
        self.btn_add_engine.setToolTip("Browse and load a custom UCI chess engine (.exe)")
        self.btn_add_engine.setStyleSheet(
            "QPushButton { background-color: #1E1E2E; border: 1px dashed #6C7086; color: #BAC2DE; "
            "font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600; } "
            "QPushButton:hover { background-color: #313244; border-color: #89B4FA; color: #CDD6F4; }"
        )

        self._populate_engine_combos()

        # Vision Calibration Elements
        self.lbl_vision_status = QLabel("Vision: Idle")
        self.lbl_vision_status.setStyleSheet("color: #F9E2AF; font-weight: bold;")

        self.lbl_roi_coords = QLabel("ROI: Not calibrated")
        self.lbl_roi_coords.setStyleSheet("color: #A6ADC8; font-size: 11px;")

        # Vision Interval
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.1, 5.0)
        self.spin_interval.setSingleStep(0.1)
        self.spin_interval.setValue(0.15)
        self.spin_interval.setStyleSheet("QDoubleSpinBox { background-color: #181825; color: #CDD6F4; border: 1px solid #45475A; border-radius: 3px; font-size: 10px; }")
        self.spin_interval.setToolTip("Vision Update Interval (Seconds)")

        self.lbl_preview_thumb = QLabel()
        self.lbl_preview_thumb.setFixedSize(120, 120)
        self.lbl_preview_thumb.setStyleSheet("background-color: #11111B; border: 1px dashed #45475A; border-radius: 4px;")
        self.lbl_preview_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_thumb.setText("No Signal")

        # Move History Table
        self.move_table = QTableWidget(0, 3)
        self.move_table.setHorizontalHeaderLabels(["#", "White", "Black"])
        self.move_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.move_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.move_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.move_table.verticalHeader().setVisible(False)
        self.move_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.move_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # Notation Tools
        self.txt_fen = QLineEdit()
        self.txt_fen.setReadOnly(True)
        self.txt_fen.setText(self.game_state.get_fen())
        self.btn_copy_fen = QPushButton("Copy FEN")
        self.btn_copy_pgn = QPushButton("Copy PGN")

        self.lbl_game_status = QLabel("Game in progress (White to move)")
        self.lbl_game_status.setStyleSheet("color: #FAB387; font-weight: bold; font-size: 12px;")

    def _build_layout(self) -> None:
        """Constructs strictly bounded layout inside 995 x 960 window."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(14)

        # ---------------------------------------------------------------------
        # LEFT COLUMN: ~600px Width (Board, Player Badges, Actions)
        # ---------------------------------------------------------------------
        left_container = QWidget()
        left_container.setFixedWidth(590)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Black Player Row
        black_row = QHBoxLayout()
        black_icon = QLabel("⚫")
        black_icon.setFont(QFont("Segoe UI", 12))
        black_row.addWidget(black_icon)
        black_row.addWidget(self.lbl_black_player)
        black_row.addStretch()
        black_row.addWidget(self.lbl_black_material)
        left_layout.addLayout(black_row)

        # Board + Eval Bar Row
        board_row = QHBoxLayout()
        board_row.setSpacing(8)
        board_row.addWidget(self.eval_bar)
        board_row.addWidget(self.board_mirror)
        left_layout.addLayout(board_row)

        # White Player Row
        white_row = QHBoxLayout()
        white_icon = QLabel("⚪")
        white_icon.setFont(QFont("Segoe UI", 12))
        white_row.addWidget(white_icon)
        white_row.addWidget(self.lbl_white_player)
        white_row.addStretch()
        white_row.addWidget(self.lbl_white_material)
        left_layout.addLayout(white_row)

        # Board Control Action Grid
        action_card = QFrame()
        action_card.setObjectName("action_card")
        action_card.setStyleSheet("background-color: #181825; border: 1px solid #313244; border-radius: 8px; padding: 6px;")
        action_layout = QGridLayout(action_card)
        action_layout.setContentsMargins(6, 6, 6, 6)
        action_layout.setSpacing(6)

        self.btn_select_roi.setProperty("class", "primary")
        self.btn_select_roi.setStyleSheet("background-color: #1E66F5; color: #FFFFFF; font-weight: bold;")
        action_layout.addWidget(self.btn_select_roi, 0, 0)
        action_layout.addWidget(self.btn_auto_detect, 0, 1)

        self.btn_toggle_tracking.setStyleSheet("background-color: #A6E3A1; color: #11111B; font-weight: bold;")
        action_layout.addWidget(self.btn_toggle_tracking, 0, 2)

        action_layout.addWidget(self.btn_flip_board, 1, 0)
        action_layout.addWidget(self.btn_reset_game, 1, 1)
        action_layout.addWidget(self.lbl_game_status, 1, 2)

        # Row 3: Undo, Sync, Hybrid
        action_layout.addWidget(self.btn_undo_move, 2, 0)
        action_layout.addWidget(self.btn_sync_board, 2, 1)
        action_layout.addWidget(self.chk_hybrid_mode, 2, 2)

        left_layout.addWidget(action_card)
        left_layout.addStretch()

        root_layout.addWidget(left_container)

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: ~367px Width (Engine Readout, Vision CV, Move Table)
        # ---------------------------------------------------------------------
        right_container = QWidget()
        right_container.setFixedWidth(SIDEBAR_WIDTH)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 1. Engine Analysis Card
        engine_card = QFrame()
        engine_card.setProperty("class", "card")
        engine_layout = QVBoxLayout(engine_card)
        engine_layout.setContentsMargins(10, 10, 10, 10)
        engine_layout.setSpacing(6)

        engine_header = QHBoxLayout()
        lbl_eng_title = QLabel("Tactical Engine Analysis")
        lbl_eng_title.setProperty("class", "title")
        engine_header.addWidget(lbl_eng_title)
        engine_header.addStretch()
        self.lbl_eng_name = QLabel("Stockfish 15.1")
        self.lbl_eng_name.setProperty("class", "badge")
        engine_header.addWidget(self.lbl_eng_name)
        engine_layout.addLayout(engine_header)

        eval_row = QHBoxLayout()
        eval_row.addWidget(self.lbl_eval_score)
        eval_row.addSpacing(16)
        eval_row.addWidget(self.lbl_best_move)
        eval_row.addStretch()
        engine_layout.addLayout(eval_row)

        metrics_row = QHBoxLayout()
        metrics_row.addWidget(self.lbl_engine_depth)
        metrics_row.addSpacing(12)
        metrics_row.addWidget(self.lbl_engine_nps)
        metrics_row.addStretch()
        engine_layout.addLayout(metrics_row)

        # Engine Assignment Rows (Trio)
        engine_assign_layout = QVBoxLayout()
        engine_assign_layout.setSpacing(4)
        
        row1 = QHBoxLayout()
        self.lbl_engine_sel_1 = QLabel("⚪")
        self.lbl_engine_sel_1.setFont(QFont("Segoe UI", 10))
        row1.addWidget(self.lbl_engine_sel_1)
        row1.addWidget(self.cmb_white_engine, stretch=1)
        row1.addWidget(QLabel("Arrows:"))
        row1.addWidget(self.spin_lines_1)
        engine_assign_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        self.lbl_engine_sel_2 = QLabel("⚫")
        self.lbl_engine_sel_2.setFont(QFont("Segoe UI", 10))
        row2.addWidget(self.lbl_engine_sel_2)
        row2.addWidget(self.cmb_black_engine, stretch=1)
        row2.addWidget(QLabel("Lns:"))
        row2.addWidget(self.spin_lines_2)
        engine_assign_layout.addLayout(row2)
        
        self.row3 = QHBoxLayout()
        self.lbl_engine_sel_3 = QLabel("Ter:")
        self.lbl_engine_sel_3.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.row3.addWidget(self.lbl_engine_sel_3)
        self.row3.addWidget(self.cmb_tertiary_engine, stretch=1)
        self.row3.addWidget(QLabel("Lns:"))
        self.row3.addWidget(self.spin_lines_3)
        engine_assign_layout.addLayout(self.row3)
        
        # Hide the 3rd row initially (shown when Trio mode is enabled)
        for i in range(self.row3.count()):
            widget = self.row3.itemAt(i).widget()
            if widget:
                widget.setVisible(False)
                
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Aggressiveness:"))
        settings_row.addWidget(self.spin_aggressiveness)
        settings_row.addStretch()
        engine_assign_layout.addLayout(settings_row)
        
        engine_layout.addLayout(engine_assign_layout)

        engine_btn_row = QHBoxLayout()
        engine_btn_row.addWidget(self.btn_add_engine)
        engine_btn_row.addStretch()
        engine_layout.addLayout(engine_btn_row)

        engine_layout.addWidget(self.lbl_engine_pv)
        right_layout.addWidget(engine_card)

        # 2. Vision Perception & Calibration Card
        vision_card = QFrame()
        vision_card.setProperty("class", "card")
        vision_layout = QVBoxLayout(vision_card)
        vision_layout.setContentsMargins(10, 10, 10, 10)
        vision_layout.setSpacing(6)

        vision_header = QHBoxLayout()
        lbl_vis_title = QLabel("Zero-DOM Screen Perception")
        lbl_vis_title.setProperty("class", "title")
        vision_header.addWidget(lbl_vis_title)
        vision_header.addStretch()
        vision_header.addWidget(self.lbl_vision_status)
        vision_layout.addLayout(vision_header)

        cv_body = QHBoxLayout()
        cv_body.addWidget(self.lbl_preview_thumb)

        cv_info = QVBoxLayout()
        cv_info.addWidget(self.lbl_roi_coords)
        lbl_instr = QLabel("Auto-detects high-contrast chessboard contour or rubberband snips screen.")
        lbl_instr.setStyleSheet("color: #6C7086; font-size: 10px;")
        lbl_instr.setWordWrap(True)
        cv_info.addWidget(lbl_instr)
        
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Interval (s):"))
        interval_row.addWidget(self.spin_interval)
        interval_row.addStretch()
        cv_info.addLayout(interval_row)

        # Nudge Buttons for fine calibration
        nudge_row = QHBoxLayout()
        btn_nudge_up = QPushButton("▲")
        btn_nudge_down = QPushButton("▼")
        btn_nudge_left = QPushButton("◄")
        btn_nudge_right = QPushButton("►")
        for b in [btn_nudge_up, btn_nudge_down, btn_nudge_left, btn_nudge_right]:
            b.setFixedSize(26, 26)
            b.setStyleSheet("padding: 2px;")
            nudge_row.addWidget(b)
        nudge_row.addStretch()
        cv_info.addLayout(nudge_row)

        btn_nudge_up.clicked.connect(lambda: self._nudge_roi(0, -2, 0, 0))
        btn_nudge_down.clicked.connect(lambda: self._nudge_roi(0, 2, 0, 0))
        btn_nudge_left.clicked.connect(lambda: self._nudge_roi(-2, 0, 0, 0))
        btn_nudge_right.clicked.connect(lambda: self._nudge_roi(2, 0, 0, 0))

        cv_body.addLayout(cv_info)
        vision_layout.addLayout(cv_body)
        right_layout.addWidget(vision_card)

        # 3. Live Move History & PGN Card
        history_card = QFrame()
        history_card.setProperty("class", "card")
        history_layout = QVBoxLayout(history_card)
        history_layout.setContentsMargins(10, 10, 10, 10)
        history_layout.setSpacing(6)

        lbl_hist_title = QLabel("Move History & PGN Log")
        lbl_hist_title.setProperty("class", "title")
        history_layout.addWidget(lbl_hist_title)

        history_layout.addWidget(self.move_table, stretch=1)

        fen_row = QHBoxLayout()
        fen_row.addWidget(self.txt_fen)
        fen_row.addWidget(self.btn_copy_fen)
        fen_row.addWidget(self.btn_copy_pgn)
        history_layout.addLayout(fen_row)

        right_layout.addWidget(history_card, stretch=1)

        root_layout.addWidget(right_container)

    def _connect_signals(self) -> None:
        """Wires internal buttons and background worker signals."""
        self.btn_select_roi.clicked.connect(self._start_roi_selection)
        self.btn_auto_detect.clicked.connect(self._auto_detect_board)
        self.btn_toggle_tracking.clicked.connect(self._toggle_tracking)
        self.btn_flip_board.clicked.connect(self._toggle_flip)
        self.btn_reset_game.clicked.connect(self._reset_game)
        self.btn_copy_fen.clicked.connect(self._copy_fen)
        self.btn_copy_pgn.clicked.connect(self._copy_pgn)
        self.btn_undo_move.clicked.connect(self._manual_undo)
        self.btn_sync_board.clicked.connect(self._sync_board_from_screen)
        self.chk_hybrid_mode.toggled.connect(self._toggle_hybrid_mode)

        # Settings Signals
        self.spin_interval.valueChanged.connect(self._on_vision_interval_changed)
        self.spin_lines_1.valueChanged.connect(self._on_lines_changed)
        self.spin_lines_2.valueChanged.connect(self._on_lines_changed)
        self.spin_lines_3.valueChanged.connect(self._on_lines_changed)
        self.spin_aggressiveness.valueChanged.connect(self.engine_worker.set_engine_aggressiveness)

        # Vision signals
        self.vision_worker.sig_move_detected.connect(self._on_vision_move)
        self.vision_worker.sig_status.connect(self._on_vision_status)
        self.vision_worker.sig_preview_updated.connect(self._on_vision_preview)
        self.vision_worker.sig_auto_detected_box.connect(self._on_auto_detected_box)
        self.vision_worker.sig_undo_move.connect(self._on_vision_undo)

        # Engine signals
        self.engine_worker.sig_evaluation_ready.connect(self._on_engine_evaluation)
        self.engine_worker.sig_secondary_eval_ready.connect(self._on_secondary_engine_evaluation)
        self.engine_worker.sig_tertiary_eval_ready.connect(self._on_tertiary_engine_evaluation)

        # Engine selection signals
        self.cmb_white_engine.currentIndexChanged.connect(
            lambda idx: self._on_engine_changed("white", idx)
        )
        self.cmb_black_engine.currentIndexChanged.connect(
            lambda idx: self._on_engine_changed("black", idx)
        )
        self.cmb_tertiary_engine.currentIndexChanged.connect(
            lambda idx: self._on_engine_changed("tertiary", idx)
        )
        self.btn_add_engine.clicked.connect(self._on_add_engine_clicked)

    # -------------------------------------------------------------------------
    # Slots & Handlers
    # -------------------------------------------------------------------------

    def _on_vision_interval_changed(self, val: float) -> None:
        """Updates the background polling interval of the screen capture."""
        self.vision_worker.poll_interval_ms = int(val * 1000)

    def _on_lines_changed(self, _value: int = 0) -> None:
        """Syncs engine MultiPV and board arrow limits when any lines spinbox changes."""
        l1 = self.spin_lines_1.value()
        l2 = self.spin_lines_2.value()
        l3 = self.spin_lines_3.value()

        # Update engine MultiPV counts
        self.engine_worker.set_engine_multipv(
            self.cmb_white_engine.currentData() or "stockfish", 0, l1
        )
        self.engine_worker.set_engine_multipv(
            self.cmb_black_engine.currentData() or "stockfish", 1, l2
        )
        self.engine_worker.set_engine_multipv(
            self.cmb_tertiary_engine.currentData() or "none", 2, l3
        )

        # Update board mirror arrow drawing limits
        self.board_mirror.set_arrow_limits(l1, l2, l3)

        # Re-request analysis so the engines send back the right number of lines
        self._request_analysis()

    def _start_roi_selection(self) -> None:
        """Opens snipping overlay."""
        self.screen_selector.start_selection()

    def _on_region_selected(self, x: int, y: int, w: int, h: int) -> None:
        """Applies user selected screen bounding box."""
        self.vision_worker.set_region(x, y, w, h)
        self.lbl_roi_coords.setText(f"ROI: {w}x{h} at ({x}, {y})")
        # Automatically enable tracking once selected
        self.btn_toggle_tracking.setChecked(True)
        self.btn_toggle_tracking.setText("Stop Tracking")
        self.btn_toggle_tracking.setStyleSheet("background-color: #C62828; color: #FFFFFF; font-weight: bold;")
        self.vision_worker.set_tracking(True)

    def _auto_detect_board(self) -> None:
        """Triggers screen auto-contour detection."""
        self.vision_worker.trigger_auto_detect()

    def _on_auto_detected_box(self, x: int, y: int, w: int, h: int) -> None:
        """Handles auto-detected ROI box."""
        self.lbl_roi_coords.setText(f"ROI: {w}x{h} at ({x}, {y})")
        self.btn_toggle_tracking.setChecked(True)
        self.btn_toggle_tracking.setText("Stop Tracking")
        self.btn_toggle_tracking.setStyleSheet("background-color: #C62828; color: #FFFFFF; font-weight: bold;")
        self.vision_worker.set_tracking(True)

    def _nudge_roi(self, dx: int, dy: int, dw: int, dh: int) -> None:
        """Fine tunes the active bounding box."""
        if self.vision_worker.bbox:
            x, y, w, h = self.vision_worker.bbox
            new_box = (max(0, x + dx), max(0, y + dy), max(40, w + dw), max(40, h + dh))
            self.vision_worker.set_region(*new_box)
            self.lbl_roi_coords.setText(f"ROI: {new_box[2]}x{new_box[3]} at ({new_box[0]}, {new_box[1]})")

    def _toggle_tracking(self) -> None:
        """Starts or pauses live screen perception."""
        is_tracking = self.btn_toggle_tracking.isChecked()
        self.vision_worker.set_tracking(is_tracking)
        if is_tracking:
            self.btn_toggle_tracking.setText("Stop Tracking")
            self.btn_toggle_tracking.setStyleSheet("background-color: #C62828; color: #FFFFFF; font-weight: bold;")
        else:
            self.btn_toggle_tracking.setText("Start Tracking")
            self.btn_toggle_tracking.setStyleSheet("background-color: #A6E3A1; color: #11111B; font-weight: bold;")

    def _toggle_flip(self) -> None:
        """Flips board mirror and perception perspective."""
        new_flip = not self.board_mirror.is_flipped
        self.board_mirror.set_flipped(new_flip)
        self.vision_worker.set_flipped(new_flip)

    def _reset_game(self) -> None:
        """Resets board state to starting position."""
        self.game_state.reset()
        self.board_mirror.set_board(self.game_state.board, last_move=None, engine_move=None, vision_move=None)
        self.board_mirror.clear_nn_arrows()
        self.vision_worker.reset_vision()
        self._refresh_move_table()
        self._refresh_game_status()
        self.txt_fen.setText(self.game_state.get_fen())
        self._request_analysis()

    def _manual_undo(self) -> None:
        """Manually undoes the last move via UI button."""
        if self.game_state.undo_last_move():
            logger.info("Manual undo. Board now: %s", self.game_state.get_fen())
            self.vision_worker.reset_vision()
            last = self.game_state.last_move
            self.board_mirror.set_board(self.game_state.board, last_move=last)
            self._refresh_move_table()
            self._refresh_game_status()
            self.txt_fen.setText(self.game_state.get_fen())
            self._request_analysis()
            self.lbl_game_status.setText("⟲ Undid last move")
        else:
            self.lbl_game_status.setText("No moves to undo")

    def _sync_board_from_screen(self) -> None:
        """Force-captures the external board and resets internal state to match.

        This re-reads the entire visual board from the ROI, reconstructs the
        position via the vision detector's full-board scan, and resets the
        internal game state to match. Useful when the tracker has drifted.
        """
        if not self.vision_worker.bbox:
            self.lbl_game_status.setText("Select screen area first")
            return

        try:
            x, y, w, h = self.vision_worker.bbox
            frame = self.vision_worker.capture.capture_region(x, y, w, h)
            if frame is not None and frame.size > 0:
                from chess_tracker.vision.capture import refine_board_grid
                inner_offset = refine_board_grid(frame)
                ix, iy, iw, ih = inner_offset
                board_frame = frame[iy:iy+ih, ix:ix+iw]
                if board_frame.size == 0:
                    board_frame = frame

                # Full board state scan via the detector
                full_board = self.vision_worker.detector.scan_full_board(
                    board_frame, is_flipped=self.board_mirror.is_flipped
                )
                if full_board is not None:
                    self.game_state.board = full_board.copy()
                    self.game_state.last_move = None
                    self.vision_worker.reset_vision()
                    self.board_mirror.set_board(self.game_state.board, last_move=None)
                    self._refresh_move_table()
                    self._refresh_game_status()
                    self.txt_fen.setText(self.game_state.get_fen())
                    self._request_analysis()
                    self.lbl_game_status.setText("📷 Board synced from screen")
                    logger.info("Board synced from screen: %s", self.game_state.get_fen())
                else:
                    self.lbl_game_status.setText("Sync failed: could not read board")
        except Exception as e:
            logger.error("Sync board failed: %s", e)
            self.lbl_game_status.setText(f"Sync error: {e}")

    def _toggle_hybrid_mode(self, enabled: bool) -> None:
        """Enables or disables Trio-Engine hybrid analysis mode."""
        self.engine_worker.set_hybrid_mode(enabled)
        
        for i in range(self.row3.count()):
            widget = self.row3.itemAt(i).widget()
            if widget:
                widget.setVisible(enabled)

        if enabled:
            # Change UI labels for combo boxes
            self.lbl_engine_sel_1.setText("Pri:")
            self.lbl_engine_sel_1.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.lbl_engine_sel_2.setText("Sec:")
            self.lbl_engine_sel_2.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

            # Auto-detect best available engines for hybrid mode if they haven't been picked
            primary_id = self.cmb_white_engine.currentData() or "stockfish"
            secondary_id = self.cmb_black_engine.currentData()
            tertiary_id = self.cmb_tertiary_engine.currentData()
            
            if secondary_id == primary_id or not secondary_id:
                secondary_candidates = ["patricia", "berserk", "opental", "reckless"]
                secondary_id = "none"
                from chess_tracker.core.engine_registry import get_all_available_engines
                available_ids = {e.id for e in get_all_available_engines()}
                for cand in secondary_candidates:
                    if cand in available_ids:
                        secondary_id = cand
                        break
                if secondary_id == "none":
                    secondary_id = "stockfish"

                idx = self.cmb_black_engine.findData(secondary_id)
                if idx >= 0:
                    self.cmb_black_engine.setCurrentIndex(idx)
                    
            if tertiary_id == primary_id or tertiary_id == secondary_id or not tertiary_id:
                tertiary_id = "none"
                idx = self.cmb_tertiary_engine.findData(tertiary_id)
                if idx >= 0:
                    self.cmb_tertiary_engine.setCurrentIndex(idx)

            self.engine_worker.set_trio_engines(primary_id, secondary_id, tertiary_id)
            self.lbl_game_status.setText("🔀 Trio-Engine Mode Active")
        else:
            # Revert UI labels
            self.lbl_engine_sel_1.setText("⚪")
            self.lbl_engine_sel_1.setFont(QFont("Segoe UI", 10))
            self.lbl_engine_sel_2.setText("⚫")
            self.lbl_engine_sel_2.setFont(QFont("Segoe UI", 10))
            self.board_mirror.clear_nn_arrows()
            self.lbl_game_status.setText("Standard engine mode")

            # Force re-sync of per-side engines based on current selections
            self._on_engine_changed("white", self.cmb_white_engine.currentIndex())
            self._on_engine_changed("black", self.cmb_black_engine.currentIndex())

        self._request_analysis()

    @pyqtSlot(chess.Move)
    def _on_vision_move(self, move: chess.Move) -> None:
        """Applies visually verified legal move."""
        if self.game_state.apply_move(move):
            self.board_mirror.set_board(self.game_state.board, last_move=move, vision_move=move)
            self._refresh_move_table()
            self._refresh_game_status()
            self.txt_fen.setText(self.game_state.get_fen())
            self._request_analysis()

    @pyqtSlot()
    def _on_vision_undo(self) -> None:
        """Self-correction: undoes the last move when board divergence is detected.

        Triggered by the vision worker when visual occupancy consistently disagrees
        with the internal board state for 15+ consecutive frames (~2.2 seconds).
        This handles false positive move detections by rolling back and allowing
        the vision to re-sync with the actual board.
        """
        if self.game_state.undo_last_move():
            logger.info("Self-correction: undid last move. Board now: %s", self.game_state.get_fen())
            self.vision_worker.reset_vision()
            last = self.game_state.last_move
            self.board_mirror.set_board(self.game_state.board, last_move=last)
            self._refresh_move_table()
            self._refresh_game_status()
            self.txt_fen.setText(self.game_state.get_fen())
            self._request_analysis()
            self.lbl_game_status.setText("⟲ Self-corrected: undid false detection")

    @pyqtSlot(str, bool)
    def _on_vision_status(self, text: str, is_active: bool) -> None:
        """Updates vision status pill."""
        self.lbl_vision_status.setText(text)
        color = "#A6E3A1" if is_active else "#F9E2AF"
        self.lbl_vision_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    @pyqtSlot(QImage, str)
    def _on_vision_preview(self, qimg: QImage, status_text: str) -> None:
        """Renders live thumbnail of captured ROI."""
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_preview_thumb.setPixmap(pixmap)

    @pyqtSlot(object)
    def _on_engine_evaluation(self, res: EvaluationResult) -> None:
        """Updates engine readout, tactical multi-PV arrows, and engine name badge."""
        self.eval_bar.set_evaluation(res)
        self.lbl_eval_score.setText(res.display_score)
        color = "#A6E3A1" if (res.score_cp or 0) >= 0 else "#F38BA8"
        self.lbl_eval_score.setStyleSheet(f"color: {color};")

        if res.best_move_san:
            self.lbl_best_move.setText(f"Best: {res.best_move_san}")
        else:
            self.lbl_best_move.setText("Best: --")

        self.lbl_engine_depth.setText(f"Depth: {res.depth}")
        self.lbl_engine_nps.setText(f"NPS: {res.nps // 1000}k" if res.nps else "NPS: --")
        self.lbl_eng_name.setText(res.engine_name)

        if res.pv_san:
            self.lbl_engine_pv.setText("PV: " + " ".join(res.pv_san[:6]))

        # Multi-PV: set up to 3 candidate move arrows on the board mirror
        if res.top_moves:
            top_move_objs = [m for m, sc, pv in res.top_moves]
            top_scores = [sc for m, sc, pv in res.top_moves]
            self.board_mirror.set_engine_top_moves(top_move_objs, top_scores)
        elif res.best_move:
            self.board_mirror.set_engine_top_moves([res.best_move], [res.score_cp or 0])

    @pyqtSlot(object)
    def _on_secondary_engine_evaluation(self, res: EvaluationResult) -> None:
        """Updates secondary engine arrows in hybrid mode."""
        if not self.engine_worker.is_hybrid_mode():
            return

        if res.top_moves:
            nn_moves = [m for m, sc, pv in res.top_moves]
            nn_scores = [sc for m, sc, pv in res.top_moves]
            self.board_mirror.set_nn_top_moves(nn_moves, nn_scores)
        elif res.best_move:
            self.board_mirror.set_nn_top_moves([res.best_move], [res.score_cp or 0])

    @pyqtSlot(object)
    def _on_tertiary_engine_evaluation(self, res: EvaluationResult) -> None:
        """Updates tertiary engine arrows in hybrid mode."""
        if not self.engine_worker.is_hybrid_mode():
            return

        if res.top_moves:
            tert_moves = [m for m, sc, pv in res.top_moves]
            tert_scores = [sc for m, sc, pv in res.top_moves]
            self.board_mirror.set_tert_top_moves(tert_moves, tert_scores)
        elif res.best_move:
            self.board_mirror.set_tert_top_moves([res.best_move], [res.score_cp or 0])

    def _populate_engine_combos(self) -> None:
        """Populates all engine dropdowns with all available engines."""
        engines = get_all_available_engines()
        self.cmb_white_engine.blockSignals(True)
        self.cmb_black_engine.blockSignals(True)
        self.cmb_tertiary_engine.blockSignals(True)

        cur_white_id = self.cmb_white_engine.currentData() or "stockfish"
        cur_black_id = self.cmb_black_engine.currentData() or "stockfish"
        cur_tert_id = self.cmb_tertiary_engine.currentData() or "none"

        self.cmb_white_engine.clear()
        self.cmb_black_engine.clear()
        self.cmb_tertiary_engine.clear()

        for eng in engines:
            self.cmb_white_engine.addItem(eng.name, userData=eng.id)
            self.cmb_black_engine.addItem(eng.name, userData=eng.id)
            self.cmb_tertiary_engine.addItem(eng.name, userData=eng.id)

        idx_w = self.cmb_white_engine.findData(cur_white_id)
        self.cmb_white_engine.setCurrentIndex(idx_w if idx_w >= 0 else 0)

        idx_b = self.cmb_black_engine.findData(cur_black_id)
        self.cmb_black_engine.setCurrentIndex(idx_b if idx_b >= 0 else 0)
        
        idx_t = self.cmb_tertiary_engine.findData(cur_tert_id)
        self.cmb_tertiary_engine.setCurrentIndex(idx_t if idx_t >= 0 else 0)

        self.cmb_white_engine.blockSignals(False)
        self.cmb_black_engine.blockSignals(False)
        self.cmb_tertiary_engine.blockSignals(False)

    def _on_engine_changed(self, side: str, index: int) -> None:
        """Handles engine selection from dropdown (per-side or hybrid mode)."""
        if side == "white":
            combo = self.cmb_white_engine
        elif side == "black":
            combo = self.cmb_black_engine
        else:
            combo = self.cmb_tertiary_engine
            
        engine_id = combo.itemData(index)
        if not engine_id:
            return

        if self.engine_worker.is_hybrid_mode():
            primary_id = self.cmb_white_engine.currentData() or "stockfish"
            secondary_id = self.cmb_black_engine.currentData() or "none"
            tertiary_id = self.cmb_tertiary_engine.currentData() or "none"
            self.engine_worker.set_trio_engines(primary_id, secondary_id, tertiary_id)
            logger.info("Hybrid engines updated: primary=%s, secondary=%s", primary_id, secondary_id)
        else:
            # Standard per-side mode
            self.engine_worker.set_engine_for_side(side, engine_id)
            logger.info("Engine for %s set to: %s", side, engine_id)
            
        self._request_analysis()

    def _on_add_engine_clicked(self) -> None:
        """Opens file dialog for user to add an external UCI engine binary."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select UCI Chess Engine Executable",
            "",
            "Executable Files (*.exe);;All Files (*)",
        )
        if not file_path:
            return

        # Probe binary to check validity and retrieve engine name
        is_valid, detected_name, err = probe_uci_engine(file_path)
        if not is_valid:
            QMessageBox.warning(
                self,
                "Invalid Chess Engine",
                f"Could not connect to engine as a valid UCI engine:\n\n{err}\n\nPlease select a valid UCI chess engine executable.",
            )
            return

        import os
        import uuid
        default_name = detected_name or os.path.splitext(os.path.basename(file_path))[0].title()
        name, ok = QInputDialog.getText(
            self,
            "Register Custom Engine",
            "Engine Display Name:",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not ok or not name.strip():
            return

        engine_name = name.strip()
        engine_id = f"custom_{uuid.uuid4().hex[:6]}"
        eng_def = EngineDefinition(
            id=engine_id,
            name=engine_name,
            binary_path=file_path,
            engine_type="uci",
            is_builtin=False,
            description=f"Custom engine loaded from {file_path}",
        )
        if save_custom_engine(eng_def):
            self._populate_engine_combos()
            # Select new engine for White
            idx = self.cmb_white_engine.findData(engine_id)
            if idx >= 0:
                self.cmb_white_engine.setCurrentIndex(idx)
            QMessageBox.information(
                self,
                "Engine Added",
                f"Successfully registered '{engine_name}'!\n\nIt is now active and selectable for White and Black.",
            )
            self._request_analysis()
        else:
            QMessageBox.critical(self, "Error", "Failed to save engine configuration.")

    def _request_analysis(self) -> None:
        """Sends current FEN to engine worker."""
        fen = self.game_state.get_fen()
        self.engine_worker.request_analysis(fen)

    def _refresh_move_table(self) -> None:
        """Re-populates move history table."""
        rows = self.game_state.get_formatted_move_table()
        self.move_table.setRowCount(len(rows))
        for i, (num, w, b) in enumerate(rows):
            self.move_table.setItem(i, 0, QTableWidgetItem(f"{num}."))
            self.move_table.setItem(i, 1, QTableWidgetItem(w))
            self.move_table.setItem(i, 2, QTableWidgetItem(b))
        self.move_table.scrollToBottom()

    def _refresh_game_status(self) -> None:
        """Updates check status, player turn, and material balance."""
        self.lbl_game_status.setText(self.game_state.get_status_text())

        diff, white_lost, black_lost = self.game_state.get_material_balance()
        if diff > 0:
            self.lbl_white_material.setText(f"+{diff}")
            self.lbl_black_material.setText("")
        elif diff < 0:
            self.lbl_black_material.setText(f"+{abs(diff)}")
            self.lbl_white_material.setText("")
        else:
            self.lbl_white_material.setText("+0")
            self.lbl_black_material.setText("+0")

    def _copy_fen(self) -> None:
        """Copies active FEN to clipboard."""
        QApplication.clipboard().setText(self.game_state.get_fen())
        self.lbl_game_status.setText("FEN copied to clipboard!")

    def _copy_pgn(self) -> None:
        """Copies active PGN to clipboard."""
        QApplication.clipboard().setText(self.game_state.get_pgn())
        self.lbl_game_status.setText("PGN copied to clipboard!")

    def closeEvent(self, event) -> None:
        """Terminates background worker threads cleanly."""
        self.vision_worker.stop()
        self.engine_worker.stop()
        event.accept()
