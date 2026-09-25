from __future__ import annotations
from typing import List, Optional
import chess

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TelemetryView(QWidget):
    """Real-time telemetry dashboard presenting move history, educational commentary, and engine evaluation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        main_layout: QVBoxLayout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # 1. Engine Recommendation Card
        self.engine_card: QFrame = QFrame()
        self.engine_card.setObjectName("engineCard")
        self.engine_card.setStyleSheet("""
            #engineCard {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 6px;
            }
        """)
        engine_layout: QVBoxLayout = QVBoxLayout(self.engine_card)
        engine_layout.setContentsMargins(10, 8, 10, 8)
        engine_layout.setSpacing(4)

        header_row = QHBoxLayout()
        rec_title = QLabel("🤖 ENGINE EVALUATION")
        rec_title.setStyleSheet("color: #89B4FA; font-weight: bold; font-size: 11px;")
        header_row.addWidget(rec_title)

        self.source_badge = QLabel("Idle")
        self.source_badge.setStyleSheet("""
            background-color: #313244;
            color: #CDD6F4;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
        """)
        header_row.addWidget(self.source_badge, alignment=Qt.AlignmentFlag.AlignRight)
        engine_layout.addLayout(header_row)

        score_row = QHBoxLayout()
        self.lbl_best_move = QLabel("Best Move: --")
        self.lbl_best_move.setStyleSheet("color: #A6E3A1; font-size: 15px; font-weight: bold;")
        score_row.addWidget(self.lbl_best_move)

        self.lbl_eval = QLabel("Score: 0.00")
        self.lbl_eval.setStyleSheet("color: #F9E2AF; font-size: 13px; font-weight: bold;")
        score_row.addWidget(self.lbl_eval, alignment=Qt.AlignmentFlag.AlignRight)
        engine_layout.addLayout(score_row)

        self.lbl_pv = QLabel("Tactical Line: --")
        self.lbl_pv.setStyleSheet("color: #BAC2DE; font-size: 11px;")
        self.lbl_pv.setWordWrap(True)
        engine_layout.addWidget(self.lbl_pv)

        main_layout.addWidget(self.engine_card)

        # 2. Educational Strategic Commentary Card
        self.commentary_card: QFrame = QFrame()
        self.commentary_card.setObjectName("commentaryCard")
        self.commentary_card.setStyleSheet("""
            #commentaryCard {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
            }
        """)
        comm_layout: QVBoxLayout = QVBoxLayout(self.commentary_card)
        comm_layout.setContentsMargins(10, 8, 10, 8)
        comm_layout.setSpacing(6)

        comm_title = QLabel("📖 STRATEGIC OPENING REPERTOIRE")
        comm_title.setStyleSheet("color: #F9E2AF; font-weight: bold; font-size: 11px;")
        comm_layout.addWidget(comm_title)

        self.lbl_opening_name = QLabel("Catalan / Caro-Kann Guide")
        self.lbl_opening_name.setStyleSheet("color: #89B4FA; font-size: 13px; font-weight: bold;")
        self.lbl_opening_name.setWordWrap(True)
        comm_layout.addWidget(self.lbl_opening_name)

        self.txt_commentary = QTextEdit()
        self.txt_commentary.setReadOnly(True)
        self.txt_commentary.setStyleSheet("""
            QTextEdit {
                background-color: #11111B;
                color: #CDD6F4;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 6px;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        self.txt_commentary.setFixedHeight(140)
        self.txt_commentary.setPlaceholderText(
            "Move-by-move master commentary covering Catalan long-diagonal dominance "
            "and Caro-Kann pawn counter-strikes will appear here."
        )
        comm_layout.addWidget(self.txt_commentary)

        main_layout.addWidget(self.commentary_card)

        # 3. Move History Table (SAN)
        history_header = QLabel("📜 MOVE HISTORY (SAN)")
        history_header.setStyleSheet("color: #CDD6F4; font-weight: bold; font-size: 11px; margin-top: 4px;")
        main_layout.addWidget(history_header)

        self.tbl_moves: QTableWidget = QTableWidget(0, 3)
        self.tbl_moves.setHorizontalHeaderLabels(["#", "White", "Black"])
        self.tbl_moves.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_moves.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tbl_moves.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tbl_moves.verticalHeader().setVisible(False)
        self.tbl_moves.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_moves.setStyleSheet("""
            QTableWidget {
                background-color: #181825;
                color: #CDD6F4;
                border: 1px solid #313244;
                border-radius: 6px;
                gridline-color: #313244;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #11111B;
                color: #BAC2DE;
                border: 1px solid #313244;
                font-weight: bold;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #45475A;
                color: #F9E2AF;
            }
        """)
        main_layout.addWidget(self.tbl_moves)

    def update_engine_eval(
        self,
        score_cp: int,
        best_move: Optional[chess.Move],
        source: str,
        title: str,
        body: str,
        pv_san: str,
    ) -> None:
        """Updates engine recommendation card and educational commentary."""
        # Update badge
        if source == "book":
            self.source_badge.setText("BOOK REPERTOIRE")
            self.source_badge.setStyleSheet("background-color: #1E66F5; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;")
        elif source == "search":
            self.source_badge.setText("ENGINE SEARCH")
            self.source_badge.setStyleSheet("background-color: #40A02B; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;")
        else:
            self.source_badge.setText("IDLE")
            self.source_badge.setStyleSheet("background-color: #313244; color: #BAC2DE; padding: 2px 6px; border-radius: 4px; font-size: 10px;")

        # Update best move and eval
        if best_move is not None:
            self.lbl_best_move.setText(f"Best Move: {best_move.uci()}")
        else:
            self.lbl_best_move.setText("Best Move: --")

        if abs(score_cp) >= 80000:
            moves = max(1, 90000 - abs(score_cp))
            self.lbl_eval.setText(f"Score: Mate in {moves}")
        else:
            pawn_val = score_cp / 100.0
            sign = "+" if pawn_val > 0 else ""
            self.lbl_eval.setText(f"Score: {sign}{pawn_val:.2f}")

        self.lbl_pv.setText(f"Tactical Line: {pv_san}" if pv_san else "Tactical Line: --")

        # Update Commentary
        if title:
            self.lbl_opening_name.setText(title)
        if body:
            self.txt_commentary.setPlainText(body)

    def update_move_history(self, board: chess.Board) -> None:
        """Reconstructs the move history table from board move stack."""
        self.tbl_moves.setRowCount(0)
        temp_board = chess.Board()

        move_pairs: List[Tuple[int, str, str]] = []
        white_san: str = ""

        for idx, move in enumerate(board.move_stack):
            san: str = temp_board.san(move)
            temp_board.push(move)

            if idx % 2 == 0:
                white_san = san
            else:
                move_num = (idx // 2) + 1
                move_pairs.append((move_num, white_san, san))
                white_san = ""

        if white_san:
            move_num = (len(board.move_stack) // 2) + 1
            move_pairs.append((move_num, white_san, ""))

        self.tbl_moves.setRowCount(len(move_pairs))
        for row_idx, (num, w_san, b_san) in enumerate(move_pairs):
            item_num = QTableWidgetItem(f"{num}.")
            item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_w = QTableWidgetItem(w_san)
            item_b = QTableWidgetItem(b_san)

            self.tbl_moves.setItem(row_idx, 0, item_num)
            self.tbl_moves.setItem(row_idx, 1, item_w)
            self.tbl_moves.setItem(row_idx, 2, item_b)

        self.tbl_moves.scrollToBottom()
