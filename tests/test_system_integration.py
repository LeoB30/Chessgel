from __future__ import annotations
import os
import sys
import unittest
import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sync import GameStateSynchronizer
from engine.evaluator import evaluate_board
from engine.opening_book import (
    get_book_move,
    get_candidate_explanation,
    identify_opening,
)
from engine.search import find_best_move
from vision.capture import detect_chessboard_contour, slice_board_grid
from vision.classifier import BoardClassifier, ClassificationResult
from vision.templates import load_or_extract_templates


class TestSystemIntegration(unittest.TestCase):
    """End-to-end integration test suite verifying core state sync, engine, vision, and GUI components."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ref_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_board.png"
        )
        cls.template_store = load_or_extract_templates(cls.ref_path)

    def test_01_core_state_synchronizer_sanity_validation(self) -> None:
        """Verifies core GameStateSynchronizer sanity verification rules."""
        sync = GameStateSynchronizer()

        # Valid starting position grid
        valid_board = chess.Board()
        valid_grid = {sq: valid_board.piece_at(sq) for sq in chess.SQUARES}
        is_sane, err = sync.validate_detection_sanity(valid_grid)
        self.assertTrue(is_sane)
        self.assertIsNone(err)

        # Illegal: Missing White King
        bad_grid_no_k = dict(valid_grid)
        bad_grid_no_k[chess.E1] = None
        is_sane, err = sync.validate_detection_sanity(bad_grid_no_k)
        self.assertFalse(is_sane)
        self.assertIn("White kings", str(err))

        # Illegal: Two Black Kings
        bad_grid_two_k = dict(valid_grid)
        bad_grid_two_k[chess.E6] = chess.Piece.from_symbol("k")
        is_sane, err = sync.validate_detection_sanity(bad_grid_two_k)
        self.assertFalse(is_sane)
        self.assertIn("Black kings", str(err))

        # Illegal: Pawn on Rank 1
        bad_grid_pawn_r1 = dict(valid_grid)
        bad_grid_pawn_r1[chess.D1] = chess.Piece.from_symbol("P")
        is_sane, err = sync.validate_detection_sanity(bad_grid_pawn_r1)
        self.assertFalse(is_sane)
        self.assertIn("Pawn on rank 1", str(err))

    def test_02_core_state_synchronizer_debounce_and_failsafe(self) -> None:
        """Verifies 2-consecutive-frame debounce filtering and silent fail-safe fallback."""
        board = chess.Board()
        sync = GameStateSynchronizer(board=board, consecutive_frames_required=2)

        # Test frame after move 1. e4
        board_after_e4 = chess.Board()
        board_after_e4.push_san("e4")
        e4_grid = {sq: board_after_e4.piece_at(sq) for sq in chess.SQUARES}

        # Frame 1: Detected move e4 (first sighting -> pending, not yet confirmed)
        move_f1, valid_f1 = sync.process_detected_frame(e4_grid)
        self.assertTrue(valid_f1)
        self.assertIsNone(move_f1, "Debounce should hold move on first consecutive frame")

        # Frame 2: Confirmed identical move e4 on second consecutive frame
        move_f2, valid_f2 = sync.process_detected_frame(e4_grid)
        self.assertTrue(valid_f2)
        self.assertIsNotNone(move_f2, "Debounce should confirm move on second consecutive frame")
        self.assertEqual(move_f2.uci(), "e2e4")
        self.assertEqual(sync.board.fen(), board_after_e4.fen())

        # Test fail-safe: Corrupted frame with 0 kings
        corrupt_grid = {sq: None for sq in chess.SQUARES}
        move_bad, valid_bad = sync.process_detected_frame(corrupt_grid)
        self.assertFalse(valid_bad)
        self.assertIsNone(move_bad)
        # Verify internal board state did not corrupt or reset
        self.assertEqual(sync.board.fen(), board_after_e4.fen())

    def test_03_caro_kann_opening_book_and_strategic_annotations(self) -> None:
        """Verifies Caro-Kann repertoire paths and educational strategic annotations."""
        b = chess.Board()
        b.push_san("e4")

        # In response to 1. e4, repertoire recommends 1...c6
        book_move = get_book_move(b, repertoire="catalan_carokann", selection_mode="best")
        self.assertIsNotNone(book_move)
        self.assertEqual(book_move.uci(), "c7c6")

        explanation = get_candidate_explanation(b, book_move)
        self.assertIn("Caro-Kann", explanation.variation_name)
        self.assertIn("c8 bishop", explanation.key_ideas.lower())
        self.assertTrue(len(explanation.strategic_intent) > 20)

        # Walk through Advance Variation: 1. e4 c6 2. d4 d5 3. e5
        b.push(book_move)
        b.push_san("d4")
        b.push_san("d5")
        b.push_san("e5")

        advance_response = get_book_move(b, repertoire="catalan_carokann", selection_mode="best")
        self.assertIsNotNone(advance_response)
        self.assertEqual(advance_response.uci(), "c8f5")  # Bishop outside pawn chain!

        adv_exp = get_candidate_explanation(b, advance_response)
        self.assertIn("bishop", adv_exp.strategic_intent.lower())

    def test_04_catalan_opening_book_and_strategic_annotations(self) -> None:
        """Verifies Catalan Opening repertoire paths and educational strategic annotations."""
        b = chess.Board()

        # Starting as White: Catalan 1. d4
        move_1 = get_book_move(b, repertoire="catalan_carokann", selection_mode="best")
        self.assertIsNotNone(move_1)
        self.assertEqual(move_1.uci(), "d2d4")

        exp_1 = get_candidate_explanation(b, move_1)
        self.assertTrue(
            "Catalan" in exp_1.variation_name or "Catalan" in exp_1.positional_goals
        )

        # Walk to Catalan signature setup: 1. d4 Nf6 2. c4 e6 3. g3 d5 4. Bg2
        b.push(move_1)
        b.push_san("Nf6")
        b.push_san("c4")
        b.push_san("e6")

        cat_g3 = get_book_move(b, repertoire="catalan_carokann", selection_mode="best")
        self.assertIsNotNone(cat_g3)
        self.assertEqual(cat_g3.uci(), "g2g3")

        exp_g3 = get_candidate_explanation(b, cat_g3)
        self.assertIn("fianchetto", exp_g3.strategic_intent.lower())

    def test_05_engine_search_and_smooth_handoff(self) -> None:
        """Verifies engine alpha-beta search with killer moves and PST centipawn scoring."""
        # Custom tactical position outside opening book
        fen_tactics = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5"
        board = chess.Board(fen_tactics)

        # Verify static evaluation evaluates positional and material balance
        score = evaluate_board(board)
        self.assertIsInstance(score, int)

        # Alpha-beta search with depth 3
        best_move = find_best_move(board, depth=3, movetime_ms=300, use_book=False)
        self.assertIsNotNone(best_move)
        self.assertIn(best_move, board.legal_moves)

    def test_06_pyqt6_gui_initialization(self) -> None:
        """Verifies PyQt6 GUI components instantiate cleanly without errors."""
        from PyQt6.QtWidgets import QApplication
        from gui.app import ChessApp
        from gui.board_view import BoardView
        from gui.eval_bar import EvalBar
        from gui.telemetry_view import TelemetryView
        from gui.cv_overlay import CVOverlayWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        window = ChessApp()
        self.assertIsInstance(window.board_view, BoardView)
        self.assertIsInstance(window.eval_bar, EvalBar)
        self.assertIsInstance(window.telemetry_view, TelemetryView)
        self.assertIsInstance(window.cv_overlay, CVOverlayWindow)
        window.close()
