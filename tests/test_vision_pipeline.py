"""Comprehensive automated test suite for the OpenCV Vision & LivePlay subsystem.

Validates template extraction, board detection, two-tier classification,
highlight resilience, legal move state synchronization, and fail-safe defenses.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
import chess
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.capture import ScreenCapture, detect_chessboard_contour, slice_board_grid
from vision.classifier import BoardClassifier, ClassificationResult
from vision.sync import BoardVisionSync
from vision.templates import (
    PieceTemplate,
    TemplateStore,
    detect_board_crop,
    extract_templates_from_board,
    load_or_extract_templates,
)


class TestVisionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.ref_path = os.path.join(cls.workspace_dir, "reference_board.png")
        if not os.path.isfile(cls.ref_path):
            raise FileNotFoundError(f"Missing ground truth baseline at {cls.ref_path}")
        cls.ref_img = cv2.imread(cls.ref_path)
        assert cls.ref_img is not None, "Failed to load reference_board.png"
        cls.template_store = load_or_extract_templates(cls.ref_path)

    def test_01_template_extraction_pipeline(self) -> None:
        """Verifies template extraction for all 12 pieces with grayscale and Canny edges."""
        store = self.template_store
        expected_symbols = {"P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"}
        self.assertEqual(set(store.symbols()), expected_symbols)

        for sym in expected_symbols:
            tmpl = store.get(sym)
            self.assertIsNotNone(tmpl)
            self.assertGreater(len(tmpl.gray_templates), 0)
            self.assertGreater(len(tmpl.edge_templates), 0)
            for g, e in zip(tmpl.gray_templates, tmpl.edge_templates):
                self.assertEqual(g.shape, (64, 64))
                self.assertEqual(e.shape, (64, 64))
                # Canny edge mask must have distinct silhouette edges
                self.assertGreater(np.count_nonzero(e), 20)

    def test_02_board_auto_localization(self) -> None:
        """Tests automatic chessboard contour localization from reference image."""
        box = detect_board_crop(self.ref_img)
        bx, by, bw, bh = box
        self.assertGreaterEqual(bw, 850)
        self.assertGreaterEqual(bh, 850)
        aspect = bw / float(bh)
        self.assertAlmostEqual(aspect, 1.0, delta=0.08)

        # Test localization within a larger synthetic background canvas
        canvas = np.full((1200, 1400, 3), (30, 30, 30), dtype=np.uint8)
        offset_y, offset_x = 100, 150
        canvas[offset_y : offset_y + self.ref_img.shape[0], offset_x : offset_x + self.ref_img.shape[1]] = self.ref_img

        detected_box = detect_chessboard_contour(canvas)
        self.assertIsNotNone(detected_box)
        dx, dy, dw, dh = detected_box
        self.assertAlmostEqual(dw, bw, delta=15)
        self.assertAlmostEqual(dh, bh, delta=15)

    def test_03_baseline_board_classification_100_percent(self) -> None:
        """Verifies 100% accurate FEN reconstruction on ground-truth reference baseline."""
        bx, by, bw, bh = detect_board_crop(self.ref_img)
        board_crop = self.ref_img[by : by + bh, bx : bx + bw]

        grid_slices = slice_board_grid(board_crop, is_flipped=False, inner_inset_pct=0.12)
        self.assertEqual(len(grid_slices), 64)

        classifier = BoardClassifier(template_store=self.template_store)
        result: ClassificationResult = classifier.classify_board(grid_slices)

        expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        self.assertEqual(result.placement_fen, expected_fen)
        self.assertTrue(result.is_legal)
        self.assertIsNone(result.validation_error)

        # Check square-by-square pieces
        starting_board = chess.Board()
        for sq in chess.SQUARES:
            exp_p = starting_board.piece_at(sq)
            det_p = result.piece_grid[sq]
            if exp_p is None:
                self.assertIsNone(det_p, f"Square {chess.square_name(sq)} expected Empty, got {det_p}")
            else:
                self.assertIsNotNone(det_p, f"Square {chess.square_name(sq)} expected {exp_p.symbol()}, got None")
                self.assertEqual(exp_p.symbol(), det_p.symbol())
            # Confidence must be high
            self.assertGreaterEqual(result.confidences[sq], 0.70)

    def test_04_flipped_perspective_mapping(self) -> None:
        """Verifies coordinate slicing when board perspective is flipped (Black at bottom)."""
        bx, by, bw, bh = detect_board_crop(self.ref_img)
        board_crop = self.ref_img[by : by + bh, bx : bx + bw]
        sq_w = bw / 8.0
        sq_h = bh / 8.0

        # Construct flipped board (Black at bottom, pieces upright)
        flipped_board = np.zeros_like(board_crop)
        for r in range(8):
            for c in range(8):
                sr, sc = 7 - r, 7 - c
                x1_s, y1_s = int(round(sc * sq_w)), int(round(sr * sq_h))
                x2_s, y2_s = int(round((sc + 1) * sq_w)), int(round((sr + 1) * sq_h))
                x1_d, y1_d = int(round(c * sq_w)), int(round(r * sq_h))
                x2_d, y2_d = int(round((c + 1) * sq_w)), int(round((r + 1) * sq_h))
                cell = board_crop[y1_s:y2_s, x1_s:x2_s]
                flipped_board[y1_d:y2_d, x1_d:x2_d] = cv2.resize(cell, (x2_d - x1_d, y2_d - y1_d))

        grid_slices_flipped = slice_board_grid(flipped_board, is_flipped=True, inner_inset_pct=0.12)
        classifier = BoardClassifier(template_store=self.template_store)
        result = classifier.classify_board(grid_slices_flipped)

        expected_flipped_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        self.assertEqual(result.placement_fen, expected_flipped_fen)

    def test_05_move_highlight_resilience(self) -> None:
        """Verifies immunity to move highlight tints on both vacated and occupied squares."""
        bx, by, bw, bh = detect_board_crop(self.ref_img)
        board_crop = self.ref_img[by : by + bh, bx : bx + bw].copy()
        sq_w = bw / 8.0
        sq_h = bh / 8.0

        # Simulate 1. e4: move pawn from e2 to e4
        # e2: row 6, col 4
        # e4: row 4, col 4
        # e3: row 5, col 4 (empty square background)
        x_e2, y_e2 = int(round(4 * sq_w)), int(round(6 * sq_h))
        x_e3, y_e3 = int(round(4 * sq_w)), int(round(5 * sq_h))
        x_e4, y_e4 = int(round(4 * sq_w)), int(round(4 * sq_h))
        cell_w, cell_h = int(round(5 * sq_w)) - x_e2, int(round(7 * sq_h)) - y_e2

        # 1. Clear e2 (vacated from-square) by copying empty square texture from e3
        board_crop[y_e2 : y_e2 + cell_h, x_e2 : x_e2 + cell_w] = board_crop[y_e3 : y_e3 + cell_h, x_e3 : x_e3 + cell_w]

        # 2. Place White Pawn on e4 (occupied to-square)
        pawn_tile = self.ref_img[by + y_e2 : by + y_e2 + cell_h, bx + x_e2 : bx + x_e2 + cell_w]
        board_crop[y_e4 : y_e4 + cell_h, x_e4 : x_e4 + cell_w] = pawn_tile

        # 3. Apply strong yellow/green move highlight overlay to both e2 and e4
        highlight_color = np.array([40, 225, 230], dtype=np.uint8)  # yellow tint
        for y, x in [(y_e2, x_e2), (y_e4, x_e4)]:
            tile = board_crop[y : y + cell_h, x : x + cell_w]
            board_crop[y : y + cell_h, x : x + cell_w] = cv2.addWeighted(
                tile, 0.65, np.full_like(tile, highlight_color), 0.35, 0
            )

        grid_slices = slice_board_grid(board_crop, is_flipped=False, inner_inset_pct=0.12)
        classifier = BoardClassifier(template_store=self.template_store)
        result = classifier.classify_board(grid_slices)

        # e2 must be classified as EMPTY despite highlight
        self.assertIsNone(result.piece_grid[chess.E2], "Vacated square e2 must be EMPTY despite move highlight")
        # e4 must be classified as White Pawn despite highlight
        self.assertIsNotNone(result.piece_grid[chess.E4], "Occupied square e4 must have piece")
        self.assertEqual(result.piece_grid[chess.E4].symbol(), "P")

        expected_fen_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        self.assertEqual(result.placement_fen, expected_fen_e4)

    def test_06_defensive_fail_safe_on_illegal_positions(self) -> None:
        """Verifies that invalid or occluded boards missing kings are flagged and discarded."""
        bx, by, bw, bh = detect_board_crop(self.ref_img)
        board_crop = self.ref_img[by : by + bh, bx : bx + bw].copy()

        # Occlude the White King (row 7, col 4) by painting black over it
        sq_w = bw / 8.0
        sq_h = bh / 8.0
        x_k, y_k = int(round(4 * sq_w)), int(round(7 * sq_h))
        cell_w, cell_h = int(round(5 * sq_w)) - x_k, int(round(8 * sq_h)) - y_k
        board_crop[y_k : y_k + cell_h, x_k : x_k + cell_w] = 0

        grid_slices = slice_board_grid(board_crop, is_flipped=False)
        classifier = BoardClassifier(template_store=self.template_store)
        result = classifier.classify_board(grid_slices)

        # Legality validation must fail due to missing White King
        self.assertFalse(result.is_legal)
        self.assertIn("White has 0 kings", str(result.validation_error))

    def test_07_legal_move_inference_and_consecutive_debounce(self) -> None:
        """Verifies state transition detection from internal python-chess Board."""
        board = chess.Board()
        detected_moves: list[chess.Move] = []
        statuses: list[str] = []
        fens: list[str] = []

        vision = BoardVisionSync(
            get_board_callback=lambda: board.copy(),
            on_move_detected=lambda m: detected_moves.append(m),
            on_status_update=lambda s, a, e: statuses.append(s),
            on_fen_detected=lambda fen, grid, conf: fens.append(fen),
            poll_interval=0.04,
            consecutive_frames_required=2,
            template_store=self.template_store,
        )

        # Generate board image after 1. e4
        bx, by, bw, bh = detect_board_crop(self.ref_img)
        board_crop = self.ref_img[by : by + bh, bx : bx + bw].copy()
        sq_w = bw / 8.0
        sq_h = bh / 8.0
        x_e2, y_e2 = int(round(4 * sq_w)), int(round(6 * sq_h))
        x_e3, y_e3 = int(round(4 * sq_w)), int(round(5 * sq_h))
        x_e4, y_e4 = int(round(4 * sq_w)), int(round(4 * sq_h))
        cell_w, cell_h = int(round(5 * sq_w)) - x_e2, int(round(7 * sq_h)) - y_e2
        board_crop[y_e2 : y_e2 + cell_h, x_e2 : x_e2 + cell_w] = board_crop[y_e3 : y_e3 + cell_h, x_e3 : x_e3 + cell_w]
        pawn_tile = self.ref_img[by + y_e2 : by + y_e2 + cell_h, bx + x_e2 : bx + x_e2 + cell_w]
        board_crop[y_e4 : y_e4 + cell_h, x_e4 : x_e4 + cell_w] = pawn_tile

        pil_frame = Image.fromarray(cv2.cvtColor(board_crop, cv2.COLOR_BGR2RGB))
        vision.mock_image_provider = lambda: pil_frame
        _ = vision.classifier  # Pre-warm classifier cache before thread starts

        started = vision.start()
        self.assertTrue(started)
        self.assertTrue(vision.is_running)

        # Allow background thread to poll cycles until move detected
        for _ in range(20):
            if len(detected_moves) >= 1:
                break
            time.sleep(0.05)
        vision.stop()

        self.assertFalse(vision.is_running)
        self.assertGreaterEqual(len(fens), 1)
        self.assertEqual(len(detected_moves), 1)
        self.assertEqual(detected_moves[0], chess.Move.from_uci("e2e4"))


if __name__ == "__main__":
    unittest.main()
