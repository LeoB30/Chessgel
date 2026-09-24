from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple
import chess
import cv2
import numpy as np
from PIL import Image

import vision.sync
from vision.capture import detect_chessboard_contour, slice_board_grid
from vision.classifier import BoardClassifier, ClassificationResult
from vision.templates import TemplateStore, load_or_extract_templates


class BoardVisionSync(vision.sync.BoardVisionSync):
    """Refactored BoardVisionSync integrating OpenCV two-tier vision with GUI callbacks."""

    @classmethod
    def slice_and_classify_board(
        cls,
        frame: Image.Image,
        is_flipped: bool = False,
        templates: Optional[TemplateStore] = None,
        outer_margin_pct: float = 0.0,
        inner_inset_pct: float = 0.12,
    ) -> Tuple[Dict[chess.Square, Optional[chess.Piece]], str]:
        """Class method for slicing and classifying an 8x8 PIL frame directly.

        Returns (piece_grid, placement_fen).
        """
        frame_rgb = np.array(frame.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        grid_slices = slice_board_grid(
            frame_bgr,
            is_flipped=is_flipped,
            outer_margin_pct=outer_margin_pct,
            inner_inset_pct=inner_inset_pct,
        )

        classifier = BoardClassifier(template_store=templates)
        result: ClassificationResult = classifier.classify_board(grid_slices)
        return result.piece_grid, result.placement_fen

    @classmethod
    def generate_debug_overlay_image(
        cls,
        frame: Image.Image,
        piece_grid: Dict[chess.Square, Optional[chess.Piece]],
        is_flipped: bool = False,
        outer_margin_pct: float = 0.0,
        inner_inset_pct: float = 0.12,
    ) -> Image.Image:
        """Class method generating an annotated debug overlay image with grid and detected pieces."""
        frame_rgb = np.array(frame.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        grid_slices = slice_board_grid(
            frame_bgr,
            is_flipped=is_flipped,
            outer_margin_pct=outer_margin_pct,
            inner_inset_pct=inner_inset_pct,
        )

        classifier = BoardClassifier()
        result = classifier.classify_board(grid_slices)

        dummy_sync = cls(
            get_board_callback=lambda: chess.Board(),
            on_move_detected=lambda m: None,
            on_status_update=lambda s, a, e: None,
            outer_margin_pct=outer_margin_pct,
            inner_inset_pct=inner_inset_pct,
        )
        dummy_sync.is_flipped = is_flipped
        return dummy_sync.generate_debug_overlay(frame_bgr, grid_slices, result)
