"""Vision & LivePlay subsystem for automated board detection and real-time state synchronization."""
from __future__ import annotations

from vision.capture import (
    ScreenCapture,
    detect_chessboard_contour,
    slice_board_grid,
)
from vision.classifier import (
    BoardClassifier,
    ClassificationResult,
)
from vision.sync import BoardVisionSync
from vision.templates import (
    PieceTemplate,
    TemplateStore,
    detect_board_crop,
    extract_templates_from_board,
    load_or_extract_templates,
)

__all__ = [
    "PieceTemplate",
    "TemplateStore",
    "detect_board_crop",
    "extract_templates_from_board",
    "load_or_extract_templates",
    "ScreenCapture",
    "detect_chessboard_contour",
    "slice_board_grid",
    "BoardClassifier",
    "ClassificationResult",
    "BoardVisionSync",
]
