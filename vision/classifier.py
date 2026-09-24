from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import chess
import cv2
import numpy as np

from vision.templates import TemplateStore, load_or_extract_templates


@dataclass(frozen=True)
class ClassificationResult:
    """Encapsulates the 64-square classification, FEN string, and confidence scores."""
    piece_grid: Dict[chess.Square, Optional[chess.Piece]]
    placement_fen: str
    confidences: Dict[chess.Square, float]
    is_legal: bool
    validation_error: Optional[str] = None


class BoardClassifier:
    """Robust two-tier chessboard classifier resilient to move highlights and style variations."""

    def __init__(
        self,
        template_store: Optional[TemplateStore] = None,
        occupancy_var_thresh: float = 175.0,
        occupancy_edge_thresh: int = 100,
        canny_thresh1: int = 50,
        canny_thresh2: int = 150,
    ) -> None:
        if template_store is None:
            self.template_store: TemplateStore = load_or_extract_templates()
        else:
            self.template_store = template_store

        self.occupancy_var_thresh: float = occupancy_var_thresh
        self.occupancy_edge_thresh: int = occupancy_edge_thresh
        self.canny_thresh1: int = canny_thresh1
        self.canny_thresh2: int = canny_thresh2
        self.target_size: Tuple[int, int] = self.template_store.target_size

    def classify_square(
        self,
        roi_bgr: np.ndarray,
        square: chess.Square,
    ) -> Tuple[Optional[chess.Piece], float]:
        """Classifies a single square using two-tier occupancy check and template matching.

        Returns (chess.Piece | None, confidence_score 0.0-1.0).
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
        h, w = gray.shape[:2]
        if h != self.target_size[1] or w != self.target_size[0]:
            gray_scaled = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            gray_scaled = gray

        # Tier 1: Occupancy Check (evaluates raw contrast against empty wood thresholds)
        var = float(np.var(gray_scaled))
        raw_edges = cv2.Canny(gray_scaled, self.canny_thresh1, self.canny_thresh2)
        edge_count = int(np.count_nonzero(raw_edges))

        if var < self.occupancy_var_thresh and edge_count < self.occupancy_edge_thresh:
            confidence = max(0.85, 1.0 - (var / (2.0 * self.occupancy_var_thresh)))
            return None, confidence

        # Tier 2: Contour & Feature Matching (normalized for silhouette shape matching)
        norm_gray = cv2.normalize(gray_scaled, None, 0, 255, cv2.NORM_MINMAX)
        sq_edges = cv2.Canny(norm_gray, self.canny_thresh1, self.canny_thresh2)
        mean_brightness = float(np.mean(gray_scaled))

        best_score: float = -999.0
        best_symbol: Optional[str] = None

        for symbol in self.template_store.symbols():
            piece_tmpl = self.template_store.get(symbol)
            if piece_tmpl is None:
                continue

            for cand_gray, cand_edge in zip(piece_tmpl.gray_templates, piece_tmpl.edge_templates):
                # Edge matching (silhouette shape)
                res_edge = cv2.matchTemplate(sq_edges, cand_edge, cv2.TM_CCOEFF_NORMED)
                edge_score = float(res_edge[0][0]) if not np.isnan(res_edge[0][0]) else 0.0

                # Grayscale normalized matching
                res_gray = cv2.matchTemplate(norm_gray, cand_gray, cv2.TM_CCOEFF_NORMED)
                gray_score = float(res_gray[0][0]) if not np.isnan(res_gray[0][0]) else 0.0

                composite = 0.65 * edge_score + 0.35 * gray_score

                # Luminance color weighting (White piece vs Black piece)
                is_white_piece = symbol.isupper()
                if is_white_piece and mean_brightness < 95.0:
                    composite -= 0.50
                elif not is_white_piece and mean_brightness > 155.0:
                    composite -= 0.50

                if composite > best_score:
                    best_score = composite
                    best_symbol = symbol

        if best_symbol is None or best_score < 0.20:
            # Low confidence fallback - treat as empty or low confidence
            return None, 0.0

        piece = chess.Piece.from_symbol(best_symbol)
        confidence = max(0.0, min(1.0, (best_score + 0.2) / 1.2))
        return piece, confidence

    def classify_board(
        self,
        grid_slices: Dict[chess.Square, Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]],
    ) -> ClassificationResult:
        """Classifies all 64 squares, constructs placement FEN, and validates legal constraints."""
        piece_grid: Dict[chess.Square, Optional[chess.Piece]] = {}
        confidences: Dict[chess.Square, float] = {}

        for sq in chess.SQUARES:
            if sq in grid_slices:
                _, _, roi_bgr = grid_slices[sq]
                piece, conf = self.classify_square(roi_bgr, sq)
                piece_grid[sq] = piece
                confidences[sq] = conf
            else:
                piece_grid[sq] = None
                confidences[sq] = 0.0

        # Construct 64-square placement FEN string
        fen_ranks: list[str] = []
        for rank in range(7, -1, -1):
            empty_count = 0
            rank_str = ""
            for file in range(8):
                sq = chess.square(file, rank)
                p = piece_grid.get(sq)
                if p is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += p.symbol()
            if empty_count > 0:
                rank_str += str(empty_count)
            fen_ranks.append(rank_str)

        placement_fen = "/".join(fen_ranks)
        is_legal, val_err = self.validate_fen_legality(placement_fen, piece_grid)

        return ClassificationResult(
            piece_grid=piece_grid,
            placement_fen=placement_fen,
            confidences=confidences,
            is_legal=is_legal,
            validation_error=val_err,
        )

    @staticmethod
    def validate_fen_legality(
        placement_fen: str,
        piece_grid: Dict[chess.Square, Optional[chess.Piece]],
    ) -> Tuple[bool, Optional[str]]:
        """Validates fundamental chess legality constraints defensively."""
        white_kings = sum(1 for p in piece_grid.values() if p is not None and p.symbol() == "K")
        black_kings = sum(1 for p in piece_grid.values() if p is not None and p.symbol() == "k")

        if white_kings != 1:
            return False, f"Illegal position: White has {white_kings} kings (expected 1)"
        if black_kings != 1:
            return False, f"Illegal position: Black has {black_kings} kings (expected 1)"

        white_pieces = sum(1 for p in piece_grid.values() if p is not None and p.color == chess.WHITE)
        black_pieces = sum(1 for p in piece_grid.values() if p is not None and p.color == chess.BLACK)

        if white_pieces > 16:
            return False, f"Illegal position: White has {white_pieces} pieces (max 16)"
        if black_pieces > 16:
            return False, f"Illegal position: Black has {black_pieces} pieces (max 16)"

        # Pawns on 1st or 8th rank check
        for file in range(8):
            rank1_piece = piece_grid.get(chess.square(file, 0))
            if rank1_piece is not None and rank1_piece.piece_type == chess.PAWN:
                return False, f"Illegal position: Pawn on rank 1 at file {file}"
            rank8_piece = piece_grid.get(chess.square(file, 7))
            if rank8_piece is not None and rank8_piece.piece_type == chess.PAWN:
                return False, f"Illegal position: Pawn on rank 8 at file {file}"

        return True, None
