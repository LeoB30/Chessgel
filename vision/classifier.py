from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import chess
import cv2
import numpy as np
from vision.templates import TemplateStore, load_or_extract_templates


@dataclass(frozen=True)
class ClassificationResult:
    """Encapsulates 64-square piece detection, placement FEN, and confidence scores."""
    piece_grid: Dict[chess.Square, Optional[chess.Piece]]
    placement_fen: str
    confidences: Dict[chess.Square, float]
    is_legal: bool
    validation_error: Optional[str] = None


class BoardClassifier:
    """Two-stage chessboard classifier: dynamic light/dark occupancy filter + contour edge matching."""

    def __init__(
        self,
        template_store: Optional[TemplateStore] = None,
        occupancy_var_light: float = 160.0,
        occupancy_var_dark: float = 190.0,
        occupancy_edge_light: int = 90,
        occupancy_edge_dark: int = 110,
        canny_thresh1: int = 50,
        canny_thresh2: int = 150,
    ) -> None:
        if template_store is None:
            self.template_store: TemplateStore = load_or_extract_templates()
        else:
            self.template_store = template_store

        self.occupancy_var_light: float = occupancy_var_light
        self.occupancy_var_dark: float = occupancy_var_dark
        self.occupancy_edge_light: int = occupancy_edge_light
        self.occupancy_edge_dark: int = occupancy_edge_dark
        self.canny_thresh1: int = canny_thresh1
        self.canny_thresh2: int = canny_thresh2
        self.target_size: Tuple[int, int] = self.template_store.target_size

    def is_light_square(self, square: chess.Square) -> bool:
        """Determines whether a square is light or dark (file + rank % 2 == 1 for light)."""
        file_idx: int = chess.square_file(square)
        rank_idx: int = chess.square_rank(square)
        return (file_idx + rank_idx) % 2 == 1

    def classify_square(
        self,
        roi_bgr: np.ndarray,
        square: chess.Square,
    ) -> Tuple[Optional[chess.Piece], float]:
        """Classifies an individual square ROI using dynamic occupancy thresholds and contour matching."""
        gray: np.ndarray = (
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
        )
        h, w = gray.shape[:2]
        if h != self.target_size[1] or w != self.target_size[0]:
            gray_scaled: np.ndarray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            gray_scaled = gray

        # Stage 1: Dynamic Occupancy Filter (differentiating light vs dark square textures)
        is_light: bool = self.is_light_square(square)
        var_thresh: float = self.occupancy_var_light if is_light else self.occupancy_var_dark
        edge_thresh: int = self.occupancy_edge_light if is_light else self.occupancy_edge_dark

        variance: float = float(np.var(gray_scaled))
        raw_edges: np.ndarray = cv2.Canny(gray_scaled, self.canny_thresh1, self.canny_thresh2)
        edge_count: int = int(np.count_nonzero(raw_edges))

        if variance < var_thresh and edge_count < edge_thresh:
            confidence: float = max(0.85, 1.0 - (variance / (2.0 * var_thresh)))
            return None, confidence

        # Stage 2: Edge/Contour Matching using Canny silhouettes (immune to wood grain and move highlights)
        norm_gray: np.ndarray = cv2.normalize(gray_scaled, None, 0, 255, cv2.NORM_MINMAX)
        sq_edges: np.ndarray = cv2.Canny(norm_gray, self.canny_thresh1, self.canny_thresh2)
        mean_brightness: float = float(np.mean(gray_scaled))

        best_score: float = -999.0
        best_symbol: Optional[str] = None

        for symbol in self.template_store.symbols():
            piece_tmpl = self.template_store.get(symbol)
            if piece_tmpl is None:
                continue

            for cand_gray, cand_edge in zip(piece_tmpl.gray_templates, piece_tmpl.edge_templates):
                # Contour silhouette matching
                res_edge: np.ndarray = cv2.matchTemplate(sq_edges, cand_edge, cv2.TM_CCOEFF_NORMED)
                edge_score: float = float(res_edge[0][0]) if not np.isnan(res_edge[0][0]) else 0.0

                # Grayscale normalized match
                res_gray: np.ndarray = cv2.matchTemplate(norm_gray, cand_gray, cv2.TM_CCOEFF_NORMED)
                gray_score: float = float(res_gray[0][0]) if not np.isnan(res_gray[0][0]) else 0.0

                composite: float = 0.65 * edge_score + 0.35 * gray_score

                # Luminance polarity weighting
                is_white_piece: bool = symbol.isupper()
                if is_white_piece and mean_brightness < 95.0:
                    composite -= 0.50
                elif not is_white_piece and mean_brightness > 155.0:
                    composite -= 0.50

                if composite > best_score:
                    best_score = composite
                    best_symbol = symbol

        if best_symbol is None or best_score < 0.20:
            return None, 0.0

        piece: chess.Piece = chess.Piece.from_symbol(best_symbol)
        confidence = max(0.0, min(1.0, (best_score + 0.2) / 1.2))
        return piece, confidence

    def classify_board(
        self,
        grid_slices: Dict[chess.Square, Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]],
    ) -> ClassificationResult:
        """Classifies all 64 squares, constructs placement FEN, and checks legality."""
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

        # Construct placement FEN string (ranks 8 down to 1)
        fen_ranks: list[str] = []
        for rank in range(7, -1, -1):
            empty_count: int = 0
            rank_str: str = ""
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

        placement_fen: str = "/".join(fen_ranks)
        is_legal, val_err = self.validate_fen_legality(piece_grid)

        return ClassificationResult(
            piece_grid=piece_grid,
            placement_fen=placement_fen,
            confidences=confidences,
            is_legal=is_legal,
            validation_error=val_err,
        )

    @staticmethod
    def validate_fen_legality(
        piece_grid: Dict[chess.Square, Optional[chess.Piece]],
    ) -> Tuple[bool, Optional[str]]:
        """Validates fundamental chess legality rules defensively."""
        white_kings: int = sum(1 for p in piece_grid.values() if p is not None and p.symbol() == "K")
        black_kings: int = sum(1 for p in piece_grid.values() if p is not None and p.symbol() == "k")

        if white_kings != 1:
            return False, f"Illegal position: White has {white_kings} kings (expected exactly 1)"
        if black_kings != 1:
            return False, f"Illegal position: Black has {black_kings} kings (expected exactly 1)"

        white_pieces: int = sum(1 for p in piece_grid.values() if p is not None and p.color == chess.WHITE)
        black_pieces: int = sum(1 for p in piece_grid.values() if p is not None and p.color == chess.BLACK)

        if white_pieces > 16:
            return False, f"Illegal position: White has {white_pieces} pieces (maximum 16)"
        if black_pieces > 16:
            return False, f"Illegal position: Black has {black_pieces} pieces (maximum 16)"

        # Pawns on 1st or 8th rank are invalid in standard chess
        for file in range(8):
            rank1_p = piece_grid.get(chess.square(file, 0))
            if rank1_p is not None and rank1_p.piece_type == chess.PAWN:
                return False, f"Illegal position: Pawn on rank 1 at file {file}"
            rank8_p = piece_grid.get(chess.square(file, 7))
            if rank8_p is not None and rank8_p.piece_type == chess.PAWN:
                return False, f"Illegal position: Pawn on rank 8 at file {file}"

        return True, None
