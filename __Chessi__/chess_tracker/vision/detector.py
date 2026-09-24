"""Computer vision detector with inner-square cropping, move highlight immunity, and delta tracking."""

from __future__ import annotations
from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple
import chess
import cv2
import numpy as np

from chess_tracker.config import (
    INNER_CROP_RATIO,
    CANNY_THRESH1,
    CANNY_THRESH2,
    DEBOUNCE_FRAMES_REQUIRED,
)

logger = logging.getLogger("ChessTracker.Detector")


@dataclass
class SquareFeature:
    square: chess.Square
    square_name: str
    row: int
    col: int
    is_light: bool
    crop_bgr: np.ndarray
    crop_gray: np.ndarray
    canny: np.ndarray
    edge_count: int
    mean_val: float
    std_val: float
    white_ratio: float
    dark_ratio: float
    is_occupied: bool
    classified_symbol: Optional[str] = None


class ChessBoardDetector:
    """Robust zero-DOM chess perception engine immune to move highlights and tile textures."""

    def __init__(
        self,
        reference_image_path: Optional[str] = None,
        inner_crop_ratio: float = INNER_CROP_RATIO,
        debounce_required: int = DEBOUNCE_FRAMES_REQUIRED,
    ) -> None:
        self.inner_crop_ratio = inner_crop_ratio
        self.debounce_required = debounce_required

        # Piece templates: symbol -> List of (norm_gray, canny_edge)
        self.templates: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self._load_or_extract_templates(reference_image_path)

        # Delta tracking state
        self.last_features: Optional[Dict[chess.Square, SquareFeature]] = None
        self._pending_move: Optional[chess.Move] = None
        self._pending_count: int = 0

    def _load_or_extract_templates(self, reference_path: Optional[str]) -> None:
        """Loads piece templates from reference board image."""
        paths_to_check = []
        if reference_path:
            paths_to_check.append(reference_path)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths_to_check.append(os.path.join(base_dir, "reference_images", "initial_board.png"))
        paths_to_check.append(os.path.join(base_dir, "reference_images", "reference_board.png"))

        resolved_path = None
        for p in paths_to_check:
            if os.path.isfile(p):
                resolved_path = p
                break

        if not resolved_path:
            logger.warning("No reference board image found; using synthetic edge templates.")
            return

        ref_img = cv2.imread(resolved_path)
        if ref_img is None:
            logger.warning("Could not read image at %s", resolved_path)
            return

        # Auto-detect or use standard board contour, then refine to actual grid
        from chess_tracker.vision.capture import detect_chessboard_contour, refine_board_grid
        box = detect_chessboard_contour(ref_img)
        if box:
            bx, by, bw, bh = box
            coarse_crop = ref_img[by : by + bh, bx : bx + bw]
        else:
            coarse_crop = ref_img

        # Pass 2: refine to actual tile grid (strips chrome like player labels)
        gx, gy, gw, gh = refine_board_grid(coarse_crop)
        board_crop = coarse_crop[gy : gy + gh, gx : gx + gw]

        self._extract_templates_from_crop(board_crop)

    def _extract_templates_from_crop(self, board_crop: np.ndarray) -> None:
        """Extracts piece templates for all 12 pieces from standard starting position."""
        bh, bw = board_crop.shape[:2]
        sq_w = bw / 8.0
        sq_h = bh / 8.0

        starting_coords = {
            "R": [(7, 0), (7, 7)],
            "N": [(7, 1), (7, 6)],
            "B": [(7, 2), (7, 5)],
            "Q": [(7, 3)],
            "K": [(7, 4)],
            "P": [(6, c) for c in range(8)],
            "r": [(0, 0), (0, 7)],
            "n": [(0, 1), (0, 6)],
            "b": [(0, 2), (0, 5)],
            "q": [(0, 3)],
            "k": [(0, 4)],
            "p": [(1, c) for c in range(8)],
        }

        for symbol, coords in starting_coords.items():
            tmpl_list = []
            for r, c in coords:
                x1 = int(round(c * sq_w))
                y1 = int(round(r * sq_h))
                x2 = int(round((c + 1) * sq_w))
                y2 = int(round((r + 1) * sq_h))
                sw, sh = x2 - x1, y2 - y1
                pad_x = int(round(sw * (1.0 - self.inner_crop_ratio) / 2.0))
                pad_y = int(round(sh * (1.0 - self.inner_crop_ratio) / 2.0))

                crop = board_crop[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
                if crop.size == 0:
                    continue
                resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                canny = cv2.Canny(norm_gray, CANNY_THRESH1, CANNY_THRESH2)
                tmpl_list.append((norm_gray, canny))

            if tmpl_list:
                self.templates[symbol] = tmpl_list
        logger.info("Extracted piece templates for %d piece symbols.", len(self.templates))

    def extract_square_features(
        self,
        board_bgr: np.ndarray,
        is_flipped: bool = False,
    ) -> Dict[chess.Square, SquareFeature]:
        """Extracts inner 60% features for all 64 squares from localized board image."""
        bh, bw = board_bgr.shape[:2]
        sq_w = bw / 8.0
        sq_h = bh / 8.0

        features: Dict[chess.Square, SquareFeature] = {}

        for r in range(8):
            for c in range(8):
                # Handle perspective: standard White bottom is row 0 = rank 8, col 0 = file a
                if is_flipped:
                    file_idx = 7 - c
                    rank_idx = r
                else:
                    file_idx = c
                    rank_idx = 7 - r

                sq = chess.square(file_idx, rank_idx)
                sq_name = chess.square_name(sq)
                is_light = (file_idx + rank_idx) % 2 == 1

                x1 = int(round(c * sq_w))
                y1 = int(round(r * sq_h))
                x2 = int(round((c + 1) * sq_w))
                y2 = int(round((r + 1) * sq_h))
                sw, sh = x2 - x1, y2 - y1

                # Inner-square cropping: sample inner 60%
                pad_x = int(round(sw * (1.0 - self.inner_crop_ratio) / 2.0))
                pad_y = int(round(sh * (1.0 - self.inner_crop_ratio) / 2.0))

                inner_crop = board_bgr[y1 + pad_y : y2 - pad_y, x1 + pad_x : x2 - pad_x]
                if inner_crop.size == 0:
                    inner_crop = board_bgr[y1:y2, x1:x2]

                resized = cv2.resize(inner_crop, (64, 64), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                canny = cv2.Canny(gray, CANNY_THRESH1, CANNY_THRESH2)

                edge_count = int(np.count_nonzero(canny))
                mean_val = float(np.mean(gray))
                std_val = float(np.std(gray))
                white_px = int(np.count_nonzero(gray > 170))
                dark_px = int(np.count_nonzero(gray < 90))
                total_px = float(gray.size)

                white_ratio = white_px / total_px
                dark_ratio = dark_px / total_px

                # Highlight-immune Occupancy Detection:
                # Highlight overlays (even saturated yellow/green) are smooth textures with std < 8 and edges < 25.
                # Actual pieces have distinct internal borders, silhouettes, and edge counts >= 35.
                is_occupied = (edge_count >= 35 and std_val >= 8.0)

                features[sq] = SquareFeature(
                    square=sq,
                    square_name=sq_name,
                    row=r,
                    col=c,
                    is_light=is_light,
                    crop_bgr=resized,
                    crop_gray=gray,
                    canny=canny,
                    edge_count=edge_count,
                    mean_val=mean_val,
                    std_val=std_val,
                    white_ratio=white_ratio,
                    dark_ratio=dark_ratio,
                    is_occupied=is_occupied,
                )

        return features

    def classify_square_symbol(
        self,
        feature: SquareFeature,
    ) -> Optional[str]:
        """Identifies piece symbol on an occupied square via contour & edge template matching."""
        if not feature.is_occupied:
            return None

        # Determine piece color:
        # White pieces have distinct bright body pixels (>170) or mean > 140
        is_white_piece = (feature.white_ratio > 0.16) or (feature.mean_val > 142.0)
        cand_symbols = ["P", "N", "B", "R", "Q", "K"] if is_white_piece else ["p", "n", "b", "r", "q", "k"]

        norm_gray = cv2.normalize(feature.crop_gray, None, 0, 255, cv2.NORM_MINMAX)
        sq_edges = feature.canny

        best_score = -999.0
        best_symbol = cand_symbols[0]

        for sym in cand_symbols:
            tmpls = self.templates.get(sym, [])
            for t_gray, t_edge in tmpls:
                res_edge = cv2.matchTemplate(sq_edges, t_edge, cv2.TM_CCOEFF_NORMED)[0][0]
                res_gray = cv2.matchTemplate(norm_gray, t_gray, cv2.TM_CCOEFF_NORMED)[0][0]
                if np.isnan(res_edge):
                    res_edge = 0.0
                if np.isnan(res_gray):
                    res_gray = 0.0

                combined = 0.65 * res_edge + 0.35 * res_gray
                if combined > best_score:
                    best_score = combined
                    best_symbol = sym

        return best_symbol

    def detect_move_delta(
        self,
        curr_features: Dict[chess.Square, SquareFeature],
        board: chess.Board,
    ) -> Tuple[Optional[chess.Move], float]:
        """Tracks structural deltas against the last stable baseline and cross-references with legal moves.

        Key design: the baseline (`last_features`) is a *sticky* snapshot of the last known stable
        board state.  It is NOT updated when a move candidate is detected — only when the board
        appears unchanged or after a move is confirmed via `process_frame`.  This prevents
        chess.com's piece-slide animation from eroding the delta signal across multiple frames.

        The occupancy gate is HARD: `from_empty AND to_occupied` must both hold.  This prevents
        click-selection highlights (where the piece is still on its square) from triggering false
        positives.  The occupancy thresholds themselves are relaxed (edge>=35, std>=8) to
        tolerate move-highlight overlays on truly empty/occupied squares.
        """
        if self.last_features is None:
            self.last_features = curr_features
            return None, 0.0

        prev_features = self.last_features

        # 1. Calculate structural differences for all 64 squares
        deltas: List[Tuple[float, chess.Square]] = []
        for sq in chess.SQUARES:
            f_prev = prev_features[sq]
            f_curr = curr_features[sq]

            edge_diff = float(np.sum(np.abs(f_curr.canny.astype(float) - f_prev.canny.astype(float)))) / 255.0
            gray_diff = float(np.mean(np.abs(f_curr.crop_gray.astype(float) - f_prev.crop_gray.astype(float))))
            occ_change = (f_curr.is_occupied != f_prev.is_occupied)

            # Combined delta score
            score = edge_diff + (gray_diff * 1.5) + (100.0 if occ_change else 0.0)
            deltas.append((score, sq))

        deltas.sort(key=lambda x: x[0], reverse=True)
        top_changed_squares = {sq for score, sq in deltas if score > 30.0}

        # If no significant change across any square, board is stable — update baseline
        if not top_changed_squares:
            self.last_features = curr_features
            return None, 0.0

        delta_map = dict(deltas)

        # 2. Match against legal moves with HARD occupancy gate
        best_candidate: Optional[chess.Move] = None
        best_score = -1.0

        for move in board.legal_moves:
            affected = {move.from_square, move.to_square}
            if board.is_castling(move):
                if move.to_square == chess.G1:
                    affected.update({chess.H1, chess.F1})
                elif move.to_square == chess.C1:
                    affected.update({chess.A1, chess.D1})
                elif move.to_square == chess.G8:
                    affected.update({chess.H8, chess.F8})
                elif move.to_square == chess.C8:
                    affected.update({chess.A8, chess.D8})
            elif board.is_en_passant(move):
                cap_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
                affected.add(cap_sq)

            # HARD occupancy gate: from_square must now be empty, to_square must be occupied.
            # This is the primary defense against click-selection false positives — clicking
            # a piece on chess.com highlights it but does NOT empty the source square.
            from_empty = not curr_features[move.from_square].is_occupied
            to_occupied = curr_features[move.to_square].is_occupied

            if not (from_empty and to_occupied):
                continue

            # Check overlap between affected squares and high-delta squares
            overlap_count = len(affected.intersection(top_changed_squares))
            if overlap_count == 0:
                continue

            # Delta energy from source and destination
            move_energy = delta_map.get(move.from_square, 0.0) + delta_map.get(move.to_square, 0.0)
            total_score = (overlap_count * 200.0) + move_energy

            if total_score > best_score:
                best_score = total_score
                best_candidate = move

        # STICKY BASELINE: Do NOT update last_features when a candidate is found.
        # This preserves the pre-move baseline so the next frame still produces a strong delta.
        if best_candidate is None or best_score < 100.0:
            # No valid move candidate — treat current frame as potential new stable state
            self.last_features = curr_features
            return None, 0.0

        return best_candidate, best_score

    def compute_occupancy_mismatch(
        self,
        features: Dict[chess.Square, SquareFeature],
        board: chess.Board,
    ) -> int:
        """Counts how many squares have visual occupancy that disagrees with the internal board.

        Used by the vision worker to detect board-state divergence and trigger self-correction.
        A high mismatch count sustained over multiple frames indicates a false positive was
        accepted and the internal state needs to be rolled back.
        """
        mismatch = 0
        for sq in chess.SQUARES:
            visual_occupied = features[sq].is_occupied
            board_occupied = board.piece_at(sq) is not None
            if visual_occupied != board_occupied:
                mismatch += 1
        return mismatch

    def process_frame(
        self,
        board_bgr: np.ndarray,
        board: chess.Board,
        is_flipped: bool = False,
    ) -> Tuple[Optional[chess.Move], Dict[chess.Square, SquareFeature]]:
        """Extracts features, evaluates move delta, and applies 3-frame debouncing.

        When a move candidate is detected, the baseline is frozen (sticky) until:
        - The candidate is confirmed (3 consecutive frames agree) → baseline advances, or
        - The candidate is abandoned (different/no candidate) → pending resets.
        """
        features = self.extract_square_features(board_bgr, is_flipped=is_flipped)
        candidate_move, score = self.detect_move_delta(features, board)

        confirmed_move: Optional[chess.Move] = None

        if candidate_move is not None:
            if candidate_move == self._pending_move:
                self._pending_count += 1
            else:
                self._pending_move = candidate_move
                self._pending_count = 1

            if self._pending_count >= self.debounce_required:
                confirmed_move = self._pending_move
                self._pending_move = None
                self._pending_count = 0
                # Advance the sticky baseline now that the move is confirmed
                self.last_features = features
                logger.info("Confirmed legal move via visual delta: %s (score=%.1f)", confirmed_move.uci(), score)
        else:
            # No candidate — hard reset pending state
            self._pending_move = None
            self._pending_count = 0

        return confirmed_move, features

    def reset_delta(self) -> None:
        """Resets tracking cache."""
        self.last_features = None
        self._pending_move = None
        self._pending_count = 0
