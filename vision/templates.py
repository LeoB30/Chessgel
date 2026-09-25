from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class PieceTemplate:
    """Stores normalized grayscale and Canny edge templates for a specific chess piece."""
    symbol: str
    gray_templates: tuple[np.ndarray, ...]
    edge_templates: tuple[np.ndarray, ...]


class TemplateStore:
    """Manages extracted piece templates and handles access by FEN piece symbol."""

    def __init__(
        self,
        templates: Dict[str, PieceTemplate],
        target_size: Tuple[int, int] = (64, 64),
        roi_ratio: float = 0.75,
    ) -> None:
        self.templates: Dict[str, PieceTemplate] = templates
        self.target_size: Tuple[int, int] = target_size
        self.roi_ratio: float = roi_ratio

    def get(self, symbol: str) -> Optional[PieceTemplate]:
        """Retrieves template by piece symbol ('P', 'n', etc.)."""
        return self.templates.get(symbol)

    def symbols(self) -> List[str]:
        """Returns all registered piece symbols."""
        return list(self.templates.keys())


def detect_board_crop(
    image: np.ndarray,
    min_area_ratio: float = 0.20,
) -> Tuple[int, int, int, int]:
    """Detects the largest high-contrast square chessboard contour in the image.

    Returns bounding box (x, y, w, h). If not found, falls back to full image bounds.
    """
    h, w = image.shape[:2]
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred: np.ndarray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges: np.ndarray = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box: Optional[Tuple[int, int, int, int]] = None
    max_area: float = 0.0
    total_area: float = float(w * h)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        aspect: float = bw / float(bh)
        area: float = cv2.contourArea(cnt)
        if area > (total_area * min_area_ratio) and 0.85 <= aspect <= 1.15:
            if area > max_area:
                max_area = area
                best_box = (bx, by, bw, bh)

    if best_box is not None:
        return best_box
    return (0, 0, w, h)


def crop_square_roi(
    board_crop: np.ndarray,
    row: int,
    col: int,
    roi_ratio: float = 0.75,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Crops the central 70-75% ROI of an individual square, excluding tile borders and coords."""
    bh, bw = board_crop.shape[:2]
    sq_w: float = bw / 8.0
    sq_h: float = bh / 8.0

    x1: int = int(round(col * sq_w))
    y1: int = int(round(row * sq_h))
    x2: int = int(round((col + 1) * sq_w))
    y2: int = int(round((row + 1) * sq_h))

    square: np.ndarray = board_crop[y1:y2, x1:x2]
    sh, sw = square.shape[:2]
    if sh < 4 or sw < 4:
        raise ValueError(f"Square slice too small at row={row}, col={col}: ({sw}x{sh})")

    pad_x: int = int(round(sw * (1.0 - roi_ratio) / 2.0))
    pad_y: int = int(round(sh * (1.0 - roi_ratio) / 2.0))

    roi: np.ndarray = square[pad_y : sh - pad_y, pad_x : sw - pad_x]
    if roi.size == 0:
        roi = square

    return cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)


def extract_templates_from_board(
    reference_image: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    roi_ratio: float = 0.75,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
) -> TemplateStore:
    """Extracts Canny edge silhouettes and normalized grayscale profiles for all 12 pieces."""
    bx, by, bw, bh = detect_board_crop(reference_image)
    board_crop: np.ndarray = reference_image[by : by + bh, bx : bx + bw]

    piece_starting_positions: Dict[str, List[Tuple[int, int]]] = {
        "r": [(0, 0), (0, 7)],
        "n": [(0, 1), (0, 6)],
        "b": [(0, 2), (0, 5)],
        "q": [(0, 3)],
        "k": [(0, 4)],
        "p": [(1, c) for c in range(8)],
        "R": [(7, 0), (7, 7)],
        "N": [(7, 1), (7, 6)],
        "B": [(7, 2), (7, 5)],
        "Q": [(7, 3)],
        "K": [(7, 4)],
        "P": [(6, c) for c in range(8)],
    }

    templates: Dict[str, PieceTemplate] = {}

    for symbol, coords in piece_starting_positions.items():
        gray_list: List[np.ndarray] = []
        edge_list: List[np.ndarray] = []

        for r, c in coords:
            roi = crop_square_roi(board_crop, r, c, roi_ratio=roi_ratio, target_size=target_size)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            edges = cv2.Canny(norm_gray, canny_thresh1, canny_thresh2)

            gray_list.append(norm_gray)
            edge_list.append(edges)

        templates[symbol] = PieceTemplate(
            symbol=symbol,
            gray_templates=tuple(gray_list),
            edge_templates=tuple(edge_list),
        )

    return TemplateStore(templates=templates, target_size=target_size, roi_ratio=roi_ratio)


_CACHED_STORE: Optional[TemplateStore] = None


def load_or_extract_templates(
    reference_path: Optional[str] = None,
    target_size: Tuple[int, int] = (64, 64),
    roi_ratio: float = 0.75,
) -> TemplateStore:
    """Loads templates from reference_board.png or returns existing cached store."""
    global _CACHED_STORE
    if _CACHED_STORE is not None:
        return _CACHED_STORE

    candidate_paths: List[str] = []
    if reference_path:
        candidate_paths.append(reference_path)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths.append(os.path.join(script_dir, "reference_board.png"))
    candidate_paths.append("reference_board.png")

    resolved_path: Optional[str] = None
    for cp in candidate_paths:
        if os.path.isfile(cp):
            resolved_path = cp
            break

    if resolved_path is None:
        raise FileNotFoundError(
            f"Reference board image not found. Checked: {candidate_paths}"
        )

    img = cv2.imread(resolved_path)
    if img is None:
        raise ValueError(f"Unable to read image at {resolved_path}")

    _CACHED_STORE = extract_templates_from_board(
        img,
        target_size=target_size,
        roi_ratio=roi_ratio,
    )
    return _CACHED_STORE
