from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import chess
import cv2
import numpy as np
from PIL import Image, ImageGrab

try:
    import mss
    import mss.exception
    HAS_MSS: bool = True
except ImportError:
    HAS_MSS = False


class ScreenCapture:
    """Low-latency screen capture using mss with seamless fallback to PIL.ImageGrab."""

    def __init__(self) -> None:
        self._sct: Optional[Any] = None
        if HAS_MSS:
            try:
                factory = getattr(mss, "MSS", mss.mss)
                self._sct = factory()
            except Exception:
                self._sct = None

    def capture_full_screen(self, monitor_index: int = 1) -> np.ndarray:
        """Captures full display as a BGR numpy array."""
        if self._sct is not None:
            try:
                monitors = self._sct.monitors
                mon = monitors[monitor_index] if monitor_index < len(monitors) else monitors[0]
                sct_img = self._sct.grab(mon)
                # Convert BGRA to BGR
                return np.ascontiguousarray(np.array(sct_img)[:, :, :3])
            except Exception:
                pass

        # Fallback to ImageGrab
        pil_img = ImageGrab.grab(all_screens=True).convert("RGB")
        rgb = np.array(pil_img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def capture_region(self, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Captures a designated (left, top, width, height) bounding box.

        Returns BGR numpy array or None on failure.
        """
        left, top, width, height = bbox
        if width <= 0 or height <= 0:
            return None

        if self._sct is not None:
            try:
                monitor = {
                    "left": int(left),
                    "top": int(top),
                    "width": int(width),
                    "height": int(height),
                }
                sct_img = self._sct.grab(monitor)
                return np.ascontiguousarray(np.array(sct_img)[:, :, :3])
            except Exception:
                pass

        try:
            bbox_box = (int(left), int(top), int(left + width), int(top + height))
            pil_img = ImageGrab.grab(bbox=bbox_box, all_screens=True).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def close(self) -> None:
        """Releases screen capture resources."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def detect_chessboard_contour(
    frame: np.ndarray,
    min_area_ratio: float = 0.05,
) -> Optional[Tuple[int, int, int, int]]:
    """Automatically localizes a chessboard contour in the captured display/window frame.

    Finds the largest high-contrast square contour with aspect ratio ~1.0.
    Returns (x, y, w, h) bounding box on success, or None if not found.
    """
    h, w = frame.shape[:2]
    total_area = float(w * h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)

    # Dilate slightly to connect segmented board borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box: Optional[Tuple[int, int, int, int]] = None
    max_area: float = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue

        aspect = bw / float(bh)
        area = cv2.contourArea(cnt)

        # Board must be reasonably large, square, and occupy significant screen area
        if area > (total_area * min_area_ratio) and 0.88 <= aspect <= 1.14:
            if area > max_area:
                max_area = area
                best_box = (bx, by, bw, bh)

    return best_box


SlicedCell = Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]


def slice_board_grid(
    board_crop: np.ndarray,
    is_flipped: bool = False,
    outer_margin_pct: float = 0.0,
    inner_inset_pct: float = 0.12,
) -> Dict[chess.Square, Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]]:
    """Divides the localized board into an exact 8x8 grid with inward-cell padding.

    Returns mapping of chess.Square -> ((x1, y1, x2, y2), full_cell_bgr, inset_roi_bgr).
    """
    bh, bw = board_crop.shape[:2]

    margin_x = int(round(bw * max(0.0, min(15.0, outer_margin_pct)) / 100.0))
    margin_y = int(round(bh * max(0.0, min(15.0, outer_margin_pct)) / 100.0))

    active_x = margin_x
    active_y = margin_y
    active_w = max(16, bw - 2 * margin_x)
    active_h = max(16, bh - 2 * margin_y)

    sq_w = active_w / 8.0
    sq_h = active_h / 8.0

    grid_slices: Dict[chess.Square, Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]] = {}

    for row in range(8):
        for col in range(8):
            # Chessboard coordinate mapping
            rank = row if is_flipped else (7 - row)
            file = (7 - col) if is_flipped else col
            sq = chess.square(file, rank)

            x1 = active_x + int(round(col * sq_w))
            y1 = active_y + int(round(row * sq_h))
            x2 = active_x + int(round((col + 1) * sq_w))
            y2 = active_y + int(round((row + 1) * sq_h))

            # Bound within crop dimensions
            x1 = max(0, min(bw - 1, x1))
            x2 = max(x1 + 1, min(bw, x2))
            y1 = max(0, min(bh - 1, y1))
            y2 = max(y1 + 1, min(bh, y2))

            cell_img = board_crop[y1:y2, x1:x2]
            ch, cw = cell_img.shape[:2]

            # Inward cell padding to exclude borders and highlight fringes
            pad_x = int(round(cw * max(0.02, min(0.25, inner_inset_pct))))
            pad_y = int(round(ch * max(0.02, min(0.25, inner_inset_pct))))

            roi_img = cell_img[pad_y : max(pad_y + 1, ch - pad_y), pad_x : max(pad_x + 1, cw - pad_x)]
            if roi_img.size == 0:
                roi_img = cell_img

            grid_slices[sq] = ((x1, y1, x2, y2), cell_img, roi_img)

    return grid_slices
