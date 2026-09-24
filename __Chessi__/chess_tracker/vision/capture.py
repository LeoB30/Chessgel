"""Zero-DOM screen capture using mss and robust chessboard grid detection.

The detection pipeline uses a two-pass strategy:
  Pass 1 (Contour): finds the largest near-square contour (catches the board or its UI wrapper).
  Pass 2 (Wood-HSV): within a captured region, isolates warm wood-tone pixels via HSV mask
         to find the ACTUAL 8x8 tile grid, stripping chrome (player labels, nav buttons, eval bar).

This ensures the returned bounding box always maps to the 8x8 playing surface regardless of how
much surrounding UI is captured in the ROI.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import cv2
import mss
import numpy as np

logger = logging.getLogger("ChessTracker.Capture")


class ScreenCapture:
    """Zero-DOM screen capture interface utilizing mss."""

    def __init__(self) -> None:
        self._sct: Optional[mss.mss] = None

    def _ensure_context(self) -> mss.mss:
        """Lazily creates / recreates the mss context if it was closed or never created."""
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def capture_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Captures localized desktop region and returns standard BGR image."""
        sct = self._ensure_context()
        monitor = {
            "top": int(y),
            "left": int(x),
            "width": int(width),
            "height": int(height),
        }
        try:
            raw = sct.grab(monitor)
        except Exception:
            # Recreate context on transient GDI failures and retry once
            self._sct = mss.mss()
            raw = self._sct.grab(monitor)
        frame = np.array(raw, dtype=np.uint8)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def capture_full_screen(self, monitor_idx: int = 1) -> np.ndarray:
        """Captures designated display (1 = primary monitor)."""
        sct = self._ensure_context()
        monitors = sct.monitors
        idx = min(monitor_idx, len(monitors) - 1)
        try:
            raw = sct.grab(monitors[idx])
        except Exception:
            self._sct = mss.mss()
            raw = self._sct.grab(self._sct.monitors[idx])
        frame = np.array(raw, dtype=np.uint8)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def close(self) -> None:
        """Closes mss context."""
        try:
            if self._sct is not None:
                self._sct.close()
                self._sct = None
        except Exception:
            self._sct = None


# ---------------------------------------------------------------------------
# Board Detection (Two-Pass: Contour → Wood-HSV Refinement)
# ---------------------------------------------------------------------------

def detect_chessboard_contour(
    image_bgr: np.ndarray,
    min_area_ratio: float = 0.12,
    max_aspect_deviation: float = 0.12,
) -> Optional[Tuple[int, int, int, int]]:
    """Pass 1: Detects the largest near-square contour on the screen.

    This may capture the board *and* surrounding chrome (player labels, buttons).
    Always refine the result through `refine_board_grid()` before dividing into 8×8.

    Returns:
        (x, y, width, height) bounding box or None if no valid candidate found.
    """
    h, w = image_bgr.shape[:2]
    total_area = float(w * h)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box: Optional[Tuple[int, int, int, int]] = None
    max_area: float = 0.0

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0 or bw == 0:
            continue
        aspect = bw / float(bh)
        area = cv2.contourArea(cnt)
        if (1.0 - max_aspect_deviation) <= aspect <= (1.0 + max_aspect_deviation):
            if area > (total_area * min_area_ratio) and area > max_area:
                max_area = area
                best_box = (bx, by, bw, bh)

    return best_box


def refine_board_grid(
    roi_bgr: np.ndarray,
    min_area_ratio: float = 0.30,
) -> Tuple[int, int, int, int]:
    """Pass 2: Finds the actual 8×8 tile grid within a captured ROI using wood-tone HSV isolation.

    Chess boards (chess.com, lichess, etc.) use wooden tile colors in the warm brown / tan
    hue range. UI chrome (player labels, navigation buttons, dark backgrounds) falls well
    outside this range. By masking wood tones and finding the bounding rectangle of the
    largest connected component, we precisely locate the playing surface.

    If no wood region is found, falls back to the full ROI dimensions.

    Returns:
        (x, y, width, height) relative to the ROI top-left corner.
    """
    h, w = roi_bgr.shape[:2]
    total_area = float(w * h)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Wood tone range 1: darker brown tiles  (Hue 8-35, moderate-high saturation)
    lower_dark_wood = np.array([8, 25, 50])
    upper_dark_wood = np.array([35, 200, 250])
    mask_dark = cv2.inRange(hsv, lower_dark_wood, upper_dark_wood)

    # Wood tone range 2: lighter beige / birch tiles  (Hue 15-40, lower saturation, high value)
    lower_light_wood = np.array([15, 10, 150])
    upper_light_wood = np.array([40, 120, 255])
    mask_light = cv2.inRange(hsv, lower_light_wood, upper_light_wood)

    combined_mask = cv2.bitwise_or(mask_dark, mask_light)

    # Morphological cleanup to merge nearby tile pixels into one blob
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box: Optional[Tuple[int, int, int, int]] = None
    max_area: float = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0 or bw == 0:
            continue
        aspect = bw / float(bh)
        # Must be roughly square and cover a significant portion of the ROI
        if 0.85 <= aspect <= 1.18 and area > (total_area * min_area_ratio) and area > max_area:
            max_area = area
            best_box = (bx, by, bw, bh)

    if best_box is not None:
        bx, by, bw, bh = best_box
        # Force the result to be perfectly square using the smaller dimension
        side = min(bw, bh)
        # Center the square crop within the detected rectangle
        cx = bx + bw // 2
        cy = by + bh // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x0 = min(x0, w - side)
        y0 = min(y0, h - side)
        logger.debug("Wood-HSV refined board grid: (%d, %d, %d, %d) within %dx%d ROI", x0, y0, side, side, w, h)
        return (x0, y0, side, side)

    # Fallback: use the full ROI
    logger.debug("Wood-HSV refinement found no grid; using full ROI (%d, %d)", w, h)
    return (0, 0, w, h)


def crop_board_from_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """Convenience: applies `refine_board_grid` and returns the cropped board image."""
    x, y, w, h = refine_board_grid(roi_bgr)
    return roi_bgr[y : y + h, x : x + w]
