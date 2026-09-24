"""Configuration constants, geometry specifications, and engine defaults."""

from __future__ import annotations
import os
from typing import Tuple

# UI Geometry & Layout Constraints (Strictly 995 x 960 Window Budget)
WINDOW_WIDTH: int = 995
WINDOW_HEIGHT: int = 960
BOARD_DISPLAY_SIZE: int = 560
EVAL_BAR_WIDTH: int = 24
SIDEBAR_WIDTH: int = 375

# Board Visual Theme (Matches Chess.com Wood + Classic / Neo Style)
COLOR_LIGHT_SQUARE: str = "#D5B88A"  # Warm Birch / Beech wood
COLOR_DARK_SQUARE: str = "#9E683E"   # Warm Walnut wood
COLOR_HIGHLIGHT_MOVE: str = "rgba(247, 206, 50, 0.65)"  # Semi-transparent amber/yellow
COLOR_HIGHLIGHT_CHECK: str = "rgba(235, 60, 60, 0.85)"  # Crimson check glow
COLOR_HIGHLIGHT_SELECT: str = "rgba(100, 180, 246, 0.70)" # Selected square glow

COLOR_ARROW_ENGINE: str = "#00D2D3"  # Cyan neon arrow for engine recommendation
COLOR_ARROW_VISION: str = "#F59E0B"  # Amber neon arrow for detected opponent move

# Vision Perception Invariants
INNER_CROP_RATIO: float = 0.60       # Sample inner 60% to eliminate highlight border bleed
CANNY_THRESH1: int = 50
CANNY_THRESH2: int = 150
DEBOUNCE_FRAMES_REQUIRED: int = 3     # Three consecutive stable frames to confirm move
POLL_INTERVAL_MS: int = 150          # ~6.6 FPS screen polling rate

# Engine Settings & Detection Paths
DEFAULT_ENGINE_CANDIDATES = [
    r"C:\Users\leoba\Desktop\OldDesktop\Banksia\BanksiaGui-0.58-win64\bsg-engines\stockfish_15.1_x64_bmi2.exe",
    r"stockfish.exe",
    r"stockfish",
]

def find_stockfish_binary() -> str | None:
    """Locates an available Stockfish executable or returns None."""
    for path in DEFAULT_ENGINE_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    # Check PATH
    import shutil
    which_path = shutil.which("stockfish")
    if which_path:
        return which_path
    return None

STOCKFISH_PATH: str | None = find_stockfish_binary()
ENGINE_ANALYSIS_DEPTH: int = 14
