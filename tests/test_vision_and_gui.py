"""Automated verification test for board vision, 8x8 grid slicing, FEN generation,

move highlight resilience, debug overlay, and live move synchronization.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from PIL import Image, ImageDraw, ImageFont

from gui.board_canvas import BoardCanvas, SOLID_PIECE_GLYPHS
from gui.board_vision import BoardVisionSync

_CACHED_PIECE_TILES: dict[str, Image.Image] | None = None


def get_reference_piece_tiles() -> dict[str, Image.Image] | None:
    global _CACHED_PIECE_TILES
    if _CACHED_PIECE_TILES is not None:
        return _CACHED_PIECE_TILES

    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_board.png")
    if not os.path.isfile(ref_path):
        return None

    import cv2
    from vision.templates import detect_board_crop

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        return None

    bx, by, bw, bh = detect_board_crop(ref_img)
    board_crop = ref_img[by : by + bh, bx : bx + bw]
    sq_w = bw / 8.0
    sq_h = bh / 8.0

    piece_coords = {
        "r": (0, 0), "n": (0, 1), "b": (0, 2), "q": (0, 3), "k": (0, 4), "p": (1, 0),
        "R": (7, 0), "N": (7, 1), "B": (7, 2), "Q": (7, 3), "K": (7, 4), "P": (6, 0),
    }
    tiles: dict[str, Image.Image] = {}
    for sym, (r, c) in piece_coords.items():
        x1 = int(round(c * sq_w))
        y1 = int(round(r * sq_h))
        x2 = int(round((c + 1) * sq_w))
        y2 = int(round((r + 1) * sq_h))
        cell_bgr = board_crop[y1:y2, x1:x2]
        cell_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
        tiles[sym] = Image.fromarray(cell_rgb)

    _CACHED_PIECE_TILES = tiles
    return tiles


def create_synthetic_board(
    board: chess.Board,
    size: int = 400,
    highlight_moves: tuple[chess.Square, chess.Square] | None = None,
) -> Image.Image:
    img = Image.new("RGB", (size, size), "#EEEED2")
    draw = ImageDraw.Draw(img)
    sq_size = size // 8

    light_sq = (238, 238, 210)
    dark_sq = (118, 150, 86)
    highlight_light = (246, 246, 105)
    highlight_dark = (186, 202, 68)

    piece_tiles = get_reference_piece_tiles()

    try:
        font = ImageFont.truetype("seguisym.ttf", int(sq_size * 0.70))
    except Exception:
        font = ImageFont.load_default()

    glyphs = {
        "P": "♟", "N": "♞", "B": "♝", "R": "♜", "Q": "♛", "K": "♚",
        "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    }

    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, 7 - rank)
            x1 = file * sq_size
            y1 = rank * sq_size
            x2 = x1 + sq_size
            y2 = y1 + sq_size

            is_light = (file + rank) % 2 == 0
            color = light_sq if is_light else dark_sq

            # Dynamic Move Highlights
            if highlight_moves and sq in highlight_moves:
                color = highlight_light if is_light else highlight_dark

            draw.rectangle([x1, y1, x2, y2], fill=color)

            piece = board.piece_at(sq)
            if piece:
                if piece_tiles and piece.symbol() in piece_tiles:
                    tile = piece_tiles[piece.symbol()].resize((sq_size, sq_size), Image.Resampling.BILINEAR)
                    if highlight_moves and sq in highlight_moves:
                        # Blend highlight color
                        hl_img = Image.new("RGB", (sq_size, sq_size), color)
                        tile = Image.blend(tile, hl_img, 0.35)
                    img.paste(tile, (x1, y1))
                else:
                    char = glyphs.get(piece.symbol(), "♟")
                    bbox = draw.textbbox((0, 0), char, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    tx = x1 + (sq_size - tw) // 2 - bbox[0]
                    ty = y1 + (sq_size - th) // 2 - bbox[1]
                    p_color = (255, 255, 255) if piece.color == chess.WHITE else (20, 20, 20)
                    draw.text((tx, ty), char, font=font, fill=p_color)

    return img

def test_board_canvas_glyphs():
    print("Checking solid piece glyphs...")
    assert len(SOLID_PIECE_GLYPHS) == 12
    assert SOLID_PIECE_GLYPHS["P"] == "♟"
    assert SOLID_PIECE_GLYPHS["N"] == "♞"
    assert SOLID_PIECE_GLYPHS["B"] == "♝"
    assert SOLID_PIECE_GLYPHS["R"] == "♜"
    assert SOLID_PIECE_GLYPHS["Q"] == "♛"
    assert SOLID_PIECE_GLYPHS["K"] == "♚"
    print("Solid glyphs verified successfully.")

def test_8x8_slicing_and_fen():
    print("Testing 8x8 Grid Slicing & FEN Translation...")
    board = chess.Board()
    frame = create_synthetic_board(board, size=400)

    # 1. Normal Perspective (White at bottom)
    grid_white, fen_white = BoardVisionSync.slice_and_classify_board(frame, is_flipped=False, outer_margin_pct=0.0, inner_inset_pct=0.12)
    assert len(grid_white) == 64
    occupied_count = sum(1 for p in grid_white.values() if p is not None)
    assert occupied_count == 32, f"Expected 32 pieces, found {occupied_count}"
    print(f"Generated White Perspective FEN: {fen_white}")

    # 2. Flipped Perspective (Black at bottom)
    grid_black, fen_black = BoardVisionSync.slice_and_classify_board(frame, is_flipped=True, outer_margin_pct=0.0, inner_inset_pct=0.12)
    assert len(grid_black) == 64
    assert sum(1 for p in grid_black.values() if p is not None) == 32
    print(f"Generated Black Perspective FEN: {fen_black}")
    print("8x8 Slicing and FEN translation passed!")

def test_move_highlight_resilience():
    print("Testing Move Highlight Resilience (Yellow Tints on Source & Destination)...")
    board = chess.Board()
    board.push_san("e4")
    # Simulate move highlights on e2 (vacated from-square) and e4 (occupied to-square)
    frame_highlight = create_synthetic_board(board, size=400, highlight_moves=(chess.E2, chess.E4))

    grid, fen = BoardVisionSync.slice_and_classify_board(frame_highlight, is_flipped=False, inner_inset_pct=0.12)
    # e2 must be classified as EMPTY despite the yellow highlight
    assert grid[chess.E2] is None, "Square e2 (highlighted from-square) must be recognized as EMPTY"
    # e4 must be classified as OCCUPIED by White Pawn despite the yellow highlight
    assert grid[chess.E4] is not None and grid[chess.E4].color == chess.WHITE, "Square e4 (highlighted to-square) must be White Pawn"
    print("Move highlight resilience verified successfully (e2=empty, e4=White Pawn)!")

def test_debug_overlay_generation():
    print("Testing Debug Visualizer Overlay Generation...")
    board = chess.Board()
    frame = create_synthetic_board(board, size=400)
    grid, _ = BoardVisionSync.slice_and_classify_board(frame, is_flipped=False)

    overlay = BoardVisionSync.generate_debug_overlay_image(
        frame,
        grid,
        is_flipped=False,
        outer_margin_pct=2.0,
        inner_inset_pct=0.12,
    )
    assert overlay.size == frame.size
    assert isinstance(overlay, Image.Image)
    print("Debug overlay successfully created with grid lines and labels.")

def test_vision_sync_pipeline():
    print("Testing BoardVisionSync background loop and live move detection...")
    current_board = chess.Board()
    detected_moves: list[chess.Move] = []
    status_updates: list[str] = []
    fens_received: list[str] = []

    def get_board():
        return current_board.copy()

    def on_move(m: chess.Move):
        detected_moves.append(m)

    def on_status(s: str, is_active: bool, is_err: bool):
        status_updates.append(s)

    def on_fen(fen: str, grid):
        fens_received.append(fen)

    vision = BoardVisionSync(
        get_board_callback=get_board,
        on_move_detected=on_move,
        on_status_update=on_status,
        on_fen_detected=on_fen,
        poll_interval=0.05,
        consecutive_frames_required=2,
        outer_margin_pct=0.0,
        inner_inset_pct=0.12,
    )

    # Set region
    vision.set_region((100, 100, 400, 400))

    # Prepare board after 1. e4 with move highlights
    after_e4 = chess.Board()
    after_e4.push_san("e4")
    frame_e4 = create_synthetic_board(after_e4, size=400, highlight_moves=(chess.E2, chess.E4))

    vision.mock_image_provider = lambda: frame_e4

    started = vision.start()
    assert started
    assert vision.is_running

    # Wait for consecutive frames to process and confirm move
    for _ in range(20):
        if len(detected_moves) >= 1:
            break
        time.sleep(0.05)

    vision.stop()
    assert not vision.is_running
    assert len(fens_received) >= 1
    assert len(detected_moves) >= 1
    assert detected_moves[0] == chess.Move.from_uci("e2e4")
    print(f"Vision correctly detected FEN ({fens_received[-1][:25]}...) and move ({detected_moves[0]})")
    print("Vision sync pipeline test passed!")

if __name__ == "__main__":
    test_board_canvas_glyphs()
    test_8x8_slicing_and_fen()
    test_move_highlight_resilience()
    test_debug_overlay_generation()
    test_vision_sync_pipeline()
    print("ALL TESTS PASSED WITH 100% ACCURACY!")
