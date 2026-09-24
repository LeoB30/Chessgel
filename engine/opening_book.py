from __future__ import annotations
import random
import chess

# EPD -> list of (uci_move_str, weight)
OpeningBookDict = dict[str, list[tuple[str, int]]]

OPENING_BOOK: OpeningBookDict = {}


def register_move(board: chess.Board, move_uci: str, weight: int = 1) -> None:
    """Registers a single UCI move for the given board position with a weight."""
    epd_key: str = board.epd()
    if epd_key not in OPENING_BOOK:
        OPENING_BOOK[epd_key] = []

    # If the move is already in the list, update weight, else append
    for idx, (m, w) in enumerate(OPENING_BOOK[epd_key]):
        if m == move_uci:
            OPENING_BOOK[epd_key][idx] = (m, max(w, weight))
            return

    OPENING_BOOK[epd_key].append((move_uci, weight))


def register_line(move_sequence: list[str], weight: int = 1) -> None:
    """Registers an entire line of moves (in UCI or SAN notation).

    Walks a board from the start position and adds each move to the book.
    """
    board: chess.Board = chess.Board()
    for move_str in move_sequence:
        move_str = move_str.strip()
        if not move_str:
            continue

        try:
            # Try UCI first (e.g. 'e2e4', 'g1f3')
            move: chess.Move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                move = board.parse_san(move_str)
        except ValueError:
            # Fall back to SAN (e.g. 'e4', 'Nf6', 'O-O')
            move = board.parse_san(move_str)

        register_move(board, move.uci(), weight=weight)
        board.push(move)


def get_book_move(board: chess.Board) -> chess.Move | None:
    """Retrieves a weighted random move from the opening book if the position exists.

    Returns None if no book move is found or no candidate is legal.
    """
    epd_key: str = board.epd()
    candidates: list[tuple[str, int]] | None = OPENING_BOOK.get(epd_key)
    if not candidates:
        return None

    valid_moves: list[chess.Move] = []
    weights: list[int] = []

    for move_uci, weight in candidates:
        try:
            m: chess.Move = chess.Move.from_uci(move_uci)
            if m in board.legal_moves:
                valid_moves.append(m)
                weights.append(weight)
        except ValueError:
            continue

    if not valid_moves:
        return None

    # Weighted random selection
    chosen_move: chess.Move = random.choices(valid_moves, weights=weights, k=1)[0]
    return chosen_move


# =====================================================================
# DEFAULT OPENING REPERTOIRE
# =====================================================================

def init_default_book() -> None:
    """Populates opening repertoire with common lines including Caro-Kann and Catalan."""
    OPENING_BOOK.clear()

    # --- 1. Start Position Choices ---
    # 1. e4 (weight 10), 1. d4 (weight 10), 1. c4 (weight 4), 1. Nf3 (weight 4)
    register_line(["e2e4"], weight=10)
    register_line(["d2d4"], weight=10)
    register_line(["c2c4"], weight=4)
    register_line(["g1f3"], weight=4)

    # --- 2. Caro-Kann Defense (for Black: 1. e4 c6) ---
    # Classical Variation: 1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5
    register_line(["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5"], weight=8)
    # Advance Variation: 1. e4 c6 2. d4 d5 3. e5 Bf5 4. Nf3 e6
    register_line(["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6"], weight=6)
    # Panov-Botvinnik Attack: 1. e4 c6 2. d4 d5 3. exd5 cxd5 4. c4 Nf6
    register_line(["e4", "c6", "d4", "d5", "exd5", "cxd5", "c4", "Nf6"], weight=5)

    # --- 3. Catalan Opening (1. d4 Nf6 2. c4 e6 3. g3 d5 4. Bg2) ---
    # Open Catalan: 4... dxc4 5. Nf3 a6 6. O-O
    register_line(["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "dxc4", "Nf3", "a6", "O-O"], weight=9)
    # Closed Catalan: 4... Be7 5. Nf3 O-O 6. O-O Nbd7 7. Qc2 c6
    register_line(["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O", "Nbd7", "Qc2", "c6"], weight=8)

    # --- 4. Queen's Gambit ---
    # QGD: 1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O
    register_line(["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"], weight=7)
    # Slav Defense: 1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4
    register_line(["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "dxc4"], weight=6)

    # --- 5. Open Game (1. e4 e5) ---
    # Italian Game: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6
    register_line(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6"], weight=7)
    # Ruy Lopez: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O
    register_line(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"], weight=8)

    # --- 6. Sicilian Defense (1. e4 c5) ---
    # Open Sicilian / Najdorf: 1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6
    register_line(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"], weight=8)


# Initialize default repertoire on module load
init_default_book()
