from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Literal
import chess

# EPD -> list of (uci_move_str, weight)
OpeningBookDict = dict[str, list[tuple[str, int]]]

REPERTOIRE_NAMES = ("catalan_carokann", "catalan_white", "carokann_black", "tournament")
RepertoireType = Literal["catalan_carokann", "catalan_white", "carokann_black", "tournament"]
SelectionMode = Literal["best", "weighted", "random"]

# Storage for isolated repertoires
REPERTOIRES: dict[str, OpeningBookDict] = {
    "catalan_carokann": {},
    "catalan_white": {},
    "carokann_black": {},
    "tournament": {},
}

# Position to descriptive opening classification
OPENING_CLASSIFICATIONS: dict[str, str] = {}


@dataclass(frozen=True)
class MoveExplanation:
    """Educational metadata detailing strategic intent, positional nuances, and master ideas."""
    move_san: str
    variation_name: str
    strategic_intent: str
    key_ideas: str
    positional_goals: str


# Lookups for educational explanations
# Key: (epd_before_move, move_uci) -> MoveExplanation
MOVE_EXPLANATIONS: dict[tuple[str, str], MoveExplanation] = {}
# Key: epd_after_move -> MoveExplanation
POSITION_EXPLANATIONS: dict[str, MoveExplanation] = {}


def register_move(
    repertoire_key: str,
    board: chess.Board,
    move_uci: str,
    weight: int = 1,
) -> None:
    """Registers a single UCI move for the given board position in a specific repertoire."""
    book: OpeningBookDict = REPERTOIRES.setdefault(repertoire_key, {})
    epd_key: str = board.epd()
    if epd_key not in book:
        book[epd_key] = []

    for idx, (m, w) in enumerate(book[epd_key]):
        if m == move_uci:
            book[epd_key][idx] = (m, max(w, weight))
            return

    book[epd_key].append((move_uci, weight))


def register_line(
    repertoire_keys: list[str],
    move_sequence: list[str],
    weight: int = 10,
    opening_name: str | None = None,
    side: chess.Color | None = None,
) -> None:
    """Walks a board from the start position and adds each move into specified repertoires."""
    board: chess.Board = chess.Board()
    for move_str in move_sequence:
        move_str = move_str.strip()
        if not move_str:
            continue

        try:
            move: chess.Move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                move = board.parse_san(move_str)
        except Exception:
            move = board.parse_san(move_str)

        uci_str: str = move.uci()
        # Only register move if side matches or side is unconstrained
        if side is None or board.turn == side:
            for rep_key in repertoire_keys:
                register_move(rep_key, board, uci_str, weight=weight)

        board.push(move)

    if opening_name:
        OPENING_CLASSIFICATIONS[board.epd()] = opening_name


def register_explanation(
    line_moves: list[str],
    move_index: int,
    explanation: MoveExplanation,
) -> None:
    """Walks line up to move_index and registers educational explanation for that move."""
    board: chess.Board = chess.Board()
    for idx, move_str in enumerate(line_moves):
        try:
            move: chess.Move = chess.Move.from_uci(move_str)
            if move not in board.legal_moves:
                move = board.parse_san(move_str)
        except Exception:
            move = board.parse_san(move_str)

        if idx == move_index:
            epd_before: str = board.epd()
            MOVE_EXPLANATIONS[(epd_before, move.uci())] = explanation
            board.push(move)
            epd_after: str = board.epd()
            POSITION_EXPLANATIONS[epd_after] = explanation
            return

        board.push(move)


def get_book_candidates(
    board: chess.Board,
    repertoire: str = "catalan_carokann",
) -> list[tuple[chess.Move, int, float]]:
    """Retrieves legal book candidates as (Move, weight, probability_pct) for current position."""
    normalized_rep: str = repertoire.lower().replace("-", "_").replace(" ", "_")
    book: OpeningBookDict = REPERTOIRES.get(normalized_rep, REPERTOIRES["catalan_carokann"])
    candidates: list[tuple[str, int]] | None = book.get(board.epd())

    if not candidates and normalized_rep != "tournament":
        candidates = REPERTOIRES["tournament"].get(board.epd())

    if not candidates:
        return []

    valid_candidates: list[tuple[chess.Move, int]] = []
    total_weight: int = 0

    for move_uci, weight in candidates:
        try:
            m: chess.Move = chess.Move.from_uci(move_uci)
            if m in board.legal_moves:
                valid_candidates.append((m, weight))
                total_weight += weight
        except ValueError:
            continue

    if not valid_candidates or total_weight <= 0:
        return []

    results: list[tuple[chess.Move, int, float]] = []
    for move, weight in valid_candidates:
        pct: float = (weight / total_weight) * 100.0
        results.append((move, weight, round(pct, 1)))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def get_book_move(
    board: chess.Board,
    repertoire: str = "catalan_carokann",
    selection_mode: str = "best",
) -> chess.Move | None:
    """Selects a move from the opening book based on repertoire and selection strategy."""
    candidates = get_book_candidates(board, repertoire=repertoire)
    if not candidates:
        return None

    norm_mode: str = selection_mode.lower()
    if norm_mode in ("best", "bestmove", "deterministic"):
        return candidates[0][0]

    moves: list[chess.Move] = [c[0] for c in candidates]
    weights: list[int] = [c[1] for c in candidates]

    if norm_mode == "random":
        return random.choice(moves)

    # Default: 'weighted'
    return random.choices(moves, weights=weights, k=1)[0]


def get_move_explanation_for_position(board: chess.Board) -> MoveExplanation | None:
    """Returns educational explanation for the move that produced the current board position."""
    epd: str = board.epd()
    if epd in POSITION_EXPLANATIONS:
        return POSITION_EXPLANATIONS[epd]
    return None


def get_candidate_explanation(board: chess.Board, move: chess.Move) -> MoveExplanation:
    """Returns educational explanation for a candidate move played from the current position."""
    key = (board.epd(), move.uci())
    if key in MOVE_EXPLANATIONS:
        return MOVE_EXPLANATIONS[key]

    b_copy = board.copy()
    b_copy.push(move)
    if b_copy.epd() in POSITION_EXPLANATIONS:
        return POSITION_EXPLANATIONS[b_copy.epd()]

    return get_educational_summary(b_copy)


def get_educational_summary(board: chess.Board, last_move: chess.Move | None = None) -> MoveExplanation:
    """Returns an educational explanation for the current position, using exact annotations

    or synthesizing structured theoretical guidance based on opening detection.
    """
    if board.move_stack:
        exp = get_move_explanation_for_position(board)
        if exp is not None:
            return exp

    opening_name: str = identify_opening(board)
    move_san: str = "-"
    if board.move_stack:
        try:
            prev = board.copy()
            last_m = prev.pop()
            move_san = prev.san(last_m)
        except Exception:
            move_san = board.peek().uci()

    if "Caro-Kann" in opening_name or "Slav" in opening_name:
        return MoveExplanation(
            move_san=move_san,
            variation_name=opening_name,
            strategic_intent="Maintain pawn structure integrity, prevent kingside weaknesses, and challenge White's central space.",
            key_ideas="Black coordinates minor pieces behind the c6/d5 pawn barrier, preparing active breaks like ...c5 or ...e5.",
            positional_goals="Trade into a favorable endgame where Black's pawn structure proves superior, or exploit White's overextension.",
        )
    elif "Catalan" in opening_name or "Queen's" in opening_name or "Indian" in opening_name:
        return MoveExplanation(
            move_san=move_san,
            variation_name=opening_name,
            strategic_intent="Maximize diagonal laser pressure with the g2 bishop, controlling e4 and restraining Black's queenside expansion.",
            key_ideas="White combines central restraint with long-range fianchetto power, frequently winning the two bishops or queenside space.",
            positional_goals="Sustain lasting pressure on the queenside (b7/c6), recapture on c4 favorably, and control the central files.",
        )
    elif len(board.move_stack) == 0:
        return MoveExplanation(
            move_san="Start",
            variation_name="Starting Position",
            strategic_intent="Initial setup. Preferred White repertoire: 1. d4 (Catalan); Preferred Black response to 1. e4: 1...c6 (Caro-Kann).",
            key_ideas="White seeks long-range Catalan fianchetto dominance; Black seeks rock-solid Caro-Kann pawn symmetry.",
            positional_goals="Control the four central squares (e4, d4, e5, d5), facilitate rapid piece development, and ensure king safety.",
        )

    return MoveExplanation(
        move_san=move_san,
        variation_name=opening_name,
        strategic_intent="Opening line concluded or off-book position reached. Deep Alpha-Beta search evaluation is active.",
        key_ideas="Search evaluates tactical threats, king safety, piece activity, and pawn structures via Negamax with Quiescence search.",
        positional_goals="Calculate concrete tactical sequences, avoid blunders, and capitalize on positional imbalances.",
    )


def identify_opening(board: chess.Board) -> str:
    """Returns detected opening name from position classification or move history signature."""
    moves: list[str] = [m.uci() for m in board.move_stack]
    n: int = len(moves)

    if n == 0:
        return "Starting Position"

    # Caro-Kann family (1. e4 c6)
    if n >= 2 and moves[0] == "e2e4" and moves[1] == "c7c6":
        if n >= 4 and moves[2] == "d2d4" and moves[3] == "d7d5":
            if n >= 5 and moves[4] == "e4e5":
                if n >= 6 and moves[5] == "c8f5":
                    if n >= 8 and moves[6] == "g1f3" and moves[7] == "e7e6":
                        return "Caro-Kann Defense: Advance Variation (Short System)"
                    if n >= 7 and moves[6] == "h2h4":
                        return "Caro-Kann Defense: Advance (Shirov-Anand 4. h4)"
                    if n >= 7 and moves[6] == "b1c3":
                        return "Caro-Kann Defense: Advance (Bayonet Attack)"
                return "Caro-Kann Defense: Advance Variation"
            if n >= 6 and moves[4] == "e4d5" and moves[5] == "c6d5":
                if n >= 7 and moves[6] == "c2c4":
                    return "Caro-Kann Defense: Panov-Botvinnik Attack"
                return "Caro-Kann Defense: Exchange Variation"
            if n >= 6 and moves[4] in ("b1c3", "b1d2") and moves[5] == "d5e4":
                if n >= 8 and moves[7] == "c8f5":
                    return "Caro-Kann Defense: Classical (Capablanca) Variation"
                if n >= 8 and moves[7] == "b8d7":
                    return "Caro-Kann Defense: Karpov Variation (4...Nd7)"
                if n >= 8 and moves[7] == "g8f6":
                    return "Caro-Kann Defense: Tartakower Variation"
                return "Caro-Kann Defense: Main Line (3. Nc3)"
        if n >= 4 and moves[2] == "b1c3" and moves[3] == "d7d5":
            return "Caro-Kann Defense: Two Knights Variation"
        if n >= 4 and moves[2] == "f2f3" and moves[3] == "d7d5":
            return "Caro-Kann Defense: Fantasy Variation"
        return "Caro-Kann Defense"

    # Catalan family (1. d4 ...)
    if moves[0] == "d2d4":
        if n >= 2 and moves[1] == "g8f6":
            if n >= 4 and moves[2] == "c2c4" and moves[3] == "e7e6":
                if n >= 5 and moves[4] == "g2g3":
                    if n >= 8 and moves[5] == "d7d5" and moves[6] == "f1g2" and moves[7] == "d5c4":
                        return "Catalan Opening: Open Defense"
                    if n >= 8 and moves[5] == "d7d5" and moves[6] == "f1g2" and moves[7] == "f8e7":
                        return "Catalan Opening: Closed Defense"
                    if n >= 6 and moves[5] == "f8b4":
                        return "Catalan Opening: Bogo-Catalan Variation"
                    return "Catalan Opening"
                return "Indian Defense (Catalan Setup)"
            return "Indian Defense"
        if n >= 2 and moves[1] == "d7d5":
            if n >= 4 and moves[2] == "c2c4" and moves[3] == "c7c6":
                return "Slav Defense (Caro Pawn Structure)"
            if n >= 4 and moves[2] == "c2c4" and moves[3] == "e7e6":
                if n >= 6 and moves[4] == "g1f3" and moves[6] == "g2g3":
                    return "Catalan Opening"
                return "Queen's Gambit Declined"
            return "Queen's Gambit"
        return "Queen's Pawn Opening"

    if moves[0] == "e2e4":
        if n >= 2 and moves[1] == "c7c5":
            return "Sicilian Defense"
        if n >= 2 and moves[1] == "e7e5":
            return "Open Game (1. e4 e5)"
        if n >= 2 and moves[1] == "e7e6":
            return "French Defense"
        return "King's Pawn Opening"

    epd: str = board.epd()
    if epd in OPENING_CLASSIFICATIONS:
        return OPENING_CLASSIFICATIONS[epd]

    return "Custom Position / Out of Book"


# =====================================================================
# INITIALIZE SPECIALIZED REPERTOIRES & PEDAGOGICAL EXPLANATIONS
# =====================================================================

def init_repertoire() -> None:
    """Populates opening books with deep Catalan and Caro-Kann lines and explanations."""
    for rep in REPERTOIRES.values():
        rep.clear()
    OPENING_CLASSIFICATIONS.clear()
    MOVE_EXPLANATIONS.clear()
    POSITION_EXPLANATIONS.clear()

    white_only: list[str] = ["catalan_carokann", "catalan_white", "tournament"]
    black_only: list[str] = ["catalan_carokann", "carokann_black", "tournament"]
    tourney_only: list[str] = ["tournament"]

    # -----------------------------------------------------------------
    # 1. CATALAN OPENING (WHITE) LINES & EXPLANATIONS
    # -----------------------------------------------------------------
    register_line(white_only, ["d4"], weight=100, opening_name="Queen's Pawn Opening (Catalan Setup)", side=chess.WHITE)

    # 1. Open Catalan Main Line
    cat_open_main = ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O", "dxc4", "Qc2", "a6", "Qxc4", "b5", "Qc2", "Bb7", "Bd2"]
    register_line(white_only, cat_open_main, weight=25, opening_name="Catalan Opening: Open Defense (Main Line)", side=chess.WHITE)

    # 2. Open Catalan Modern 6...Nc6
    cat_open_nc6 = ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "dxc4", "Nf3", "a6", "O-O", "Nc6", "e3", "Rb8", "Nfd2", "e5", "Bxc6+", "bxc6", "dxe5"]
    register_line(white_only, cat_open_nc6, weight=20, opening_name="Catalan Opening: Modern 6...Nc6", side=chess.WHITE)

    # 3. Open Catalan 6...Bd7
    cat_open_bd7 = ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "dxc4", "Nf3", "a6", "O-O", "Bd7", "Ne5", "Bc6", "Nxc6", "Nxc6", "e3"]
    register_line(white_only, cat_open_bd7, weight=18, opening_name="Catalan Opening: 6...Bd7 Variation", side=chess.WHITE)

    # 4. Closed Catalan Main Line
    cat_closed_main = ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O", "c6", "Qc2", "Nbd7", "Nbd2", "b6", "e4"]
    register_line(white_only, cat_closed_main, weight=25, opening_name="Catalan Opening: Closed Defense (Main Line)", side=chess.WHITE)

    # 5. Closed Catalan 6...Nbd7 with 8. Rd1
    cat_closed_nbd7 = ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O", "Nbd7", "Qc2", "c6", "Rd1", "b6", "b3", "Bb7", "Nc3"]
    register_line(white_only, cat_closed_nbd7, weight=20, opening_name="Catalan Opening: Closed 6...Nbd7", side=chess.WHITE)

    # 6. Bogo-Catalan 3...Bb4+ 4. Bd2
    cat_bogo = ["d4", "Nf6", "c4", "e6", "g3", "Bb4+", "Bd2", "Be7", "Bg2", "d5", "Nf3", "O-O", "O-O", "c6", "Qc2"]
    register_line(white_only, cat_bogo, weight=22, opening_name="Catalan Opening: Bogo-Catalan Variation", side=chess.WHITE)

    # 7. Catalan vs Benoni 3...c5
    cat_benoni = ["d4", "Nf6", "c4", "e6", "g3", "c5", "Nf3", "cxd4", "Nxd4", "d5", "Bg2", "e5", "Nf3", "d4", "O-O", "Nc6"]
    register_line(white_only, cat_benoni, weight=18, opening_name="Catalan Opening: Benoni Structure", side=chess.WHITE)

    # 8. Catalan via 1...d5 move order
    cat_d5_order = ["d4", "d5", "c4", "e6", "Nf3", "Nf6", "g3", "Be7", "Bg2", "O-O", "O-O", "dxc4", "Qc2", "a6", "Qxc4"]
    register_line(white_only, cat_d5_order, weight=24, opening_name="Catalan Opening: 1...d5 Move Order", side=chess.WHITE)

    # 9. Catalan vs Slav Setup
    cat_slav = ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "g3", "dxc4", "Bg2", "b5", "a4", "Bb7", "O-O"]
    register_line(white_only, cat_slav, weight=20, opening_name="Catalan Opening: vs Slav Setup", side=chess.WHITE)

    # Educational Annotations for Catalan Moves
    register_explanation(cat_open_main, 0, MoveExplanation(
        move_san="1. d4",
        variation_name="Queen's Pawn Opening (Catalan Anchor)",
        strategic_intent="White stakes claim to the central squares d4 and e5, freeing diagonals for Queen and c1 bishop.",
        key_ideas="Establishes solid central presence without obstructing the c-pawn, which will actively challenge d5 via c4.",
        positional_goals="Spatial expansion, preparing 2. c4 and the signature Catalan fianchetto battery 3. g3.",
    ))

    register_explanation(cat_open_main, 1, MoveExplanation(
        move_san="1... Nf6",
        variation_name="Indian Defense",
        strategic_intent="Flexible response preventing White from establishing a dual-pawn center with 2. e4.",
        key_ideas="Controls e4 without committing central pawns prematurely, allowing flexible transitions into Indian setups.",
        positional_goals="Dynamic counterplay, rapid kingside piece development.",
    ))

    register_explanation(cat_open_main, 2, MoveExplanation(
        move_san="2. c4",
        variation_name="Queen's Gambit / Catalan Wedge",
        strategic_intent="Strikes at the d5 square, establishing a classical central pawn wedge and preparing Nc3.",
        key_ideas="Creates tension against Black's future central counter-thrusts while fighting for spatial superiority.",
        positional_goals="Reinforce White's central dominance and expand queenside territory.",
    ))

    register_explanation(cat_open_main, 3, MoveExplanation(
        move_san="2... e6",
        variation_name="Nimzo / Catalan Preparation",
        strategic_intent="Solidifies support for ...d5 while preparing kingside minor piece development (Be7 or Bb4+).",
        key_ideas="Maintains central tension while keeping options open between the Queen's Gambit Declined, Nimzo-Indian, or Catalan.",
        positional_goals="Solid, harmonious kingside piece coordination.",
    ))

    register_explanation(cat_open_main, 4, MoveExplanation(
        move_san="3. g3",
        variation_name="The Catalan Hallmark",
        strategic_intent="The defining move of the Catalan Opening! Prepares to fianchetto the light-squared bishop to g2.",
        key_ideas="The g2 bishop exercises laser-like diagonal control across the h1-a8 diagonal, exerting pressure on Black's queenside (b7, c6, a8).",
        positional_goals="Long-term diagonal dominance that persists throughout middlegame and endgame phases.",
    ))

    register_explanation(cat_open_main, 5, MoveExplanation(
        move_san="3... d5",
        variation_name="Catalan Central Clash",
        strategic_intent="Black establishes a firm pawn stake in the center, challenging White's spatial superiority directly.",
        key_ideas="Counters the g3 fianchetto by claiming d5 with the support of e6, challenging White to clarify the c4 tension.",
        positional_goals="Equalize central space and prevent White from easily pushing e4.",
    ))

    register_explanation(cat_open_main, 6, MoveExplanation(
        move_san="4. Bg2",
        variation_name="Catalan Bishop Battery Deployed",
        strategic_intent="Stations the light-squared bishop on its optimal diagonal, surveying central and queenside territory.",
        key_ideas="Pressures d5, restrains ...e5, and prepares kingside castling without blocking any center pawns.",
        positional_goals="Unmatched long-range diagonal activity and rapid King safety.",
    ))

    register_explanation(cat_open_main, 7, MoveExplanation(
        move_san="4... Be7",
        variation_name="Closed Catalan Setup",
        strategic_intent="Classical development prioritizing king safety and blunting White's bishop before committing to central liquidation.",
        key_ideas="Prepares immediate kingside castling while retaining the d5 central pawn anchor.",
        positional_goals="Solid defensive coordination before launching queenside breaks.",
    ))

    register_explanation(cat_open_main, 11, MoveExplanation(
        move_san="5... dxc4 / 6... dxc4",
        variation_name="Open Catalan Liquidation",
        strategic_intent="Black gives up the central pawn wedge to activate queenside piece play and gain tempos on White's queen.",
        key_ideas="Tempts White into spending time recovering c4 (via Qc2 or Qa4+), allowing Black rapid ...b5, ...a6, or ...Bb7 counterplay.",
        positional_goals="Unbalance the position and contest the long diagonal with active piece counter-punches.",
    ))

    register_explanation(cat_open_main, 12, MoveExplanation(
        move_san="7. Qc2",
        variation_name="Catalan Queen Recapture Battery",
        strategic_intent="Centralizes the queen, eyes the c4 pawn for recapture, and prepares e2-e4 central dominance.",
        key_ideas="Maintains pressure on the c-file while coordinating with the g2 bishop against Black's queenside expansion.",
        positional_goals="Recapture c4 favorably and command the center with e4.",
    ))

    register_explanation(cat_closed_main, 11, MoveExplanation(
        move_san="6... c6",
        variation_name="Closed Catalan Pawn Fortress",
        strategic_intent="Constructs an unshakeable pawn triangle (c6-d5-e6) that completely blunts White's g2 bishop.",
        key_ideas="Solidifies d5, allowing Black to develop ...Nbd7 and ...b6 without fearing tactical pin collapses.",
        positional_goals="Withstand White's central pressure and prepare ...e5 or ...c5 counter-breaks.",
    ))

    register_explanation(cat_bogo, 5, MoveExplanation(
        move_san="3... Bb4+",
        variation_name="Bogo-Catalan Check",
        strategic_intent="Active check that forces White to interpose a piece, disrupting natural piece harmony.",
        key_ideas="Forces 4. Bd2 or 4. Nd2, allowing Black to trade pieces and ease congested development.",
        positional_goals="Alleviate spatial disadvantage through targeted piece exchanges.",
    ))

    # -----------------------------------------------------------------
    # 2. CARO-KANN DEFENSE (BLACK) LINES & EXPLANATIONS
    # -----------------------------------------------------------------
    register_line(black_only, ["e4", "c6"], weight=100, opening_name="Caro-Kann Defense", side=chess.BLACK)

    # 1. Classical Capablanca Main Line
    caro_classical = ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5", "Ng3", "Bg6", "h4", "h6", "Nf3", "Nd7", "h5", "Bh7", "Bd3", "Bxd3", "Qxd3", "e6", "Bd2", "Ngf6", "O-O-O", "Be7", "Kb1", "O-O"]
    register_line(black_only, caro_classical, weight=30, opening_name="Caro-Kann Defense: Classical (Capablanca) Variation", side=chess.BLACK)

    # 2. Modern Karpov 4...Nd7
    caro_karpov = ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nd7", "Nf3", "Ngf6", "Nxf6+", "Nxf6", "Bc4", "Bf5", "O-O", "e6"]
    register_line(black_only, caro_karpov, weight=25, opening_name="Caro-Kann Defense: Karpov Variation (4...Nd7)", side=chess.BLACK)

    # 3. Tartakower 4...Nf6
    caro_tartakower = ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nf6", "Nxf6+", "exf6", "c3", "Bd6", "Bd3", "O-O", "Qc2", "Re8+", "Ne2", "h5"]
    register_line(black_only, caro_tartakower, weight=20, opening_name="Caro-Kann Defense: Tartakower Variation", side=chess.BLACK)

    # 4. Advance Short System
    caro_advance_short = ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6", "Be2", "c5", "Be3", "Qb6", "Nc3", "Nc6", "O-O", "Qxb2"]
    register_line(black_only, caro_advance_short, weight=30, opening_name="Caro-Kann Defense: Advance Variation (Short System)", side=chess.BLACK)

    # 5. Advance 4. h4 Shirov-Anand
    caro_advance_h4 = ["e4", "c6", "d4", "d5", "e5", "Bf5", "h4", "h5", "c4", "e6", "Nc3", "Ne7", "Nge2", "Nd7", "Ng3", "Bg6"]
    register_line(black_only, caro_advance_h4, weight=22, opening_name="Caro-Kann Defense: Advance 4. h4", side=chess.BLACK)

    # 6. Advance Bayonet Attack
    caro_advance_bayonet = ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nc3", "e6", "g4", "Bg6", "Nge2", "c5", "h4", "h6", "Be3", "Nc6"]
    register_line(black_only, caro_advance_bayonet, weight=20, opening_name="Caro-Kann Defense: Advance Bayonet", side=chess.BLACK)

    # 7. Panov-Botvinnik Attack
    caro_panov = ["e4", "c6", "d4", "d5", "exd5", "cxd5", "c4", "Nf6", "Nc3", "e6", "Nf3", "Be7", "cxd5", "Nxd5", "Bd3", "Nc6", "O-O", "O-O"]
    register_line(black_only, caro_panov, weight=25, opening_name="Caro-Kann Defense: Panov-Botvinnik Attack", side=chess.BLACK)

    # 8. Exchange Variation
    caro_exchange = ["e4", "c6", "d4", "d5", "exd5", "cxd5", "Bd3", "Nc6", "c3", "Nf6", "Bf4", "Bg4", "Qb3", "Qc8", "Nd2", "e6", "Ngf3", "Be7", "O-O", "O-O"]
    register_line(black_only, caro_exchange, weight=22, opening_name="Caro-Kann Defense: Exchange Variation", side=chess.BLACK)

    # 9. Two Knights Variation
    caro_two_knights = ["e4", "c6", "Nc3", "d5", "Nf3", "Bg4", "h3", "Bxf3", "Qxf3", "e6", "d4", "Nf6", "Bd3", "dxe4", "Nxe4", "Qxd4"]
    register_line(black_only, caro_two_knights, weight=20, opening_name="Caro-Kann Defense: Two Knights Variation", side=chess.BLACK)

    # 10. Fantasy Variation
    caro_fantasy = ["e4", "c6", "f3", "d5", "Nc3", "dxe4", "fxe4", "e5", "Nf3", "Bg4", "Bc4", "Nd7", "d3", "Bc5"]
    register_line(black_only, caro_fantasy, weight=15, opening_name="Caro-Kann Defense: Fantasy Variation", side=chess.BLACK)

    # Educational Annotations for Caro-Kann Moves
    register_explanation(caro_classical, 1, MoveExplanation(
        move_san="1... c6",
        variation_name="Caro-Kann Defense Foundation",
        strategic_intent="Black's cornerstone move! Prepares 2...d5 while keeping the c8-h3 diagonal completely open for the light-squared bishop.",
        key_ideas="Unlike the French Defense (1...e6) which imprisons the c8 bishop behind pawns, the Caro-Kann allows ...Bf5 or ...Bg4 before ...e6 is played.",
        positional_goals="Construct an uncompromised, rock-solid pawn structure, challenge e4 immediately, and achieve effortless endgame equality.",
    ))

    register_explanation(caro_classical, 3, MoveExplanation(
        move_san="2... d5",
        variation_name="The Caro-Kann Central Counter-Strike",
        strategic_intent="Directly attacks White's e4 central pawn, demanding an immediate response from White.",
        key_ideas="Forces White to choose between defending with 3. Nc3, pushing with 3. e5, or capturing with 3. exd5.",
        positional_goals="Stake equal claim in the center and clarify central pawn tension on Black's terms.",
    ))

    register_explanation(caro_classical, 5, MoveExplanation(
        move_san="3... dxe4",
        variation_name="Classical Center Liquidation",
        strategic_intent="Eliminates White's central e4 pawn, granting Black access to the prime d5 outpost for piece maneuvering.",
        key_ideas="Removes White's dual pawn center, opening the d-file and preparing active minor piece deployment.",
        positional_goals="Neutralize White's space advantage and prepare ...Bf5 or ...Nd7.",
    ))

    register_explanation(caro_classical, 7, MoveExplanation(
        move_san="4... Bf5",
        variation_name="Capablanca Classical Bishop Activation",
        strategic_intent="The gold standard of the Caro-Kann! Activates the light-squared bishop outside the pawn chain before playing ...e6.",
        key_ideas="Puts direct pressure on White's active knight on e4, avoids any 'bad bishop' handicap, and prepares harmonious piece play.",
        positional_goals="Complete piece development with zero pawn weaknesses and optimal bishop placement.",
    ))

    register_explanation(caro_classical, 9, MoveExplanation(
        move_san="5... Bg6",
        variation_name="Bishop Anchor & Safe Retreat",
        strategic_intent="Retreats the bishop to safety along the h7-b1 diagonal while maintaining pressure against White's center.",
        key_ideas="Bishop remains an active nuisance to White, anchoring the kingside and awaiting White's flank attempts.",
        positional_goals="Control the h7-b1 diagonal and absorb White's kingside pawn pushes.",
    ))

    register_explanation(caro_classical, 11, MoveExplanation(
        move_san="6... h6",
        variation_name="Essential Luft for Bishop Safety",
        strategic_intent="Creates a critical escape square on h7 for the bishop against White's threatening h4-h5 pawn thrust.",
        key_ideas="Completely neutralizes White's tactical trap of trapping the bishop on g6 with h5.",
        positional_goals="Guarantee the permanent survival and active utility of the light-squared bishop on h7.",
    ))

    register_explanation(caro_classical, 13, MoveExplanation(
        move_san="7... Nd7",
        variation_name="Karpovian Clamp on e5",
        strategic_intent="Covers the critical e5 square, preventing White's knight from embedding itself as an aggressive outpost.",
        key_ideas="Prepares ...Ngf6 and ...e6 without allowing an annoying Ne5 fork or pin; textbook positional mastery.",
        positional_goals="Absolute harmonic coordination and control over central outposts.",
    ))

    register_explanation(caro_advance_short, 5, MoveExplanation(
        move_san="3... Bf5",
        variation_name="Advance Variation - Bishop Breakout",
        strategic_intent="White seized space with 3. e5; Black immediately stations the bishop outside the pawn chain before locking center with ...e6.",
        key_ideas="Avoids the cramped French Defense bishop dilemma, preparing to strike at White's d4 pawn base with ...c5.",
        positional_goals="Break down White's pawn chain with ...c5, ...Nc6, and ...Qb6.",
    ))

    register_explanation(caro_advance_short, 7, MoveExplanation(
        move_san="4... e6",
        variation_name="Advance - Fortifying the Structure",
        strategic_intent="Locks the central pawn chain securely while opening the diagonal for the dark-squared bishop (...Be7 or ...Bd6).",
        key_ideas="Black has already solved the problem of the light-squared bishop; now development proceeds smoothly.",
        positional_goals="Prepares the central counter-punch ...c5 to dismantle White's d4 anchor.",
    ))

    register_explanation(caro_advance_short, 9, MoveExplanation(
        move_san="5... c5",
        variation_name="Thematic Caro-Kann Counter-Punch",
        strategic_intent="The key thematic strike in the Advance Caro-Kann! Attacks the base of White's pawn chain on d4.",
        key_ideas="Forces White to defend d4 with pieces or pawns, dissolving White's central space and winning queenside file access.",
        positional_goals="Shatter White's central pawn wedge and launch active queenside operations.",
    ))

    register_explanation(caro_panov, 6, MoveExplanation(
        move_san="4. c4",
        variation_name="Panov-Botvinnik IQP Transformation",
        strategic_intent="White converts the game into an Isolated Queen's Pawn (IQP) structure to unleash aggressive piece play.",
        key_ideas="White gains open c- and e-files with active diagonal play; Black focuses on blockading the d4 pawn.",
        positional_goals="Black aims to blockade d4, trade active minor pieces, and win the weakened isolated pawn in the endgame.",
    ))

    register_explanation(caro_karpov, 7, MoveExplanation(
        move_san="4... Nd7",
        variation_name="Karpov Modern Variation",
        strategic_intent="Deeply positional setup preparing ...Ngf6 without allowing doubled pawns, avoiding bishop displacement.",
        key_ideas="Black plays for harmonious piece trades, unassailable central squares, and superior endgame piece play.",
        positional_goals="Effortless structural equality and master-level positional durability.",
    ))

    register_explanation(caro_tartakower, 9, MoveExplanation(
        move_san="5... exf6",
        variation_name="Tartakower Asymmetric Pawn Structure",
        strategic_intent="Black accepts doubled f-pawns in exchange for concrete dynamic advantages: an open e-file and rapid piece activity.",
        key_ideas="The e-file offers immediate pressure against White's center; the dark-squared bishop activates effortlessly on d6.",
        positional_goals="Dynamic imbalance, rapid kingside piece activity, and rook centralization.",
    ))

    # Tournament baseline general lines
    register_line(tourney_only, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"], weight=10, opening_name="Ruy Lopez")
    register_line(tourney_only, ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4"], weight=10, opening_name="Italian Game")
    register_line(tourney_only, ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"], weight=10, opening_name="Sicilian Defense: Najdorf")
    register_line(tourney_only, ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"], weight=10, opening_name="Queen's Gambit Declined")


# Initialize on module import
init_repertoire()
