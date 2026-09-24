from __future__ import annotations
import sys
import time
import chess
from engine.search import find_best_move

ENGINE_NAME: str = "MiniChess Pro"
ENGINE_AUTHOR: str = "Antigravity Chess AI"


class UCIEngine:
    """Handles Universal Chess Interface (UCI) protocol communication via stdin/stdout."""

    def __init__(self) -> None:
        self.board: chess.Board = chess.Board()
        self.use_book: bool = True
        self.book_repertoire: str = "catalan_carokann"
        self.book_selection: str = "best"
        self.book_max_ply: int = 24
        self.default_depth: int = 3
        self.default_movetime_ms: int | None = None

    def run(self) -> None:
        """Main UCI command processing loop."""
        while True:
            try:
                line: str = sys.stdin.readline()
                if not line:
                    break

                command_line: str = line.strip()
                if not command_line:
                    continue

                if not self.handle_command(command_line):
                    break
            except (KeyboardInterrupt, SystemExit):
                break

    def handle_command(self, command_line: str) -> bool:
        """Parses and executes a single UCI command line.

        Returns False if the engine should terminate ('quit'), True otherwise.
        """
        tokens: list[str] = command_line.split()
        if not tokens:
            return True

        cmd: str = tokens[0]

        if cmd == "uci":
            self._send_response(f"id name {ENGINE_NAME}")
            self._send_response(f"id author {ENGINE_AUTHOR}")
            self._send_response("option name OwnBook type check default true")
            self._send_response(
                "option name BookRepertoire type combo default Catalan_CaroKann "
                "var Catalan_CaroKann var Catalan_White var CaroKann_Black var Tournament"
            )
            self._send_response(
                "option name BookSelection type combo default BestMove "
                "var BestMove var Weighted var Random"
            )
            self._send_response("option name BookMaxPly type spin default 24 min 1 max 60")
            self._send_response("option name SearchDepth type spin default 3 min 1 max 10")
            self._send_response("option name MoveTime type spin default 0 min 0 max 60000")
            self._send_response("uciok")

        elif cmd == "setoption":
            self._handle_setoption(tokens[1:])

        elif cmd == "isready":
            self._send_response("readyok")

        elif cmd == "ucinewgame":
            self.board.reset()

        elif cmd == "position":
            self._handle_position(tokens[1:])

        elif cmd == "go":
            self._handle_go(tokens[1:])

        elif cmd == "quit":
            return False

        return True

    def _handle_setoption(self, args: list[str]) -> None:
        """Parses 'setoption name <name> value <val>'."""
        try:
            if "name" in args and "value" in args:
                name_idx: int = args.index("name") + 1
                val_idx: int = args.index("value") + 1
                if name_idx < len(args) and val_idx < len(args):
                    opt_name: str = args[name_idx].lower()
                    opt_val: str = " ".join(args[val_idx:]).strip()

                    if opt_name == "ownbook":
                        self.use_book = opt_val.lower() in ("true", "1", "yes", "on")
                    elif opt_name == "bookrepertoire":
                        self.book_repertoire = opt_val.lower()
                    elif opt_name == "bookselection":
                        self.book_selection = opt_val.lower()
                    elif opt_name == "bookmaxply" and opt_val.isdigit():
                        self.book_max_ply = max(1, int(opt_val))
                    elif opt_name == "searchdepth" and opt_val.isdigit():
                        self.default_depth = max(1, int(opt_val))
                    elif opt_name == "movetime" and opt_val.isdigit():
                        ms: int = int(opt_val)
                        self.default_movetime_ms = ms if ms > 0 else None
        except Exception:
            pass

    def _handle_position(self, args: list[str]) -> None:
        """Parses 'position startpos [moves ...]' or 'position fen <FEN> [moves ...]'."""
        if not args:
            return

        if args[0] == "startpos":
            self.board.reset()
            moves_index: int = 1
        elif args[0] == "fen":
            fen_tokens: list[str] = []
            moves_index = 1
            while moves_index < len(args) and args[moves_index] != "moves":
                fen_tokens.append(args[moves_index])
                moves_index += 1
            fen_string: str = " ".join(fen_tokens)
            self.board.set_fen(fen_string)
        else:
            return

        if moves_index < len(args) and args[moves_index] == "moves":
            for move_str in args[moves_index + 1 :]:
                move: chess.Move = chess.Move.from_uci(move_str)
                if move in self.board.legal_moves:
                    self.board.push(move)

    def _handle_go(self, args: list[str]) -> None:
        """Executes search and reports 'bestmove'."""
        depth: int = self.default_depth
        movetime: int | None = self.default_movetime_ms

        if "depth" in args:
            depth_idx: int = args.index("depth") + 1
            if depth_idx < len(args) and args[depth_idx].isdigit():
                depth = int(args[depth_idx])

        if "movetime" in args:
            mt_idx: int = args.index("movetime") + 1
            if mt_idx < len(args) and args[mt_idx].isdigit():
                movetime = int(args[mt_idx])

        start_time: float = time.perf_counter()

        def on_search_info(d: int, score: int, nodes: int, move: chess.Move, source: str) -> None:
            elapsed: float = time.perf_counter() - start_time
            time_ms: int = int(elapsed * 1000)
            nps: int = int(nodes / elapsed) if elapsed > 0.001 else nodes * 1000
            if source == "book":
                self._send_response(
                    f"info depth 1 score cp 0 time 0 nodes 1 nps 0 string Book move ({move.uci()}) pv {move.uci()}"
                )
            else:
                self._send_response(
                    f"info depth {d} score cp {score} time {time_ms} nodes {nodes} nps {nps} pv {move.uci()}"
                )

        best_move: chess.Move | None = find_best_move(
            self.board,
            depth=depth,
            movetime_ms=movetime,
            use_book=self.use_book,
            book_repertoire=self.book_repertoire,
            book_selection=self.book_selection,
            book_max_ply=self.book_max_ply,
            info_callback=on_search_info,
        )
        if best_move is not None:
            self._send_response(f"bestmove {best_move.uci()}")
        else:
            self._send_response("bestmove (none)")

    @staticmethod
    def _send_response(message: str) -> None:
        """Outputs response to stdout and flushes immediately."""
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
