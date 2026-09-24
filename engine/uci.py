from __future__ import annotations
import sys
import time
import chess
from engine.search import find_best_move

ENGINE_NAME: str = "MiniChess Python"
ENGINE_AUTHOR: str = "Antigravity Mentor"


class UCIEngine:
    """Handles Universal Chess Interface (UCI) protocol communication via stdin/stdout."""

    def __init__(self) -> None:
        self.board: chess.Board = chess.Board()
        self.use_book: bool = True

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
                    opt_val: str = args[val_idx].lower()
                    if opt_name == "ownbook":
                        self.use_book = opt_val in ("true", "1", "yes", "on")
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
            # Collect FEN parts (6 tokens)
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
        depth: int = 3
        # Check if depth parameter was passed
        if "depth" in args:
            depth_idx: int = args.index("depth") + 1
            if depth_idx < len(args) and args[depth_idx].isdigit():
                depth = int(args[depth_idx])

        start_time: float = time.perf_counter()

        def on_search_info(d: int, score: int, nodes: int, move: chess.Move) -> None:
            elapsed: float = time.perf_counter() - start_time
            time_ms: int = int(elapsed * 1000)
            nps: int = int(nodes / elapsed) if elapsed > 0.001 else nodes * 1000
            self._send_response(
                f"info depth {d} score cp {score} time {time_ms} nodes {nodes} nps {nps} pv {move.uci()}"
            )

        best_move: chess.Move | None = find_best_move(
            self.board,
            depth=depth,
            use_book=self.use_book,
            info_callback=on_search_info,
        )
        if best_move is not None:
            self._send_response(f"bestmove {best_move.uci()}")
        else:
            # Fallback when no move exists
            self._send_response("bestmove (none)")

    @staticmethod
    def _send_response(message: str) -> None:
        """Outputs response to stdout and flushes immediately."""
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
