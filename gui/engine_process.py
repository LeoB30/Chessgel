from __future__ import annotations
import os
import queue
import subprocess
import sys
import threading
from typing import Callable


class EngineProcess:
    """Manages an external or local UCI chess engine process asynchronously."""

    def __init__(self, engine_cmd: list[str] | None = None) -> None:
        if engine_cmd is None:
            self.engine_cmd: list[str] = self._resolve_default_engine_cmd()
        else:
            self.engine_cmd = engine_cmd

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        self.is_running: bool = False
        self.log_callback: Callable[[str, bool], None] | None = None  # (message, is_sent_command)

    @staticmethod
    def _resolve_default_engine_cmd() -> list[str]:
        """Resolves the executable command used to run the UCI chess engine."""
        # 1. When bundled as a frozen PyInstaller executable
        if getattr(sys, "frozen", False):
            exe_dir: str = os.path.dirname(sys.executable)
            local_minichess: str = os.path.join(exe_dir, "minichess.exe")
            if os.path.isfile(local_minichess):
                return [local_minichess]
            dist_v3_minichess: str = os.path.join(exe_dir, "dist_v3", "minichess", "minichess.exe")
            if os.path.isfile(dist_v3_minichess):
                return [dist_v3_minichess]
            dist_minichess: str = os.path.join(exe_dir, "dist", "minichess", "minichess.exe")
            if os.path.isfile(dist_minichess):
                return [dist_minichess]
            arena_minichess: str = os.path.join(exe_dir, "ArenaChess", "minichess.exe")
            if os.path.isfile(arena_minichess):
                return [arena_minichess]
            return [sys.executable, "--engine"]

        # 2. Running in development/script mode
        base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_py: str = os.path.join(base_dir, "main.py")
        py_exe: str = sys.executable
        if py_exe.lower().endswith("pythonw.exe"):
            candidate_exe: str = py_exe[:-5] + ".exe"
            if os.path.exists(candidate_exe):
                py_exe = candidate_exe
        return [py_exe, main_py]

    def start(self) -> None:
        """Launches the engine process and starts background stdout reader thread."""
        if self.is_running:
            return

        # Hide console window on Windows
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE

        self.process = subprocess.Popen(
            self.engine_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            startupinfo=startupinfo,
        )
        self.is_running = True

        self.reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader_thread.start()

        # Initial UCI handshake
        self.send("uci")
        self.send("isready")

    def _read_stdout(self) -> None:
        """Background thread target that buffers lines from engine stdout into queue."""
        if self.process is None or self.process.stdout is None:
            return

        for line in self.process.stdout:
            cleaned: str = line.strip()
            if cleaned:
                self.output_queue.put(cleaned)
                if self.log_callback is not None:
                    self.log_callback(cleaned, False)

        self.is_running = False

    def send(self, command: str) -> None:
        """Sends a single UCI command line to the engine's stdin."""
        if self.process is None or self.process.stdin is None or not self.is_running:
            return

        try:
            self.process.stdin.write(f"{command}\n")
            self.process.stdin.flush()
            if self.log_callback is not None:
                self.log_callback(command, True)
        except (BrokenPipeError, OSError):
            self.is_running = False

    def set_option(self, name: str, value: str | bool | int) -> None:
        """Sets a UCI option."""
        val_str: str = "true" if value is True else ("false" if value is False else str(value))
        self.send(f"setoption name {name} value {val_str}")

    def set_book_options(
        self,
        own_book: bool,
        repertoire: str = "catalan_carokann",
        selection: str = "best",
        max_ply: int = 24,
    ) -> None:
        """Configures all opening book options in one call."""
        self.set_option("OwnBook", own_book)
        self.set_option("BookRepertoire", repertoire)
        self.set_option("BookSelection", selection)
        self.set_option("BookMaxPly", max_ply)

    def new_game(self) -> None:
        """Notifies engine of a new game."""
        self.send("ucinewgame")
        self.send("isready")

    def set_position(self, moves: list[str], fen: str | None = None) -> None:
        """Sets up board state using UCI position command."""
        cmd: str = "position "
        if fen:
            cmd += f"fen {fen}"
        else:
            cmd += "startpos"

        if moves:
            cmd += " moves " + " ".join(moves)

        self.send(cmd)

    def go(self, depth: int = 3, movetime_ms: int | None = None) -> None:
        """Instructs engine to start search with given depth and/or movetime limits."""
        cmd: str = f"go depth {depth}"
        if movetime_ms is not None and movetime_ms > 0:
            cmd += f" movetime {movetime_ms}"
        self.send(cmd)

    def stop(self) -> None:
        """Signals engine to stop search immediately and emit bestmove."""
        self.send("stop")

    def poll_messages(self) -> list[str]:
        """Drains and returns all pending lines received from engine stdout."""
        lines: list[str] = []
        while not self.output_queue.empty():
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def terminate(self) -> None:
        """Safely shuts down the engine process."""
        if not self.is_running or self.process is None:
            return

        try:
            self.send("quit")
            self.process.terminate()
            self.process.wait(timeout=1.0)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.is_running = False
