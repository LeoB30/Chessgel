from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import chess
from gui.board_canvas import BoardCanvas
from gui.engine_process import EngineProcess


class ChessApp:
    """Main desktop GUI application for the MiniChess engine."""

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Root = root
        self.root.title("MiniChess - UCI Engine & Analysis Studio")
        self.root.geometry("1100x740")
        self.root.minsize(980, 680)
        self.root.configure(bg="#1E1E2E")

        # Game state
        self.board: chess.Board = chess.Board()
        self.engine: EngineProcess = EngineProcess()
        self.move_history_uci: list[str] = []
        self.is_engine_thinking: bool = False
        self.game_mode: tk.StringVar = tk.StringVar(value="Human (White) vs Engine")
        self.depth_var: tk.IntVar = tk.IntVar(value=3)
        self.book_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.delay_var: tk.IntVar = tk.IntVar(value=500)

        # Style configuration
        self._configure_styles()

        # Build UI layout
        self._build_ui()

        # Start engine
        self.engine.log_callback = self._on_engine_log
        self.engine.start()
        self.engine.set_option("OwnBook", self.book_var.get())

        # Start periodic engine poll loop
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_engine)

        self._update_status()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1E1E2E")
        style.configure("TLabel", background="#1E1E2E", foreground="#CDD6F4", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#89B4FA")
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("TCheckbutton", background="#1E1E2E", foreground="#CDD6F4")
        style.configure("TCombobox", padding=4)

    def _build_ui(self) -> None:
        # Top Header Bar
        header_frame: tk.Frame = tk.Frame(self.root, bg="#181825", height=45)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        self.status_label: tk.Label = tk.Label(
            header_frame,
            text="White to move",
            font=("Segoe UI", 13, "bold"),
            bg="#181825",
            fg="#A6E3A1",
            padx=15,
            pady=8,
        )
        self.status_label.pack(side=tk.LEFT)

        # Main Workspace Container
        main_container: tk.Frame = tk.Frame(self.root, bg="#1E1E2E")
        main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Left Column: Chess Board
        board_container: tk.Frame = tk.Frame(main_container, bg="#1E1E2E")
        board_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 12))

        self.canvas: BoardCanvas = BoardCanvas(board_container, board_size=560, on_move=self._on_player_move)
        self.canvas.pack(side=tk.TOP)

        # Board Controls Bar
        board_ctrl: tk.Frame = tk.Frame(board_container, bg="#1E1E2E", pady=8)
        board_ctrl.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            board_ctrl,
            text="🔄 Flip Board",
            command=self._flip_board,
            bg="#313244",
            fg="#CDD6F4",
            activebackground="#45475A",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            board_ctrl,
            text="↩ Undo Move",
            command=self._undo_move,
            bg="#313244",
            fg="#CDD6F4",
            activebackground="#45475A",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            board_ctrl,
            text="⚡ Make Engine Move",
            command=self._trigger_engine_turn,
            bg="#45475A",
            fg="#CDD6F4",
            activebackground="#585B70",
            relief=tk.FLAT,
            padx=10,
            pady=4,
        ).pack(side=tk.LEFT, padx=4)

        # Right Column: Notebook Tabs (Dashboard / Controls / History / Logs)
        right_container: tk.Frame = tk.Frame(main_container, bg="#181825", relief=tk.FLAT)
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(right_container)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab 1: Controls & Engine Telemetry
        telemetry_tab: tk.Frame = tk.Frame(notebook, bg="#181825", padx=10, pady=10)
        notebook.add(telemetry_tab, text=" Dashboard & Controls ")

        # Tab 2: Logs & Move History
        log_tab: tk.Frame = tk.Frame(notebook, bg="#181825", padx=10, pady=10)
        notebook.add(log_tab, text=" Game History & UCI Log ")

        self._build_telemetry_tab(telemetry_tab)
        self._build_log_tab(log_tab)

    def _build_telemetry_tab(self, parent: tk.Frame) -> None:
        # 1. Game Setup Panel
        setup_group = tk.LabelFrame(
            parent,
            text=" Game Configuration ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=10,
            pady=8,
        )
        setup_group.pack(fill=tk.X, pady=(0, 10))

        tk.Label(setup_group, text="Mode:", bg="#181825", fg="#CDD6F4").grid(row=0, column=0, sticky="w", pady=4)
        mode_cb = ttk.Combobox(
            setup_group,
            textvariable=self.game_mode,
            values=[
                "Human (White) vs Engine",
                "Engine (White) vs Human",
                "Human vs Human",
                "Engine vs Engine (Spectator)",
            ],
            state="readonly",
            width=26,
        )
        mode_cb.grid(row=0, column=1, sticky="w", pady=4, padx=6)
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        tk.Label(setup_group, text="Search Depth:", bg="#181825", fg="#CDD6F4").grid(row=1, column=0, sticky="w", pady=4)
        depth_spin = tk.Spinbox(
            setup_group,
            from_=1,
            to=6,
            textvariable=self.depth_var,
            width=6,
            bg="#313244",
            fg="#CDD6F4",
            buttonbackground="#45475A",
        )
        depth_spin.grid(row=1, column=1, sticky="w", pady=4, padx=6)

        book_check = tk.Checkbutton(
            setup_group,
            text="Use Opening Book",
            variable=self.book_var,
            command=self._on_book_toggle,
            bg="#181825",
            fg="#CDD6F4",
            selectcolor="#313244",
            activebackground="#181825",
            activeforeground="#CDD6F4",
        )
        book_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        btn_row = tk.Frame(setup_group, bg="#181825")
        btn_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2))

        tk.Button(
            btn_row,
            text="✨ New Game",
            command=self._new_game,
            bg="#A6E3A1",
            fg="#11111B",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_row,
            text="⏹ Stop Engine",
            command=self._stop_engine,
            bg="#F38BA8",
            fg="#11111B",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
        ).pack(side=tk.LEFT)

        # 2. Live Engine Telemetry Panel
        telem_group = tk.LabelFrame(
            parent,
            text=" Live UCI Engine Telemetry ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=10,
            pady=8,
        )
        telem_group.pack(fill=tk.BOTH, expand=True)

        # Visual Advantage Gauge
        tk.Label(telem_group, text="Evaluation Balance:", bg="#181825", fg="#CDD6F4").pack(anchor="w")
        self.eval_canvas: tk.Canvas = tk.Canvas(telem_group, height=18, bg="#313244", highlightthickness=0)
        self.eval_canvas.pack(fill=tk.X, pady=(4, 10))
        self._update_eval_bar(0)

        # Telemetry Metrics Grid
        grid_frame = tk.Frame(telem_group, bg="#181825")
        grid_frame.pack(fill=tk.X, pady=4)

        self.telem_depth = self._add_stat(grid_frame, "Search Depth:", "0", row=0, col=0)
        self.telem_eval = self._add_stat(grid_frame, "Evaluation:", "0.00", row=0, col=1)
        self.telem_nodes = self._add_stat(grid_frame, "Nodes:", "0", row=1, col=0)
        self.telem_nps = self._add_stat(grid_frame, "Speed (NPS):", "0", row=1, col=1)
        self.telem_time = self._add_stat(grid_frame, "Time:", "0 ms", row=2, col=0)
        self.telem_book = self._add_stat(grid_frame, "Source:", "Search", row=2, col=1)

        # Principal Variation Line (PV)
        tk.Label(telem_group, text="Principal Variation (PV):", bg="#181825", fg="#CDD6F4").pack(anchor="w", pady=(8, 2))
        self.pv_label: tk.Label = tk.Label(
            telem_group,
            text="-",
            bg="#11111B",
            fg="#F9E2AF",
            font=("Consolas", 10),
            wraplength=380,
            justify=tk.LEFT,
            padx=8,
            pady=6,
            relief=tk.FLAT,
        )
        self.pv_label.pack(fill=tk.X)

    def _build_log_tab(self, parent: tk.Frame) -> None:
        # Move History
        tk.Label(parent, text="Move History (SAN):", bg="#181825", fg="#89B4FA", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.history_text: tk.Text = tk.Text(parent, height=8, bg="#11111B", fg="#CDD6F4", font=("Consolas", 10), relief=tk.FLAT)
        self.history_text.pack(fill=tk.X, pady=(4, 10))

        # UCI Debug Stream
        tk.Label(parent, text="UCI Protocol Stream:", bg="#181825", fg="#89B4FA", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.log_text: tk.Text = tk.Text(parent, height=14, bg="#11111B", fg="#A6ADC8", font=("Consolas", 9), relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=4)

    def _add_stat(self, parent: tk.Frame, label_text: str, default_val: str, row: int, col: int) -> tk.Label:
        box: tk.Frame = tk.Frame(parent, bg="#11111B", padx=8, pady=6, relief=tk.FLAT)
        box.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)
        parent.columnconfigure(col, weight=1)

        tk.Label(box, text=label_text, font=("Segoe UI", 8), bg="#11111B", fg="#A6ADC8").pack(anchor="w")
        val_label: tk.Label = tk.Label(box, text=default_val, font=("Segoe UI", 11, "bold"), bg="#11111B", fg="#89B4FA")
        val_label.pack(anchor="w")
        return val_label

    def _update_eval_bar(self, centipawns: int) -> None:
        """Draws visual balance bar where 50% is 0.00, White advantage is white, Black is dark."""
        self.eval_canvas.delete("all")
        width: int = self.eval_canvas.winfo_width()
        if width <= 1:
            width = 380
        height: int = 18

        # Clamp eval to -1000 to +1000 centipawns for bar display
        clamped: int = max(-1000, min(1000, centipawns))
        # Center = 0.5
        fraction: float = (clamped + 1000) / 2000.0
        split_x: int = int(width * fraction)

        # White side (left) vs Black side (right)
        self.eval_canvas.create_rectangle(0, 0, split_x, height, fill="#FFFFFF", outline="")
        self.eval_canvas.create_rectangle(split_x, 0, width, height, fill="#313244", outline="")
        self.eval_canvas.create_line(width // 2, 0, width // 2, height, fill="#89B4FA", width=2)

    def _on_player_move(self, move: chess.Move) -> None:
        """Invoked when user completes a move on the board canvas."""
        if self.is_engine_thinking or self.board.is_game_over():
            return

        self.board.push(move)
        self.move_history_uci.append(move.uci())
        self.canvas.set_board(self.board, last_move=move)
        self._update_history_text()
        self._update_status()

        # Check if engine should reply
        if not self.board.is_game_over():
            if self._is_engine_turn():
                self._trigger_engine_turn()

    def _trigger_engine_turn(self) -> None:
        """Sends position and go command to engine."""
        if self.is_engine_thinking or self.board.is_game_over():
            return

        self.is_engine_thinking = True
        self.status_label.config(text=f"Engine calculating ({'White' if self.board.turn == chess.WHITE else 'Black'})...", fg="#F9E2AF")

        self.engine.set_position(self.move_history_uci)
        self.engine.go(depth=self.depth_var.get())

    def _poll_engine(self) -> None:
        """Polls messages from engine queue every 50ms."""
        messages: list[str] = self.engine.poll_messages()
        for msg in messages:
            self._handle_engine_message(msg)

        self.root.after(50, self._poll_engine)

    def _handle_engine_message(self, line: str) -> None:
        """Parses UCI 'info' and 'bestmove' output lines."""
        tokens: list[str] = line.split()
        if not tokens:
            return

        if tokens[0] == "info":
            # Parse depth, score, nodes, nps, pv
            if "depth" in tokens:
                d_idx: int = tokens.index("depth") + 1
                if d_idx < len(tokens):
                    self.telem_depth.config(text=tokens[d_idx])

            if "score" in tokens:
                s_idx: int = tokens.index("score") + 1
                if s_idx < len(tokens):
                    score_type: str = tokens[s_idx]
                    val: int = int(tokens[s_idx + 1]) if s_idx + 1 < len(tokens) else 0
                    if score_type == "cp":
                        # Adjust score relative to White
                        display_score: float = val / 100.0 if self.board.turn == chess.WHITE else -val / 100.0
                        self.telem_eval.config(text=f"{display_score:+.2f}")
                        self._update_eval_bar(int(display_score * 100))
                    elif score_type == "mate":
                        self.telem_eval.config(text=f"M{val}")

            if "nodes" in tokens:
                n_idx: int = tokens.index("nodes") + 1
                if n_idx < len(tokens):
                    node_cnt: int = int(tokens[n_idx])
                    self.telem_nodes.config(text=f"{node_cnt:,}")
                    if node_cnt == 1:
                        self.telem_book.config(text="Book Move", fg="#A6E3A1")
                    else:
                        self.telem_book.config(text="Alpha-Beta", fg="#89B4FA")

            if "nps" in tokens:
                nps_idx: int = tokens.index("nps") + 1
                if nps_idx < len(tokens):
                    self.telem_nps.config(text=f"{int(tokens[nps_idx]):,}")

            if "time" in tokens:
                t_idx: int = tokens.index("time") + 1
                if t_idx < len(tokens):
                    self.telem_time.config(text=f"{tokens[t_idx]} ms")

            if "pv" in tokens:
                pv_idx: int = tokens.index("pv") + 1
                pv_moves: str = " ".join(tokens[pv_idx:pv_idx + 6])
                self.pv_label.config(text=pv_moves)

        elif tokens[0] == "bestmove":
            self.is_engine_thinking = False
            best_move_str: str = tokens[1]
            if best_move_str != "(none)":
                move: chess.Move = chess.Move.from_uci(best_move_str)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.move_history_uci.append(move.uci())
                    self.canvas.set_board(self.board, last_move=move)
                    self._update_history_text()

            self._update_status()

            # If Spectator mode (Engine vs Engine), trigger next turn after delay
            if not self.board.is_game_over() and self.game_mode.get() == "Engine vs Engine (Spectator)":
                self.root.after(self.delay_var.get(), self._trigger_engine_turn)

    def _is_engine_turn(self) -> bool:
        mode: str = self.game_mode.get()
        if mode == "Human (White) vs Engine":
            return self.board.turn == chess.BLACK
        elif mode == "Engine (White) vs Human":
            return self.board.turn == chess.WHITE
        elif mode == "Engine vs Engine (Spectator)":
            return True
        return False  # Human vs Human

    def _on_mode_change(self, event=None) -> None:
        mode: str = self.game_mode.get()
        if mode == "Engine (White) vs Human":
            self.canvas.set_flipped(True)
        else:
            self.canvas.set_flipped(False)

        if not self.board.is_game_over() and self._is_engine_turn():
            self._trigger_engine_turn()

    def _on_book_toggle(self) -> None:
        self.engine.set_option("OwnBook", self.book_var.get())

    def _flip_board(self) -> None:
        self.canvas.set_flipped(not self.canvas.is_flipped)

    def _undo_move(self) -> None:
        if self.is_engine_thinking:
            return

        if len(self.board.move_stack) >= 2 and self.game_mode.get() in ("Human (White) vs Engine", "Engine (White) vs Human"):
            # Undo both engine and player move
            self.board.pop()
            self.board.pop()
            self.move_history_uci.pop()
            self.move_history_uci.pop()
        elif len(self.board.move_stack) >= 1:
            self.board.pop()
            self.move_history_uci.pop()

        last_move: chess.Move | None = self.board.peek() if self.board.move_stack else None
        self.canvas.set_board(self.board, last_move=last_move)
        self._update_history_text()
        self._update_status()

    def _new_game(self) -> None:
        if self.is_engine_thinking:
            self._stop_engine()

        self.board.reset()
        self.move_history_uci.clear()
        self.canvas.set_board(self.board)
        self.engine.new_game()
        self.history_text.delete("1.0", tk.END)
        self._update_eval_bar(0)
        self.pv_label.config(text="-")
        self._update_status()

        if self._is_engine_turn():
            self.root.after(200, self._trigger_engine_turn)

    def _stop_engine(self) -> None:
        self.engine.stop()
        self.is_engine_thinking = False
        self._update_status()

    def _update_status(self) -> None:
        if self.board.is_checkmate():
            winner: str = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_label.config(text=f"Checkmate! {winner} wins! 🏆", fg="#F38BA8")
        elif self.board.is_stalemate():
            self.status_label.config(text="Game Drawn by Stalemate.", fg="#FAB387")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Game Drawn by Insufficient Material.", fg="#FAB387")
        elif self.board.can_claim_threefold_repetition():
            self.status_label.config(text="Game Drawn by Threefold Repetition.", fg="#FAB387")
        elif self.board.is_check():
            self.status_label.config(text=f"Check! ({'White' if self.board.turn == chess.WHITE else 'Black'} to move)", fg="#F38BA8")
        else:
            turn_str: str = "White to move" if self.board.turn == chess.WHITE else "Black to move"
            self.status_label.config(text=turn_str, fg="#A6E3A1")

    def _update_history_text(self) -> None:
        """Renders move list in standard SAN format (e.g. 1. e4 c6  2. d4 d5)."""
        temp_board: chess.Board = chess.Board()
        san_tokens: list[str] = []

        for idx, uci_move in enumerate(self.move_history_uci):
            move = chess.Move.from_uci(uci_move)
            san_str: str = temp_board.san(move)
            temp_board.push(move)

            if idx % 2 == 0:
                move_num: int = (idx // 2) + 1
                san_tokens.append(f"{move_num}. {san_str}")
            else:
                san_tokens[-1] += f"  {san_str}"

        self.history_text.delete("1.0", tk.END)
        self.history_text.insert(tk.END, "\n".join(san_tokens))
        self.history_text.see(tk.END)

    def _on_engine_log(self, text: str, is_command: bool) -> None:
        """Appends engine communication to debug console."""
        prefix: str = ">> " if is_command else "<< "
        color: str = "#89DCEB" if is_command else "#A6ADC8"
        self.log_text.insert(tk.END, f"{prefix}{text}\n")
        self.log_text.see(tk.END)

    def _on_close(self) -> None:
        """Clean shutdown handler."""
        self.engine.terminate()
        self.root.destroy()
