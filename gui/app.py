from __future__ import annotations
import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import chess
from engine.opening_book import (
    MoveExplanation,
    get_book_candidates,
    get_candidate_explanation,
    get_educational_summary,
    identify_opening,
)
from gui.board_canvas import BoardCanvas
from gui.board_vision import BoardVisionSync
from gui.engine_process import EngineProcess
from gui.grid_inspector import GridAlignmentInspector
from gui.screen_selector import ScreenRegionSelector


class ChessApp:
    """Professional Desktop GUI for MiniChess engine, analysis studio, and opening repertoire."""

    def __init__(self, root: tk.Tk) -> None:
        self.root: tk.Tk = root
        self.root.title("MiniChess Studio - Professional Analysis & Opening Suite")
        self.root.geometry("1180x780")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#1E1E2E")

        # Game and engine state
        self.board: chess.Board = chess.Board()
        self.engine: EngineProcess = EngineProcess()
        self.move_history_uci: list[str] = []
        self.history_index: int = 0
        self.is_engine_thinking: bool = False
        self.current_eval_cp: int = 0
        self.current_mate_score: int | None = None
        self.suggested_move: chess.Move | None = None

        # Configuration variables
        self.game_mode: tk.StringVar = tk.StringVar(value="Human (White) vs Engine")
        self.depth_var: tk.IntVar = tk.IntVar(value=3)
        self.time_mode_var: tk.StringVar = tk.StringVar(value="Fixed Depth")
        self.movetime_var: tk.IntVar = tk.IntVar(value=1000)
        self.delay_var: tk.IntVar = tk.IntVar(value=500)

        # Opening book variables
        self.book_var: tk.BooleanVar = tk.BooleanVar(value=True)
        self.repertoire_var: tk.StringVar = tk.StringVar(value="Catalan & Caro-Kann (Preferred)")
        self.selection_var: tk.StringVar = tk.StringVar(value="Best Move (Deterministic)")
        self.max_ply_var: tk.IntVar = tk.IntVar(value=24)
        self.show_arrow_var: tk.BooleanVar = tk.BooleanVar(value=True)

        # Vision synchronization state
        self.grid_inspector: GridAlignmentInspector | None = None
        self.vision_fen_var: tk.StringVar = tk.StringVar(value=chess.STARTING_FEN)
        self.vision_margin_var: tk.DoubleVar = tk.DoubleVar(value=0.0)
        self.vision_inset_var: tk.DoubleVar = tk.DoubleVar(value=12.0)
        self.vision_sync: BoardVisionSync = BoardVisionSync(
            get_board_callback=lambda: self.board.copy(),
            on_move_detected=self._on_vision_move_detected,
            on_status_update=self._on_vision_status_update,
            on_fen_detected=self._on_vision_fen_detected,
            on_debug_frame=self._on_vision_debug_frame,
            outer_margin_pct=0.0,
            inner_inset_pct=0.12,
        )
        self.last_detected_opponent_move: chess.Move | None = None
        self.engine_counter_move: chess.Move | None = None
        self.is_vision_counter_eval: bool = False
        self.vision_auto_play_var: tk.BooleanVar = tk.BooleanVar(value=False)
        self.vision_perspective_var: tk.StringVar = tk.StringVar(value="White at Bottom (Normal)")

        # Global Hotkey Binds for Screen Region Selector
        self.root.bind("<Control-s>", lambda event: self._launch_screen_selector())
        self.root.bind("<F9>", lambda event: self._launch_screen_selector())

        # Apply UI themes and styling
        self._configure_styles()

        # Build main interface
        self._build_ui()

        # Start background engine
        self.engine.log_callback = self._on_engine_log
        self.engine.start()
        self._sync_engine_options()

        # Setup periodic engine polling and window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_engine)

        self._update_board_and_diagnostics()

    def _configure_styles(self) -> None:
        """Applies modern dark-palette ttk styles."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1E1E2E")
        style.configure("TLabel", background="#1E1E2E", foreground="#CDD6F4", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#89B4FA")
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=5)
        style.configure("TCheckbutton", background="#181825", foreground="#CDD6F4")
        style.configure("TCombobox", padding=4)
        style.configure("Treeview", background="#11111B", foreground="#CDD6F4", fieldbackground="#11111B", rowheight=22)
        style.map("Treeview", background=[("selected", "#45475A")], foreground=[("selected", "#89B4FA")])

    def _build_ui(self) -> None:
        """Constructs top status bar, board area with vertical eval bar, and tabbed control panels."""
        # Top Header & Status Bar
        header_frame: tk.Frame = tk.Frame(self.root, bg="#181825", height=48)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        self.status_label: tk.Label = tk.Label(
            header_frame,
            text="White to move",
            font=("Segoe UI", 13, "bold"),
            bg="#181825",
            fg="#A6E3A1",
            padx=16,
            pady=10,
        )
        self.status_label.pack(side=tk.LEFT)

        self.opening_banner: tk.Label = tk.Label(
            header_frame,
            text="Opening: Starting Position",
            font=("Segoe UI", 11, "italic"),
            bg="#181825",
            fg="#89B4FA",
            padx=16,
            pady=10,
        )
        self.opening_banner.pack(side=tk.RIGHT)

        # Main Workspace Container
        main_container: tk.Frame = tk.Frame(self.root, bg="#1E1E2E")
        main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Left Column: Vertical Eval Bar + Chess Board Canvas
        board_section: tk.Frame = tk.Frame(main_container, bg="#1E1E2E")
        board_section.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 12))

        board_row: tk.Frame = tk.Frame(board_section, bg="#1E1E2E")
        board_row.pack(side=tk.TOP)

        # Vertical Evaluation Bar
        eval_bar_frame: tk.Frame = tk.Frame(board_row, bg="#181825", width=34, height=560)
        eval_bar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        eval_bar_frame.pack_propagate(False)

        self.eval_badge: tk.Label = tk.Label(
            eval_bar_frame,
            text="0.0",
            font=("Segoe UI", 9, "bold"),
            bg="#181825",
            fg="#CDD6F4",
            pady=2,
        )
        self.eval_badge.pack(side=tk.TOP, fill=tk.X)

        self.vertical_eval_canvas: tk.Canvas = tk.Canvas(
            eval_bar_frame,
            width=26,
            height=530,
            bg="#181825",
            highlightthickness=1,
            highlightbackground="#45475A",
        )
        self.vertical_eval_canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Chessboard Canvas
        self.canvas: BoardCanvas = BoardCanvas(board_row, board_size=560, on_move=self._on_player_move)
        self.canvas.pack(side=tk.LEFT)

        # Board Controls Toolbar
        board_ctrl: tk.Frame = tk.Frame(board_section, bg="#1E1E2E", pady=8)
        board_ctrl.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            board_ctrl,
            text="🔄 Flip Board",
            command=self._flip_board,
            bg="#313244",
            fg="#CDD6F4",
            activebackground="#45475A",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            board_ctrl,
            text="↩ Undo Move",
            command=self._undo_move,
            bg="#313244",
            fg="#CDD6F4",
            activebackground="#45475A",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            board_ctrl,
            text="⚡ Move Now",
            command=self._trigger_engine_turn,
            bg="#45475A",
            fg="#89B4FA",
            activebackground="#585B70",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            board_ctrl,
            text="🏳 Resign",
            command=self._resign_game,
            bg="#313244",
            fg="#F38BA8",
            activebackground="#45475A",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            board_ctrl,
            text="🎯 Vision (F9)",
            command=self._launch_screen_selector,
            bg="#313244",
            fg="#A6E3A1",
            activebackground="#45475A",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        # Right Column: Notebook Tabs
        right_container: tk.Frame = tk.Frame(main_container, bg="#181825", relief=tk.FLAT)
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(right_container)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab 1: Engine Controls & Options
        controls_tab: tk.Frame = tk.Frame(notebook, bg="#181825", padx=10, pady=10)
        notebook.add(controls_tab, text=" Engine & Book Options ")

        # Tab 2: Notation & PGN Manager
        notation_tab: tk.Frame = tk.Frame(notebook, bg="#181825", padx=10, pady=10)
        notebook.add(notation_tab, text=" Notation & PGN ")

        # Tab 3: Opening Book Explorer & Educational Coach
        book_tab: tk.Frame = tk.Frame(notebook, bg="#181825", padx=10, pady=10)
        notebook.add(book_tab, text=" Book & Strategy Coach ")

        # Tab 4: Live Telemetry & UCI Logs
        log_tab: tk.Frame = tk.Frame(notebook, bg="#181825", padx=10, pady=10)
        notebook.add(log_tab, text=" Telemetry & Logs ")

        # Tab 5: Screen Vision & Live Play
        vision_tab: tk.Frame = tk.Frame(notebook, bg="#181825", padx=10, pady=10)
        notebook.add(vision_tab, text=" 🎯 Vision & Live Play ")

        self._build_controls_tab(controls_tab)
        self._build_notation_tab(notation_tab)
        self._build_book_tab(book_tab)
        self._build_log_tab(log_tab)
        self._build_vision_tab(vision_tab)

    def _build_controls_tab(self, parent: tk.Frame) -> None:
        """Constructs engine settings, depth/time controls, and opening book preferences."""
        # 1. Match Settings
        match_group = tk.LabelFrame(
            parent,
            text=" Match Configuration ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=10,
            pady=8,
        )
        match_group.pack(fill=tk.X, pady=(0, 8))

        tk.Label(match_group, text="Game Mode:", bg="#181825", fg="#CDD6F4").grid(row=0, column=0, sticky="w", pady=4)
        mode_cb = ttk.Combobox(
            match_group,
            textvariable=self.game_mode,
            values=[
                "Human (White) vs Engine",
                "Engine (White) vs Human",
                "Human vs Human",
                "Engine vs Engine (Spectator)",
            ],
            state="readonly",
            width=28,
        )
        mode_cb.grid(row=0, column=1, sticky="w", pady=4, padx=6)
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        tk.Label(match_group, text="Time Mode:", bg="#181825", fg="#CDD6F4").grid(row=1, column=0, sticky="w", pady=4)
        time_mode_cb = ttk.Combobox(
            match_group,
            textvariable=self.time_mode_var,
            values=["Fixed Depth", "Move Time Limit"],
            state="readonly",
            width=28,
        )
        time_mode_cb.grid(row=1, column=1, sticky="w", pady=4, padx=6)
        time_mode_cb.bind("<<ComboboxSelected>>", self._on_time_mode_change)

        tk.Label(match_group, text="Search Depth (1-8):", bg="#181825", fg="#CDD6F4").grid(row=2, column=0, sticky="w", pady=4)
        depth_spin = tk.Spinbox(
            match_group,
            from_=1,
            to=8,
            textvariable=self.depth_var,
            width=6,
            bg="#313244",
            fg="#CDD6F4",
            buttonbackground="#45475A",
            command=self._sync_engine_options,
        )
        depth_spin.grid(row=2, column=1, sticky="w", pady=4, padx=6)

        tk.Label(match_group, text="Move Time (ms):", bg="#181825", fg="#CDD6F4").grid(row=3, column=0, sticky="w", pady=4)
        time_spin = tk.Spinbox(
            match_group,
            from_=100,
            to=15000,
            increment=250,
            textvariable=self.movetime_var,
            width=6,
            bg="#313244",
            fg="#CDD6F4",
            buttonbackground="#45475A",
            command=self._sync_engine_options,
        )
        time_spin.grid(row=3, column=1, sticky="w", pady=4, padx=6)

        # 2. Opening Book Options
        book_group = tk.LabelFrame(
            parent,
            text=" Opening Repertoire Settings ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=10,
            pady=8,
        )
        book_group.pack(fill=tk.X, pady=(0, 8))

        book_check = tk.Checkbutton(
            book_group,
            text="Enable Opening Book",
            variable=self.book_var,
            command=self._sync_engine_options,
            bg="#181825",
            fg="#CDD6F4",
            selectcolor="#313244",
            activebackground="#181825",
            activeforeground="#CDD6F4",
        )
        book_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=3)

        tk.Label(book_group, text="Preferred Repertoire:", bg="#181825", fg="#CDD6F4").grid(row=1, column=0, sticky="w", pady=4)
        rep_cb = ttk.Combobox(
            book_group,
            textvariable=self.repertoire_var,
            values=[
                "Catalan & Caro-Kann (Preferred)",
                "Catalan Only (White)",
                "Caro-Kann Only (Black)",
                "Full Tournament Repertoire",
            ],
            state="readonly",
            width=28,
        )
        rep_cb.grid(row=1, column=1, sticky="w", pady=4, padx=6)
        rep_cb.bind("<<ComboboxSelected>>", lambda e: self._sync_engine_options())

        tk.Label(book_group, text="Selection Strategy:", bg="#181825", fg="#CDD6F4").grid(row=2, column=0, sticky="w", pady=4)
        sel_cb = ttk.Combobox(
            book_group,
            textvariable=self.selection_var,
            values=[
                "Best Move (Deterministic)",
                "Weighted Random",
                "Uniform Random",
            ],
            state="readonly",
            width=28,
        )
        sel_cb.grid(row=2, column=1, sticky="w", pady=4, padx=6)
        sel_cb.bind("<<ComboboxSelected>>", lambda e: self._sync_engine_options())

        tk.Label(book_group, text="Book Max Ply:", bg="#181825", fg="#CDD6F4").grid(row=3, column=0, sticky="w", pady=4)
        ply_spin = tk.Spinbox(
            book_group,
            from_=2,
            to=50,
            textvariable=self.max_ply_var,
            width=6,
            bg="#313244",
            fg="#CDD6F4",
            buttonbackground="#45475A",
            command=self._sync_engine_options,
        )
        ply_spin.grid(row=3, column=1, sticky="w", pady=4, padx=6)

        # 3. Educational Coach Mini Banner on Dashboard
        coach_mini = tk.LabelFrame(
            parent,
            text=" 💡 Opening Strategic Insight ",
            font=("Segoe UI", 9, "bold"),
            bg="#181825",
            fg="#A6E3A1",
            padx=10,
            pady=6,
        )
        coach_mini.pack(fill=tk.X, pady=(0, 8))

        self.mini_coach_title: tk.Label = tk.Label(
            coach_mini,
            text="Starting Position",
            font=("Segoe UI", 9, "bold"),
            bg="#181825",
            fg="#89B4FA",
            anchor="w",
        )
        self.mini_coach_title.pack(fill=tk.X)

        self.mini_coach_text: tk.Label = tk.Label(
            coach_mini,
            text="White aims for Catalan (1. d4 / 3. g3); Black aims for Caro-Kann (1. e4 c6).",
            font=("Segoe UI", 9),
            bg="#181825",
            fg="#CDD6F4",
            wraplength=460,
            justify=tk.LEFT,
            anchor="w",
        )
        self.mini_coach_text.pack(fill=tk.X, pady=(2, 0))

        # 4. Action Buttons & Display Options
        btn_frame = tk.Frame(parent, bg="#181825")
        btn_frame.pack(fill=tk.X, pady=(2, 0))

        tk.Button(
            btn_frame,
            text="✨ New Game",
            command=self._new_game,
            bg="#A6E3A1",
            fg="#11111B",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=5,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_frame,
            text="⏹ Stop Calculation",
            command=self._stop_engine,
            bg="#F38BA8",
            fg="#11111B",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=5,
        ).pack(side=tk.LEFT, padx=(0, 6))

        arrow_check = tk.Checkbutton(
            btn_frame,
            text="Show Best Move Arrow",
            variable=self.show_arrow_var,
            command=self._on_toggle_arrow,
            bg="#181825",
            fg="#CDD6F4",
            selectcolor="#313244",
            activebackground="#181825",
            activeforeground="#CDD6F4",
        )
        arrow_check.pack(side=tk.RIGHT)

    def _build_notation_tab(self, parent: tk.Frame) -> None:
        """Constructs move notation history table, move navigation toolbar, and PGN export buttons."""
        top_bar = tk.Frame(parent, bg="#181825")
        top_bar.pack(fill=tk.X, pady=(0, 6))

        tk.Label(top_bar, text="Played Moves (SAN):", bg="#181825", fg="#89B4FA", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        # Move Navigation Toolbar
        nav_frame = tk.Frame(top_bar, bg="#181825")
        nav_frame.pack(side=tk.RIGHT)

        tk.Button(nav_frame, text="⏮ First", command=self._nav_first, bg="#313244", fg="#CDD6F4", relief=tk.FLAT, padx=6).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="◀ Prev", command=self._nav_prev, bg="#313244", fg="#CDD6F4", relief=tk.FLAT, padx=6).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="Next ▶", command=self._nav_next, bg="#313244", fg="#CDD6F4", relief=tk.FLAT, padx=6).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="Current ⏭", command=self._nav_last, bg="#313244", fg="#CDD6F4", relief=tk.FLAT, padx=6).pack(side=tk.LEFT, padx=2)

        # Move Notation Table
        tree_container = tk.Frame(parent, bg="#11111B")
        tree_container.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        columns = ("move_num", "white_move", "black_move")
        self.move_tree = ttk.Treeview(tree_container, columns=columns, show="headings", height=12)
        self.move_tree.heading("move_num", text="#")
        self.move_tree.heading("white_move", text="White")
        self.move_tree.heading("black_move", text="Black")
        self.move_tree.column("move_num", width=45, anchor="center")
        self.move_tree.column("white_move", width=110, anchor="center")
        self.move_tree.column("black_move", width=110, anchor="center")

        tree_scroll = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.move_tree.yview)
        self.move_tree.configure(yscrollcommand=tree_scroll.set)
        self.move_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.move_tree.bind("<<TreeviewSelect>>", self._on_tree_move_selected)

        # PGN & FEN Management Buttons
        export_group = tk.LabelFrame(
            parent,
            text=" PGN & FEN Export / Import ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=8,
            pady=6,
        )
        export_group.pack(fill=tk.X)

        tk.Button(
            export_group,
            text="📋 Copy PGN to Clipboard",
            command=self._copy_pgn_clipboard,
            bg="#313244",
            fg="#CDD6F4",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            export_group,
            text="💾 Save PGN to File",
            command=self._save_pgn_file,
            bg="#313244",
            fg="#CDD6F4",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            export_group,
            text="📋 Copy FEN",
            command=self._copy_fen,
            bg="#313244",
            fg="#CDD6F4",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            export_group,
            text="📥 Load FEN",
            command=self._load_fen,
            bg="#313244",
            fg="#CDD6F4",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=3)

    def _build_book_tab(self, parent: tk.Frame) -> None:
        """Constructs opening diagnostics, book candidate moves table, and the Educational Coach card."""
        info_box = tk.Frame(parent, bg="#11111B", padx=10, pady=6)
        info_box.pack(fill=tk.X, pady=(0, 6))

        self.book_status_label = tk.Label(
            info_box,
            text="📖 Book Status: Active (In Book)",
            font=("Segoe UI", 10, "bold"),
            bg="#11111B",
            fg="#A6E3A1",
        )
        self.book_status_label.pack(anchor="w")

        self.book_opening_desc = tk.Label(
            info_box,
            text="Repertoire: Catalan (White 1. d4) / Caro-Kann (Black 1...c6)",
            font=("Segoe UI", 9),
            bg="#11111B",
            fg="#CDD6F4",
        )
        self.book_opening_desc.pack(anchor="w", pady=(2, 0))

        tk.Label(parent, text="Available Book Candidates (Click to Preview Strategy):", bg="#181825", fg="#89B4FA", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))

        # Candidates Table
        cols = ("san", "uci", "weight", "prob")
        self.cand_tree = ttk.Treeview(parent, columns=cols, show="headings", height=5)
        self.cand_tree.heading("san", text="Move (SAN)")
        self.cand_tree.heading("uci", text="UCI")
        self.cand_tree.heading("weight", text="Priority Weight")
        self.cand_tree.heading("prob", text="Probability (%)")
        self.cand_tree.column("san", width=95, anchor="center")
        self.cand_tree.column("uci", width=75, anchor="center")
        self.cand_tree.column("weight", width=95, anchor="center")
        self.cand_tree.column("prob", width=95, anchor="center")
        self.cand_tree.pack(fill=tk.X, pady=(0, 6))
        self.cand_tree.bind("<<TreeviewSelect>>", self._on_candidate_selected)

        # Educational Grandmaster Coach Card
        coach_card = tk.LabelFrame(
            parent,
            text=" 🎓 Grandmaster Strategic Coach & Move Analysis ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#A6E3A1",
            padx=10,
            pady=8,
        )
        coach_card.pack(fill=tk.BOTH, expand=True)
        coach_card.bind("<Configure>", self._on_coach_resize)

        self.coach_header: tk.Label = tk.Label(
            coach_card,
            text="Move: - | Variation: -",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            anchor="w",
        )
        self.coach_header.pack(fill=tk.X, pady=(0, 4))

        tk.Label(coach_card, text="🎯 Strategic Intent:", font=("Segoe UI", 9, "bold"), bg="#181825", fg="#F9E2AF", anchor="w").pack(fill=tk.X)
        self.coach_intent: tk.Label = tk.Label(
            coach_card,
            text="-",
            font=("Segoe UI", 9),
            bg="#181825",
            fg="#CDD6F4",
            wraplength=480,
            justify=tk.LEFT,
            anchor="w",
        )
        self.coach_intent.pack(fill=tk.X, pady=(0, 4))

        tk.Label(coach_card, text="💡 Key Positional Ideas:", font=("Segoe UI", 9, "bold"), bg="#181825", fg="#F9E2AF", anchor="w").pack(fill=tk.X)
        self.coach_ideas: tk.Label = tk.Label(
            coach_card,
            text="-",
            font=("Segoe UI", 9),
            bg="#181825",
            fg="#CDD6F4",
            wraplength=480,
            justify=tk.LEFT,
            anchor="w",
        )
        self.coach_ideas.pack(fill=tk.X, pady=(0, 4))

        tk.Label(coach_card, text="🚩 Positional Goals & Future Plans:", font=("Segoe UI", 9, "bold"), bg="#181825", fg="#F9E2AF", anchor="w").pack(fill=tk.X)
        self.coach_goals: tk.Label = tk.Label(
            coach_card,
            text="-",
            font=("Segoe UI", 9),
            bg="#181825",
            fg="#CDD6F4",
            wraplength=480,
            justify=tk.LEFT,
            anchor="w",
        )
        self.coach_goals.pack(fill=tk.X)

    def _on_coach_resize(self, event: tk.Event) -> None:
        """Dynamically adjusts text wraplength to parent frame width."""
        wrap_width: int = max(320, event.width - 24)
        self.coach_intent.config(wraplength=wrap_width)
        self.coach_ideas.config(wraplength=wrap_width)
        self.coach_goals.config(wraplength=wrap_width)

    def _build_log_tab(self, parent: tk.Frame) -> None:
        """Constructs engine telemetry metrics grid and raw UCI communications log."""
        metrics_frame = tk.LabelFrame(
            parent,
            text=" Real-Time Search Metrics ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=8,
            pady=6,
        )
        metrics_frame.pack(fill=tk.X, pady=(0, 8))

        grid = tk.Frame(metrics_frame, bg="#181825")
        grid.pack(fill=tk.X)

        self.telem_depth = self._add_stat(grid, "Search Depth:", "0", 0, 0)
        self.telem_eval = self._add_stat(grid, "Centipawn Score:", "0.00", 0, 1)
        self.telem_nodes = self._add_stat(grid, "Nodes Evaluated:", "0", 1, 0)
        self.telem_nps = self._add_stat(grid, "Speed (NPS):", "0", 1, 1)
        self.telem_time = self._add_stat(grid, "Time Taken:", "0 ms", 2, 0)
        self.telem_source = self._add_stat(grid, "Move Source:", "Standby", 2, 1)

        tk.Label(metrics_frame, text="Principal Variation (PV):", bg="#181825", fg="#CDD6F4").pack(anchor="w", pady=(6, 2))
        self.pv_label = tk.Label(
            metrics_frame,
            text="-",
            bg="#11111B",
            fg="#F9E2AF",
            font=("Consolas", 10),
            wraplength=450,
            justify=tk.LEFT,
            padx=8,
            pady=4,
        )
        self.pv_label.pack(fill=tk.X)

        # Raw UCI Protocol Stream
        tk.Label(parent, text="UCI Protocol Console:", bg="#181825", fg="#89B4FA", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))
        self.log_text: tk.Text = tk.Text(parent, height=9, bg="#11111B", fg="#A6ADC8", font=("Consolas", 9), relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _add_stat(self, parent: tk.Frame, label_text: str, default_val: str, row: int, col: int) -> tk.Label:
        box: tk.Frame = tk.Frame(parent, bg="#11111B", padx=8, pady=4, relief=tk.FLAT)
        box.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)
        parent.columnconfigure(col, weight=1)

        tk.Label(box, text=label_text, font=("Segoe UI", 8), bg="#11111B", fg="#A6ADC8").pack(anchor="w")
        val_label: tk.Label = tk.Label(box, text=default_val, font=("Segoe UI", 10, "bold"), bg="#11111B", fg="#89B4FA")
        val_label.pack(anchor="w")
        return val_label

    def _sync_engine_options(self) -> None:
        """Sends all current configuration settings to the UCI engine process."""
        rep_map: dict[str, str] = {
            "Catalan & Caro-Kann (Preferred)": "Catalan_CaroKann",
            "Catalan Only (White)": "Catalan_White",
            "Caro-Kann Only (Black)": "CaroKann_Black",
            "Full Tournament Repertoire": "Tournament",
        }
        sel_map: dict[str, str] = {
            "Best Move (Deterministic)": "BestMove",
            "Weighted Random": "Weighted",
            "Uniform Random": "Random",
        }

        uci_rep: str = rep_map.get(self.repertoire_var.get(), "Catalan_CaroKann")
        uci_sel: str = sel_map.get(self.selection_var.get(), "BestMove")

        self.engine.set_book_options(
            own_book=self.book_var.get(),
            repertoire=uci_rep,
            selection=uci_sel,
            max_ply=self.max_ply_var.get(),
        )
        self.engine.set_option("SearchDepth", self.depth_var.get())

        movetime: int = self.movetime_var.get() if self.time_mode_var.get() == "Move Time Limit" else 0
        self.engine.set_option("MoveTime", movetime)

        self._refresh_book_candidates_panel()

    def _update_board_and_diagnostics(self) -> None:
        """Synchronizes board graphics, evaluation bar, opening classification, and educational displays."""
        last_move: chess.Move | None = self.board.peek() if self.board.move_stack else None
        hint: chess.Move | None = self.suggested_move if self.show_arrow_var.get() else None
        self.canvas.set_board(self.board, last_move=last_move, hint_move=hint)

        # Update detected opening banner
        opening_title: str = identify_opening(self.board)
        self.opening_banner.config(text=f"Opening: {opening_title}")

        # Update vertical and badge evaluation
        self._update_eval_display()

        # Update move notation list
        self._update_notation_table()

        # Update candidates list
        self._refresh_book_candidates_panel()

        # Update educational coach displays
        self._update_educational_coach(last_move)

        # Update turn status
        self._update_status()

    def _update_educational_coach(self, last_move: chess.Move | None = None) -> None:
        """Retrieves and displays educational strategic explanation for current position or last move."""
        summary: MoveExplanation = get_educational_summary(self.board, last_move=last_move)
        self._render_coach_card(summary, is_candidate=False)

        # Update dashboard mini coach card
        self.mini_coach_title.config(text=f"{summary.variation_name} ({summary.move_san})")
        self.mini_coach_text.config(text=summary.strategic_intent)

    def _render_coach_card(self, exp: MoveExplanation, is_candidate: bool = False) -> None:
        """Renders explanation content onto the Coach Card."""
        prefix: str = "Candidate Preview: " if is_candidate else "Played Move: "
        self.coach_header.config(
            text=f"{prefix}{exp.move_san}  |  {exp.variation_name}",
            fg="#F9E2AF" if is_candidate else "#89B4FA",
        )
        self.coach_intent.config(text=exp.strategic_intent)
        self.coach_ideas.config(text=exp.key_ideas)
        self.coach_goals.config(text=exp.positional_goals)

    def _on_candidate_selected(self, event=None) -> None:
        """Invoked when user clicks a candidate move in the Book Explorer table."""
        selected = self.cand_tree.selection()
        if not selected:
            return

        item = self.cand_tree.item(selected[0])
        values = item.get("values", [])
        if len(values) >= 2:
            uci_str: str = str(values[1])
            try:
                move: chess.Move = chess.Move.from_uci(uci_str)
                exp: MoveExplanation | None = get_candidate_explanation(self.board, move)
                if exp is not None:
                    self._render_coach_card(exp, is_candidate=True)
            except Exception:
                pass

    def _update_eval_display(self) -> None:
        """Renders the graphical vertical evaluation bar and numerical readout badge."""
        self.vertical_eval_canvas.delete("all")
        height: int = self.vertical_eval_canvas.winfo_height()
        width: int = self.vertical_eval_canvas.winfo_width()
        if height <= 1:
            height = 530
        if width <= 1:
            width = 26

        if self.current_mate_score is not None:
            badge_text: str = f"M{self.current_mate_score}"
            fraction: float = 1.0 if self.current_mate_score > 0 else 0.0
        else:
            score_cp: int = max(-1000, min(1000, self.current_eval_cp))
            badge_text = f"{self.current_eval_cp / 100.0:+.2f}"
            fraction = (score_cp + 1000) / 2000.0

        self.eval_badge.config(text=badge_text)

        if not self.canvas.is_flipped:
            white_h: int = int(height * fraction)
            black_h: int = height - white_h
            self.vertical_eval_canvas.create_rectangle(0, 0, width, black_h, fill="#313244", outline="")
            self.vertical_eval_canvas.create_rectangle(0, black_h, width, height, fill="#FFFFFF", outline="")
            self.vertical_eval_canvas.create_line(0, height // 2, width, height // 2, fill="#89B4FA", width=2)
        else:
            black_h = int(height * fraction)
            white_h = height - black_h
            self.vertical_eval_canvas.create_rectangle(0, 0, width, white_h, fill="#FFFFFF", outline="")
            self.vertical_eval_canvas.create_rectangle(0, white_h, width, height, fill="#313244", outline="")
            self.vertical_eval_canvas.create_line(0, height // 2, width, height // 2, fill="#89B4FA", width=2)

    def _refresh_book_candidates_panel(self) -> None:
        """Populates the Opening Book Explorer table with candidate moves for current position."""
        for item in self.cand_tree.get_children():
            self.cand_tree.delete(item)

        if not self.book_var.get() or self.board.ply() >= self.max_ply_var.get():
            self.book_status_label.config(text="⚙️ Book Status: Search Active (Out of Book)", fg="#89B4FA")
            return

        rep_map = {
            "Catalan & Caro-Kann (Preferred)": "catalan_carokann",
            "Catalan Only (White)": "catalan_white",
            "Caro-Kann Only (Black)": "carokann_black",
            "Full Tournament Repertoire": "tournament",
        }
        rep_key: str = rep_map.get(self.repertoire_var.get(), "catalan_carokann")
        candidates = get_book_candidates(self.board, repertoire=rep_key)

        if candidates:
            self.book_status_label.config(text=f"📖 Book Status: Active ({len(candidates)} candidate moves)", fg="#A6E3A1")
            for move, weight, prob in candidates:
                san_move: str = self.board.san(move)
                self.cand_tree.insert("", tk.END, values=(san_move, move.uci(), weight, f"{prob}%"))
        else:
            self.book_status_label.config(text="⚙️ Book Status: Line Concluded (Handing off to Alpha-Beta)", fg="#FAB387")

    def _update_notation_table(self) -> None:
        """Rebuilds the SAN move history table."""
        for item in self.move_tree.get_children():
            self.move_tree.delete(item)

        temp_board = chess.Board()
        rows: list[tuple[str, str, str]] = []

        for idx, uci_move in enumerate(self.move_history_uci):
            try:
                move = chess.Move.from_uci(uci_move)
                san = temp_board.san(move)
                temp_board.push(move)

                move_idx: int = idx // 2
                if idx % 2 == 0:
                    rows.append((f"{move_idx + 1}.", san, ""))
                else:
                    curr = rows[-1]
                    rows[-1] = (curr[0], curr[1], san)
            except Exception:
                pass

        for row in rows:
            self.move_tree.insert("", tk.END, values=row)

        children = self.move_tree.get_children()
        if children:
            self.move_tree.see(children[-1])

    def _on_player_move(self, move: chess.Move) -> None:
        """Handles a completed legal move made by the human player."""
        if self.is_engine_thinking or self.board.is_game_over():
            return

        self.board.push(move)
        self.move_history_uci.append(move.uci())
        self.history_index = len(self.move_history_uci)
        self.suggested_move = None

        self._update_board_and_diagnostics()

        if not self.board.is_game_over() and self._is_engine_turn():
            self._trigger_engine_turn()

    def _trigger_engine_turn(self) -> None:
        """Sends board position and search command to UCI engine."""
        if self.is_engine_thinking or self.board.is_game_over():
            return

        self.is_engine_thinking = True
        side: str = "White" if self.board.turn == chess.WHITE else "Black"
        self.status_label.config(text=f"Engine calculating ({side})...", fg="#F9E2AF")

        self.engine.set_position(self.move_history_uci)
        movetime: int | None = self.movetime_var.get() if self.time_mode_var.get() == "Move Time Limit" else None
        self.engine.go(depth=self.depth_var.get(), movetime_ms=movetime)

    def _poll_engine(self) -> None:
        """Periodically reads engine messages from the subprocess output queue."""
        messages: list[str] = self.engine.poll_messages()
        for msg in messages:
            self._handle_engine_message(msg)

        self.root.after(50, self._poll_engine)

    def _handle_engine_message(self, line: str) -> None:
        """Parses engine telemetry and bestmove outputs."""
        tokens: list[str] = line.split()
        if not tokens:
            return

        if tokens[0] == "info":
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
                        norm_score: int = val if self.board.turn == chess.WHITE else -val
                        self.current_eval_cp = norm_score
                        self.current_mate_score = None
                        self.telem_eval.config(text=f"{norm_score / 100.0:+.2f}")
                    elif score_type == "mate":
                        self.current_mate_score = val if self.board.turn == chess.WHITE else -val
                        self.telem_eval.config(text=f"M{self.current_mate_score}")
                    self._update_eval_display()

            if "nodes" in tokens:
                n_idx: int = tokens.index("nodes") + 1
                if n_idx < len(tokens):
                    cnt: int = int(tokens[n_idx])
                    self.telem_nodes.config(text=f"{cnt:,}")
                    if cnt == 1:
                        self.telem_source.config(text="Opening Book", fg="#A6E3A1")
                    else:
                        self.telem_source.config(text="Alpha-Beta Search", fg="#89B4FA")

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
                pv_str: str = " ".join(tokens[pv_idx : pv_idx + 6])
                self.pv_label.config(text=pv_str)
                if pv_idx < len(tokens):
                    try:
                        self.suggested_move = chess.Move.from_uci(tokens[pv_idx])
                        if self.show_arrow_var.get():
                            self.canvas.set_hint_move(self.suggested_move)
                    except Exception:
                        pass

        elif tokens[0] == "bestmove":
            self.is_engine_thinking = False
            best_move_str: str = tokens[1]
            if best_move_str != "(none)":
                try:
                    move: chess.Move = chess.Move.from_uci(best_move_str)
                    if move in self.board.legal_moves:
                        if getattr(self, "is_vision_counter_eval", False):
                            self.is_vision_counter_eval = False
                            self.engine_counter_move = move
                            san_counter = self.board.san(move)
                            if hasattr(self, "vision_counter_move_label"):
                                self.vision_counter_move_label.config(
                                    text=f"{san_counter} ({move.uci()})",
                                    fg="#00D2D3",
                                )
                            self.status_label.config(
                                text=f"Counter-move: {san_counter} ({move.uci()})",
                                fg="#00D2D3",
                            )
                            # Display both opponent move (amber) and engine counter-move (cyan) live on canvas
                            self.canvas.set_vision_moves(self.last_detected_opponent_move, move)

                            if self.vision_auto_play_var.get():
                                self.board.push(move)
                                self.move_history_uci.append(move.uci())
                                self.history_index = len(self.move_history_uci)
                        else:
                            self.board.push(move)
                            self.move_history_uci.append(move.uci())
                            self.history_index = len(self.move_history_uci)
                except Exception:
                    pass

            self.suggested_move = None
            self._update_board_and_diagnostics()

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
        return False

    def _on_mode_change(self, event=None) -> None:
        mode: str = self.game_mode.get()
        if mode == "Engine (White) vs Human":
            self.canvas.set_flipped(True)
        else:
            self.canvas.set_flipped(False)

        self._update_board_and_diagnostics()
        if not self.board.is_game_over() and self._is_engine_turn():
            self._trigger_engine_turn()

    def _on_time_mode_change(self, event=None) -> None:
        self._sync_engine_options()

    def _on_toggle_arrow(self) -> None:
        self.canvas.show_arrow = self.show_arrow_var.get()
        self.canvas.redraw()

    def _flip_board(self) -> None:
        self.canvas.set_flipped(not self.canvas.is_flipped)
        self._update_eval_display()

    def _undo_move(self) -> None:
        if self.is_engine_thinking:
            return

        if len(self.board.move_stack) >= 2 and self.game_mode.get() in ("Human (White) vs Engine", "Engine (White) vs Human"):
            self.board.pop()
            self.board.pop()
            self.move_history_uci.pop()
            self.move_history_uci.pop()
        elif len(self.board.move_stack) >= 1:
            self.board.pop()
            self.move_history_uci.pop()

        self.history_index = len(self.move_history_uci)
        self.suggested_move = None
        self._update_board_and_diagnostics()

    def _resign_game(self) -> None:
        if self.board.is_game_over():
            return
        loser: str = "White" if self.board.turn == chess.WHITE else "Black"
        winner: str = "Black" if loser == "White" else "White"
        if messagebox.askyesno("Resign Game", f"Confirm resignation by {loser}?"):
            self.status_label.config(text=f"{loser} resigned. {winner} wins! 🏆", fg="#F38BA8")
            self._stop_engine()

    def _new_game(self) -> None:
        if self.is_engine_thinking:
            self._stop_engine()

        self.board.reset()
        self.move_history_uci.clear()
        self.history_index = 0
        self.suggested_move = None
        self.current_eval_cp = 0
        self.current_mate_score = None

        self.engine.new_game()
        self.pv_label.config(text="-")
        self._sync_engine_options()
        self._update_board_and_diagnostics()

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
            turn_name: str = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"Check! ({turn_name} to move)", fg="#F38BA8")
        else:
            turn_str: str = "White to move" if self.board.turn == chess.WHITE else "Black to move"
            self.status_label.config(text=turn_str, fg="#A6E3A1")

    # Move History Review Navigation
    def _nav_first(self) -> None:
        self._display_historical_ply(0)

    def _nav_prev(self) -> None:
        self._display_historical_ply(max(0, self.history_index - 1))

    def _nav_next(self) -> None:
        self._display_historical_ply(min(len(self.move_history_uci), self.history_index + 1))

    def _nav_last(self) -> None:
        self._display_historical_ply(len(self.move_history_uci))

    def _display_historical_ply(self, ply: int) -> None:
        self.history_index = ply
        display_board = chess.Board()
        last_move: chess.Move | None = None
        for i in range(ply):
            m = chess.Move.from_uci(self.move_history_uci[i])
            display_board.push(m)
            last_move = m
        self.canvas.set_board(display_board, last_move=last_move)
        self.canvas.redraw()
        self._update_educational_coach(last_move)

    def _on_tree_move_selected(self, event=None) -> None:
        selected = self.move_tree.selection()
        if not selected:
            return
        item_id = selected[0]
        row_idx: int = self.move_tree.index(item_id)
        target_ply: int = min(len(self.move_history_uci), (row_idx + 1) * 2)
        self._display_historical_ply(target_ply)

    # PGN & FEN Management
    def _generate_pgn_text(self) -> str:
        """Generates standard PGN text including metadata headers."""
        headers: list[str] = [
            '[Event "MiniChess Championship Match"]',
            '[Site "MiniChess Studio"]',
            f'[Date "{datetime.datetime.now().strftime("%Y.%m.%d")}"]',
            '[Round "1"]',
            '[White "Player White"]',
            '[Black "Player Black"]',
        ]
        result: str = "*"
        if self.board.is_checkmate():
            result = "1-0" if self.board.turn == chess.BLACK else "0-1"
        elif self.board.is_game_over():
            result = "1/2-1/2"

        headers.append(f'[Result "{result}"]')
        headers.append(f'[ECO "{identify_opening(self.board)}"]')
        headers.append("")

        temp_board = chess.Board()
        move_tokens: list[str] = []
        for idx, uci_move in enumerate(self.move_history_uci):
            m = chess.Move.from_uci(uci_move)
            san = temp_board.san(m)
            temp_board.push(m)
            if idx % 2 == 0:
                move_tokens.append(f"{(idx // 2) + 1}. {san}")
            else:
                move_tokens.append(f"{san}")

        move_tokens.append(result)
        body: str = " ".join(move_tokens)
        return "\n".join(headers) + "\n" + body

    def _copy_pgn_clipboard(self) -> None:
        pgn: str = self._generate_pgn_text()
        self.root.clipboard_clear()
        self.root.clipboard_append(pgn)
        messagebox.showinfo("PGN Copied", "Current game PGN successfully copied to clipboard.")

    def _save_pgn_file(self) -> None:
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pgn",
            filetypes=[("PGN Files", "*.pgn"), ("All Files", "*.*")],
            title="Export Game PGN",
        )
        if not filepath:
            return

        try:
            pgn: str = self._generate_pgn_text()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pgn)
            messagebox.showinfo("PGN Saved", f"Game PGN exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save PGN file:\n{e}")

    def _copy_fen(self) -> None:
        fen: str = self.board.fen()
        self.root.clipboard_clear()
        self.root.clipboard_append(fen)
        messagebox.showinfo("FEN Copied", f"Current FEN copied to clipboard:\n{fen}")

    def _load_fen(self) -> None:
        raw_fen: str | None = simpledialog.askstring("Load FEN", "Paste standard FEN position:")
        if not raw_fen:
            return

        try:
            new_board = chess.Board(raw_fen.strip())
            self.board = new_board
            self.move_history_uci.clear()
            self.history_index = 0
            self.suggested_move = None
            self.engine.set_position([], fen=raw_fen.strip())
            self._update_board_and_diagnostics()
            messagebox.showinfo("Position Loaded", "FEN loaded successfully.")
        except Exception as e:
            messagebox.showerror("Invalid FEN", f"Could not parse FEN string:\n{e}")

    def _on_engine_log(self, text: str, is_command: bool) -> None:
        prefix: str = ">> " if is_command else "<< "
        self.log_text.insert(tk.END, f"{prefix}{text}\n")
        self.log_text.see(tk.END)

    def _build_vision_tab(self, parent: tk.Frame) -> None:
        """Constructs screen-capture region selection, live board recognition, and auto-counter controls."""
        # 1. Screen Region Selection Frame
        region_group = tk.LabelFrame(
            parent,
            text=" 🎯 Screen Region Selection ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=10,
            pady=8,
        )
        region_group.pack(fill=tk.X, pady=(0, 8))

        btn_row = tk.Frame(region_group, bg="#181825")
        btn_row.pack(fill=tk.X, pady=(0, 6))

        tk.Button(
            btn_row,
            text="🎯 Drag-Select (Ctrl+S/F9)",
            command=self._launch_screen_selector,
            bg="#45475A",
            fg="#89B4FA",
            activebackground="#585B70",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=5,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_row,
            text="🤖 Auto-Detect Board",
            command=self._auto_detect_screen_board,
            bg="#313244",
            fg="#A6E3A1",
            activebackground="#45475A",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=5,
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.vision_region_label: tk.Label = tk.Label(
            btn_row,
            text="Region: Not Selected",
            font=("Segoe UI", 9, "italic"),
            bg="#181825",
            fg="#BAC2DE",
        )
        self.vision_region_label.pack(side=tk.LEFT, pady=4)

        # Perspective & Alignment Configuration
        orient_row = tk.Frame(region_group, bg="#181825")
        orient_row.pack(fill=tk.X, pady=4)

        tk.Label(orient_row, text="Perspective:", bg="#181825", fg="#CDD6F4").pack(side=tk.LEFT, padx=(0, 4))
        orient_cb = ttk.Combobox(
            orient_row,
            textvariable=self.vision_perspective_var,
            values=["White at Bottom (Normal)", "Black at Bottom (Flipped)"],
            state="readonly",
            width=20,
        )
        orient_cb.pack(side=tk.LEFT, padx=(0, 8))
        orient_cb.bind("<<ComboboxSelected>>", self._on_vision_perspective_change)

        tk.Label(orient_row, text="Margin %:", bg="#181825", fg="#89B4FA").pack(side=tk.LEFT, padx=(0, 2))
        self.vision_margin_spin = ttk.Spinbox(
            orient_row,
            from_=0.0,
            to=15.0,
            increment=0.5,
            textvariable=self.vision_margin_var,
            width=5,
            command=self._on_margin_changed,
        )
        self.vision_margin_spin.pack(side=tk.LEFT, padx=(0, 8))
        self.vision_margin_spin.bind("<Return>", lambda e: self._on_margin_changed())

        tk.Label(orient_row, text="Inset %:", bg="#181825", fg="#A6E3A1").pack(side=tk.LEFT, padx=(0, 2))
        self.vision_inset_spin = ttk.Spinbox(
            orient_row,
            from_=5.0,
            to=25.0,
            increment=1.0,
            textvariable=self.vision_inset_var,
            width=5,
            command=self._on_inset_changed,
        )
        self.vision_inset_spin.pack(side=tk.LEFT, padx=(0, 10))
        self.vision_inset_spin.bind("<Return>", lambda e: self._on_inset_changed())

        tk.Button(
            orient_row,
            text="📷 Grid Inspector",
            command=self._toggle_grid_inspector,
            bg="#45475A",
            fg="#89B4FA",
            activebackground="#585B70",
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            padx=6,
            pady=2,
        ).pack(side=tk.LEFT)

        # 2. Live Capture & Polling Engine Frame
        engine_group = tk.LabelFrame(
            parent,
            text=" ⚡ Live Capture & Recognition Engine ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#A6E3A1",
            padx=10,
            pady=8,
        )
        engine_group.pack(fill=tk.X, pady=(0, 8))

        ctrl_row = tk.Frame(engine_group, bg="#181825")
        ctrl_row.pack(fill=tk.X, pady=(0, 6))

        self.vision_toggle_btn: tk.Button = tk.Button(
            ctrl_row,
            text="▶ Start Live Capture",
            command=self._toggle_vision_capture,
            bg="#2ECC71",
            fg="#11111B",
            activebackground="#27AE60",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=6,
        )
        self.vision_toggle_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.vision_status_badge: tk.Label = tk.Label(
            ctrl_row,
            text="● Vision Idle",
            font=("Segoe UI", 10, "bold"),
            bg="#11111B",
            fg="#BAC2DE",
            padx=10,
            pady=4,
            relief=tk.RIDGE,
        )
        self.vision_status_badge.pack(side=tk.LEFT, padx=(0, 10))

        tk.Checkbutton(
            engine_group,
            text="Auto-play engine counter-move on internal board",
            variable=self.vision_auto_play_var,
            bg="#181825",
            fg="#CDD6F4",
            activebackground="#181825",
            selectcolor="#11111B",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=4)

        # 3. Live Move Detection & Engine Counter-Response Frame
        move_group = tk.LabelFrame(
            parent,
            text=" 🧠 Move Detection & Engine Counter-Response ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#F9E2AF",
            padx=10,
            pady=8,
        )
        move_group.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        card_container = tk.Frame(move_group, bg="#11111B", padx=12, pady=10)
        card_container.pack(fill=tk.BOTH, expand=True)

        # Opponent move display
        tk.Label(
            card_container,
            text="Detected Opponent Move:",
            font=("Segoe UI", 9, "bold"),
            bg="#11111B",
            fg="#FFA500",
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))

        self.vision_opp_move_label: tk.Label = tk.Label(
            card_container,
            text="None yet (waiting for move)",
            font=("Segoe UI", 14, "bold"),
            bg="#11111B",
            fg="#FFA500",
        )
        self.vision_opp_move_label.grid(row=1, column=0, sticky="w", pady=(0, 10))

        # Counter-move display
        tk.Label(
            card_container,
            text="Recommended Engine Counter-Move:",
            font=("Segoe UI", 9, "bold"),
            bg="#11111B",
            fg="#00D2D3",
        ).grid(row=2, column=0, sticky="w", pady=(0, 2))

        self.vision_counter_move_label: tk.Label = tk.Label(
            card_container,
            text="None yet",
            font=("Segoe UI", 14, "bold"),
            bg="#11111B",
            fg="#00D2D3",
        )
        self.vision_counter_move_label.grid(row=3, column=0, sticky="w", pady=(0, 10))

        # Reset & Resync buttons
        action_row = tk.Frame(card_container, bg="#11111B")
        action_row.grid(row=4, column=0, sticky="w", pady=(8, 0))

        tk.Button(
            action_row,
            text="🔄 Reset Game & Resync",
            command=self._reset_game,
            bg="#313244",
            fg="#CDD6F4",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            action_row,
            text="⚡ Compute Counter-Move Now",
            command=self._compute_vision_counter_move,
            bg="#45475A",
            fg="#89B4FA",
            relief=tk.FLAT,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT)

        # 4. Live Screen FEN Debug View & Synchronization Frame
        fen_group = tk.LabelFrame(
            parent,
            text=" 🔍 Live Screen FEN Debug View ",
            font=("Segoe UI", 10, "bold"),
            bg="#181825",
            fg="#89B4FA",
            padx=10,
            pady=8,
        )
        fen_group.pack(fill=tk.X, pady=(0, 4))

        fen_row1 = tk.Frame(fen_group, bg="#181825")
        fen_row1.pack(fill=tk.X, pady=(0, 4))

        tk.Label(fen_row1, text="Detected Screen FEN:", font=("Segoe UI", 9, "bold"), bg="#181825", fg="#CDD6F4").pack(side=tk.LEFT)
        self.vision_fen_match_label: tk.Label = tk.Label(
            fen_row1,
            text="● Synchronized",
            font=("Segoe UI", 9, "bold"),
            bg="#181825",
            fg="#A6E3A1",
        )
        self.vision_fen_match_label.pack(side=tk.RIGHT)

        self.vision_fen_entry: tk.Entry = tk.Entry(
            fen_group,
            textvariable=self.vision_fen_var,
            font=("Consolas", 9),
            bg="#11111B",
            fg="#89B4FA",
            relief=tk.FLAT,
            readonlybackground="#11111B",
            state="readonly",
        )
        self.vision_fen_entry.pack(fill=tk.X, pady=(2, 6))

        fen_btn_row = tk.Frame(fen_group, bg="#181825")
        fen_btn_row.pack(fill=tk.X)

        tk.Button(
            fen_btn_row,
            text="📥 Force Sync Board to Screen FEN",
            command=self._force_sync_to_screen_fen,
            bg="#313244",
            fg="#A6E3A1",
            activebackground="#45475A",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=3,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            fen_btn_row,
            text="📋 Copy FEN",
            command=lambda: self._copy_text(self.vision_fen_var.get()),
            bg="#313244",
            fg="#CDD6F4",
            activebackground="#45475A",
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            padx=8,
            pady=3,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            fen_btn_row,
            text="📷 Grid Inspector",
            command=self._toggle_grid_inspector,
            bg="#45475A",
            fg="#89B4FA",
            activebackground="#585B70",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=3,
        ).pack(side=tk.LEFT)

    def _launch_screen_selector(self) -> None:
        """Opens full-screen overlay letting the user drag a bounding box around any screen chessboard."""
        ScreenRegionSelector(self.root, on_region_selected=self._on_screen_region_selected)

    def _on_screen_region_selected(self, bbox: tuple[int, int, int, int]) -> None:
        """Invoked when user confirms a bounding box selection."""
        left, top, width, height = bbox
        self.vision_sync.set_region(bbox)
        self.vision_region_label.config(text=f"Region: ({left}, {top}, {width}x{height} px)")
        self.status_label.config(text=f"Screen Region Set: {width}x{height} px", fg="#A6E3A1")
        if not self.vision_sync.is_running:
            self._toggle_vision_capture()

    def _auto_detect_screen_board(self) -> None:
        """Automatically scans display to localize high-contrast chessboard contour."""
        self.status_label.config(text="Scanning display for chessboard...", fg="#89B4FA")
        self.root.update_idletasks()
        best_box = self.vision_sync.auto_detect_board()
        if best_box is not None:
            left, top, width, height = best_box
            self.vision_region_label.config(text=f"Region: ({left}, {top}, {width}x{height} px) [Auto-Detected]")
            self.status_label.config(text=f"Chessboard detected: {width}x{height} px at ({left}, {top})", fg="#A6E3A1")
            if not self.vision_sync.is_running:
                self._toggle_vision_capture()
        else:
            self.status_label.config(text="No board contour detected. Use manual drag-select.", fg="#F38BA8")
            messagebox.showinfo(
                "Auto-Detect Result",
                "Could not find a distinct square chessboard contour on the current display.\n\nPlease ensure your chess board is visible on screen or use 'Drag-Select'.",
            )

    def _toggle_vision_capture(self) -> None:
        """Toggles background screen capture and board move synchronization."""
        if self.vision_sync.is_running:
            self.vision_sync.stop()
            self.vision_toggle_btn.config(
                text="▶ Start Live Capture",
                bg="#2ECC71",
                activebackground="#27AE60",
            )
            self.vision_status_badge.config(
                text="● Vision Stopped",
                fg="#BAC2DE",
            )
        else:
            started: bool = self.vision_sync.start()
            if started:
                self.vision_toggle_btn.config(
                    text="⏹ Stop Live Capture",
                    bg="#E74C3C",
                    activebackground="#C0392B",
                )
                self.vision_status_badge.config(
                    text="● Scanning Screen (Active)",
                    fg="#A6E3A1",
                )

    def _on_vision_status_update(self, status: str, is_active: bool, is_error: bool) -> None:
        """Receives status updates from the background vision thread."""
        def update_ui() -> None:
            if not hasattr(self, "vision_status_badge"):
                return
            fg_col: str = "#F38BA8" if is_error else ("#A6E3A1" if is_active else "#BAC2DE")
            self.vision_status_badge.config(text=f"● {status}", fg=fg_col)
            if is_error:
                self.status_label.config(text=status, fg="#F38BA8")
        self.root.after(0, update_ui)

    def _on_vision_move_detected(self, move: chess.Move) -> None:
        """Thread-safe callback triggered by BoardVisionSync when a legal move is detected on screen."""
        self.root.after(0, lambda: self._handle_vision_move(move))

    def _handle_vision_move(self, move: chess.Move) -> None:
        """Processes a detected opponent move on the main GUI thread."""
        if move not in self.board.legal_moves:
            return

        san_str: str = self.board.san(move)
        self.board.push(move)
        self.move_history_uci.append(move.uci())
        self.history_index = len(self.move_history_uci)
        self.last_detected_opponent_move = move

        self.vision_opp_move_label.config(
            text=f"{san_str} ({move.uci()})",
            fg="#FFA500",
        )
        self.vision_status_badge.config(
            text=f"● Opponent played {san_str}",
            fg="#A6E3A1",
        )
        self.status_label.config(
            text=f"Opponent move detected: {san_str}",
            fg="#FFA500",
        )

        # Update board canvas with opponent move arrow and refresh board
        self.canvas.detected_opponent_move = move
        self.canvas.hint_move = None
        self._update_board_and_diagnostics()

        if self.board.is_game_over():
            return

        # Automatically calculate engine counter-move
        self._compute_vision_counter_move()

    def _compute_vision_counter_move(self) -> None:
        """Triggers the engine to search for the optimal counter-move response."""
        if self.board.is_game_over():
            return

        self.status_label.config(text="Computing counter-move...", fg="#89B4FA")
        self.engine.set_position(self.move_history_uci)
        movetime: int | None = self.movetime_var.get() if self.time_mode_var.get() == "Move Time Limit" else None
        self.is_vision_counter_eval = True
        self.engine.go(depth=self.depth_var.get(), movetime_ms=movetime)

    def _on_vision_perspective_change(self, event=None) -> None:
        """Synchronizes perspective orientation between vision engine and board canvas."""
        flipped: bool = "Black" in self.vision_perspective_var.get()
        self.vision_sync.set_flipped(flipped)
        self.canvas.set_flipped(flipped)

    def _reset_game(self) -> None:
        """Resets the internal board, move history, engine state, and vision status to starting position."""
        self.board = chess.Board()
        self.move_history_uci.clear()
        self.history_index = 0
        self.suggested_move = None
        self.last_detected_opponent_move = None
        self.engine_counter_move = None
        self.canvas.detected_opponent_move = None
        self.canvas.hint_move = None
        self.engine.new_game()
        self.engine.set_position([])
        if hasattr(self, "vision_opp_move_label"):
            self.vision_opp_move_label.config(text="None yet (waiting for move)", fg="#FFA500")
            self.vision_counter_move_label.config(text="None yet", fg="#00D2D3")
        self._update_board_and_diagnostics()

    def _on_vision_fen_detected(
        self,
        fen: str,
        piece_grid: dict[chess.Square, chess.Piece | None],
        confidences: dict[chess.Square, float] | None = None,
    ) -> None:
        """Thread-safe callback updating the live FEN debug view in the UI."""
        def update_ui() -> None:
            if not hasattr(self, "vision_fen_var"):
                return
            self.vision_fen_var.set(fen)
            internal_placement: str = self.board.fen().split()[0]
            detected_placement: str = fen.split()[0]
            avg_conf_str = ""
            if confidences:
                avg_conf = (sum(confidences.values()) / len(confidences)) * 100.0
                avg_conf_str = f" ({avg_conf:.0f}% conf)"

            if internal_placement == detected_placement:
                self.vision_fen_match_label.config(text=f"● Synchronized{avg_conf_str}", fg="#A6E3A1")
            else:
                self.vision_fen_match_label.config(text=f"● Diff Detected{avg_conf_str}", fg="#FFA500")

        self.root.after(0, update_ui)

    def _force_sync_to_screen_fen(self) -> None:
        """Forces the internal board state to match the live detected screen FEN."""
        raw_fen: str = self.vision_fen_var.get().strip() if hasattr(self, "vision_fen_var") else ""
        if not raw_fen:
            return

        try:
            new_board: chess.Board = chess.Board(raw_fen)
            self.board = new_board
            self.move_history_uci.clear()
            self.history_index = 0
            self.suggested_move = None
            self.last_detected_opponent_move = None
            self.engine_counter_move = None
            self.canvas.detected_opponent_move = None
            self.canvas.hint_move = None
            self.engine.new_game()
            self.engine.set_position([], fen=raw_fen)
            self._update_board_and_diagnostics()
            self.status_label.config(text="Synchronized internal board to screen FEN", fg="#A6E3A1")
        except Exception as e:
            messagebox.showerror("Sync Error", f"Cannot parse detected FEN:\n{raw_fen}\n\nError: {e}")

    def _copy_text(self, text: str) -> None:
        """Copies given text to system clipboard."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_label.config(text="Copied to clipboard", fg="#89B4FA")

    def _toggle_grid_inspector(self) -> None:
        """Opens or focuses the Grid Alignment Inspector window showing the 8x8 overlay."""
        if self.grid_inspector is not None and self.grid_inspector.is_alive():
            try:
                self.grid_inspector.window.lift()
                self.grid_inspector.window.focus_set()
            except Exception:
                pass
            return

        self.grid_inspector = GridAlignmentInspector(
            parent=self.root,
            get_outer_margin=lambda: self.vision_sync.outer_margin_pct,
            set_outer_margin=self._set_outer_margin,
            get_inner_inset=lambda: self.vision_sync.inner_inset_pct,
            set_inner_inset=self._set_inner_inset,
            on_close_callback=self._on_grid_inspector_closed,
            on_toggle_overlay=lambda enabled: setattr(self.vision_sync, "show_debug_overlay", enabled),
        )

    def _set_outer_margin(self, val: float) -> None:
        """Updates outer margin percentage in vision engine and synchronizes UI controls."""
        self.vision_sync.set_outer_margin(val)
        if hasattr(self, "vision_margin_var"):
            self.vision_margin_var.set(val)

    def _set_inner_inset(self, val: float) -> None:
        """Updates inner inset percentage in vision engine and synchronizes UI controls."""
        self.vision_sync.set_inner_inset(val)
        if hasattr(self, "vision_inset_var"):
            self.vision_inset_var.set(val * 100.0)

    def _on_margin_changed(self) -> None:
        """Triggered when the user adjusts outer margin percentage in the main GUI."""
        try:
            val = float(self.vision_margin_var.get())
            self.vision_sync.set_outer_margin(val)
            if self.grid_inspector is not None and self.grid_inspector.is_alive():
                self.grid_inspector.outer_margin_var.set(val)
        except Exception:
            pass

    def _on_inset_changed(self) -> None:
        """Triggered when the user adjusts inner cell inset percentage in the main GUI."""
        try:
            val = float(self.vision_inset_var.get()) / 100.0
            self.vision_sync.set_inner_inset(val)
            if self.grid_inspector is not None and self.grid_inspector.is_alive():
                self.grid_inspector.inner_inset_var.set(val * 100.0)
        except Exception:
            pass

    def _on_grid_inspector_closed(self) -> None:
        """Callback invoked when the user closes the Grid Alignment Inspector."""
        self.grid_inspector = None

    def _on_vision_debug_frame(self, overlay_img: Image.Image) -> None:
        """Dispatches the live debug overlay frame to the Grid Alignment Inspector."""
        if self.grid_inspector is not None and self.grid_inspector.is_alive():
            self.root.after(
                0,
                lambda: self.grid_inspector.update_frame(overlay_img)
                if self.grid_inspector and self.grid_inspector.is_alive()
                else None,
            )

    def _on_close(self) -> None:
        """Terminates engine, inspector, and vision background threads cleanly."""
        if self.grid_inspector is not None and self.grid_inspector.is_alive():
            try:
                self.grid_inspector.window.destroy()
            except Exception:
                pass
        if hasattr(self, "vision_sync"):
            self.vision_sync.stop()
        self.engine.terminate()
        self.root.destroy()
