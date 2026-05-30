"""
Strip Packing Solver & Visualizer
==================================
Animated visualization of strip packing solvers.

Log format (JSONL):
    heuristic: start -> search* -> place -> ... -> done
    exact:     start -> done (placements embedded in done event)

Playback speed: each step is shown for
    max(slider_floor_ms,  inter_event_elapsed_ms / 0.75)
so the animation runs at 75% of real solve speed, with a human-visible floor.

Adding a new solver:
    1. Write a run(instance, sort_key, log_path) -> None function.
    2. Append a dict to SOLVER_REGISTRY below.

Run:    python visualizer.py
"""

import json
import threading
import tkinter as tk
from collections import deque
from tkinter import ttk

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from benchmarks import get_all_benchmarks, generate_random_instance
from solver_heuristic import solve_blf
from solver_exact import solve_exact
from solver_shelf import solve_nfdh, solve_ffdh
from solver_skyline import solve_skyline
from solver_sa import solve_sa

LOG_PATH = "strip_packing_vis.log"
_PAD = 32

_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#86BCB6",
    "#499894", "#E4A96A", "#D37295", "#A0CBE8", "#FFBE7D",
    "#8CD17D", "#B6992D", "#F1CE63", "#D4A6C8", "#FABFD2",
]

# -- Solver runner functions (called in background threads) -------------------

def _run_heuristic(instance, sort_key, log_path):
    """Run BLF. If sort_key='best', silently find the winner then re-run with logging."""
    if sort_key == "best":
        best_key, best_h = "height", float("inf")
        for sk in ("height", "width", "area", "perimeter"):
            r = solve_blf(instance, sort_key=sk, log_path=None)
            if r["height"] < best_h:
                best_h, best_key = r["height"], sk
        open(log_path, "w").close()
        solve_blf(instance, sort_key=best_key, log_path=log_path)
    else:
        solve_blf(instance, sort_key=sort_key, log_path=log_path)


def _run_exact(instance, sort_key, log_path):
    """Run CP-SAT exact solver. sort_key is ignored."""
    solve_exact(instance, time_limit_seconds=60.0, log_path=log_path)


def _run_nfdh(instance, sort_key, log_path):
    """Run Next Fit Decreasing Height shelf algorithm."""
    solve_nfdh(instance, log_path=log_path)


def _run_ffdh(instance, sort_key, log_path):
    """Run First Fit Decreasing Height shelf algorithm."""
    solve_ffdh(instance, log_path=log_path)


def _run_ml_advisor(instance, sort_key, log_path):
    """
    Use the trained ML model to predict the best heuristic, then run it
    with full step-by-step logging so the visualizer animates the placement.
    sort_key encodes the model type: "RF" (Random Forest) or "MLP" (neural net).
    Falls back to best-of-BLF if the model file is not found.
    """
    model_type = sort_key  # "RF" or "MLP"
    chosen_id = "blf_height"  # safe default

    try:
        import joblib
        import numpy as np
        from instance_features import extract_features

        if model_type == "MLP":
            import torch
            import torch.nn as nn
            bundle = joblib.load("best_solver_model_nn.pkl")

            class _MLP(nn.Module):
                def __init__(self, inp, nc):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(inp, 64), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, nc),
                    )
                def forward(self, x): return self.net(x)

            mlp = _MLP(bundle["input_dim"], bundle["n_classes"])
            mlp.load_state_dict(bundle["state_dict"])
            mlp.eval()
            feats = bundle["scaler"].transform([extract_features(instance)])
            with torch.no_grad():
                idx = int(mlp(torch.tensor(feats, dtype=torch.float32)).argmax(dim=1).item())
            chosen_id = bundle["solver_names"][idx]
        else:
            bundle = joblib.load("best_solver_model.pkl")
            feats = np.array(extract_features(instance), dtype=float).reshape(1, -1)
            idx = int(bundle["model"].predict(feats)[0])
            chosen_id = bundle["solver_names"][idx]
    except FileNotFoundError:
        best_h = float("inf")
        for sk in ("height", "width", "area", "perimeter"):
            r = solve_blf(instance, sort_key=sk)
            if r["height"] < best_h:
                best_h, chosen_id = r["height"], f"blf_{sk}"
    except Exception:
        pass  # use default

    # Emit an ml_choice event so the visualizer can show the oracle comparison.
    with open(log_path, "a", buffering=1) as _lf:
        _lf.write(json.dumps({"event": "ml_choice", "chosen_id": chosen_id,
                               "model_type": model_type}) + "\n")

    # Dispatch to the chosen solver WITH log_path so events are emitted
    # and the visualizer can animate them step by step.
    if chosen_id == "nfdh":
        solve_nfdh(instance, log_path=log_path)
    elif chosen_id == "ffdh":
        solve_ffdh(instance, log_path=log_path)
    elif chosen_id.startswith("blf_"):
        sk = chosen_id[4:]
        solve_blf(instance, sort_key=sk, log_path=log_path)
    elif chosen_id.startswith("sky_"):
        sk = chosen_id[4:]
        solve_skyline(instance, sort_key=sk, log_path=log_path)
    else:
        solve_blf(instance, sort_key="height", log_path=log_path)


def _run_skyline(instance, sort_key, log_path):
    """Run Skyline heuristic. If sort_key='best', try all 4 sorts and replay the winner."""
    if sort_key == "best":
        best_key, best_h = "height", float("inf")
        for sk in ("height", "width", "area", "perimeter"):
            r = solve_skyline(instance, sort_key=sk)
            if r["height"] < best_h:
                best_h, best_key = r["height"], sk
        open(log_path, "w").close()
        solve_skyline(instance, sort_key=best_key, log_path=log_path)
    else:
        solve_skyline(instance, sort_key=sort_key, log_path=log_path)


def _run_sa(instance, sort_key, log_path):
    """Run Simulated Annealing (10 s budget, instant-mode render)."""
    solve_sa(instance, log_path=log_path)


# -- Solver Registry -----------------------------------------------------------
# To add a new solver: append a dict with these keys:
#   id     – unique string
#   label  – button text
#   bg     – button background colour
#   mode   – "animated" (step-by-step log replay) | "instant" (draw all at once)
#   run    – callable(instance, sort_key, log_path) that runs the solver

SOLVER_REGISTRY = [
    {
        "id":    "heuristic",
        "label": "▶  BLF",
        "bg":    "#3A8E4E",
        "mode":  "animated",
        "run":   _run_heuristic,
    },
    {
        "id":    "nfdh",
        "label": "▶  NFDH",
        "bg":    "#7D3C98",
        "mode":  "animated",
        "run":   _run_nfdh,
    },
    {
        "id":    "ffdh",
        "label": "▶  FFDH",
        "bg":    "#1A5276",
        "mode":  "animated",
        "run":   _run_ffdh,
    },
    {
        "id":    "skyline",
        "label": "▶  Skyline",
        "bg":    "#1A6A5A",
        "mode":  "animated",
        "run":   _run_skyline,
    },
    {
        "id":    "sa",
        "label": "▶  SA (10s)",
        "bg":    "#7A4F1D",
        "mode":  "instant",
        "run":   _run_sa,
    },
    {
        "id":    "exact",
        "label": "▶  Exact",
        "bg":    "#2471A3",
        "mode":  "instant",
        "run":   _run_exact,
    },
    {
        "id":    "ml",
        "label": "▶  ML Select",
        "bg":    "#6E2F8E",
        "mode":  "animated",
        "run":   _run_ml_advisor,
    },
]


class VisualizerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Strip Packing Solver & Visualizer")
        self.root.configure(bg="#2A2A3A")
        self.root.minsize(1020, 660)

        self._instances = {i.name: i for i in get_all_benchmarks()}
        self._inst_names = sorted(self._instances)

        # Runtime state
        self._running = False
        self._current_mode = "animated"
        self._event_queue: deque = deque()
        self._log_offset = 0
        self._solver_thread: threading.Thread | None = None
        self._scale = 1.0
        self._strip_W = 10
        self._area_lb = 1
        self._color_idx = 0
        self._current_height = 0
        self._placements_list = []   # [(x, y, w, h, color, step)]
        self._ghost_map = {}         # step -> (x, y, w, h)

        # Zoom / pan state
        self._zoom = 1.0          # multiplier relative to fit-to-window
        self._pan_x = 0.0         # horizontal pan offset in canvas pixels
        self._pan_y = 0.0         # vertical pan offset in canvas pixels
        self._fit_scale = 1.0     # scale that fits the whole strip (no zoom)
        self._drag_start = None   # (canvas_x, canvas_y, pan_x0, pan_y0)

        # Comparison panel state (shown when ML Select or SA is active)
        self._comparison_visible = False
        self._comparison_mpl_canvas = None
        self._ml_chosen_sk = None
        self._ml_chosen_id = None   # full solver id from ml_choice event
        self._oracle_id = None      # actual best solver from comparison data
        self._last_comparison_data = None  # (heights_dict, times_dict)
        self._current_solver_id = ""       # id of the solver currently running
        self._sa_result = None             # (height, time) stored by SA done event

        self._build_ui()

    # -- UI ----------------------------------------------------------------

    def _build_ui(self):
        BG, FG, DIM = "#2A2A3A", "#DDDDDD", "#888899"

        # -- Left panel ---------------------------------------------------
        left = tk.Frame(self.root, bg=BG, padx=14, pady=14, width=262)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="Strip Packing\nSolver", bg=BG, fg="#7EB8F7",
                 font=("Helvetica", 13, "bold"), justify="left").pack(anchor="w", pady=(0, 14))

        # Instance selector
        tk.Label(left, text="INSTANCE", bg=BG, fg=DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self._inst_var = tk.StringVar(value=self._inst_names[0])
        inst_row = tk.Frame(left, bg=BG)
        inst_row.pack(anchor="w", pady=(2, 2), fill=tk.X)
        self._inst_combo = ttk.Combobox(
            inst_row, textvariable=self._inst_var, values=self._inst_names,
            width=21, state="readonly",
        )
        self._inst_combo.pack(side=tk.LEFT)
        tk.Button(
            inst_row, text="🎲", width=3, bg="#2E5E4E", fg="white",
            font=("Helvetica", 10), relief=tk.FLAT, cursor="hand2",
            command=self._on_random_instance,
        ).pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(inst_row, text="rand", bg=BG, fg="#556677",
                 font=("Helvetica", 7)).pack(side=tk.LEFT, padx=(2, 0))

        tk.Frame(left, bg=BG, height=6).pack()  # small spacer

        # Sort strategy
        tk.Label(left, text="SORT STRATEGY  (heuristic only)", bg=BG, fg=DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self._sort_var = tk.StringVar(value="best")
        for val, lbl in [
            ("best",      "Best of all (auto)"),
            ("height",    "Decreasing Height"),
            ("width",     "Decreasing Width"),
            ("area",      "Decreasing Area"),
            ("perimeter", "Decreasing Perimeter"),
        ]:
            tk.Radiobutton(left, text=lbl, variable=self._sort_var, value=val,
                           bg=BG, fg=FG, selectcolor="#4E79A7",
                           activebackground=BG, activeforeground="#FFF",
                           font=("Helvetica", 9)).pack(anchor="w")

        # ML model selector
        tk.Frame(left, bg="#44445A", height=1).pack(fill=tk.X, pady=(10, 4))
        tk.Label(left, text="ML MODEL", bg=BG, fg=DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self._ml_model_var = tk.StringVar(value="RF")
        for val, lbl in [("RF", "Random Forest (RF)"), ("MLP", "Neural Network (MLP)")]:
            tk.Radiobutton(left, text=lbl, variable=self._ml_model_var, value=val,
                           bg=BG, fg=FG, selectcolor="#6E2F8E",
                           activebackground=BG, activeforeground="#FFF",
                           font=("Helvetica", 9)).pack(anchor="w")

        # Speed slider
        tk.Label(left, text="MIN STEP DELAY  (ms)", bg=BG, fg=DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(12, 0))
        self._delay_var = tk.IntVar(value=150)
        tk.Scale(left, variable=self._delay_var, from_=0, to=600,
                 orient=tk.HORIZONTAL, length=210, bg=BG, fg=FG,
                 troughcolor="#44445A", highlightthickness=0,
                 activebackground="#4E79A7").pack(anchor="w")
        tk.Label(left, text="Playback = 75% real speed, ≥ floor above",
                 bg=BG, fg="#555577", font=("Helvetica", 7)).pack(anchor="w")

        # Divider
        tk.Frame(left, bg="#44445A", height=1).pack(fill=tk.X, pady=10)

        # Statistics
        tk.Label(left, text="STATISTICS", bg=BG, fg=DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self._stat_n    = self._stat_row(left, "Items (n)")
        self._stat_W    = self._stat_row(left, "Strip width W")
        self._stat_lb   = self._stat_row(left, "Area lower bound")
        self._stat_h    = self._stat_row(left, "Current height")
        self._stat_gap  = self._stat_row(left, "Gap to area LB")
        self._stat_time = self._stat_row(left, "Solve time")
        self._stat_sort = self._stat_row(left, "Sort / status")

        tk.Frame(left, bg="#44445A", height=1).pack(fill=tk.X, pady=8)
        tk.Label(left, text="Gap = (H − LB) / LB\narea_LB = ⌈Σ(w·h) / W⌉",
                 bg=BG, fg="#666688", font=("Helvetica", 8), justify="left").pack(anchor="w")

        # -- Main area ----------------------------------------------------
        main = tk.Frame(self.root, bg="#1A1A2B")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # -- Bottom toolbar (pack first so it anchors to bottom) ----------
        bottom = tk.Frame(main, bg="#111122", pady=7, padx=6)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self._solver_btns = []
        for cfg in SOLVER_REGISTRY:
            btn = tk.Button(
                bottom, text=cfg["label"], width=13,
                bg=cfg["bg"], fg="white", font=("Helvetica", 10, "bold"),
                relief=tk.FLAT, cursor="hand2",
                command=lambda c=cfg: self._on_solve(c),
            )
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self._solver_btns.append(btn)

        self._stop_btn = tk.Button(
            bottom, text="■  Stop", width=8,
            bg="#A93226", fg="white", font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, cursor="hand2", state=tk.DISABLED,
            command=self._on_stop,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 12))

        self._status = tk.StringVar(value="Select an instance and press a solver button.")
        tk.Label(bottom, textvariable=self._status, bg="#111122", fg="#888899",
                 font=("Helvetica", 9), anchor="w").pack(side=tk.LEFT)

        # -- Decision log (above bottom bar) -----------------------------
        log_frame = tk.Frame(main, bg="#0D0D1A", height=105)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        log_frame.pack_propagate(False)
        tk.Label(log_frame, text="DECISION LOG", bg="#0D0D1A", fg="#334455",
                 font=("Helvetica", 7, "bold"), anchor="w", padx=6,
                 pady=2).pack(fill=tk.X)
        self._log_text = tk.Text(
            log_frame, bg="#0D0D1A", fg="#8899AA",
            font=("Courier", 8), state=tk.DISABLED,
            height=5, wrap=tk.WORD, bd=0, pady=2,
        )
        log_sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                               command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4)

        # -- Content area (canvas + comparison panel side by side) ---------
        content_frame = tk.Frame(main, bg="#1A1A2B")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # -- Comparison panel (right, shown only when ML Select is active) -
        self._comparison_panel = tk.Frame(content_frame, bg="#12121F", width=400)
        self._comparison_panel.pack_propagate(False)
        tk.Label(
            self._comparison_panel,
            text="ALL SOLVERS COMPARISON",
            bg="#12121F", fg="#334455",
            font=("Helvetica", 7, "bold"),
        ).pack(anchor="w", padx=6, pady=(6, 0))
        self._comparison_loading_label = tk.Label(
            self._comparison_panel,
            text="Running all solvers…",
            bg="#12121F", fg="#888899", font=("Helvetica", 9),
        )
        self._comparison_loading_label.pack(pady=20)
        self._oracle_label = tk.Label(
            self._comparison_panel, text="",
            bg="#12121F", fg="#59A14F",
            font=("Helvetica", 10, "bold"), wraplength=390,
        )
        self._oracle_label.pack(side=tk.BOTTOM, padx=6, pady=4)

        # -- Canvas -------------------------------------------------------
        self._canvas_frame = tk.Frame(content_frame, bg="#1A1A2B")
        self._canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                padx=(6, 0), pady=(6, 0))

        self._canvas = tk.Canvas(self._canvas_frame, bg="#1A1A2B", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", lambda e: self._on_canvas_resize())
        # Zoom with mouse wheel
        self._canvas.bind("<MouseWheel>",  self._on_mousewheel)   # Windows
        self._canvas.bind("<Button-4>",    self._on_mousewheel)   # Linux scroll up
        self._canvas.bind("<Button-5>",    self._on_mousewheel)   # Linux scroll down
        # Pan by click-drag
        self._canvas.bind("<ButtonPress-1>",   self._on_drag_start)
        self._canvas.bind("<B1-Motion>",        self._on_drag)
        self._canvas.bind("<ButtonRelease-1>",  self._on_drag_end)
        # Double-click resets zoom/pan
        self._canvas.bind("<Double-Button-1>",  self._on_zoom_reset)

        self._canvas.create_text(
            350, 200,
            text="Press  ▶ Heuristic  or  ▶ Exact (CP-SAT)  to start",
            fill="#2A2A44", font=("Helvetica", 13), tags="hint",
        )

    def _stat_row(self, parent, label: str) -> tk.StringVar:
        row = tk.Frame(parent, bg="#2A2A3A")
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text=f"{label}:", width=17, anchor="w",
                 bg="#2A2A3A", fg="#666688",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        val = tk.StringVar(value="—")
        tk.Label(row, textvariable=val, anchor="w",
                 bg="#2A2A3A", fg="#DDDDFF",
                 font=("Courier", 9, "bold")).pack(side=tk.LEFT)
        return val

    # -- Solve dispatch ----------------------------------------------------

    def _on_solve(self, solver_cfg):
        inst = self._instances[self._inst_var.get()]

        # Reset state
        self._running = True
        self._current_mode = solver_cfg["mode"]
        self._event_queue.clear()
        self._log_offset = 0
        self._color_idx = 0
        self._current_height = 0
        self._strip_W = inst.strip_width
        self._area_lb = inst.area_lower_bound
        self._placements_list = []
        self._ghost_map = {}

        for sv in (self._stat_n, self._stat_W, self._stat_lb,
                   self._stat_h, self._stat_gap, self._stat_time, self._stat_sort):
            sv.set("—")
        self._stat_n.set(str(inst.n))
        self._stat_W.set(str(inst.strip_width))
        self._stat_lb.set(str(inst.area_lower_bound))

        cw = self._canvas.winfo_width() or 700
        self._fit_scale = (cw - 2 * _PAD) / inst.strip_width
        self._scale = self._fit_scale
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        self._canvas.delete("all")
        self._draw_border(0)
        self._log_clear()

        for btn in self._solver_btns:
            btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        lbl = solver_cfg["label"].strip().lstrip("▶").strip()
        self._status.set(f"Running {lbl} on {inst.name} …")

        open(LOG_PATH, "w").close()

        self._current_solver_id = solver_cfg["id"]

        # Show comparison panel for ML Select and SA; hide for all other solvers
        if solver_cfg["id"] in ("ml", "sa"):
            self._ml_chosen_sk = None
            self._ml_chosen_id = None
            self._oracle_id = None
            self._oracle_label.config(text="")
            self._last_comparison_data = None
            self._sa_result = None
            self._show_comparison_panel()
            threading.Thread(
                target=self._run_all_solvers_thread,
                args=(inst,),
                daemon=True,
            ).start()
        else:
            self._hide_comparison_panel()

        # For ML solver, pass the chosen model type via sort_key.
        effective_sort_key = (
            self._ml_model_var.get()
            if solver_cfg["id"] == "ml"
            else self._sort_var.get()
        )

        self._solver_thread = threading.Thread(
            target=self._run_solver_thread,
            args=(solver_cfg, inst, effective_sort_key),
            daemon=True,
        )
        self._solver_thread.start()

        self.root.after(80, self._poll_log)
        self.root.after(max(self._delay_var.get(), 1), self._animate_next)

    def _on_stop(self):
        self._running = False
        for btn in self._solver_btns:
            btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._status.set("Stopped.")

    def _on_random_instance(self):
        """Generate a fresh random instance and select it in the combobox."""
        import random as _rnd
        seed = _rnd.randint(1, 999_999)
        inst = generate_random_instance(seed=seed)
        self._instances[inst.name] = inst
        self._inst_names = sorted(self._instances)
        self._inst_combo.configure(values=self._inst_names)
        self._inst_var.set(inst.name)
        self._status.set(
            f"Random instance '{inst.name}' — n={inst.n}, W={inst.strip_width}"
        )

    # -- Solver thread -----------------------------------------------------

    def _run_solver_thread(self, solver_cfg, inst, sort_key):
        try:
            solver_cfg["run"](inst, sort_key, LOG_PATH)
        except Exception as e:
            with open(LOG_PATH, "a", buffering=1) as f:
                f.write(json.dumps({"event": "error", "msg": str(e)}) + "\n")

    # -- Log polling -------------------------------------------------------

    def _poll_log(self):
        try:
            with open(LOG_PATH, "r") as f:
                # Detect truncation: the ML advisor writes ml_choice with "a" mode,
                # then the chosen solver re-opens the file with "w" mode (truncating
                # it).  If the poll already advanced _log_offset past the ml_choice
                # event before the truncation, it would seek past the new content
                # and miss all start/place/done events.  Reset offset when we
                # detect that the file is shorter than our current position.
                f.seek(0, 2)          # seek to end to measure file size
                file_size = f.tell()
                if file_size < self._log_offset:
                    self._log_offset = 0
                f.seek(self._log_offset)
                for raw in f:
                    raw = raw.strip()
                    if raw:
                        try:
                            self._event_queue.append(json.loads(raw))
                        except json.JSONDecodeError:
                            pass
                self._log_offset = f.tell()
        except FileNotFoundError:
            pass

        alive = self._solver_thread and self._solver_thread.is_alive()
        if self._running or alive:
            self.root.after(60, self._poll_log)

    # -- Animation loop ----------------------------------------------------

    def _animate_next(self):
        if not self._running:
            return

        if not self._event_queue:
            self.root.after(30, self._animate_next)
            return

        evt = self._event_queue.popleft()
        self._handle_event(evt)

        # Delay for next step: max(floor, inter-event time / 0.75)
        cur_el = evt.get("elapsed", 0.0)
        next_el = self._event_queue[0].get("elapsed", cur_el) if self._event_queue else cur_el
        floor_ms = self._delay_var.get()
        speed_ms = (next_el - cur_el) * 1000 / 0.75

        if evt.get("event") == "search":
            delay = int(max(max(floor_ms // 3, 25), speed_ms))
        elif evt.get("event") == "improvement":
            # Show SA improvements at real speed but cap the wait so the
            # animation doesn't stall when improvements are far apart.
            delay = int(max(floor_ms, min(speed_ms, 400)))
        else:
            delay = int(max(floor_ms, speed_ms))

        self.root.after(delay, self._animate_next)

    # -- Event handlers ----------------------------------------------------

    def _handle_event(self, evt):
        etype = evt.get("event")

        if etype == "start":
            self._status.set(
                f"Solving {evt['name']}  |  W={evt['W']}  "
                f"n={evt['n']}  area_LB={evt['area_LB']}"
            )
            self._log_line(
                f"START  {evt['name']}  W={evt['W']}  "
                f"n={evt['n']}  area_LB={evt['area_LB']}"
            )

        elif etype == "search":
            x, y, w, h = evt["x"], evt["y"], evt["w"], evt["h"]
            self._ghost_map[evt["step"]] = (x, y, w, h)
            self._redraw_canvas()
            self._log_line(
                f"  step {evt['step']:>3d}:  try  ({x:>4d},{y:>4d})  "
                f"{w}×{h}  -> overlap, skip"
            )

        elif etype == "place":
            self._ghost_map.pop(evt["step"], None)
            x, y, w, h = evt["x"], evt["y"], evt["w"], evt["h"]
            color = _COLORS[self._color_idx % len(_COLORS)]
            self._color_idx += 1
            self._current_height = evt.get("current_height", y + h)
            self._placements_list.append((x, y, w, h, color, evt["step"]))
            self._redraw_canvas()
            gap = (self._current_height - self._area_lb) / self._area_lb * 100
            self._stat_h.set(str(self._current_height))
            self._stat_gap.set(f"{gap:.1f}%")
            self._log_line(
                f"  step {evt['step']:>3d}:  place ({x:>4d},{y:>4d})  "
                f"{w}×{h}  H -> {self._current_height}"
            )

        elif etype == "done":
            # Instant mode: build placements list and redraw with correct scale
            if self._current_mode == "instant" and evt.get("placements"):
                self._placements_list = []
                for i, p in enumerate(evt["placements"]):
                    self._placements_list.append(
                        (p[0], p[1], p[2], p[3], _COLORS[i % len(_COLORS)], i + 1)
                    )
                if evt["placements"]:
                    self._current_height = max(p[1] + p[3] for p in evt["placements"])
                self._redraw_canvas()

            h = evt.get("height") or self._current_height
            t = evt.get("wall_time", 0.0)
            sk = evt.get("sort_key", evt.get("status", "—"))
            gap = (h - self._area_lb) / self._area_lb * 100 if self._area_lb else 0
            self._stat_h.set(str(h))
            self._stat_gap.set(f"{gap:.1f}%")
            self._stat_time.set(f"{t:.4f}s")
            self._stat_sort.set(sk)
            self._draw_border(h)
            # Refresh comparison panel if visible
            if self._comparison_visible:
                self._ml_chosen_sk = sk
                if self._current_solver_id == "sa":
                    # SA finished: store its result and add it to the chart
                    self._sa_result = (h, t)
                    self._ml_chosen_id = "sa"
                    if self._last_comparison_data is not None:
                        heights_d, times_d = self._last_comparison_data
                        heights_d["sa"] = h
                        times_d["sa"] = t
                        self._draw_comparison_chart(heights_d, times_d)
                elif self._last_comparison_data is not None:
                    self._draw_comparison_chart(*self._last_comparison_data)
            self._log_line(
                f"DONE   height={h}  gap={gap:.1f}%  "
                f"sort/status={sk}  time={t:.4f}s"
            )
            self._status.set(
                f"Done — Height={h}  Gap={gap:.1f}%  Time={t:.4f}s"
            )
            self._running = False
            for btn in self._solver_btns:
                btn.config(state=tk.NORMAL)
            self._stop_btn.config(state=tk.DISABLED)

        elif etype == "improvement":
            # SA found a better permutation: replace the entire packing display.
            self._placements_list = [
                (p[0], p[1], p[2], p[3], _COLORS[i % len(_COLORS)], i + 1)
                for i, p in enumerate(evt.get("placements", []))
            ]
            h = evt.get("height", self._current_height)
            self._current_height = h
            gap = (h - self._area_lb) / self._area_lb * 100 if self._area_lb else 0
            self._stat_h.set(str(h))
            self._stat_gap.set(f"{gap:.1f}%")
            iters = evt.get("iteration", 0)
            self._stat_sort.set("SA  init" if iters == 0 else f"SA  iter {iters}")
            self._redraw_canvas()
            self._log_line(
                f"  SA {'init' if iters == 0 else f'iter {iters:>5}'}:  "
                f"H \u2192 {h}  gap={gap:.1f}%"
            )

        elif etype == "ml_choice":
            self._ml_chosen_id = evt.get("chosen_id", "?")
            mt = evt.get("model_type", "RF")
            self._log_line(f"ℹ  ML ({mt}) chose: {self._ml_chosen_id}")
            self._status.set(f"ML ({mt}) selected: {self._ml_chosen_id} — running…")
            if self._last_comparison_data is not None:
                heights_d, times_d = self._last_comparison_data
                self._draw_comparison_chart(heights_d, times_d)

        elif etype == "info":
            msg = evt.get("msg", "")
            self._log_line(f"\u2139  {msg}")
            self._status.set(msg)

        elif etype == "error":
            self._log_line(f"ERROR: {evt.get('msg', '?')}")
            self._status.set(f"Error: {evt.get('msg', '?')}")
            self._running = False
            for btn in self._solver_btns:
                btn.config(state=tk.NORMAL)
            self._stop_btn.config(state=tk.DISABLED)

    # -- Drawing -----------------------------------------------------------

    def _canvas_y(self, strip_y: float) -> float:
        """Strip y=0 is bottom; canvas y=0 is top."""
        ch = max(self._canvas.winfo_height(), 100)
        return ch - _PAD - strip_y * self._scale + self._pan_y

    def _draw_rect(self, x, y, w, h, fill, step, ghost=False, tag=None):
        s = self._scale
        cx0 = _PAD + x * s + self._pan_x
        cx1 = cx0 + w * s
        cy0 = self._canvas_y(y + h)
        cy1 = self._canvas_y(y)
        tags = ("rect", tag) if tag else ("rect",)
        kw = {"fill": fill, "outline": "#FFFFFF", "width": 1, "tags": tags}
        if ghost:
            kw["stipple"] = "gray25"
            kw["outline"] = "#FF4444"
        self._canvas.create_rectangle(cx0, cy0, cx1, cy1, **kw)
        if not ghost:
            pw, ph = cx1 - cx0, cy1 - cy0
            if pw > 12 and ph > 9:
                fs = max(6, min(int(ph * 0.38), 12))
                self._canvas.create_text(
                    (cx0 + cx1) / 2, (cy0 + cy1) / 2,
                    text=str(step), fill="white",
                    font=("Helvetica", fs, "bold"), tags=tags,
                )

    def _redraw_canvas(self):
        """Recompute fit scale, apply zoom, then redraw everything."""
        cw = max(self._canvas.winfo_width(), 200)
        ch = max(self._canvas.winfo_height(), 200)
        cur_h = max(self._current_height, 1)
        self._fit_scale = min(
            (cw - 2 * _PAD) / self._strip_W,
            (ch - 2 * _PAD) / cur_h,
        )
        self._scale = self._fit_scale * self._zoom
        self._canvas.delete("all")
        for (x, y, w, h, color, step) in self._placements_list:
            self._draw_rect(x, y, w, h, color, step)
        for step, (x, y, w, h) in self._ghost_map.items():
            self._draw_rect(x, y, w, h, "#662222", step,
                            ghost=True, tag=f"ghost_{step}")
        self._draw_border(self._current_height)
        # Zoom indicator overlay
        if self._zoom != 1.0:
            self._canvas.create_text(
                cw - 6, 6,
                text=f"zoom {self._zoom:.1f}×  (scroll=zoom  drag=pan  dbl-click=reset)",
                anchor="ne", fill="#7777AA",
                font=("Helvetica", 7), tags="zoomlabel",
            )
        else:
            self._canvas.create_text(
                cw - 6, 6,
                text="scroll to zoom · drag to pan · dbl-click to reset",
                anchor="ne", fill="#44445A",
                font=("Helvetica", 7), tags="zoomlabel",
            )

    def _on_canvas_resize(self):
        """Keep the whole strip in view when the window is resized."""
        if self._placements_list or self._ghost_map:
            self._redraw_canvas()

    def _draw_border(self, height):
        self._canvas.delete("border")
        s = self._scale
        x0 = _PAD + self._pan_x
        x1 = _PAD + self._strip_W * s + self._pan_x
        y_bot = self._canvas_y(0)
        y_top = self._canvas_y(max(height, 1))
        self._canvas.create_rectangle(
            x0, y_top, x1, y_bot,
            outline="#5555AA", width=2, dash=(8, 4), tags="border",
        )
        self._canvas.create_text(
            (x0 + x1) / 2, y_bot + 16,
            text=f"W = {self._strip_W}",
            fill="#6666AA", font=("Helvetica", 9), tags="border",
        )
        if height > 0:
            self._canvas.create_text(
                x0 - 6, (y_top + y_bot) / 2,
                text=f"H={height}", fill="#6666AA",
                font=("Helvetica", 8), anchor="e", tags="border",
            )

    # -- Decision log ------------------------------------------------------

    def _log_line(self, text: str):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, text + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _log_clear(self):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    # -- Zoom / pan handlers -------------------------------------------------

    def _on_mousewheel(self, event):
        """Zoom in/out centred on the cursor position."""
        if not self._placements_list:
            return
        # Determine scroll direction (Windows: event.delta; Linux: event.num)
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            factor = 1 / 1.15
        else:
            factor = 1.15
        new_zoom = max(0.5, min(self._zoom * factor, 30.0))
        actual = new_zoom / self._zoom
        mx, my = event.x, event.y
        ch = max(self._canvas.winfo_height(), 100)
        # Adjust pan so the strip point under the cursor stays fixed
        self._pan_x = mx - _PAD - (mx - _PAD - self._pan_x) * actual
        self._pan_y = (my - ch + _PAD) + (ch - _PAD + self._pan_y - my) * actual
        self._zoom = new_zoom
        self._redraw_canvas()

    def _on_drag_start(self, event):
        if not self._placements_list:
            return
        self._drag_start = (event.x, event.y, self._pan_x, self._pan_y)
        self._canvas.config(cursor="fleur")

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        sx, sy, px0, py0 = self._drag_start
        self._pan_x = px0 + (event.x - sx)
        self._pan_y = py0 + (event.y - sy)
        self._redraw_canvas()

    def _on_drag_end(self, event):
        self._drag_start = None
        self._canvas.config(cursor="")

    def _on_zoom_reset(self, event):
        """Double-click: reset zoom and pan to fit-to-window."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._redraw_canvas()

    # -- Comparison panel --------------------------------------------------

    def _show_comparison_panel(self):
        if not self._comparison_visible:
            # Pack comparison panel to RIGHT before canvas so it claims right edge
            self._canvas_frame.pack_forget()
            self._comparison_panel.pack(side=tk.RIGHT, fill=tk.Y,
                                        padx=(0, 4), pady=(6, 0))
            self._canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                    padx=(6, 0), pady=(6, 0))
            self._comparison_visible = True
        # Reset to loading state for each new run
        if self._comparison_mpl_canvas is not None:
            self._comparison_mpl_canvas.get_tk_widget().destroy()
            self._comparison_mpl_canvas = None
        self._comparison_loading_label.pack(pady=20)

    def _hide_comparison_panel(self):
        if self._comparison_visible:
            self._comparison_panel.pack_forget()
            self._comparison_visible = False

    def _run_all_solvers_thread(self, inst):
        """Run all fast heuristic solvers (BLF + Shelf + Skyline) silently."""
        candidates = [
            ("blf_height",    lambda i: solve_blf(i, sort_key="height")),
            ("blf_width",     lambda i: solve_blf(i, sort_key="width")),
            ("blf_area",      lambda i: solve_blf(i, sort_key="area")),
            ("blf_perimeter", lambda i: solve_blf(i, sort_key="perimeter")),
            ("nfdh",          solve_nfdh),
            ("ffdh",          solve_ffdh),
            ("sky_height",    lambda i: solve_skyline(i, sort_key="height")),
            ("sky_width",     lambda i: solve_skyline(i, sort_key="width")),
            ("sky_area",      lambda i: solve_skyline(i, sort_key="area")),
            ("sky_perimeter", lambda i: solve_skyline(i, sort_key="perimeter")),
        ]
        heights, times = {}, {}
        for name, solver in candidates:
            try:
                result = solver(inst)
                heights[name] = result["height"]
                times[name] = result["wall_time"]
            except Exception:
                heights[name] = 0
                times[name] = 0.0
        self.root.after(0, self._update_comparison_chart, heights, times)

    def _update_comparison_chart(self, heights, times):
        """Store comparison results, compute oracle, and render the chart."""
        self._last_comparison_data = (heights, times)
        # oracle = best among the fast heuristics (SA excluded from oracle)
        self._oracle_id = min(heights, key=lambda k: (heights[k], k))
        # If SA already finished before the heuristics thread completed, add it
        if self._sa_result is not None and self._current_solver_id == "sa":
            heights["sa"] = self._sa_result[0]
            times["sa"] = self._sa_result[1]
            self._ml_chosen_id = "sa"
        self._draw_comparison_chart(heights, times)

    def _draw_comparison_chart(self, heights, times):
        """Render two bar charts (height + time) in the right comparison panel."""
        if not self._comparison_visible:
            return

        if not _HAS_MATPLOTLIB:
            self._comparison_loading_label.config(
                text="Install matplotlib\nto see the comparison chart.",
                fg="#FF8866",
            )
            return

        chosen_name = self._ml_chosen_id   # full solver id, e.g. "blf_width"
        oracle_name = self._oracle_id       # actual best, e.g. "ffdh"

        names  = list(heights.keys())
        h_vals = [heights[n] for n in names]
        t_vals = [times[n] * 1000 for n in names]   # -> milliseconds

        short_labels = {
            "blf_height": "BLF\nH", "blf_width": "BLF\nW",
            "blf_area":   "BLF\nA", "blf_perimeter": "BLF\nP",
            "nfdh": "NFDH",         "ffdh": "FFDH",
            "sky_height": "Sky\nH", "sky_width": "Sky\nW",
            "sky_area":   "Sky\nA", "sky_perimeter": "Sky\nP",
            "sa": "SA\n10s",
        }
        labels = [short_labels.get(n, n) for n in names]

        # Colour coding:
        #   gold (#EDC948)  — ML chose this AND it is the oracle / SA beats oracle
        #   orange (#F28E2B) — ML chose this but it is NOT the oracle (wrong)
        #   brown (#7A4F1D) — SA result (when SA panel is shown and SA is worse)
        #   green (#59A14F)  — oracle (actual best), not ML/SA choice
        #   blue (#4E79A7)   — all others
        def _bar_color(n):
            if n == "sa":
                sa_h  = heights.get("sa", float("inf"))
                ora_h = heights.get(oracle_name, float("inf")) if oracle_name else float("inf")
                return "#EDC948" if sa_h <= ora_h else "#7A4F1D"
            is_ml     = (n == chosen_name)
            is_oracle = (n == oracle_name)
            if is_ml and is_oracle:
                return "#EDC948"
            if is_ml:
                return "#F28E2B"
            if is_oracle:
                return "#59A14F"
            return "#4E79A7"

        bar_colors = [_bar_color(n) for n in names]

        # Tear down previous figure
        if self._comparison_mpl_canvas is not None:
            self._comparison_mpl_canvas.get_tk_widget().destroy()
            self._comparison_mpl_canvas = None
        self._comparison_loading_label.pack_forget()

        fig = Figure(figsize=(4.0, 5.2), dpi=90, facecolor="#12121F")
        fig.subplots_adjust(left=0.16, right=0.97, top=0.91,
                            bottom=0.14, hspace=0.58)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # -- Height chart --------------------------------------------------
        bars1 = ax1.bar(range(len(names)), h_vals, color=bar_colors,
                        edgecolor="#2A2A3A", linewidth=0.6)
        ax1.axhline(self._area_lb, color="#E15759", linestyle="--",
                    linewidth=1.0, label=f"LB={self._area_lb}")
        ax1.set_title("Packing Height", color="#CCCCDD", fontsize=8, pad=3)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(labels, fontsize=5.5, color="#AAAACC")
        ax1.set_ylabel("Height", color="#AAAACC", fontsize=7)
        ax1.tick_params(axis="both", colors="#AAAACC", labelsize=5.5)
        ax1.set_facecolor("#1A1A2B")
        for sp in ax1.spines.values():
            sp.set_color("#334455")
        ax1.legend(fontsize=6, labelcolor="#CCCCDD",
                   facecolor="#12121F", edgecolor="#334455", loc="upper right")
        max_h = max(h_vals) if h_vals else 1
        for bar, val in zip(bars1, h_vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max_h * 0.015,
                     str(val), ha="center", va="bottom",
                     fontsize=5.5, color="#DDDDEE")

        # -- Time chart ----------------------------------------------------
        bars2 = ax2.bar(range(len(names)), t_vals, color=bar_colors,
                        edgecolor="#2A2A3A", linewidth=0.6)
        ax2.set_title("Solve Time (ms)", color="#CCCCDD", fontsize=8, pad=3)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(labels, fontsize=5.5, color="#AAAACC")
        ax2.set_ylabel("Time (ms)", color="#AAAACC", fontsize=7)
        ax2.tick_params(axis="both", colors="#AAAACC", labelsize=5.5)
        ax2.set_facecolor("#1A1A2B")
        for sp in ax2.spines.values():
            sp.set_color("#334455")
        max_t = max(t_vals) if t_vals else 1
        for bar, val in zip(bars2, t_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max_t * 0.015,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=5.5, color="#DDDDEE")

        # -- Supra-title ---------------------------------------------------
        if self._current_solver_id == "sa" and "sa" in heights:
            sa_h  = heights["sa"]
            ora_h = heights.get(oracle_name, float("inf")) if oracle_name else float("inf")
            title_text  = f"SA: H={sa_h}  (oracle {oracle_name}: H={ora_h})"
            title_color = "#EDC948" if sa_h <= ora_h else "#7A4F1D"
        elif chosen_name:
            title_text = f"ML chose: {chosen_name}"
            title_color = "#EDC948" if chosen_name == oracle_name else "#F28E2B"
        else:
            title_text = "Running comparison…"
            title_color = "#CCCCDD"
        fig.text(0.5, 0.965, title_text, ha="center", va="top",
                 fontsize=7.5, color=title_color, fontweight="bold")

        mpl_canvas = FigureCanvasTkAgg(fig, master=self._comparison_panel)
        mpl_canvas.draw()
        mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 2))
        self._comparison_mpl_canvas = mpl_canvas

        # -- Oracle verdict label ------------------------------------------
        if self._current_solver_id == "sa" and "sa" in heights and oracle_name:
            sa_h  = heights["sa"]
            ora_h = heights[oracle_name]
            if sa_h <= ora_h:
                self._oracle_label.config(
                    text=f"✓ SA wins — oracle: {oracle_name} (H={ora_h})",
                    fg="#EDC948",
                )
            else:
                self._oracle_label.config(
                    text=f"SA: H={sa_h}  |  oracle: {oracle_name} (H={ora_h})",
                    fg="#F28E2B",
                )
        elif chosen_name and oracle_name:
            if chosen_name == oracle_name:
                self._oracle_label.config(
                    text=f"✓ ML correct — oracle: {oracle_name}",
                    fg="#59A14F",
                )
            else:
                self._oracle_label.config(
                    text=f"✗ ML wrong — oracle: {oracle_name}",
                    fg="#E15759",
                )
        else:
            self._oracle_label.config(text="")


def main():
    root = tk.Tk()
    VisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


