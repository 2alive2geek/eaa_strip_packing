"""
Strip Packing Solver & Visualizer
==================================
Animated visualization of strip packing solvers.

Log format (JSONL):
    heuristic: start → search* → place → ... → done
    exact:     start → done (placements embedded in done event)

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

from benchmarks import get_all_benchmarks
from solver_heuristic import solve_blf
from solver_exact import solve_exact

LOG_PATH = "strip_packing_vis.log"
_PAD = 32

_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#86BCB6",
    "#499894", "#E4A96A", "#D37295", "#A0CBE8", "#FFBE7D",
    "#8CD17D", "#B6992D", "#F1CE63", "#D4A6C8", "#FABFD2",
]

# ── Solver runner functions (called in background threads) ───────────────────

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


# ── Solver Registry ───────────────────────────────────────────────────────────
# To add a new solver: append a dict with these keys:
#   id     – unique string
#   label  – button text
#   bg     – button background colour
#   mode   – "animated" (step-by-step log replay) | "instant" (draw all at once)
#   run    – callable(instance, sort_key, log_path) that runs the solver

SOLVER_REGISTRY = [
    {
        "id":    "heuristic",
        "label": "▶  Heuristic",
        "bg":    "#3A8E4E",
        "mode":  "animated",
        "run":   _run_heuristic,
    },
    {
        "id":    "exact",
        "label": "▶  Exact (CP-SAT)",
        "bg":    "#2471A3",
        "mode":  "instant",
        "run":   _run_exact,
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

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        BG, FG, DIM = "#2A2A3A", "#DDDDDD", "#888899"

        # ── Left panel ───────────────────────────────────────────────────
        left = tk.Frame(self.root, bg=BG, padx=14, pady=14, width=262)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="Strip Packing\nSolver", bg=BG, fg="#7EB8F7",
                 font=("Helvetica", 13, "bold"), justify="left").pack(anchor="w", pady=(0, 14))

        # Instance selector
        tk.Label(left, text="INSTANCE", bg=BG, fg=DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")
        self._inst_var = tk.StringVar(value=self._inst_names[0])
        ttk.Combobox(left, textvariable=self._inst_var, values=self._inst_names,
                     width=27, state="readonly").pack(anchor="w", pady=(2, 12))

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

        # ── Main area ────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1A1A2B")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Bottom toolbar (pack first so it anchors to bottom) ──────────
        bottom = tk.Frame(main, bg="#111122", pady=7, padx=6)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self._solver_btns = []
        for cfg in SOLVER_REGISTRY:
            btn = tk.Button(
                bottom, text=cfg["label"], width=16,
                bg=cfg["bg"], fg="white", font=("Helvetica", 10, "bold"),
                relief=tk.FLAT, cursor="hand2",
                command=lambda c=cfg: self._on_solve(c),
            )
            btn.pack(side=tk.LEFT, padx=(0, 6))
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

        # ── Decision log (above bottom bar) ─────────────────────────────
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

        # ── Canvas ───────────────────────────────────────────────────────
        canvas_frame = tk.Frame(main, bg="#1A1A2B")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=(6, 0), pady=(6, 0))

        self._canvas = tk.Canvas(canvas_frame, bg="#1A1A2B", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", lambda e: self._on_canvas_resize())

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

    # ── Solve dispatch ────────────────────────────────────────────────────

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
        self._scale = (cw - 2 * _PAD) / inst.strip_width

        self._canvas.delete("all")
        self._draw_border(0)
        self._log_clear()

        for btn in self._solver_btns:
            btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        lbl = solver_cfg["label"].strip().lstrip("▶").strip()
        self._status.set(f"Running {lbl} on {inst.name} …")

        open(LOG_PATH, "w").close()

        self._solver_thread = threading.Thread(
            target=self._run_solver_thread,
            args=(solver_cfg, inst, self._sort_var.get()),
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

    # ── Solver thread ─────────────────────────────────────────────────────

    def _run_solver_thread(self, solver_cfg, inst, sort_key):
        try:
            solver_cfg["run"](inst, sort_key, LOG_PATH)
        except Exception as e:
            with open(LOG_PATH, "a", buffering=1) as f:
                f.write(json.dumps({"event": "error", "msg": str(e)}) + "\n")

    # ── Log polling ───────────────────────────────────────────────────────

    def _poll_log(self):
        try:
            with open(LOG_PATH, "r") as f:
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

    # ── Animation loop ────────────────────────────────────────────────────

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
        else:
            delay = int(max(floor_ms, speed_ms))

        self.root.after(delay, self._animate_next)

    # ── Event handlers ────────────────────────────────────────────────────

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
                f"{w}×{h}  → overlap, skip"
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
                f"{w}×{h}  H → {self._current_height}"
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

        elif etype == "error":
            self._log_line(f"ERROR: {evt.get('msg', '?')}")
            self._status.set(f"Error: {evt.get('msg', '?')}")
            self._running = False
            for btn in self._solver_btns:
                btn.config(state=tk.NORMAL)
            self._stop_btn.config(state=tk.DISABLED)

    # ── Drawing ───────────────────────────────────────────────────────────

    def _canvas_y(self, strip_y: float) -> float:
        """Strip y=0 is bottom; canvas y=0 is top."""
        ch = max(self._canvas.winfo_height(), 100)
        return ch - _PAD - strip_y * self._scale

    def _draw_rect(self, x, y, w, h, fill, step, ghost=False, tag=None):
        s = self._scale
        cx0 = _PAD + x * s
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
        """Recompute scale to fit the whole strip, then redraw everything."""
        cw = max(self._canvas.winfo_width(), 200)
        ch = max(self._canvas.winfo_height(), 200)
        cur_h = max(self._current_height, 1)
        self._scale = min(
            (cw - 2 * _PAD) / self._strip_W,
            (ch - 2 * _PAD) / cur_h,
        )
        self._canvas.delete("all")
        for (x, y, w, h, color, step) in self._placements_list:
            self._draw_rect(x, y, w, h, color, step)
        for step, (x, y, w, h) in self._ghost_map.items():
            self._draw_rect(x, y, w, h, "#662222", step,
                            ghost=True, tag=f"ghost_{step}")
        self._draw_border(self._current_height)

    def _on_canvas_resize(self):
        """Keep the whole strip in view when the window is resized."""
        if self._placements_list or self._ghost_map:
            self._redraw_canvas()

    def _draw_border(self, height):
        self._canvas.delete("border")
        s = self._scale
        x0, x1 = _PAD, _PAD + self._strip_W * s
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

    # ── Decision log ──────────────────────────────────────────────────────

    def _log_line(self, text: str):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, text + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _log_clear(self):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    VisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


