# Two-Dimensional Strip Packing (2SP) — Experimental Analysis

This repository implements and compares several algorithms for the **2D Strip Packing Problem (2SP)**: given a set of axis-aligned rectangles and a strip of fixed width *W*, pack all rectangles without overlap and minimise the total height *H*.

---

## Project Structure

| File | Purpose |
|---|---|
| `benchmarks.py` | Benchmark instance generator (Bengtsson, Berkey & Wang, Martello & Vigo) |
| `solver_heuristic.py` | Bottom-Left-Fill (BLF) heuristic — 4 sort strategies |
| `solver_shelf.py` | Shelf heuristics: NFDH and FFDH |
| `solver_exact.py` | Exact solver via OR-Tools CP-SAT |
| `instance_features.py` | Extracts 17 numeric features from an instance (used by ML model) |
| `train_model.py` | Trains a RandomForest classifier to select the best heuristic per instance |
| `main.py` | Experiment runner — runs all solvers and saves `results.csv` |
| `visualizer.py` | Live animated GUI — shows each solver placing rectangles step by step |
| `results.csv` | Output of `main.py` — all benchmark results |
| `best_solver_model.pkl` | Trained ML model (created by `train_model.py`) |
| `strip_packing_sota.tex` | LaTeX SOTA review (Overleaf-ready) |
| `report.tex` | LaTeX experimental report with result tables and diagrams (Overleaf-ready) |

---

## Implemented Methods

### 1. Bottom-Left-Fill (BLF) — `solver_heuristic.py`

A classical constructive heuristic (Baker et al., 1980; Chazelle, 1983).

**How it works:**
1. Sort rectangles by a chosen criterion (see below).
2. For each rectangle, scan candidate y-levels `{0} ∪ {y_j + h_j}` (bottom of strip, top edges of placed rects).
3. At each y-level, try candidate x-positions `{0} ∪ {x_j + w_j}`.
4. Place the rectangle at the **lowest, then leftmost** feasible position — the "bottom-left" position.

**Sort strategies:**
| ID | Name | Key |
|---|---|---|
| `height` | Decreasing Height (DH) | `−h` |
| `width` | Decreasing Width (DW) | `−w` |
| `area` | Decreasing Area (DA) | `−w·h` |
| `perimeter` | Decreasing Perimeter (DP) | `−2(w+h)` |

**Best-of-4 mode:** The solver runs all four strategies and returns the one with the smallest height.

**Complexity:** O(n³) — for each of n items, scans O(n) y-levels and O(n) x-positions, with an O(n) overlap check.

---

### 2. Next Fit Decreasing Height (NFDH) — `solver_shelf.py`

A very fast shelf-based heuristic (Coffman et al., 1980).

**How it works:**
1. Sort rectangles by decreasing height.
2. Maintain one open "shelf" at height y₀.
3. Try to place the next item on the current shelf (left to right). If it doesn't fit horizontally, **close the shelf** and open a new one above it.

**Approximation ratio:** ≤ 2·OPT + h_max.  
**Complexity:** O(n log n) — dominated by sorting.

---

### 3. First Fit Decreasing Height (FFDH) — `solver_shelf.py`

An improved shelf heuristic (Coffman et al., 1980).

**How it works:**
1. Sort rectangles by decreasing height.
2. For each item scan **all open shelves from the bottom** and place it on the **first shelf where it fits** horizontally.
3. If no existing shelf fits, open a new shelf above the current stack.

**Approximation ratio:** ≤ 17/10·OPT + h_max (tighter than NFDH).  
**Complexity:** O(n²) worst case.

---

### 4. Exact Solver (CP-SAT) — `solver_exact.py`

An exact method using Google OR-Tools CP-SAT constraint programming.

**Formulation:**
```
minimise H
subject to:
    xᵢ + wᵢ ≤ W       for all i
    yᵢ + hᵢ ≤ H       for all i
    NoOverlap2D({(xᵢ, xᵢ+wᵢ, yᵢ, yᵢ+hᵢ)})
    xᵢ, yᵢ ≥ 0
```

Uses `AddNoOverlap2D` with interval variables for efficient propagation. A configurable time limit (default 60 s) is applied because 2SP is NP-hard.

**Returns:** `OPTIMAL`, `FEASIBLE` (time limit hit), or `UNKNOWN`.

---

### 5. ML Algorithm Selector — `train_model.py` + `instance_features.py`

A **RandomForest classifier** that predicts which of the 6 candidate heuristics (BLF×4, NFDH, FFDH) will produce the smallest packing height for a given instance.

**Features extracted (`instance_features.py`):**

| # | Feature | Description |
|---|---|---|
| 1 | `n` | Number of rectangles |
| 2 | `W` | Strip width |
| 3 | `area_lb` | Area lower bound |
| 4–7 | `mean/std/max/min_w/W` | Width statistics, normalised by W |
| 8–11 | `mean/std/max/min_h/lb` | Height statistics, normalised by area lower bound |
| 12–13 | `mean/std_aspect` | Width-to-height ratio statistics |
| 14 | `mean_item_area/strip_area` | Average item area relative to strip |
| 15 | `density` | Total area / strip area at lower bound |
| 16 | `frac_wide` | Fraction of items wider than W/2 |
| 17 | `frac_tall` | Fraction of items taller than mean height |

**Training (`train_model.py`):**
- Labels each instance with the best-performing heuristic.
- Trains `RandomForestClassifier(n_estimators=300)` with 5-fold stratified CV.
- Saves model to `best_solver_model.pkl` (loaded at visualizer runtime).

---

## Where Are the Results?

### `results.csv`

Produced by running `python main.py`. Columns:

| Column | Meaning |
|---|---|
| `name` | Instance name |
| `n` | Number of rectangles |
| `W` | Strip width |
| `area_LB` | Theoretical minimum height = ⌈Σ(wᵢhᵢ)/W⌉ |
| `blf_height` | Height achieved by the best BLF strategy |
| `blf_sort` | Which sort strategy won (height/width/area/perimeter) |
| `blf_time` | BLF wall-clock time (seconds) |
| `blf_gap_pct` | Gap to area lower bound = (H−LB)/LB × 100% |
| `exact_height` | Height from CP-SAT (`SKIPPED` if n > limit) |
| `exact_status` | OPTIMAL / FEASIBLE / SKIPPED |
| `exact_time` | CP-SAT wall-clock time (seconds) |
| `exact_gap_pct` | CP-SAT gap to area lower bound |

> **Note:** NFDH and FFDH results are visible in the visualizer but are not currently written to `results.csv`. The shelf solvers can be added to `main.py` following the same pattern as BLF.

### Visualizer

The GUI (`python visualizer.py`) shows real-time animated placement for every solver. The left panel shows live statistics (height, gap, solve time, sort key used) for the currently running solver.

---

## Usage

```bash
# Install dependencies
pip install ortools scikit-learn joblib

# Run the full benchmark experiment (BLF + Exact where n ≤ 25)
python main.py

# Run exact solver on larger instances (slower)
python main.py --exact-limit 40 --time-limit 120

# Train the ML model (needs scikit-learn + joblib)
python train_model.py

# Open the animated visualizer
python visualizer.py
```

### Visualizer Solver Buttons

| Button | Mode | Description |
|---|---|---|
| ▶ BLF | Animated | Bottom-Left-Fill, best-of-4 sort strategies |
| ▶ NFDH | Animated | Next Fit Decreasing Height (shelf) |
| ▶ FFDH | Animated | First Fit Decreasing Height (shelf) |
| ▶ Exact (CP-SAT) | Instant | OR-Tools CP-SAT exact solver |
| 🤖 ML Select | Animated | ML predicts best heuristic, then runs it animated |

---

## Benchmark Instances

60 instances total across three families:

| Family | Instances | n range | W range | Reference |
|---|---|---|---|---|
| Bengtsson | 10 | 20–200 | 25–40 | Bengtsson (1982) |
| Berkey & Wang | 30 (6 classes × 5 sizes) | 20–100 | 10–300 | Berkey & Wang (1987) |
| Martello & Vigo | 20 (4 classes × 5 sizes) | 20–100 | 10–100 | Martello & Vigo (1998) |

The exact solver is run only on instances with n ≤ 25 (default) due to NP-hardness.
