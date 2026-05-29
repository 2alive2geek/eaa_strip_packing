"""
Simulated Annealing (SA) for the 2D Strip Packing Problem.

SA is a metaheuristic that escapes local optima by occasionally accepting
worse solutions with a probability that decreases as the "temperature" cools.

Representation:
  A permutation of the n rectangle indices defines the order in which items
  are fed to the Skyline greedy heuristic.  Different orderings yield
  different strip heights, so the search space is S_n (all permutations).

Algorithm:
  1. Build an initial permutation by sorting items by decreasing height.
  2. Evaluate its strip height h via the Skyline heuristic (_eval_perm).
  3. Repeat until temperature drops below T_min or the time budget elapses:
       a. Perturbation: swap two randomly chosen positions in the permutation.
       b. Evaluate the new height h'.
       c. Accept if h' ≤ h (improvement), or with probability exp(-Δ/T)
          where Δ = h' - h (Metropolis criterion).
       d. Update the global best if h' < best_h.
       e. Cool: T ← T × α.

SA parameters (tuned for strip packing):
  T0        = 50     initial temperature  (set so ~80% of +5-unit moves accepted)
  alpha     = 0.9997 cooling rate — very slow, so time_limit is the dominant stop
  T_min     = 0.01   halting temperature  (rarely reached; time_limit fires first)
  time_limit = 10 s  wall-clock budget (primary stopping criterion)

Perturbation:
  Two moves are tried with equal probability:
    - swap:   exchange two random positions in the permutation.
    - insert: remove one item from its position and re-insert it elsewhere.
  Together these explore the full space of permutations effectively.

Design notes:
  - The Skyline inner evaluator is imported from solver_skyline.py.
  - SA is NOT a ML-selector candidate (10 s runtime incompatible with fast
    inference), but it is available as a standalone solver in the visualizer.
  - For small instances (n ≤ 30) many thousands of iterations run in < 1 s;
    for large instances (n ≥ 150) the 10-second budget is the binding limit.
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Optional

from benchmarks import Instance
from solver_skyline import _find_skyline_position, _update_skyline


# ---------------------------------------------------------------------------
# Inner evaluator (permutation → strip height)
# ---------------------------------------------------------------------------

def _eval_perm(
    perm: list[int],
    rects: list[tuple[int, int]],
    W: int,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """
    Apply the Skyline greedy heuristic to rectangles in the order given by
    *perm* and return (strip_height, placements).

    Args:
        perm:   list of rectangle indices defining packing order.
        rects:  list of (width, height) pairs.
        W:      strip width.

    Returns:
        (height, placements) where placements is a list of (x, y, w, h).
    """
    skyline: list[tuple[int, int]] = [(0, 0)]
    placements: list[tuple[int, int, int, int]] = []

    for idx in perm:
        wi, hi = rects[idx]
        bx, by = _find_skyline_position(wi, hi, W, skyline)
        placements.append((bx, by, wi, hi))
        skyline = _update_skyline(skyline, bx, by, wi, hi, W)

    height = max(y + h for (_, y, _, h) in placements) if placements else 0
    return height, placements


# ---------------------------------------------------------------------------
# Public solver API
# ---------------------------------------------------------------------------

def solve_sa(
    instance: Instance,
    time_limit: float = 10.0,
    seed: Optional[int] = None,
    log_path: Optional[str] = None,
) -> dict:
    """
    Solve a 2D Strip Packing instance using Simulated Annealing.

    Args:
        instance:   the benchmark instance.
        time_limit: maximum wall-clock seconds (default 10 s).
        seed:       random seed for reproducibility (default None = system random).
        log_path:   optional path; writes JSONL start + done events for the
                    visualizer (no step-by-step events — SA is too fast).

    Returns:
        dict with keys:
            height      – best strip height found
            placements  – corresponding list of (x, y, w, h)
            status      – 'SA'
            wall_time   – elapsed seconds
            iterations  – SA iterations performed
            init_height – Skyline height before SA improvement
    """
    if seed is not None:
        random.seed(seed)

    W = instance.strip_width
    rects = list(instance.rectangles)
    n = len(rects)

    # -- logging --------------------------------------------------------------
    _log = None
    if log_path is not None:
        _log = open(log_path, "w", buffering=1)
        _log.write(json.dumps({
            "event":   "start",
            "name":    instance.name,
            "W":       W,
            "n":       n,
            "area_LB": instance.area_lower_bound,
            "elapsed": 0.0,
        }) + "\n")

    t0 = time.perf_counter()

    # -- initial solution: decreasing-height permutation -------------------
    current_perm = sorted(range(n), key=lambda i: -rects[i][1])

    if n < 2:
        # Cannot perform any swaps; just return the greedy result.
        init_h, init_placements = _eval_perm(current_perm, rects, W)
        elapsed = time.perf_counter() - t0
        if _log is not None:
            _log.write(json.dumps({
                "event":      "improvement",
                "iteration":  0,
                "height":     init_h,
                "elapsed":    elapsed,
                "placements": [list(p) for p in init_placements],
            }) + "\n")
            _log.write(json.dumps({
                "event":       "done",
                "height":      init_h,
                "init_height": init_h,
                "iterations":  0,
                "wall_time":   elapsed,
                "elapsed":     elapsed,
                "status":      "SA",
                "placements":  [list(p) for p in init_placements],
            }) + "\n")
            _log.close()
        return {
            "height":      init_h,
            "placements":  init_placements,
            "status":      "SA",
            "wall_time":   elapsed,
            "iterations":  0,
            "init_height": init_h,
        }

    current_h, current_plac = _eval_perm(current_perm, rects, W)
    init_h = current_h
    best_h = current_h
    best_placements: list[tuple[int, int, int, int]] = []

    # Keep a separate best permutation so we can reconstruct placements once
    # at the end rather than copying them on every improvement.
    best_perm = current_perm[:]

    # Show the initial greedy arrangement in the visualizer immediately.
    if _log is not None:
        _log.write(json.dumps({
            "event":      "improvement",
            "iteration":  0,
            "height":     best_h,
            "elapsed":    time.perf_counter() - t0,
            "placements": [list(p) for p in current_plac],
        }) + "\n")

    # -- SA main loop ------------------------------------------------------
    # Slow cooling ensures time_limit is the binding constraint on all but
    # the smallest instances.  T0=50 makes ~80 % of 5-unit uphill moves
    # acceptable initially, dropping to <5 % once T < 1.
    T = 50.0
    alpha = 0.9997
    T_min = 0.01
    iteration = 0

    while T > T_min and (time.perf_counter() - t0) < time_limit:
        # Perturbation: alternate between swap and random-insertion moves.
        if random.random() < 0.5:
            # Swap move: exchange two random positions.
            i, j = random.sample(range(n), 2)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            new_h, new_plac = _eval_perm(current_perm, rects, W)
            delta = new_h - current_h
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_h = new_h
                if new_h < best_h:
                    best_h = new_h
                    best_perm = current_perm[:]
                    if _log is not None:
                        _log.write(json.dumps({
                            "event":      "improvement",
                            "iteration":  iteration,
                            "height":     best_h,
                            "elapsed":    time.perf_counter() - t0,
                            "placements": [list(p) for p in new_plac],
                        }) + "\n")
            else:
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        else:
            # Insert move: remove item at position i, re-insert at position j.
            i = random.randrange(n)
            j = random.randrange(n - 1)
            if j >= i:
                j += 1
            item = current_perm.pop(i)
            current_perm.insert(j, item)
            new_h, new_plac = _eval_perm(current_perm, rects, W)
            delta = new_h - current_h
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_h = new_h
                if new_h < best_h:
                    best_h = new_h
                    best_perm = current_perm[:]
                    if _log is not None:
                        _log.write(json.dumps({
                            "event":      "improvement",
                            "iteration":  iteration,
                            "height":     best_h,
                            "elapsed":    time.perf_counter() - t0,
                            "placements": [list(p) for p in new_plac],
                        }) + "\n")
            else:
                # Revert insertion: remove from j, put back at i.
                current_perm.pop(j)
                current_perm.insert(i, item)

        T *= alpha
        iteration += 1

    # Reconstruct best placements from best permutation.
    _, best_placements = _eval_perm(best_perm, rects, W)

    elapsed = time.perf_counter() - t0

    if _log is not None:
        _log.write(json.dumps({
            "event":       "done",
            "height":      best_h,
            "init_height": init_h,
            "iterations":  iteration,
            "wall_time":   elapsed,
            "elapsed":     elapsed,
            "status":      "SA",
            "placements":  [list(p) for p in best_placements],
        }) + "\n")
        _log.close()

    return {
        "height":      best_h,
        "placements":  best_placements,
        "status":      "SA",
        "wall_time":   elapsed,
        "iterations":  iteration,
        "init_height": init_h,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from benchmarks import get_all_benchmarks
    from solver_heuristic import solve_blf_best
    from solver_skyline import solve_skyline_best

    demo = Instance(
        name="demo_sa",
        strip_width=10,
        rectangles=[(5, 3), (4, 4), (6, 2), (3, 5), (7, 3)],
    )
    print(f"SA on demo instance (W={demo.strip_width}, n={demo.n}):")
    r = solve_sa(demo, time_limit=2.0, seed=42)
    print(
        f"  SA height={r['height']}  init={r['init_height']}"
        f"  iters={r['iterations']}  time={r['wall_time']:.3f}s"
    )

    print("\nBenchmark comparison (SA vs Skyline vs BLF) — first 10 instances:")
    header = f"{'Instance':30s}  {'SA':>5}  {'Skyline':>7}  {'BLF':>5}  {'SA vs BLF':>9}  {'iters':>6}"
    print(header)
    print("-" * len(header))
    for inst in get_all_benchmarks()[:10]:
        sa  = solve_sa(inst, time_limit=5.0, seed=42)
        sk  = solve_skyline_best(inst)
        bl  = solve_blf_best(inst)
        imp = bl["height"] - sa["height"]
        print(
            f"  {inst.name:30s}  {sa['height']:5d}  {sk['height']:7d}"
            f"  {bl['height']:5d}  {imp:+9d}  {sa['iterations']:6d}"
        )
