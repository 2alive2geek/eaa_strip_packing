"""
Shelf-based heuristics for 2D Strip Packing.

NFDH – Next Fit Decreasing Height
    Sort items by decreasing height.  Maintain one open shelf; if the next
    item does not fit horizontally, close it and open a new shelf above.
    Approximation ratio: 2 × OPT + h_max  (Coffman et al., 1980).
    Time complexity: O(n log n).

FFDH – First Fit Decreasing Height
    Sort items by decreasing height.  For each item scan shelves from the
    bottom and place it on the first shelf where it fits; if none, open a
    new shelf above the current stack.
    Approximation ratio: 17/10 × OPT + h_max  (Coffman et al., 1980).
    Time complexity: O(n²) worst case.

Log format (JSONL, identical to solver_heuristic.py):
    start  – instance metadata
    place  – each rectangle placed (step, x, y, w, h, current_height, elapsed)
    done   – final height, algorithm name, wall_time
"""

import json
import time

from benchmarks import Instance


def _open_log(log_path: str, instance: Instance, algo: str):
    """Open log file, write the start event, return (file_handle, t0)."""
    t0 = time.perf_counter()
    if log_path is None:
        return None, t0
    f = open(log_path, "w", buffering=1)
    f.write(json.dumps({
        "event":    "start",
        "name":     instance.name,
        "W":        instance.strip_width,
        "n":        instance.n,
        "area_LB":  instance.area_lower_bound,
        "elapsed":  0.0,
    }) + "\n")
    return f, t0


def _write_place(f, step: int, x, y, w, h, cur_h, t0: float):
    f.write(json.dumps({
        "event":          "place",
        "step":           step,
        "x": x, "y": y,
        "w": w, "h": h,
        "current_height": cur_h,
        "elapsed":        time.perf_counter() - t0,
    }) + "\n")


# ── NFDH ─────────────────────────────────────────────────────────────────────

def solve_nfdh(instance: Instance, log_path=None) -> dict:
    """
    Next Fit Decreasing Height heuristic.

    Args:
        instance: the benchmark instance.
        log_path: if provided, write JSONL events so a visualizer can animate.

    Returns:
        dict with keys: height, placements, status, wall_time.
    """
    W = instance.strip_width
    rects = sorted(instance.rectangles, key=lambda r: (-r[1], -r[0]))

    _log, t0 = _open_log(log_path, instance, "nfdh")

    placements = []
    shelf_x = 0   # x cursor on the current shelf
    shelf_y = 0   # y coordinate of the current shelf's bottom
    shelf_h = 0   # height of the tallest item on the current shelf

    for step, (wi, hi) in enumerate(rects, start=1):
        # If item doesn't fit on the current shelf, open a new one above
        if shelf_h > 0 and shelf_x + wi > W:
            shelf_y += shelf_h
            shelf_x = 0
            shelf_h = 0

        x, y = shelf_x, shelf_y
        shelf_x += wi
        shelf_h = max(shelf_h, hi)
        cur_h = shelf_y + shelf_h

        placements.append((x, y, wi, hi))
        if _log is not None:
            _write_place(_log, step, x, y, wi, hi, cur_h, t0)

    height = max(y + h for _, y, _, h in placements) if placements else 0
    elapsed = time.perf_counter() - t0

    if _log is not None:
        _log.write(json.dumps({
            "event":     "done",
            "height":    height,
            "sort_key":  "nfdh",
            "wall_time": elapsed,
            "elapsed":   elapsed,
        }) + "\n")
        _log.close()

    return {
        "height":     height,
        "placements": placements,
        "status":     "HEURISTIC",
        "wall_time":  elapsed,
    }


# ── FFDH ─────────────────────────────────────────────────────────────────────

def solve_ffdh(instance: Instance, log_path=None) -> dict:
    """
    First Fit Decreasing Height heuristic.

    Args:
        instance: the benchmark instance.
        log_path: if provided, write JSONL events so a visualizer can animate.

    Returns:
        dict with keys: height, placements, status, wall_time.
    """
    W = instance.strip_width
    rects = sorted(instance.rectangles, key=lambda r: (-r[1], -r[0]))

    _log, t0 = _open_log(log_path, instance, "ffdh")

    # Each shelf: [y_bottom, shelf_height, x_cursor]
    # shelves are stacked; a new shelf always starts at y = total_h
    shelves = []
    total_h = 0   # sum of all open shelves' heights = y of next new shelf
    placements = []

    for step, (wi, hi) in enumerate(rects, start=1):
        placed = False
        for shelf in shelves:
            if shelf[2] + wi <= W:          # fits horizontally
                x, y = shelf[2], shelf[0]
                shelf[2] += wi              # advance x cursor (shelves are lists)
                placed = True
                break

        if not placed:
            # Open a new shelf on top of the current stack
            x, y = 0, total_h
            total_h += hi
            shelves.append([y, hi, wi])

        cur_h = total_h
        placements.append((x, y, wi, hi))
        if _log is not None:
            _write_place(_log, step, x, y, wi, hi, cur_h, t0)

    height = max(y + h for _, y, _, h in placements) if placements else 0
    elapsed = time.perf_counter() - t0

    if _log is not None:
        _log.write(json.dumps({
            "event":     "done",
            "height":    height,
            "sort_key":  "ffdh",
            "wall_time": elapsed,
            "elapsed":   elapsed,
        }) + "\n")
        _log.close()

    return {
        "height":     height,
        "placements": placements,
        "status":     "HEURISTIC",
        "wall_time":  elapsed,
    }
