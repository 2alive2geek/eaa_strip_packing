"""
Bottom-Left-Fill (BLF) Heuristic for the 2D Strip Packing Problem.

Based on the constructive heuristic family introduced by Baker et al. (1980)
and extended by Chazelle (1983).

Algorithm:
  1. Sort rectangles by a chosen criterion (e.g., decreasing height).
  2. For each rectangle, find the lowest position (y) where it can be placed
     without overlapping any already-placed rectangle, then push it as far
     left (x) as possible within that row.
  3. The strip height is the maximum (y + h) across all placed rectangles.

This file implements BLF with multiple sorting strategies to compare their
effect on solution quality:
  - Decreasing Height (DH)
  - Decreasing Width (DW)
  - Decreasing Area (DA)
  - Decreasing Perimeter (DP)

Design choices:
  - Placed rectangles are stored in a list; for each candidate position we
    scan existing placements to check feasibility. This is O(n^2) per item,
    O(n^3) total, which is acceptable for the benchmark sizes (n <= 200).
  - We discretize candidate y-positions to the set {0} ∪ {y_j + h_j} for
    already-placed rectangles, because an optimal BLF placement always has
    each rectangle touching either the bottom of the strip or the top edge
    of another rectangle.
"""

from benchmarks import Instance


def _no_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    """Return True if two axis-aligned rectangles do not overlap."""
    return x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1


def _find_bottom_left(wi, hi, W, placements):
    """
    Find the BLF position for a rectangle (wi x hi) in a strip of width W
    given the current list of placements [(x, y, w, h), ...].

    Returns (best_x, best_y).
    """
    # Candidate y-levels: bottom of strip + top edges of placed rects
    y_candidates = sorted({0} | {y + h for (_, y, _, h) in placements})

    best_x, best_y = 0, sum(h for (_, _, _, h) in placements) + hi  # fallback

    for y_cand in y_candidates:
        if y_cand + hi > best_y:
            break  # no improvement possible at higher y
        # Try pushing as far left as possible at this y level
        for x_cand in sorted({0} | {x + w for (x, _, w, _) in placements}):
            if x_cand + wi > W:
                continue
            # Check no overlap with all placed rectangles
            ok = all(
                _no_overlap(x_cand, y_cand, wi, hi, px, py, pw, ph)
                for (px, py, pw, ph) in placements
            )
            if ok:
                if (y_cand, x_cand) < (best_y, best_x):
                    best_x, best_y = x_cand, y_cand
                break  # leftmost valid x at this y; move to next y

    return best_x, best_y


def _find_bottom_left_verbose(wi, hi, W, placements, log_file, step, t0):
    """
    Like _find_bottom_left but emits 'search' log events for up to 4 rejected
    candidate positions, so the visualizer can animate the algorithm searching.
    """
    import json
    import time

    y_candidates = sorted({0} | {y + h for (_, y, _, h) in placements})
    best_x = 0
    best_y = (sum(h for (_, _, _, h) in placements) or 0) + hi
    searched = 0

    for y_cand in y_candidates:
        if y_cand + hi > best_y:
            break
        for x_cand in sorted({0} | {x + w for (x, _, w, _) in placements}):
            if x_cand + wi > W:
                continue
            ok = all(
                _no_overlap(x_cand, y_cand, wi, hi, px, py, pw, ph)
                for (px, py, pw, ph) in placements
            )
            if ok:
                if (y_cand, x_cand) < (best_y, best_x):
                    best_x, best_y = x_cand, y_cand
                break  # leftmost valid x at this y; move to next y
            else:
                if searched < 4:
                    log_file.write(json.dumps({
                        "event": "search",
                        "step": step,
                        "x": x_cand, "y": y_cand,
                        "w": wi, "h": hi,
                        "elapsed": time.perf_counter() - t0,
                    }) + "\n")
                    searched += 1

    return best_x, best_y


def solve_blf(instance: Instance, sort_key="height", log_path=None):
    """
    Solve a 2D Strip Packing instance using the Bottom-Left-Fill heuristic.

    Args:
        instance: the benchmark instance.
        sort_key: sorting criterion for rectangles.
            'height'    - decreasing height (DH)
            'width'     - decreasing width  (DW)
            'area'      - decreasing area   (DA)
            'perimeter' - decreasing perim. (DP)
        log_path: if provided, write JSONL placement events to this file
            in real-time so a visualizer can tail it while solving.

    Returns:
        dict with keys:
            height: strip height achieved
            placements: list of (x, y, w, h) for each rectangle
            status: always 'HEURISTIC'
            wall_time: elapsed time in seconds
    """
    import json
    import time

    W = instance.strip_width
    rects = list(instance.rectangles)

    # Sort rectangles
    key_fns = {
        "height": lambda r: (-r[1], -r[0]),
        "width": lambda r: (-r[0], -r[1]),
        "area": lambda r: -(r[0] * r[1]),
        "perimeter": lambda r: -(2 * r[0] + 2 * r[1]),
    }
    rects.sort(key=key_fns[sort_key])

    # Open log file with line-buffering so each event is flushed immediately
    _log = None
    if log_path is not None:
        _log = open(log_path, "w", buffering=1)
        _log.write(json.dumps({
            "event": "start",
            "name": instance.name,
            "W": instance.strip_width,
            "n": instance.n,
            "area_LB": instance.area_lower_bound,
            "elapsed": 0.0,
        }) + "\n")

    placements = []
    t0 = time.perf_counter()

    for step, (wi, hi) in enumerate(rects, start=1):
        if _log is not None:
            bx, by = _find_bottom_left_verbose(wi, hi, W, placements, _log, step, t0)
        else:
            bx, by = _find_bottom_left(wi, hi, W, placements)
        placements.append((bx, by, wi, hi))
        current_h = max(y + h for (_, y, _, h) in placements)

        if _log is not None:
            _log.write(json.dumps({
                "event": "place",
                "step": step,
                "x": bx, "y": by,
                "w": wi, "h": hi,
                "current_height": current_h,
                "elapsed": time.perf_counter() - t0,
            }) + "\n")

    elapsed = time.perf_counter() - t0
    height = max(y + h for (_, y, _, h) in placements) if placements else 0

    if _log is not None:
        _log.write(json.dumps({
            "event": "done",
            "height": height,
            "sort_key": sort_key,
            "wall_time": elapsed,
            "elapsed": elapsed,
        }) + "\n")
        _log.close()

    return {
        "height": height,
        "placements": placements,
        "status": "HEURISTIC",
        "wall_time": elapsed,
    }


def solve_blf_best(instance: Instance):
    """Run BLF with all four sorting strategies, return the best result."""
    best = None
    for key in ("height", "width", "area", "perimeter"):
        result = solve_blf(instance, sort_key=key)
        result["sort_key"] = key
        if best is None or result["height"] < best["height"]:
            best = result
    return best


if __name__ == "__main__":
    demo = Instance(
        name="demo_blf",
        strip_width=10,
        rectangles=[(5, 3), (4, 4), (6, 2), (3, 5), (7, 3)],
    )
    print(f"Solving {demo} with Bottom-Left-Fill heuristic...")
    for key in ("height", "width", "area", "perimeter"):
        result = solve_blf(demo, sort_key=key)
        print(
            f"  Sort={key:10s}  Height={result['height']:4d}  "
            f"Time={result['wall_time']:.4f}s"
        )
    print()
    best = solve_blf_best(demo)
    print(f"  Best: sort={best['sort_key']}, height={best['height']}")
    if best["placements"]:
        print("  Placements (x, y, w, h):")
        for p in best["placements"]:
            print(f"    {p}")
