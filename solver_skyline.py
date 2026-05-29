"""
Skyline Heuristic for the 2D Strip Packing Problem.

The Skyline algorithm maintains a step-function profile (the "skyline") that
describes the top surface of the current packing.  Each new rectangle is placed
at the lowest feasible y-coordinate reachable by scanning every horizontal
segment of the skyline as a candidate left-edge position.

Compared with Bottom-Left-Fill (BLF), the Skyline heuristic:
  - Fills concavities in the packing profile more aggressively.
  - Achieves equal or better strip heights on the vast majority of benchmark
    instances while retaining the same O(n^2) worst-case complexity per item.
  - Is the standard inner evaluator used by the Simulated Annealing solver.

Sorting strategies (same four as BLF):
  - Decreasing Height  (DH)  -- usually best for tall items
  - Decreasing Width   (DW)
  - Decreasing Area    (DA)
  - Decreasing Perimeter (DP)

Skyline representation:
  A sorted list of (x_start, height) pairs.  Segment i covers the horizontal
  range [x_i, x_{i+1}) with the last segment covering [x_k, W).
  Initial state: [(0, 0)]  (flat bottom at height 0).
"""

import json
import time

from benchmarks import Instance


# ---------------------------------------------------------------------------
# Skyline data-structure helpers
# ---------------------------------------------------------------------------

def _skyline_max_height(skyline: list, x: int, w: int) -> int:
    """Return the maximum skyline height in the horizontal range [x, x+w)."""
    x_end = x + w
    max_h = 0
    for i, (sx, sh) in enumerate(skyline):
        seg_end = skyline[i + 1][0] if i + 1 < len(skyline) else float("inf")
        if seg_end <= x:
            continue
        if sx >= x_end:
            break
        if sh > max_h:
            max_h = sh
    return max_h


def _find_skyline_position(wi: int, hi: int, W: int, skyline: list):
    """
    Find the best placement position for a rectangle of size (wi x hi).

    Tries every skyline-segment left edge as a candidate x.  The effective
    placement y at position x equals the maximum skyline height in [x, x+wi).
    The position with (lowest y, leftmost x) is returned.

    Returns (best_x, best_y).
    """
    best_x, best_y = 0, float("inf")

    for sx, _ in skyline:
        if sx + wi > W:
            continue
        eff_y = _skyline_max_height(skyline, sx, wi)
        if eff_y < best_y or (eff_y == best_y and sx < best_x):
            best_x, best_y = sx, eff_y

    return best_x, best_y


def _update_skyline(skyline: list, x: int, y: int, w: int, h: int, W: int) -> list:
    """
    Update the skyline after placing a rectangle of size (w x h) at (x, y).

    The entire footprint [x, x+w) is raised to height y + h.  Adjacent
    segments with equal height are merged to keep the list compact.
    """
    new_top = y + h
    x_end = x + w
    result = []
    new_seg_added = False

    for i, (sx, sh) in enumerate(skyline):
        seg_end = skyline[i + 1][0] if i + 1 < len(skyline) else W

        if seg_end <= x:
            # Segment entirely to the left of placement: keep unchanged.
            result.append((sx, sh))
        elif sx >= x_end:
            # Segment entirely to the right: flush new segment first, then keep.
            if not new_seg_added:
                result.append((x, new_top))
                new_seg_added = True
            result.append((sx, sh))
        else:
            # Segment overlaps with placement footprint.
            if sx < x:
                result.append((sx, sh))          # left overhang: keep [sx, x)
            if not new_seg_added:
                result.append((x, new_top))      # new raised segment
                new_seg_added = True
            if seg_end > x_end:
                result.append((x_end, sh))       # right overhang: keep [x_end, seg_end)

    if not new_seg_added:
        result.append((x, new_top))

    # Merge consecutive segments with the same height.
    merged = [result[0]]
    for sx, sh in result[1:]:
        if sh != merged[-1][1]:
            merged.append((sx, sh))

    return merged


# ---------------------------------------------------------------------------
# Public solver API
# ---------------------------------------------------------------------------

def solve_skyline(instance: Instance, sort_key: str = "height", log_path=None) -> dict:
    """
    Solve a 2D Strip Packing instance using the Skyline heuristic.

    Args:
        instance:  the benchmark instance.
        sort_key:  sorting criterion.
                   'height'    – decreasing height (DH)
                   'width'     – decreasing width  (DW)
                   'area'      – decreasing area   (DA)
                   'perimeter' – decreasing perimeter (DP)
        log_path:  optional path for real-time JSONL animation log.

    Returns:
        dict with keys:
            height     – strip height achieved
            placements – list of (x, y, w, h) for each rectangle
            status     – 'HEURISTIC'
            wall_time  – elapsed time in seconds
            sort_key   – sorting strategy used
    """
    W = instance.strip_width
    rects = list(instance.rectangles)

    key_fns = {
        "height":    lambda r: (-r[1], -r[0]),
        "width":     lambda r: (-r[0], -r[1]),
        "area":      lambda r: -(r[0] * r[1]),
        "perimeter": lambda r: -(2 * r[0] + 2 * r[1]),
    }
    rects.sort(key=key_fns[sort_key])

    _log = None
    if log_path is not None:
        _log = open(log_path, "w", buffering=1)
        _log.write(json.dumps({
            "event":   "start",
            "name":    instance.name,
            "W":       W,
            "n":       instance.n,
            "area_LB": instance.area_lower_bound,
            "elapsed": 0.0,
        }) + "\n")

    skyline = [(0, 0)]      # initial flat bottom
    placements = []
    current_h = 0
    t0 = time.perf_counter()

    for step, (wi, hi) in enumerate(rects, start=1):
        bx, by = _find_skyline_position(wi, hi, W, skyline)
        placements.append((bx, by, wi, hi))
        skyline = _update_skyline(skyline, bx, by, wi, hi, W)
        current_h = max(current_h, by + hi)

        if _log is not None:
            _log.write(json.dumps({
                "event":          "place",
                "step":           step,
                "x":              bx,
                "y":              by,
                "w":              wi,
                "h":              hi,
                "current_height": current_h,
                "elapsed":        time.perf_counter() - t0,
            }) + "\n")

    elapsed = time.perf_counter() - t0
    height = max(y + h for (_, y, _, h) in placements) if placements else 0

    if _log is not None:
        _log.write(json.dumps({
            "event":     "done",
            "height":    height,
            "sort_key":  sort_key,
            "wall_time": elapsed,
            "elapsed":   elapsed,
        }) + "\n")
        _log.close()

    return {
        "height":     height,
        "placements": placements,
        "status":     "HEURISTIC",
        "wall_time":  elapsed,
        "sort_key":   sort_key,
    }


def solve_skyline_best(instance: Instance) -> dict:
    """Run Skyline with all four sorting strategies; return the best result."""
    best = None
    for key in ("height", "width", "area", "perimeter"):
        result = solve_skyline(instance, sort_key=key)
        if best is None or result["height"] < best["height"]:
            best = result
    return best


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from benchmarks import get_all_benchmarks
    from solver_heuristic import solve_blf_best

    demo = Instance(
        name="demo_skyline",
        strip_width=10,
        rectangles=[(5, 3), (4, 4), (6, 2), (3, 5), (7, 3)],
    )
    print(f"Demo instance: W={demo.strip_width}, n={demo.n}")
    for key in ("height", "width", "area", "perimeter"):
        r = solve_skyline(demo, sort_key=key)
        print(f"  sort={key:10s}  height={r['height']}")

    print("\nBenchmark comparison (Skyline vs BLF) — first 20 instances:")
    sky_wins, blf_wins, ties = 0, 0, 0
    for inst in get_all_benchmarks()[:20]:
        sk = solve_skyline_best(inst)
        bl = solve_blf_best(inst)
        diff = sk["height"] - bl["height"]
        if diff < 0:
            sky_wins += 1
            mark = "SKY ✓"
        elif diff == 0:
            ties += 1
            mark = "TIE  "
        else:
            blf_wins += 1
            mark = "BLF ✓"
        print(
            f"  {inst.name:30s}  skyline={sk['height']:4d}  blf={bl['height']:4d}"
            f"  Δ={diff:+3d}  {mark}"
        )
    print(f"\nSkyline wins: {sky_wins}  BLF wins: {blf_wins}  Ties: {ties}")
