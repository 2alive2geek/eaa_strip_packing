"""
Exact Solver for the 2D Strip Packing Problem using OR-Tools CP-SAT.

Formulation:
  min  H
  s.t. x_i + w_i <= W              for all i
       y_i + h_i <= H              for all i
       no overlap between any pair for all i != j
       x_i, y_i >= 0               for all i

The no-overlap constraint is modeled with OR-Tools' AddNoOverlap2D,
which internally uses interval variables and efficient propagation.

This is an exact method: it finds the provably optimal solution, but
runtime grows exponentially with the number of rectangles (NP-hard).
A time limit is used to keep experiments tractable.
"""

from ortools.sat.python import cp_model
from benchmarks import Instance


def solve_exact(instance: Instance, time_limit_seconds: float = 60.0, log_path=None):
    """
    Solve a 2D Strip Packing instance exactly using CP-SAT.

    Args:
        instance: the benchmark instance to solve.
        time_limit_seconds: maximum solver time.
        log_path: if provided, write JSONL start/done events so a visualizer
            can show the result immediately after solving.

    Returns:
        dict with keys:
            height: optimal (or best found) strip height
            placements: list of (x, y, w, h) for each rectangle
            status: 'OPTIMAL', 'FEASIBLE', or 'INFEASIBLE'/'UNKNOWN'
            wall_time: solver wall-clock time in seconds
    """
    import json

    _log = None
    if log_path is not None:
        _log = open(log_path, "w", buffering=1)
        _log.write(json.dumps({
            "event": "start",
            "name": instance.name,
            "W": instance.strip_width,
            "n": instance.n,
            "area_LB": instance.area_lower_bound,
        }) + "\n")
    model = cp_model.CpModel()
    W = instance.strip_width
    rectangles = instance.rectangles

    # Upper bound on height: stack all rectangles vertically
    H_ub = sum(h for _, h in rectangles)

    # Decision variable for total strip height
    H = model.NewIntVar(0, H_ub, "H")

    x_vars = []
    y_vars = []
    x_intervals = []
    y_intervals = []

    for i, (wi, hi) in enumerate(rectangles):
        xi = model.NewIntVar(0, W - wi, f"x_{i}")
        yi = model.NewIntVar(0, H_ub - hi, f"y_{i}")
        x_vars.append(xi)
        y_vars.append(yi)

        # Interval variables for the NoOverlap2D constraint
        x_iv = model.NewFixedSizeIntervalVar(xi, wi, f"xiv_{i}")
        y_iv = model.NewFixedSizeIntervalVar(yi, hi, f"yiv_{i}")
        x_intervals.append(x_iv)
        y_intervals.append(y_iv)

        # Rectangle must fit within strip height
        model.Add(yi + hi <= H)

    # No two rectangles may overlap
    model.AddNoOverlap2D(x_intervals, y_intervals)

    # Objective: minimize strip height
    model.Minimize(H)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = 4

    status = solver.Solve(model)

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, "UNKNOWN")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        height = solver.Value(H)
        placements = []
        for i, (wi, hi) in enumerate(rectangles):
            placements.append((solver.Value(x_vars[i]), solver.Value(y_vars[i]), wi, hi))
        if _log is not None:
            _log.write(json.dumps({
                "event": "done",
                "height": height,
                "status": status_name,
                "wall_time": solver.WallTime(),
                "placements": placements,
            }) + "\n")
            _log.close()
        return {
            "height": height,
            "placements": placements,
            "status": status_name,
            "wall_time": solver.WallTime(),
        }
    else:
        if _log is not None:
            _log.write(json.dumps({
                "event": "done",
                "height": None,
                "status": status_name,
                "wall_time": solver.WallTime(),
                "placements": [],
            }) + "\n")
            _log.close()
        return {
            "height": None,
            "placements": [],
            "status": status_name,
            "wall_time": solver.WallTime(),
        }


if __name__ == "__main__":
    # Quick demo with a small instance
    demo = Instance(
        name="demo_exact",
        strip_width=10,
        rectangles=[(5, 3), (4, 4), (6, 2), (3, 5), (7, 3)],
    )
    print(f"Solving {demo} with CP-SAT exact solver...")
    result = solve_exact(demo, time_limit_seconds=30)
    print(f"  Status : {result['status']}")
    print(f"  Height : {result['height']}")
    print(f"  Time   : {result['wall_time']:.2f}s")
    if result["placements"]:
        print("  Placements (x, y, w, h):")
        for p in result["placements"]:
            print(f"    {p}")
