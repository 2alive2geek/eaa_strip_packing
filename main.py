"""
Strip Packing Experiment Runner
================================
Runs both the CP-SAT exact solver and the Bottom-Left-Fill heuristic on
the benchmark instances (Bengtsson, Berkey & Wang, Martello & Vigo) and
reports comparative results.

Usage:
    python main.py                   # run all benchmarks (heuristic only for n>25)
    python main.py --exact-limit 30  # run exact solver on instances with n <= 30
    python main.py --time-limit 120  # set exact solver time limit to 120s

Results are printed as formatted tables and saved to results.csv.
"""

import argparse
import csv
import time

from benchmarks import (
    generate_bengtsson,
    generate_berkey_wang,
    generate_martello_vigo,
)
from solver_exact import solve_exact
from solver_heuristic import solve_blf, solve_blf_best


def run_experiment(instances, exact_n_limit=25, exact_time_limit=60.0):
    """
    Run both solvers on a list of instances.

    Args:
        instances: list of Instance objects.
        exact_n_limit: only run exact solver on instances with n <= this value.
        exact_time_limit: CP-SAT time limit in seconds.

    Returns:
        list of result dicts.
    """
    results = []
    for inst in instances:
        row = {
            "name": inst.name,
            "n": inst.n,
            "W": inst.strip_width,
            "area_LB": inst.area_lower_bound,
        }

        # --- Heuristic ---
        heur = solve_blf_best(inst)
        row["blf_height"] = heur["height"]
        row["blf_sort"] = heur.get("sort_key", "?")
        row["blf_time"] = heur["wall_time"]
        row["blf_gap_pct"] = (
            (heur["height"] - inst.area_lower_bound)
            / inst.area_lower_bound
            * 100
        )

        # --- Exact ---
        if inst.n <= exact_n_limit:
            exact = solve_exact(inst, time_limit_seconds=exact_time_limit)
            row["exact_height"] = exact["height"]
            row["exact_status"] = exact["status"]
            row["exact_time"] = exact["wall_time"]
            if exact["height"] is not None:
                row["exact_gap_pct"] = (
                    (exact["height"] - inst.area_lower_bound)
                    / inst.area_lower_bound
                    * 100
                )
            else:
                row["exact_gap_pct"] = None
        else:
            row["exact_height"] = None
            row["exact_status"] = "SKIPPED"
            row["exact_time"] = None
            row["exact_gap_pct"] = None

        results.append(row)
    return results


def print_results(results, suite_name):
    """Print results as a formatted table."""
    print()
    print("=" * 110)
    print(f"  {suite_name}")
    print("=" * 110)
    header = (
        f"{'Instance':<22s} {'n':>4s} {'W':>4s} {'LB':>6s} | "
        f"{'BLF H':>6s} {'Gap%':>7s} {'Time':>8s} {'Sort':>6s} | "
        f"{'Exact H':>7s} {'Gap%':>7s} {'Time':>8s} {'Status':>8s}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        exact_h = f"{r['exact_height']:>7d}" if r["exact_height"] is not None else "     - "
        exact_gap = f"{r['exact_gap_pct']:>6.1f}%" if r["exact_gap_pct"] is not None else "      -"
        exact_t = f"{r['exact_time']:>7.2f}s" if r["exact_time"] is not None else "      - "
        exact_s = f"{r['exact_status']:>8s}"

        print(
            f"{r['name']:<22s} {r['n']:>4d} {r['W']:>4d} {r['area_LB']:>6d} | "
            f"{r['blf_height']:>6d} {r['blf_gap_pct']:>6.1f}% {r['blf_time']:>7.4f}s {r['blf_sort']:>6s} | "
            f"{exact_h} {exact_gap} {exact_t} {exact_s}"
        )

    print("-" * 110)


def save_csv(all_results, path="results.csv"):
    """Save all results to a CSV file."""
    if not all_results:
        return
    fieldnames = all_results[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="2D Strip Packing - Experimental Comparison"
    )
    parser.add_argument(
        "--exact-limit",
        type=int,
        default=25,
        help="Run exact solver only on instances with n <= this value (default: 25)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="CP-SAT time limit in seconds (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for benchmark generation (default: 42)",
    )
    args = parser.parse_args()

    print("2D Strip Packing Problem - Experimental Analysis")
    print(f"  Exact solver limit: n <= {args.exact_limit}")
    print(f"  CP-SAT time limit : {args.time_limit}s")
    print(f"  Random seed       : {args.seed}")

    all_results = []

    suites = [
        ("Bengtsson (1982)", generate_bengtsson(args.seed)),
        ("Berkey & Wang (1987)", generate_berkey_wang(args.seed)),
        ("Martello & Vigo (1998)", generate_martello_vigo(args.seed)),
    ]

    for suite_name, instances in suites:
        results = run_experiment(
            instances,
            exact_n_limit=args.exact_limit,
            exact_time_limit=args.time_limit,
        )
        print_results(results, suite_name)
        all_results.extend(results)

    save_csv(all_results, "results.csv")

    # Summary statistics
    heur_gaps = [r["blf_gap_pct"] for r in all_results]
    exact_gaps = [r["exact_gap_pct"] for r in all_results if r["exact_gap_pct"] is not None]
    exact_optimal = sum(1 for r in all_results if r["exact_status"] == "OPTIMAL")
    exact_run = sum(1 for r in all_results if r["exact_status"] != "SKIPPED")

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total instances         : {len(all_results)}")
    print(f"  BLF avg gap to LB       : {sum(heur_gaps)/len(heur_gaps):.1f}%")
    if exact_gaps:
        print(f"  Exact avg gap to LB     : {sum(exact_gaps)/len(exact_gaps):.1f}%")
    print(f"  Exact solved optimally  : {exact_optimal}/{exact_run}")
    print("=" * 60)


if __name__ == "__main__":
    main()
