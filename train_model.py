"""
Algorithm-Selection Model for Strip Packing
============================================

Trains a RandomForest classifier to predict which heuristic will achieve
the smallest packing height for a given instance, using features derived
from the instance geometry.

Usage
-----
    pip install scikit-learn joblib
    python train_model.py

Output
------
    best_solver_model.pkl  – joblib bundle containing:
        model          RandomForestClassifier (fitted)
        solver_names   list of candidate solver ids (label mapping)
        feature_names  list of feature names (for inspection)

The saved model is loaded at runtime by the visualizer's  ML Select  button.
"""

import sys
from collections import Counter

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import joblib
except ImportError:
    sys.exit(
        "scikit-learn and joblib are required.\n"
        "Install with:  pip install scikit-learn joblib"
    )

from benchmarks import get_all_benchmarks
from instance_features import extract_features, FEATURE_NAMES
from solver_heuristic import solve_blf
from solver_shelf import solve_nfdh, solve_ffdh

# ── Candidate solvers ─────────────────────────────────────────────────────────
# Each entry: (solver_id, callable(instance) -> dict with "height" key)
# The ML model predicts the *index* into this list.

SOLVER_CANDIDATES = [
    ("blf_height",    lambda i: solve_blf(i, sort_key="height")),
    ("blf_width",     lambda i: solve_blf(i, sort_key="width")),
    ("blf_area",      lambda i: solve_blf(i, sort_key="area")),
    ("blf_perimeter", lambda i: solve_blf(i, sort_key="perimeter")),
    ("nfdh",          lambda i: solve_nfdh(i)),
    ("ffdh",          lambda i: solve_ffdh(i)),
]

SOLVER_NAMES = [name for name, _ in SOLVER_CANDIDATES]

MODEL_PATH = "best_solver_model.pkl"


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_data(instances):
    """Run every candidate solver on every instance.  Return (X, y, results)."""
    X, y = [], []
    results = {}   # inst.name -> list of heights (one per candidate)

    n_solvers = len(SOLVER_CANDIDATES)
    n_inst    = len(instances)
    w_name    = max(len(i.name) for i in instances)

    print(f"Running {n_solvers} solvers × {n_inst} instances …\n")
    header = "  ".join(f"{name:>13}" for name in SOLVER_NAMES)
    print(f"{'Instance':>{w_name}}  {header}  → best")
    print("─" * (w_name + n_solvers * 15 + 10))

    for inst in instances:
        heights = [solver(inst)["height"] for _, solver in SOLVER_CANDIDATES]
        best_idx = heights.index(min(heights))

        results[inst.name] = heights
        X.append(extract_features(inst))
        y.append(best_idx)

        row = "  ".join(f"{h:>13d}" for h in heights)
        print(f"{inst.name:>{w_name}}  {row}  → {SOLVER_NAMES[best_idx]}")

    return np.array(X, dtype=float), np.array(y, dtype=int), results


def train_and_evaluate(X, y):
    """Train RandomForest, cross-validate, fit on full data, return model."""
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"\n5-fold stratified CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)
    return clf


def print_summary(y, clf):
    print("\nBest-solver distribution across all instances:")
    for idx, count in sorted(Counter(y).items(), key=lambda kv: -kv[1]):
        bar = "█" * count
        print(f"  {SOLVER_NAMES[idx]:20s}  {count:3d}  {bar}")

    print("\nFeature importances (top 10):")
    pairs = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    for name, imp in pairs[:10]:
        bar = "█" * max(1, int(imp * 60))
        print(f"  {name:30s}  {imp:.4f}  {bar}")


def main():
    instances = get_all_benchmarks()
    X, y, _ = collect_data(instances)

    print_summary(y, train_and_evaluate.__wrapped__ if hasattr(train_and_evaluate, "__wrapped__") else
                  # just show distribution before training
                  type("_", (), {"feature_importances_": [0]*len(FEATURE_NAMES)})())

    clf = train_and_evaluate(X, y)
    print_summary(y, clf)

    bundle = {
        "model":         clf,
        "solver_names":  SOLVER_NAMES,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved → {MODEL_PATH}")
    print("Run the visualizer and press  🤖 ML Select  to use the model.")


if __name__ == "__main__":
    main()
