"""
Algorithm-Selection Model for Strip Packing
============================================

Trains two classifier models to predict which heuristic will achieve the
smallest packing height for a given instance:

  1. RandomForest (RF) – ensemble of 300 decision trees, no feature scaling.
  2. Multi-Layer Perceptron (MLP) – a 3-layer neural network (PyTorch).

Both models use the same 17 dimensionless instance features (see
instance_features.py) and are evaluated with 5-fold stratified cross-
validation on the 60 training instances, then tested on a separate hold-out
set of 100 randomly generated instances.

Candidate solvers (10 total):
  - BLF  × 4 sort strategies  (height / width / area / perimeter)
  - NFDH, FFDH
  - Skyline × 4 sort strategies

SA is intentionally excluded from the candidate set (too slow for inference).

Usage
-----
    pip install scikit-learn joblib torch
    python train_model.py

Output files
------------
    best_solver_model.pkl     – RF bundle  (model, solver_names, feature_names)
    best_solver_model_nn.pkl  – MLP bundle (state_dict, scaler, solver_names,
                                            feature_names, architecture)
    test_results.csv          – per-instance accuracy on the 100-instance test set
"""

import csv
import sys
from collections import Counter

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError:
    sys.exit("scikit-learn and joblib are required.\n"
             "Install with:  pip install scikit-learn joblib")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not found – MLP training will be skipped.\n"
          "Install with:  pip install torch")

from benchmarks import get_all_benchmarks, generate_test_set
from instance_features import extract_features, FEATURE_NAMES
from solver_heuristic import solve_blf
from solver_shelf import solve_nfdh, solve_ffdh
from solver_skyline import solve_skyline

# ── Candidate solvers ─────────────────────────────────────────────────────────
# Each entry: (solver_id, callable(instance) → dict with "height" key).
# Indices 0-9 are the class labels for both the RF and MLP classifiers.

SOLVER_CANDIDATES = [
    ("blf_height",      lambda i: solve_blf(i, sort_key="height")),
    ("blf_width",       lambda i: solve_blf(i, sort_key="width")),
    ("blf_area",        lambda i: solve_blf(i, sort_key="area")),
    ("blf_perimeter",   lambda i: solve_blf(i, sort_key="perimeter")),
    ("nfdh",            lambda i: solve_nfdh(i)),
    ("ffdh",            lambda i: solve_ffdh(i)),
    ("sky_height",      lambda i: solve_skyline(i, sort_key="height")),
    ("sky_width",       lambda i: solve_skyline(i, sort_key="width")),
    ("sky_area",        lambda i: solve_skyline(i, sort_key="area")),
    ("sky_perimeter",   lambda i: solve_skyline(i, sort_key="perimeter")),
]

SOLVER_NAMES = [name for name, _ in SOLVER_CANDIDATES]

RF_MODEL_PATH  = "best_solver_model.pkl"
MLP_MODEL_PATH = "best_solver_model_nn.pkl"


# ── PyTorch MLP architecture ──────────────────────────────────────────────────

class StripPackingMLP(nn.Module):
    """
    3-layer feed-forward network for algorithm selection.

    Architecture:
        Input  (17)
        → Linear(64) → ReLU → Dropout(p=0.2)
        → Linear(32) → ReLU
        → Linear(n_classes)   [raw logits; softmax applied by loss / inference]
    """

    def __init__(self, input_dim: int = 17, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data collection ───────────────────────────────────────────────────────────

def collect_data(instances):
    """
    Run every SOLVER_CANDIDATE on every instance.

    Returns:
        X         (n_instances × 17)  numpy feature matrix
        y         (n_instances,)      numpy label vector (index of best solver)
        all_heights {inst.name: [height_per_solver]}
    """
    X, y = [], []
    all_heights = {}

    n_solvers = len(SOLVER_CANDIDATES)
    n_inst    = len(instances)
    w_name    = max(len(i.name) for i in instances)

    print(f"Running {n_solvers} solvers × {n_inst} instances …\n")
    header = "  ".join(f"{name:>13}" for name in SOLVER_NAMES)
    print(f"{'Instance':>{w_name}}  {header}  → best")
    print("─" * (w_name + n_solvers * 15 + 10))

    for inst in instances:
        heights = [fn(inst)["height"] for _, fn in SOLVER_CANDIDATES]
        best_idx = heights.index(min(heights))

        all_heights[inst.name] = heights
        X.append(extract_features(inst))
        y.append(best_idx)

        row = "  ".join(f"{h:>13d}" for h in heights)
        print(f"{inst.name:>{w_name}}  {row}  → {SOLVER_NAMES[best_idx]}")

    return np.array(X, dtype=float), np.array(y, dtype=int), all_heights


# ── Random Forest training ────────────────────────────────────────────────────

def train_rf(X, y):
    """Train RandomForestClassifier with 3-fold CV; return fitted model."""
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    # 3 folds: the rarest class (ffdh, 3 examples) fits exactly into 3 folds.
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"\n[RF]  3-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    clf.fit(X, y)
    return clf


# ── MLP training ──────────────────────────────────────────────────────────────

def train_mlp(X: np.ndarray, y: np.ndarray, epochs: int = 300):
    """
    Train a StripPackingMLP with 5-fold CV evaluation, then refit on all data.

    Features are z-score normalised per fold (and on the full dataset for the
    final model).  The scaler fitted on the full dataset is stored with the
    model weights so inference requires no separate normalisation step.

    Returns:
        (mlp, scaler)  where mlp is a fitted StripPackingMLP (CPU) and
        scaler is the StandardScaler fitted on the full training data.
    """
    if not TORCH_AVAILABLE:
        print("[MLP] Skipped (PyTorch not available).")
        return None, None

    # Always use the full candidate count so label indices stay aligned even
    # when some solvers never win on the (small) training set.
    n_classes = len(SOLVER_CANDIDATES)
    device    = torch.device("cpu")

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    # Use 3-fold CV because the rarest class (ffdh) has only 3 training
    # examples; StratifiedKFold requires at least n_splits members per class.
    cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    fold_accs = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        scaler_fold = StandardScaler().fit(X[tr_idx])
        X_tr = torch.tensor(scaler_fold.transform(X[tr_idx]), dtype=torch.float32)
        X_va = torch.tensor(scaler_fold.transform(X[va_idx]), dtype=torch.float32)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
        y_va = torch.tensor(y[va_idx], dtype=torch.long)

        model_fold = StripPackingMLP(input_dim=X.shape[1], n_classes=n_classes).to(device)
        opt   = torch.optim.Adam(model_fold.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        loader  = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

        for _ in range(epochs):
            model_fold.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model_fold(xb), yb).backward()
                opt.step()

        model_fold.eval()
        with torch.no_grad():
            preds = model_fold(X_va).argmax(dim=1)
        acc = (preds == y_va).float().mean().item()
        fold_accs.append(acc)
        print(f"  [MLP] fold {fold}: val_acc={acc:.3f}")

    print(f"\n[MLP] 5-fold CV accuracy: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")

    # ── Final model on full dataset ───────────────────────────────────────────
    scaler = StandardScaler().fit(X)
    X_all  = torch.tensor(scaler.transform(X), dtype=torch.float32)
    y_all  = torch.tensor(y,                   dtype=torch.long)

    mlp     = StripPackingMLP(input_dim=X.shape[1], n_classes=n_classes).to(device)
    opt     = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader  = DataLoader(TensorDataset(X_all, y_all), batch_size=16, shuffle=True)

    for epoch in range(epochs):
        mlp.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            l = loss_fn(mlp(xb), yb)
            l.backward()
            opt.step()
            epoch_loss += l.item() * len(xb)
        if (epoch + 1) % 100 == 0:
            print(f"  [MLP] epoch {epoch + 1}/{epochs}  loss={epoch_loss/len(y):.4f}")

    return mlp, scaler


def _mlp_predict(mlp, scaler, features_2d: np.ndarray) -> int:
    """Return the predicted solver index for a single feature row."""
    x = torch.tensor(
        scaler.transform(features_2d), dtype=torch.float32
    )
    mlp.eval()
    with torch.no_grad():
        return int(mlp(x).argmax(dim=1).item())


# ── Test-set evaluation ───────────────────────────────────────────────────────

def evaluate_on_test_set(rf_clf, mlp, scaler, n_test: int = 100):
    """
    Generate n_test random instances (seed=999), run all SOLVER_CANDIDATES
    as an oracle, then compare RF and MLP predictions.

    Metrics per model:
      - Accuracy  : fraction of instances where the predicted solver equals
                    the oracle (actual best) solver.
      - Mean regret: mean (chosen_height − oracle_height) / oracle_height × 100 %.

    Saves results to test_results.csv and returns a summary dict.
    """
    from benchmarks import generate_test_set

    print("\n" + "═" * 70)
    print("Test-set evaluation (100 random instances, seed=999)")
    print("═" * 70)

    test_instances = generate_test_set(n=n_test, seed=999)
    n_solvers = len(SOLVER_CANDIDATES)

    rf_correct = 0
    mlp_correct = 0
    rf_regret_sum = 0.0
    mlp_regret_sum = 0.0
    rows = []

    for k, inst in enumerate(test_instances, start=1):
        print(f"  [{k:3d}/{n_test}] {inst.name:12s}  n={inst.n:3d}  W={inst.strip_width:4d} … ", end="", flush=True)

        heights = [fn(inst)["height"] for _, fn in SOLVER_CANDIDATES]
        oracle_h   = min(heights)
        oracle_idx = heights.index(oracle_h)

        feats = np.array([extract_features(inst)], dtype=float)

        # RF prediction
        rf_idx = int(rf_clf.predict(feats)[0])
        rf_h   = heights[rf_idx]
        rf_reg = (rf_h - oracle_h) / oracle_h * 100 if oracle_h > 0 else 0.0

        # MLP prediction (may be None if torch unavailable)
        if mlp is not None and scaler is not None:
            mlp_idx = _mlp_predict(mlp, scaler, feats)
            mlp_h   = heights[mlp_idx]
            mlp_reg = (mlp_h - oracle_h) / oracle_h * 100 if oracle_h > 0 else 0.0
        else:
            mlp_idx, mlp_h, mlp_reg = -1, -1, float("nan")

        rf_correct  += int(rf_idx  == oracle_idx)
        mlp_correct += int(mlp_idx == oracle_idx) if mlp is not None else 0
        rf_regret_sum  += rf_reg
        mlp_regret_sum += mlp_reg if not np.isnan(mlp_reg) else 0.0

        print(f"oracle={SOLVER_NAMES[oracle_idx]:14s}  "
              f"RF={SOLVER_NAMES[rf_idx]:14s}({'✓' if rf_idx==oracle_idx else '✗'})  "
              f"MLP={SOLVER_NAMES[mlp_idx] if mlp_idx>=0 else 'N/A':14s}({'✓' if mlp_idx==oracle_idx else '✗'})")

        rows.append({
            "instance":      inst.name,
            "n":             inst.n,
            "W":             inst.strip_width,
            "oracle_solver": SOLVER_NAMES[oracle_idx],
            "oracle_height": oracle_h,
            "rf_solver":     SOLVER_NAMES[rf_idx],
            "rf_height":     rf_h,
            "rf_gap_pct":    round(rf_reg, 3),
            "rf_correct":    int(rf_idx == oracle_idx),
            "mlp_solver":    SOLVER_NAMES[mlp_idx] if mlp_idx >= 0 else "N/A",
            "mlp_height":    mlp_h,
            "mlp_gap_pct":   round(mlp_reg, 3) if not np.isnan(mlp_reg) else "",
            "mlp_correct":   int(mlp_idx == oracle_idx) if mlp is not None else "",
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(test_instances)
    print("\n" + "─" * 50)
    print(f"  RF  accuracy: {rf_correct}/{n} = {rf_correct/n:.1%}   "
          f"mean regret: {rf_regret_sum/n:.2f} %")
    if mlp is not None:
        print(f"  MLP accuracy: {mlp_correct}/{n} = {mlp_correct/n:.1%}   "
              f"mean regret: {mlp_regret_sum/n:.2f} %")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = "test_results.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved → {csv_path}")

    return {
        "rf_accuracy":   rf_correct / n,
        "mlp_accuracy":  mlp_correct / n if mlp is not None else None,
        "rf_regret":     rf_regret_sum / n,
        "mlp_regret":    mlp_regret_sum / n if mlp is not None else None,
    }


# ── Summaries ─────────────────────────────────────────────────────────────────

def print_summary(y, clf):
    print("\nBest-solver distribution across training instances:")
    for idx, count in sorted(Counter(y).items(), key=lambda kv: -kv[1]):
        bar = "█" * count
        print(f"  {SOLVER_NAMES[idx]:20s}  {count:3d}  {bar}")

    print("\nFeature importances (top 10):")
    pairs = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    for name, imp in pairs[:10]:
        bar = "█" * max(1, int(imp * 60))
        print(f"  {name:30s}  {imp:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Collect training data ──────────────────────────────────────────────
    instances = get_all_benchmarks()
    X, y, _ = collect_data(instances)

    # ── 2. Train Random Forest ────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("Training Random Forest …")
    print("═" * 70)
    rf_clf = train_rf(X, y)
    print_summary(y, rf_clf)

    rf_bundle = {
        "model":         rf_clf,
        "solver_names":  SOLVER_NAMES,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(rf_bundle, RF_MODEL_PATH)
    print(f"\nSaved → {RF_MODEL_PATH}")

    # ── 3. Train PyTorch MLP ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("Training PyTorch MLP …")
    print("═" * 70)
    mlp, scaler = train_mlp(X, y, epochs=300)

    if mlp is not None:
        nn_bundle = {
            "model_type":    "mlp",
            "state_dict":    mlp.state_dict(),
            "scaler":        scaler,
            "solver_names":  SOLVER_NAMES,
            "feature_names": FEATURE_NAMES,
            "input_dim":     X.shape[1],
            "n_classes":     len(SOLVER_CANDIDATES),
        }
        joblib.dump(nn_bundle, MLP_MODEL_PATH)
        print(f"Saved → {MLP_MODEL_PATH}")

    # ── 4. Evaluate on held-out test set ──────────────────────────────────────
    evaluate_on_test_set(rf_clf, mlp, scaler)

    print("\nDone.  Run the visualizer and press  🤖 ML Select  to use the models.")


if __name__ == "__main__":
    main()
