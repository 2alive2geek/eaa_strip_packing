"""
Feature extraction for strip-packing instances.

Shared between train_model.py (training) and the visualizer's ML advisor
(inference at run-time).  Uses only the Python standard library so it
stays importable without numpy/scikit-learn installed.

The features are all dimensionless (normalised) so the model generalises
across different strip widths and instance sizes.

Feature names (in order):
    n, W, area_lb,
    mean_w/W,  std_w/W,  max_w/W,  min_w/W,
    mean_h/lb, std_h/lb, max_h/lb, min_h/lb,
    mean_aspect, std_aspect,
    mean_item_area/strip_area, density,
    frac_wide, frac_tall
"""

import math


def extract_features(instance) -> list:
    """
    Return a Python list of 17 numeric features for the given instance.
    Compatible with numpy (just wrap in np.array(..., dtype=float)).
    """
    W = instance.strip_width
    rects = instance.rectangles
    n = len(rects)

    widths  = [w for w, _ in rects]
    heights = [h for _, h in rects]
    areas   = [w * h for w, h in rects]
    aspects = [w / h for w, h in rects]      # w/h ratios

    alb = max(instance.area_lower_bound, 1)  # area lower bound (≥1 guard)
    total_area = sum(areas)
    strip_area = W * alb                     # area of strip at alb height

    def _std(vals, mean):
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

    mw = sum(widths)  / n
    mh = sum(heights) / n
    ma = sum(aspects) / n

    return [
        # ── instance size ──────────────────────────────────────────────
        n,
        W,
        alb,

        # ── width statistics (normalised by W) ────────────────────────
        mw / W,           # mean width ratio
        _std(widths,  mw) / W,
        max(widths) / W,  # widest item fraction
        min(widths) / W,  # narrowest item fraction

        # ── height statistics (normalised by area lower bound) ────────
        mh / alb,
        _std(heights, mh) / alb,
        max(heights) / alb,
        min(heights) / alb,

        # ── aspect-ratio statistics ───────────────────────────────────
        ma,
        _std(aspects, ma),

        # ── area / density ────────────────────────────────────────────
        (total_area / n) / strip_area,  # mean item area / strip area
        total_area / strip_area,        # packing density at alb height (≈1)

        # ── categorical fractions ─────────────────────────────────────
        sum(1 for w in widths  if w > W / 2)       / n,  # "wide" items
        sum(1 for h in heights if h > alb / n)     / n,  # "tall" items
    ]


FEATURE_NAMES = [
    "n", "W", "area_lb",
    "mean_w/W",  "std_w/W",  "max_w/W",  "min_w/W",
    "mean_h/lb", "std_h/lb", "max_h/lb", "min_h/lb",
    "mean_aspect", "std_aspect",
    "mean_item_area/strip_area", "density",
    "frac_wide", "frac_tall",
]
