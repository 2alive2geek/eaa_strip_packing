"""
Benchmark Instance Generator for the 2D Strip Packing Problem.

Generates instances based on the classical benchmarks described in the literature:
  - Bengtsson (1982): 10 instances, 20-200 rectangles, strip widths 25-40
  - Berkey & Wang (1987): 6 classes, 20-100 items, strip widths 10-100
  - Martello & Vigo (1998): derived from bin packing, 20-100 rectangles, strip widths 10-100

Each instance is a tuple: (strip_width, rectangles) 
where rectangles is a list of (width, height) tuples.
"""

import random

from dataclasses import dataclass

# instance class which stores the name, strip width, and list of rectangles
@dataclass
class Instance:
    name: str
    strip_width: int
    rectangles: list  # list of (width, height) tuples

    # get number of rectangles
    @property
    def n(self):
        return len(self.rectangles)

    # get area lower bound (ceiling of total area / strip width)
    @property
    def area_lower_bound(self):
        """Continuous lower bound: total area / strip width (ceiling)."""
        total_area = sum(rectangle_w * rectangle_h for rectangle_w, rectangle_h in self.rectangles)
        return -(-total_area // self.strip_width)  # ceiling division

    def __repr__(self):
        return f"Instance({self.name}, W={self.strip_width}, n={self.n})"


def generate_bengtsson(seed=42):
    """
    Generate instances inspired by Bengtsson (1982).
    10 instances with 20-200 rectangles, strip widths 25-40.
    Rectangles have widths in [1, W] and heights in [1, W].
    """
    rng = random.Random(seed)
    instances = []
    counts = [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    for idx, n in enumerate(counts):
        W = rng.randint(25, 40)
        rects = [(rng.randint(1, W), rng.randint(1, W)) for _ in range(n)]
        instances.append(Instance(f"bengtsson_{idx + 1}", W, rects))
    return instances


def generate_berkey_wang(seed=42):
    """
    Generate instances inspired by Berkey & Wang (1987).
    6 classes x 5 sizes (20, 40, 60, 80, 100 items) = 30 instances.

    Classes differ in how rectangle dimensions relate to strip width:
      Class 1: W=10,  w_i in [1,  10], h_i in [1,  10]
      Class 2: W=30,  w_i in [1,  10], h_i in [1,  10]
      Class 3: W=40,  w_i in [1,  35], h_i in [1,  35]
      Class 4: W=100, w_i in [1,  35], h_i in [1,  35]
      Class 5: W=100, w_i in [1,  100], h_i in [1, 100]
      Class 6: W=300, w_i in [1,  100], h_i in [1, 100]
    """
    rng = random.Random(seed)
    classes = [
        (10, 1, 10, 1, 10),
        (30, 1, 10, 1, 10),
        (40, 1, 35, 1, 35),
        (100, 1, 35, 1, 35),
        (100, 1, 100, 1, 100),
        (300, 1, 100, 1, 100),
    ]
    sizes = [20, 40, 60, 80, 100]
    instances = []
    for cls_idx, (W, w_lo, w_hi, h_lo, h_hi) in enumerate(classes):
        for n in sizes:
            rects = [
                (rng.randint(w_lo, w_hi), rng.randint(h_lo, h_hi))
                for _ in range(n)
            ]
            instances.append(
                Instance(f"bw_c{cls_idx + 1}_n{n}", W, rects)
            )
    return instances


def generate_martello_vigo(seed=42):
    """
    Generate instances inspired by Martello & Vigo (1998).
    4 classes x 5 sizes (20, 40, 60, 80, 100) = 20 instances.

    Classes control how rectangle dimensions relate to W:
      Class 1: W=10,  w_i in [1, 10],  h_i in [1, 10]
      Class 2: W=30,  w_i in [1, 30],  h_i in [1, 30]
      Class 3: W=40,  w_i in [W/3, W], h_i in [1, W/2]   (wide items)
      Class 4: W=100, w_i in [1, W/2], h_i in [W/3, W]   (tall items)
    """
    rng = random.Random(seed)
    sizes = [20, 40, 60, 80, 100]
    instances = []

    # Class 1: small items, narrow strip
    W = 10
    for n in sizes:
        rects = [(rng.randint(1, W), rng.randint(1, W)) for _ in range(n)]
        instances.append(Instance(f"mv_c1_n{n}", W, rects))

    # Class 2: small to medium items
    W = 30
    for n in sizes:
        rects = [(rng.randint(1, W), rng.randint(1, W)) for _ in range(n)]
        instances.append(Instance(f"mv_c2_n{n}", W, rects))

    # Class 3: wide items
    W = 40
    for n in sizes:
        rects = [
            (rng.randint(W // 3, W), rng.randint(1, W // 2))
            for _ in range(n)
        ]
        instances.append(Instance(f"mv_c3_n{n}", W, rects))

    # Class 4: tall items
    W = 100
    for n in sizes:
        rects = [
            (rng.randint(1, W // 2), rng.randint(W // 3, W))
            for _ in range(n)
        ]
        instances.append(Instance(f"mv_c4_n{n}", W, rects))

    return instances


def get_all_benchmarks(seed=42):
    """Return all benchmark instances."""
    return (
        generate_bengtsson(seed)
        + generate_berkey_wang(seed)
        + generate_martello_vigo(seed)
    )


def get_small_benchmarks(seed=42):
    """Return a subset of small instances suitable for exact solving."""
    all_instances = get_all_benchmarks(seed)
    return [inst for inst in all_instances if inst.n <= 20]

def generate_random_instance(seed=None) -> "Instance":
    """
    Generate a single random strip-packing instance.

    Parameter ranges cover the full span of all benchmark families:
      - n_rects ∈ [10, 150]
      - W       ∈ [10, 300]
      - widths and heights uniformly drawn from [1, W]

    Args:
        seed: random seed for reproducibility.  Pass None (default) for a
              fresh random instance each call.

    Returns:
        An Instance named ``rand_<seed>`` (or ``rand_<random-tag>`` if seed
        is None).
    """
    rng = random.Random(seed)
    n = rng.randint(10, 150)
    W = rng.randint(10, 300)
    rects = [(rng.randint(1, W), rng.randint(1, W)) for _ in range(n)]
    tag = seed if seed is not None else rng.randint(10000, 99999)
    return Instance(f"rand_{tag}", W, rects)


def generate_test_set(n: int = 100, seed: int = 999) -> list:
    """
    Generate *n* random test instances for ML model evaluation.

    Uses a different base seed (999) from all benchmark families (42) so
    there is no overlap with training data.

    Four rectangle-generation styles are sampled uniformly to ensure the
    test set covers the same diversity as the benchmark families:
      - style 0: uniform [1, W] × [1, W]          (Bengtsson / BW-C5)
      - style 1: wide items  [W/3, W] × [1, W/2]  (MV class 3)
      - style 2: tall items  [1, W/2] × [W/3, W]  (MV class 4)
      - style 3: small items [1, W/3] × [1, W/3]  (BW classes 1-2)

    Args:
        n:    number of instances to generate (default 100).
        seed: base random seed (default 999).

    Returns:
        List of n Instance objects named ``test_1`` … ``test_n``.
    """
    rng = random.Random(seed)
    instances = []
    for i in range(n):
        n_rects = rng.randint(10, 150)
        W = rng.randint(10, 300)
        style = rng.randint(0, 3)
        if style == 0:
            rects = [
                (rng.randint(1, W), rng.randint(1, W))
                for _ in range(n_rects)
            ]
        elif style == 1:
            w_lo = max(1, W // 3)
            h_hi = max(1, W // 2)
            rects = [
                (rng.randint(w_lo, W), rng.randint(1, h_hi))
                for _ in range(n_rects)
            ]
        elif style == 2:
            w_hi = max(1, W // 2)
            h_lo = max(1, W // 3)
            rects = [
                (rng.randint(1, w_hi), rng.randint(h_lo, W))
                for _ in range(n_rects)
            ]
        else:
            max_dim = max(1, W // 3)
            rects = [
                (rng.randint(1, max_dim), rng.randint(1, max_dim))
                for _ in range(n_rects)
            ]
        instances.append(Instance(f"test_{i + 1}", W, rects))
    return instances


generation_cfg_list = [
    ("Bengtsson", generate_bengtsson),
    ("Berkey & Wang", generate_berkey_wang),
    ("Martello & Vigo", generate_martello_vigo),
]

if __name__ == "__main__":
    for suite_name, gen_fn in generation_cfg_list:
        instances = gen_fn()
        print(f"\n{'=' * 50}")
        print(f" {suite_name} Benchmark Suite ({len(instances)} instances)")
        print(f"{'=' * 50}")
        for inst in instances:
            print(
                f"  {inst.name:20s}  W={inst.strip_width:>4d}  "
                f"n={inst.n:>4d}  area_LB={inst.area_lower_bound:>6d}"
            )
