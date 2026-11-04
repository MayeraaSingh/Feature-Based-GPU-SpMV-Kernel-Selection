"""
synthetic_timings.py
====================
SYNTHETIC data generators for TESTS ONLY.

Nothing here is a GPU measurement. The timings come from an invented formula
so the pipeline code paths can be exercised before Member 1 delivers real
data. These files must never be written to benchmarks/kernel_timings.csv.
Output file names contain "SYNTHETIC" on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Placeholder kernel names (the real strings come from Member 1).
SYNTHETIC_KERNELS = ["kernel_A", "kernel_B", "kernel_C"]

# (matrix_id, split, group, rows, mean_row_nnz, variance_row_nnz)
_SPEC = [
    ("m01", "train", "g1", 100, 4.0, 0.2),
    ("m02", "train", "g1", 200, 6.0, 0.5),
    ("m03", "train", "g2", 150, 30.0, 40.0),
    ("m04", "train", "g2", 300, 50.0, 90.0),
    ("m05", "val", "g3", 120, 5.0, 0.1),
    ("m06", "val", "g3", 220, 40.0, 70.0),
    ("m07", "test", "g4", 180, 4.5, 0.3),
    ("m08", "test", "g5", 260, 45.0, 80.0),
]


def make_tables() -> Dict[str, pd.DataFrame]:
    """Return consistent features / splits / catalog tables (typed, valid)."""
    feats, splits, cat = [], [], []
    for mid, split, group, rows, mean, var in _SPEC:
        nnz = int(rows * mean)
        feats.append(
            dict(matrix_id=mid, rows=rows, cols=rows, nnz=nnz,
                 mean_row_nnz=mean, variance_row_nnz=var,
                 max_row_nnz=int(mean + 3 * var ** 0.5) + 1,
                 min_row_nnz=max(0, int(mean - 3 * var ** 0.5)))
        )
        splits.append(dict(matrix_id=mid, split=split))
        cat.append(dict(matrix_id=mid, rows=rows, cols=rows, nnz=nnz, matrix_group=group))
    return {
        "features": pd.DataFrame(feats),
        "splits": pd.DataFrame(splits),
        "catalog": pd.DataFrame(cat),
    }


def make_timings(features: pd.DataFrame, kernels: List[str] = None, seed: int = 0) -> pd.DataFrame:
    """Invented timings: A likes low variance, B likes high variance, C is steady."""
    kernels = kernels or SYNTHETIC_KERNELS
    rng = np.random.default_rng(seed)
    rows = []
    for r in features.itertuples(index=False):
        base = r.nnz / 1e4
        cost = {
            "kernel_A": base * (1.0 + 0.05 * r.variance_row_nnz),
            "kernel_B": base * (1.5 + 0.005 * r.variance_row_nnz),
            "kernel_C": base * 1.6,
        }
        for k in kernels:
            c = cost.get(k, base * 2.0)
            rows.append(dict(
                matrix_id=r.matrix_id, kernel_name=k,
                kernel_time_ms=round(c * (1 + rng.uniform(-0.01, 0.01)), 6),
                preprocessing_time_ms=0.5 if k == "kernel_B" else 0.0,
                correctness_pass=True,
            ))
    return pd.DataFrame(rows)


def write_files(directory: Path | str, tables: Dict[str, pd.DataFrame] = None) -> Dict[str, Path]:
    """Write the four CSVs into ``directory`` and return their paths.

    ``tables`` may override any of features/splits/catalog/timings.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = make_tables()
    base["timings"] = make_timings(base["features"])
    base.update(tables or {})
    paths = {
        "features": directory / "features.csv",
        "splits": directory / "splits.csv",
        "catalog": directory / "matrix_catalog.csv",
        "timings": directory / "SYNTHETIC_kernel_timings.csv",
    }
    for key, path in paths.items():
        base[key].to_csv(path, index=False)
    return paths
