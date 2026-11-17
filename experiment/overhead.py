"""
overhead.py
===========
Amortised end-to-end cost of running N SpMV products on the same matrix.

For matrix i, kernel k:      T_ik(N) = preprocessing_ik + N * kernel_time_ik      (ms)
For a selection method m:    Total_m(N) = sum_i [ selection_overhead_mi + T_{i, k_m(i)}(N) ]
Oracle (upper bound only):   Oracle(N)  = sum_i min_k T_ik(N)  over correct kernels, NO overhead

``selection_overhead`` = feature-gathering time + selector inference time (both measured, both
wall clock on the host); fixed-kernel methods have zero selection overhead.  Preprocessing is
charged once per matrix (the sparsity pattern is assumed to be reused for the N products);
N = 1 is the "single multiplication" case in which nothing is amortised.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

DEFAULT_N = (1, 10, 100, 1000, 10000, 100000)


def _tables(timings: pd.DataFrame):
    ok = timings[timings["correctness_pass"].astype(bool)]
    t = ok.pivot(index="matrix_id", columns="kernel_name", values="kernel_time_ms")
    p = ok.pivot(index="matrix_id", columns="kernel_name", values="preprocessing_time_ms")
    return t, p


def oracle_total(timings: pd.DataFrame, n_iter: int, matrix_ids: Iterable[str]) -> float:
    t, p = _tables(timings)
    ids = [m for m in matrix_ids if m in t.index]
    return float((p.loc[ids] + n_iter * t.loc[ids]).min(axis=1).sum())


def method_total(
    selections: pd.DataFrame,
    timings: pd.DataFrame,
    n_iter: int,
    overhead_ms: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Total ms for ``selections`` (matrix_id, selected_kernel) at ``n_iter`` products.

    Matrices where the selected kernel did not pass correctness are not scored (their count is
    returned) rather than counted as zero.
    """
    t, p = _tables(timings)
    total, n_bad, n = 0.0, 0, 0
    for mid, k in zip(selections["matrix_id"], selections["selected_kernel"]):
        if mid not in t.index or k not in t.columns or np.isnan(t.at[mid, k]):
            n_bad += 1
            continue
        total += float(p.at[mid, k] + n_iter * t.at[mid, k]) + float((overhead_ms or {}).get(mid, 0.0))
        n += 1
    return {"total_ms": total, "n_scored": n, "n_incorrect_selection": n_bad}


def amortised_table(
    methods: Mapping[str, pd.DataFrame],
    timings: pd.DataFrame,
    overhead_ms: Mapping[str, Mapping[str, float]],
    n_values: Iterable[int] = DEFAULT_N,
) -> pd.DataFrame:
    """One row per (method, N): total ms, ratio to the N-aware oracle, and scored counts.

    ``methods``: name -> selections;  ``overhead_ms``: name -> {matrix_id: ms} (missing = 0).
    Ratios use only matrices scored for that method.
    """
    rows = []
    for name, sel in methods.items():
        for n in n_values:
            r = method_total(sel, timings, n, overhead_ms.get(name))
            orc = oracle_total(timings, n, sel["matrix_id"])
            rows.append({"method": name, "n_iterations": int(n), **r, "oracle_total_ms": orc,
                         "ratio_to_oracle": r["total_ms"] / orc if orc > 0 and r["n_scored"] else float("nan")})
    return pd.DataFrame(rows)


def break_even(
    sel_a: pd.DataFrame,
    sel_b: pd.DataFrame,
    timings: pd.DataFrame,
    overhead_a: Optional[Mapping[str, float]] = None,
    overhead_b: Optional[Mapping[str, float]] = None,
    n_max: int = 10_000_000,
) -> float:
    """Smallest N (integer) for which Total_A(N) <= Total_B(N) over the common matrices.

    Totals are linear in N, so this is solved exactly: Total(N) = c + N*s.  Returns ``inf`` if A
    never catches up (its per-iteration cost is not lower) and 1.0 if it is better already at N = 1.
    """
    ids = sorted(set(sel_a["matrix_id"]) & set(sel_b["matrix_id"]))
    a = sel_a[sel_a["matrix_id"].isin(ids)]
    b = sel_b[sel_b["matrix_id"].isin(ids)]
    ca = method_total(a, timings, 0, overhead_a)["total_ms"]
    cb = method_total(b, timings, 0, overhead_b)["total_ms"]
    sa = method_total(a, timings, 1, overhead_a)["total_ms"] - ca
    sb = method_total(b, timings, 1, overhead_b)["total_ms"] - cb
    if ca <= cb and sa <= sb:
        return 1.0
    if sa >= sb:
        return float("inf")
    n_star = (ca - cb) / (sb - sa)
    n = float(np.ceil(max(n_star, 1.0)))
    return n if n <= n_max else float("inf")
