"""
group_stats.py
==============
Group-aware (cluster) statistics for kernel-selection evaluation.

Matrices from the same family are not independent observations.  The plain bootstrap
in ``evaluation.metrics`` resamples matrices as if they were independent and therefore
under-states uncertainty when families are large.  The functions here resample (or sign
flip) whole GROUPS, so the effective sample size is the number of groups.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _members(labels: np.ndarray):
    uniq = sorted(set(labels.tolist()))
    return [np.flatnonzero(labels == u) for u in uniq]


def _clean(values, clusters):
    v = np.asarray(values, dtype=float)
    labels = np.asarray(clusters, dtype=object)
    if v.shape != labels.shape:
        raise ValueError("values and clusters must have the same length")
    keep = ~np.isnan(v)
    return v[keep], labels[keep]


def cluster_bootstrap_ci(values, clusters, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI of the mean of ``values``, resampling whole groups with replacement.

    Returns ``(low, high)``.  NaNs are dropped together with their cluster label.
    Fewer than 2 groups -> ``(nan, nan)`` (an honest "cannot tell").
    """
    v, labels = _clean(values, clusters)
    members = _members(labels)
    if len(members) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sums = np.array([v[m].sum() for m in members])
    cnts = np.array([len(m) for m in members], dtype=float)
    g = len(members)
    idx = rng.integers(0, g, size=(n_boot, g))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def group_mean(values, clusters) -> float:
    """Mean of per-group means (every group weighs the same, whatever its size)."""
    v, labels = _clean(values, clusters)
    if v.size == 0:
        return float("nan")
    return float(np.mean([v[m].mean() for m in _members(labels)]))


def paired_group_signflip_test(diff, clusters, n_perm: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Paired comparison of two methods on the same matrices, at group level.

    ``diff`` = per-matrix (method A - method B) of a loss-like quantity.  Differences are
    averaged within each group (one value per group), then a sign-flip permutation test
    of the mean of those group values is run (exact enumeration when there are at most
    14 groups, Monte-Carlo otherwise).  Returns the mean group difference, the standardised
    effect size ``dz`` (mean / sd of group values), a cluster-bootstrap CI and the
    two-sided p-value.  ``p_floor`` is the smallest attainable p-value: with few groups it
    is large, and nothing below it may be claimed.
    """
    d, labels = _clean(diff, clusters)
    members = _members(labels)
    gvals = np.array([d[m].mean() for m in members])
    g = gvals.size
    out = {"n_groups": g, "n_matrices": int(d.size), "mean_group_diff": float("nan"),
           "effect_size_dz": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
           "p_value": float("nan"), "p_floor": float("nan")}
    if g < 2:
        return out
    obs = float(gvals.mean())
    rng = np.random.default_rng(seed)
    if g <= 14:
        signs = ((np.arange(2 ** g)[:, None] >> np.arange(g)) & 1) * 2 - 1
        p_floor = 2.0 / 2 ** g
    else:
        signs = rng.choice([-1, 1], size=(n_perm, g))
        p_floor = 1.0 / n_perm
    perm = (signs * gvals).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs) - 1e-15).mean())
    sd = float(gvals.std(ddof=1))
    out.update({"mean_group_diff": obs, "effect_size_dz": obs / sd if sd > 0 else float("nan"),
                "p_value": p, "p_floor": float(p_floor)})
    out["ci_low"], out["ci_high"] = cluster_bootstrap_ci(d, labels, seed=seed)
    return out


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control). NaN stay NaN."""
    p = np.asarray(p_values, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
