"""
methods.py
==========
Selection methods compared in the experiments, all behind one interface:

    method.fit(train_rows)          # rows: features + 'label' (oracle kernel) of TRAIN matrices
    method.predict(feature_rows)    # -> array of kernel names

* ``TreeSelector``        sklearn DecisionTreeClassifier (interpretable baseline)
* ``RowLengthHeuristic``  thresholds on mean_row_nnz fitted on TRAIN timings (minimise train total time)
* ``RandomForestSelector`` is the existing ``selector.model.KernelSelector`` (not re-implemented here)

Leakage discipline: ``fit`` only receives train rows; hyper-parameters are chosen by
``tune_on_validation`` / ``grouped_cv_loss`` which are only ever handed train and validation
matrices.  Nothing in this module reads the test split.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from evaluation.metrics import score_selections
from selector.labels import DEFAULT_TIE_TOLERANCE, oracle_table


class TreeSelector:
    """Decision-tree kernel selector (same predict contract as ``KernelSelector``)."""

    def __init__(self, feature_columns: Sequence[str], seed: int = 0, **params: Any) -> None:
        self.feature_columns = list(feature_columns)
        self.seed = seed
        self.params = dict(params)
        self._model: Optional[DecisionTreeClassifier] = None
        self.classes_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "TreeSelector":
        self._model = DecisionTreeClassifier(random_state=self.seed, **self.params)
        self._model.fit(X[self.feature_columns].to_numpy(float), np.asarray(list(y), dtype=str))
        self.classes_ = [str(c) for c in self._model.classes_]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("not fitted")
        return self._model.predict(X[self.feature_columns].to_numpy(float))


class RowLengthHeuristic:
    """Piecewise-constant rule on mean_row_nnz: up to two thresholds, one kernel per band.

    Fitted on TRAIN timings only: every pair of candidate thresholds (midpoints between
    sorted distinct training values) is scored by the total train kernel time obtained when
    each band uses its best kernel; the cheapest rule wins.  This is the simplest
    "static threshold" selector and is the natural hand-crafted baseline.
    """

    feature = "mean_row_nnz"

    def __init__(self) -> None:
        self.thresholds: List[float] = []
        self.band_kernels: List[str] = []
        self.classes_: List[str] = []

    def fit(self, X: pd.DataFrame, timings: pd.DataFrame) -> "RowLengthHeuristic":
        """``X``: train feature rows (matrix_id, mean_row_nnz); ``timings``: train timings."""
        ok = timings[timings["correctness_pass"].astype(bool)]
        wide = ok.pivot(index="matrix_id", columns="kernel_name", values="kernel_time_ms")
        wide = wide.dropna(axis=0)  # matrices where every candidate was correct
        feats = X.set_index("matrix_id").loc[wide.index, self.feature].to_numpy(float)
        if len(wide) == 0:
            raise ValueError("no train matrices with all kernels correct; cannot fit heuristic")
        order = np.argsort(feats, kind="stable")
        feats, vals = feats[order], wide.to_numpy()[order]
        kernels = list(wide.columns)
        uniq = np.unique(feats)
        mids = list((uniq[:-1] + uniq[1:]) / 2.0) if len(uniq) > 1 else []
        best = (np.inf, [], None)

        def band_cost(lo: int, hi: int):
            if hi <= lo:
                return 0.0, None
            tot = vals[lo:hi].sum(axis=0)
            j = int(np.argmin(tot))
            return float(tot[j]), kernels[j]

        cuts_options: List[List[float]] = [[]] + [[m] for m in mids] + [[a, b] for a, b in itertools.combinations(mids, 2)]
        for cuts in cuts_options:
            idx = [0] + [int(np.searchsorted(feats, c, side="right")) for c in cuts] + [len(feats)]
            cost, ks = 0.0, []
            for lo, hi in zip(idx[:-1], idx[1:]):
                c, k = band_cost(lo, hi)
                cost += c
                ks.append(k)
            if cost < best[0] - 1e-12:
                best = (cost, cuts, ks)
        _, self.thresholds, ks = best
        # empty bands (possible only with duplicate cut positions) inherit a neighbour's kernel
        fallback = next(k for k in ks if k is not None)
        self.band_kernels = [k or fallback for k in ks]
        self.classes_ = sorted(set(self.band_kernels))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        v = X[self.feature].to_numpy(float)
        band = np.searchsorted(np.asarray(self.thresholds, dtype=float), v, side="right")
        return np.asarray([self.band_kernels[b] for b in band], dtype=object)

    def describe(self) -> str:
        edges = [-np.inf] + list(self.thresholds) + [np.inf]
        return "; ".join(f"{self.feature} in ({edges[i]:.3g}, {edges[i + 1]:.3g}] -> {k}"
                         for i, k in enumerate(self.band_kernels))


# ------------------------------------------------------------------ #
# Tuning helpers (train/validation only)                              #
# ------------------------------------------------------------------ #

def _labels(timings: pd.DataFrame, feats: pd.DataFrame, tol: float) -> pd.DataFrame:
    oracle, _ = oracle_table(timings, tol)
    lab = oracle[["matrix_id", "oracle_kernel"]].rename(columns={"oracle_kernel": "label"})
    return feats.merge(lab, on="matrix_id", how="inner", validate="one_to_one")


def mean_loss(selector, feats: pd.DataFrame, timings: pd.DataFrame, tol: float) -> Dict[str, float]:
    """Mean oracle-relative loss and accuracy of ``selector`` on matrices that have an oracle."""
    oracle, _ = oracle_table(timings, tol)
    feats = feats[feats["matrix_id"].isin(set(oracle["matrix_id"]))]
    if feats.empty:
        return {"n": 0, "mean_loss": float("nan"), "accuracy": float("nan"), "n_incorrect": 0}
    picks = pd.DataFrame({"matrix_id": feats["matrix_id"].to_numpy(),
                          "selected_kernel": np.asarray(selector.predict(feats), dtype=object)})
    sc = score_selections(picks, timings, oracle, tol)
    loss = sc["oracle_relative_loss"].to_numpy(float)
    return {"n": int(len(sc)), "mean_loss": float(np.nanmean(loss)) if np.isfinite(loss).any() else float("nan"),
            "accuracy": float(sc["selection_correct"].mean()),
            "n_incorrect": int((~sc["selected_correct_flag"]).sum())}


def tune_on_validation(
    factory: Callable[..., Any],
    grid: Mapping[str, Sequence[Any]],
    train_feats: pd.DataFrame,
    train_timings: pd.DataFrame,
    val_feats: pd.DataFrame,
    val_timings: pd.DataFrame,
    tol: float = DEFAULT_TIE_TOLERANCE,
):
    """Fit one model per grid point on TRAIN, rank by VALIDATION mean loss (ties: grid order).

    Returns (best_fitted_model, tuning_table).  ``factory(**params)`` must return an unfitted
    model with ``fit(X, y)`` and ``predict(X)``.
    """
    keys = sorted(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*(grid[k] for k in keys))] or [{}]
    train = _labels(train_timings, train_feats, tol)
    rows, fitted = [], []
    for order, params in enumerate(combos):
        m = factory(**params)
        m.fit(train, train["label"])
        s = mean_loss(m, val_feats, val_timings, tol)
        rows.append({"grid_order": order, **params, **{f"val_{k}": v for k, v in s.items()}})
        fitted.append(m)
    tab = pd.DataFrame(rows)
    rank = tab.assign(_l=tab["val_mean_loss"].fillna(np.inf)).sort_values(["val_n_incorrect", "_l", "grid_order"])
    best = int(rank.index[0])
    tab["chosen"] = tab.index == best
    return fitted[best], tab


def group_folds(groups: Sequence[str], k: int, seed: int = 0) -> List[List[str]]:
    """Deterministic grouped K folds: whole groups per fold, balanced by matrix count."""
    g = pd.Series(list(groups))
    counts = g.value_counts()
    names = sorted(counts.index)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    names.sort(key=lambda n: -counts[n])
    k = min(k, len(names))
    folds: List[List[str]] = [[] for _ in range(k)]
    sizes = [0] * k
    for n in names:
        i = int(np.argmin(sizes))
        folds[i].append(n)
        sizes[i] += counts[n]
    return folds


def grouped_cv_loss(
    factory: Callable[[], Any],
    feature_columns: Sequence[str],
    feats: pd.DataFrame,
    timings: pd.DataFrame,
    k: int = 5,
    seed: int = 0,
    tol: float = DEFAULT_TIE_TOLERANCE,
) -> pd.DataFrame:
    """Leave-groups-out CV on TRAIN+VALIDATION matrices (never test).

    Returns the per-matrix out-of-fold loss table (matrix_id, matrix_group, oracle_relative_loss,
    selection_correct).  Labels for fitting come only from the fold's training matrices.
    """
    folds = group_folds(feats["matrix_group"], k, seed)
    out = []
    for held in folds:
        hold_mask = feats["matrix_group"].isin(held)
        tr_f, te_f = feats[~hold_mask], feats[hold_mask]
        tr_t = timings[timings["matrix_id"].isin(set(tr_f["matrix_id"]))]
        te_t = timings[timings["matrix_id"].isin(set(te_f["matrix_id"]))]
        train = _labels(tr_t, tr_f, tol)
        if train.empty or te_f.empty:
            continue
        m = factory()
        m.feature_columns = list(feature_columns)
        m.fit(train, train["label"])
        oracle, _ = oracle_table(te_t, tol)
        te_f = te_f[te_f["matrix_id"].isin(set(oracle["matrix_id"]))]
        picks = pd.DataFrame({"matrix_id": te_f["matrix_id"].to_numpy(),
                              "selected_kernel": np.asarray(m.predict(te_f), dtype=object)})
        sc = score_selections(picks, te_t, oracle, tol)
        sc = sc.merge(te_f[["matrix_id", "matrix_group"]], on="matrix_id")
        out.append(sc[["matrix_id", "matrix_group", "oracle_relative_loss", "selection_correct"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["matrix_id", "matrix_group", "oracle_relative_loss", "selection_correct"])
