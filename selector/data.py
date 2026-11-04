"""
data.py
=======
Load, validate and join the four shared CSV contracts for the selector.

Inputs (see docs/data_dictionary.md)
------------------------------------
  features.csv         Member 2   structural features per matrix
  splits.csv           Member 2   train / val / test label per matrix
  matrix_catalog.csv   Member 2   rows, cols, nnz, matrix_group
  kernel_timings.csv   Member 1   one row per (matrix_id, kernel_name)

Design rules
------------
* Pure functions, no global state.
* Every check raises ``DataContractError`` naming the offending IDs.
  Nothing is silently dropped, repaired or re-labelled.
* Joins are always by ``matrix_id`` (``pd.merge``), never by row position,
  and results are sorted by ``matrix_id`` so input row order is irrelevant.
* Kernel names are discovered from the data, never hard-coded.
* The ``Dataset`` object only hands out timings through ``timings_for(split)``
  so training code has to ask for a split explicitly (leakage guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from data_pipeline.catalog import (
    CATALOG_HEADERS,
    FEATURES_HEADERS,
    SPLITS_HEADERS,
)

# ------------------------------------------------------------------ #
# Contract constants                                                   #
# ------------------------------------------------------------------ #

TIMINGS_HEADERS = [
    "matrix_id",
    "kernel_name",
    "kernel_time_ms",
    "preprocessing_time_ms",
    "correctness_pass",
]

# Canonical split labels. The shared contract uses "val", not "validation".
VALID_SPLITS = ("train", "val", "test")

# The numeric model inputs (everything in features.csv except matrix_id).
FEATURE_COLUMNS = [h for h in FEATURES_HEADERS if h != "matrix_id"]

_FEATURE_INT_COLUMNS = ["rows", "cols", "nnz", "max_row_nnz", "min_row_nnz"]
_FEATURE_FLOAT_COLUMNS = ["mean_row_nnz", "variance_row_nnz"]

_MAX_LISTED = 20  # cap on IDs printed in one error message


class DataContractError(ValueError):
    """Raised when an input file violates the shared CSV contract."""


# ------------------------------------------------------------------ #
# Small helpers                                                        #
# ------------------------------------------------------------------ #

def _fmt(items: Iterable) -> str:
    """Format a list of offending values for an error message."""
    items = list(items)
    shown = ", ".join(repr(str(i)) for i in items[:_MAX_LISTED])
    extra = len(items) - _MAX_LISTED
    return shown + (f" (+{extra} more)" if extra > 0 else "")


def _read_csv(path: Path | str, expected_headers: List[str], name: str) -> pd.DataFrame:
    """Read a CSV as strings and check the header matches the contract exactly.

    Everything is read as ``str`` with NA-detection off, so an empty cell stays
    ``""`` and we can report it ourselves instead of pandas guessing.
    """
    path = Path(path)
    if not path.is_file():
        raise DataContractError(f"{name}: file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise DataContractError(f"{name}: file is empty (no header): {path}")
    except pd.errors.ParserError as exc:
        raise DataContractError(f"{name}: cannot parse CSV: {exc}")
    if list(df.columns) != expected_headers:
        raise DataContractError(
            f"{name}: header mismatch.\n  expected: {expected_headers}\n"
            f"  found:    {list(df.columns)}"
        )
    return df


def _check_no_missing(df: pd.DataFrame, name: str) -> None:
    """Reject empty / whitespace-only cells in any column."""
    for col in df.columns:
        blank = df[col].str.strip() == ""
        if blank.any():
            ids = df.loc[blank, "matrix_id"] if "matrix_id" in df.columns else df.index[blank]
            raise DataContractError(
                f"{name}: missing value in column '{col}' for matrix_id(s): {_fmt(ids)}"
            )


def _check_ids(df: pd.DataFrame, name: str) -> None:
    """matrix_id must have no surrounding whitespace (it would break joins)."""
    padded = df["matrix_id"] != df["matrix_id"].str.strip()
    if padded.any():
        raise DataContractError(
            f"{name}: matrix_id with leading/trailing whitespace: "
            f"{_fmt(df.loc[padded, 'matrix_id'])}"
        )


def _check_unique(df: pd.DataFrame, keys: List[str], name: str) -> None:
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        rows = df.loc[dup, keys].drop_duplicates()
        labels = [" / ".join(r) for r in rows.itertuples(index=False)]
        raise DataContractError(f"{name}: duplicate {'+'.join(keys)}: {_fmt(labels)}")


def _to_number(df: pd.DataFrame, col: str, name: str, integer: bool = False) -> pd.Series:
    """Convert a string column to float64 (or int64), requiring finite values."""
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values.isna() | ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    if bad.any():
        raise DataContractError(
            f"{name}: column '{col}' has non-numeric or non-finite values for "
            f"matrix_id(s): {_fmt(df.loc[bad, 'matrix_id'])}"
        )
    if integer:
        frac = values != np.floor(values)
        if frac.any():
            raise DataContractError(
                f"{name}: column '{col}' must be integer for matrix_id(s): "
                f"{_fmt(df.loc[frac, 'matrix_id'])}"
            )
        return values.astype("int64")
    return values.astype("float64")


def _parse_bool(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    """Strict boolean parse: only true/false (case-insensitive) are accepted."""
    lowered = df[col].str.strip().str.lower()
    ok = lowered.isin(["true", "false"])
    if not ok.all():
        bad = df.loc[~ok]
        labels = [f"{m}/{k}={v!r}" for m, k, v in zip(bad["matrix_id"], bad["kernel_name"], bad[col])]
        raise DataContractError(
            f"{name}: column '{col}' must be True/False; offending rows: {_fmt(labels)}"
        )
    return lowered == "true"


# ------------------------------------------------------------------ #
# Per-file loaders (each validates its own file)                       #
# ------------------------------------------------------------------ #

def load_features(path: Path | str) -> pd.DataFrame:
    """Load and validate ``features.csv``. Returns a DataFrame sorted by matrix_id."""
    name = "features.csv"
    raw = _read_csv(path, FEATURES_HEADERS, name)
    _check_no_missing(raw, name)
    _check_ids(raw, name)
    _check_unique(raw, ["matrix_id"], name)

    out = pd.DataFrame({"matrix_id": raw["matrix_id"]})
    for col in _FEATURE_INT_COLUMNS:
        out[col] = _to_number(raw, col, name, integer=True)
    for col in _FEATURE_FLOAT_COLUMNS:
        out[col] = _to_number(raw, col, name)
    out = out[FEATURES_HEADERS]

    for col in FEATURE_COLUMNS:
        neg = out[col] < 0
        if neg.any():
            raise DataContractError(
                f"{name}: column '{col}' must be >= 0 for matrix_id(s): "
                f"{_fmt(out.loc[neg, 'matrix_id'])}"
            )
    bad = out["min_row_nnz"] > out["max_row_nnz"]
    if bad.any():
        raise DataContractError(
            f"{name}: min_row_nnz > max_row_nnz for matrix_id(s): "
            f"{_fmt(out.loc[bad, 'matrix_id'])}"
        )
    return out.sort_values("matrix_id").reset_index(drop=True)


def load_splits(path: Path | str) -> pd.DataFrame:
    """Load and validate ``splits.csv``. Labels must be exactly train/val/test."""
    name = "splits.csv"
    raw = _read_csv(path, SPLITS_HEADERS, name)
    _check_no_missing(raw, name)
    _check_ids(raw, name)
    _check_unique(raw, ["matrix_id"], name)
    bad = ~raw["split"].isin(VALID_SPLITS)
    if bad.any():
        pairs = [f"{m}={s!r}" for m, s in zip(raw.loc[bad, "matrix_id"], raw.loc[bad, "split"])]
        raise DataContractError(
            f"{name}: invalid split label (allowed: {list(VALID_SPLITS)}; the shared "
            f"contract uses 'val', not 'validation'): {_fmt(pairs)}"
        )
    return raw.sort_values("matrix_id").reset_index(drop=True)


def load_catalog(path: Path | str) -> pd.DataFrame:
    """Load and validate ``matrix_catalog.csv``."""
    name = "matrix_catalog.csv"
    raw = _read_csv(path, CATALOG_HEADERS, name)
    _check_no_missing(raw, name)
    _check_ids(raw, name)
    _check_unique(raw, ["matrix_id"], name)
    out = pd.DataFrame({"matrix_id": raw["matrix_id"]})
    for col in ("rows", "cols", "nnz"):
        out[col] = _to_number(raw, col, name, integer=True)
        neg = out[col] < 0
        if neg.any():
            raise DataContractError(
                f"{name}: column '{col}' must be >= 0 for matrix_id(s): "
                f"{_fmt(out.loc[neg, 'matrix_id'])}"
            )
    out["matrix_group"] = raw["matrix_group"]
    return out.sort_values("matrix_id").reset_index(drop=True)


def load_timings(path: Path | str) -> pd.DataFrame:
    """Load and validate ``kernel_timings.csv``.

    A header-only file is valid *as a file* (it is what the repo ships before
    Member 1 delivers) and returns an empty DataFrame. ``build_dataset``
    refuses to proceed on it.

    Duplicate ``(matrix_id, kernel_name)`` rows are an error: whether to
    aggregate repeats (e.g. take the median) is a protocol decision for
    Member 1, not something we do silently.
    """
    name = "kernel_timings.csv"
    raw = _read_csv(path, TIMINGS_HEADERS, name)
    _check_no_missing(raw, name)
    _check_ids(raw, name)
    padded = raw["kernel_name"] != raw["kernel_name"].str.strip()
    if padded.any():
        raise DataContractError(
            f"{name}: kernel_name with leading/trailing whitespace: "
            f"{_fmt(raw.loc[padded, 'kernel_name'])}"
        )
    _check_unique(raw, ["matrix_id", "kernel_name"], name)

    out = pd.DataFrame({"matrix_id": raw["matrix_id"], "kernel_name": raw["kernel_name"]})
    out["kernel_time_ms"] = _to_number(raw, "kernel_time_ms", name)
    out["preprocessing_time_ms"] = _to_number(raw, "preprocessing_time_ms", name)
    out["correctness_pass"] = _parse_bool(raw, "correctness_pass", name)

    bad = out["kernel_time_ms"] <= 0
    if bad.any():
        raise DataContractError(
            f"{name}: kernel_time_ms must be > 0 for: "
            f"{_fmt(out.loc[bad, 'matrix_id'] + '/' + out.loc[bad, 'kernel_name'])}"
        )
    bad = out["preprocessing_time_ms"] < 0
    if bad.any():
        raise DataContractError(
            f"{name}: preprocessing_time_ms must be >= 0 for: "
            f"{_fmt(out.loc[bad, 'matrix_id'] + '/' + out.loc[bad, 'kernel_name'])}"
        )
    return out.sort_values(["matrix_id", "kernel_name"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
# Cross-file validation                                                #
# ------------------------------------------------------------------ #

def validate_cross_file(
    features: pd.DataFrame,
    splits: pd.DataFrame,
    catalog: pd.DataFrame,
    timings: pd.DataFrame,
) -> None:
    """Check the four (already individually valid) tables agree with each other."""
    f_ids, s_ids, c_ids = (set(t["matrix_id"]) for t in (features, splits, catalog))
    for (a_name, a), (b_name, b) in (
        (("features", f_ids), ("splits", s_ids)),
        (("features", f_ids), ("catalog", c_ids)),
    ):
        if a != b:
            raise DataContractError(
                f"matrix_id sets differ between {a_name} and {b_name}.\n"
                f"  only in {a_name}: {_fmt(sorted(a - b))}\n"
                f"  only in {b_name}: {_fmt(sorted(b - a))}"
            )

    unknown = sorted(set(timings["matrix_id"]) - f_ids)
    if unknown:
        raise DataContractError(
            f"kernel_timings.csv has matrix_id(s) absent from features/splits/catalog: "
            f"{_fmt(unknown)}"
        )

    merged = features.merge(catalog, on="matrix_id", suffixes=("_feat", "_cat"), validate="one_to_one")
    for col in ("rows", "cols", "nnz"):
        diff = merged[col + "_feat"] != merged[col + "_cat"]
        if diff.any():
            raise DataContractError(
                f"features.csv and matrix_catalog.csv disagree on '{col}' for matrix_id(s): "
                f"{_fmt(merged.loc[diff, 'matrix_id'])}"
            )


def find_incomplete_matrices(
    matrix_ids: Sequence[str],
    timings: pd.DataFrame,
    candidate_kernels: Sequence[str],
) -> Dict[str, List[str]]:
    """Return ``{matrix_id: [missing kernel names]}`` for matrices lacking a candidate.

    A matrix with a timing row whose ``correctness_pass`` is False is NOT
    incomplete; the row exists. Correctness is handled when the oracle is built.
    """
    have = timings.groupby("matrix_id")["kernel_name"].agg(set).to_dict()
    cand = set(candidate_kernels)
    missing: Dict[str, List[str]] = {}
    for mid in matrix_ids:
        gap = sorted(cand - have.get(mid, set()))
        if gap:
            missing[mid] = gap
    return missing


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #

EXCLUDED_HEADERS = ["matrix_id", "split", "reason", "detail"]


@dataclass(frozen=True)
class Dataset:
    """Validated, joined data. Timings are only reachable per split.

    Attributes
    ----------
    features : DataFrame
        One row per *kept* matrix: matrix_id, split, matrix_group and the
        7 feature columns. Sorted by matrix_id.
    candidate_kernels : list of str
        Kernel names, discovered from the timings file (or given explicitly).
    excluded : DataFrame
        Matrices removed, with a reason (columns ``EXCLUDED_HEADERS``).
    """

    features: pd.DataFrame
    candidate_kernels: List[str]
    excluded: pd.DataFrame
    _timings: pd.DataFrame

    def matrix_ids(self, split: str) -> List[str]:
        self._check_split(split)
        return self.features.loc[self.features["split"] == split, "matrix_id"].tolist()

    def features_for(self, split: str) -> pd.DataFrame:
        """Feature rows (inputs only; no timings) for one split."""
        self._check_split(split)
        return self.features[self.features["split"] == split].reset_index(drop=True)

    def timings_for(self, split: str) -> pd.DataFrame:
        """Timing rows for the matrices of one split."""
        self._check_split(split)
        ids = set(self.matrix_ids(split))
        return self._timings[self._timings["matrix_id"].isin(ids)].reset_index(drop=True)

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")


def build_dataset(
    features_path: Path | str,
    splits_path: Path | str,
    catalog_path: Path | str,
    timings_path: Path | str,
    candidate_kernels: Optional[Sequence[str]] = None,
) -> Dataset:
    """Load all four files, validate them, join by ``matrix_id`` and apply the
    candidate-coverage policy.

    Policy: a matrix lacking a timing row for any candidate kernel is
    *excluded* (never imputed) and listed in ``Dataset.excluded``.

    Parameters
    ----------
    candidate_kernels : sequence of str, optional
        If omitted, every ``kernel_name`` present in the timings file is a
        candidate. Pass an explicit list to restrict the candidate set.
    """
    features = load_features(features_path)
    splits = load_splits(splits_path)
    catalog = load_catalog(catalog_path)
    timings = load_timings(timings_path)

    if timings.empty:
        raise DataContractError(
            "kernel_timings.csv has a header but no rows. Real GPU timings from "
            "Member 1 are required (synthetic timings are for tests only)."
        )

    validate_cross_file(features, splits, catalog, timings)

    discovered = sorted(timings["kernel_name"].unique())
    if candidate_kernels is None:
        candidates = discovered
    else:
        candidates = sorted(set(candidate_kernels))
        unknown = sorted(set(candidates) - set(discovered))
        if unknown:
            raise DataContractError(
                f"candidate kernel(s) never appear in kernel_timings.csv: {_fmt(unknown)}"
            )
        timings = timings[timings["kernel_name"].isin(candidates)].reset_index(drop=True)

    joined = (
        features.merge(splits, on="matrix_id", how="inner", validate="one_to_one")
        .merge(catalog[["matrix_id", "matrix_group"]], on="matrix_id", how="inner", validate="one_to_one")
    )
    joined = joined[["matrix_id", "split", "matrix_group"] + FEATURE_COLUMNS]
    joined = joined.sort_values("matrix_id").reset_index(drop=True)

    incomplete = find_incomplete_matrices(joined["matrix_id"].tolist(), timings, candidates)
    split_of = dict(zip(joined["matrix_id"], joined["split"]))
    excluded = pd.DataFrame(
        [
            {
                "matrix_id": mid,
                "split": split_of[mid],
                "reason": "missing_candidate_timings",
                "detail": ";".join(gap),
            }
            for mid, gap in sorted(incomplete.items())
        ],
        columns=EXCLUDED_HEADERS,
    )

    kept = joined[~joined["matrix_id"].isin(incomplete)].reset_index(drop=True)
    kept_timings = timings[timings["matrix_id"].isin(set(kept["matrix_id"]))].reset_index(drop=True)
    return Dataset(
        features=kept,
        candidate_kernels=candidates,
        excluded=excluded,
        _timings=kept_timings,
    )
