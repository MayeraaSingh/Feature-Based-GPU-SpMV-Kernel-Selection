"""
model.py
========
``KernelSelector``: a thin, strict wrapper around a RandomForestClassifier.

What "strict" means here
------------------------
* The model remembers the *names* and *order* of its feature columns. At
  predict time inputs are picked out by name, so a DataFrame whose columns
  are shuffled gives the same answer, and a missing column is an error.
* NaN / infinite inputs are rejected, not guessed around.
* The class list is whatever kernels were the oracle on the training split.
  A kernel that never won in training can never be predicted; this is
  expected and is surfaced later in the confusion matrix.
* Library versions are stored with the model so a Colab-vs-laptop mismatch
  is detectable.

Security note: ``save``/``load`` use joblib (pickle). Only load model files
you produced yourself.
"""

from __future__ import annotations

import platform
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from selector.data import FEATURE_COLUMNS

DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1,
    "class_weight": None,
}


def library_versions() -> Dict[str, str]:
    """Versions that can change numerical behaviour; stored with every model."""
    import scipy

    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


class KernelSelector:
    """Predict the fastest correct kernel from structural matrix features."""

    def __init__(
        self,
        feature_columns: Sequence[str] = tuple(FEATURE_COLUMNS),
        seed: int = 0,
        **params: Any,
    ) -> None:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(
                f"unknown model parameter(s): {sorted(unknown)}; allowed: {sorted(DEFAULT_PARAMS)}"
            )
        if len(set(feature_columns)) != len(feature_columns) or not feature_columns:
            raise ValueError("feature_columns must be a non-empty list without duplicates")
        self.feature_columns: List[str] = list(feature_columns)
        self.seed = int(seed)
        self.params: Dict[str, Any] = {**DEFAULT_PARAMS, **params}
        self.versions: Optional[Dict[str, str]] = None
        self._model: Optional[RandomForestClassifier] = None

    # -------------------------------------------------------------- #
    # Input handling                                                  #
    # -------------------------------------------------------------- #

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Select feature columns by name (fixed order) and validate values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with named feature columns")
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing feature column(s): {missing}")
        values = X[self.feature_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("X contains NaN or infinite values")
        return values

    def _require_fitted(self) -> RandomForestClassifier:
        if self._model is None:
            raise RuntimeError("KernelSelector is not fitted; call fit() first")
        return self._model

    # -------------------------------------------------------------- #
    # Fit / predict                                                   #
    # -------------------------------------------------------------- #

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "KernelSelector":
        values = self._matrix(X)
        labels = np.asarray(list(y), dtype=object)
        if len(labels) != len(values):
            raise ValueError(f"X has {len(values)} rows but y has {len(labels)} labels")
        if len(labels) == 0:
            raise ValueError("cannot fit on zero training matrices")
        if any((not isinstance(lab, str)) or lab == "" for lab in labels):
            raise ValueError("labels must be non-empty kernel-name strings")
        model = RandomForestClassifier(random_state=self.seed, n_jobs=1, **self.params)
        model.fit(values, labels.astype(str))
        self._model = model
        self.versions = library_versions()
        return self

    @property
    def classes_(self) -> List[str]:
        """Kernel names the model can output (those that were an oracle in training)."""
        return [str(c) for c in self._require_fitted().classes_]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        model = self._require_fitted()
        values = self._matrix(X)
        if len(values) == 0:
            return np.array([], dtype=str)
        return model.predict(values)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities; one column per kernel in ``classes_``."""
        model = self._require_fitted()
        values = self._matrix(X)
        proba = model.predict_proba(values) if len(values) else np.empty((0, len(model.classes_)))
        return pd.DataFrame(proba, columns=self.classes_, index=X.index)

    def feature_importances(self) -> pd.Series:
        """Diagnostic only: impurity-based importances, by feature name."""
        model = self._require_fitted()
        return pd.Series(model.feature_importances_, index=self.feature_columns, name="importance")

    # -------------------------------------------------------------- #
    # Persistence                                                     #
    # -------------------------------------------------------------- #

    def save(self, path: Path | str) -> Path:
        model = self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": model,
                "feature_columns": self.feature_columns,
                "seed": self.seed,
                "params": self.params,
                "versions": self.versions,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "KernelSelector":
        blob = joblib.load(Path(path))
        sel = cls(feature_columns=blob["feature_columns"], seed=blob["seed"], **blob["params"])
        sel._model = blob["estimator"]
        sel.versions = blob["versions"]
        now = library_versions()
        diff = {k: (v, now.get(k)) for k, v in (sel.versions or {}).items()
                if k in ("scikit-learn", "numpy") and now.get(k) != v}
        if diff:
            warnings.warn(
                f"model was saved with different library versions (saved, current): {diff}. "
                "Predictions may not be bit-identical.",
                RuntimeWarning,
                stacklevel=2,
            )
        return sel
