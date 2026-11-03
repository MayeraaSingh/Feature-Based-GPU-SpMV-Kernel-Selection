"""
loader.py
=========
Matrix loading for the Member 2 data pipeline.

Supported source formats
------------------------
  .mtx            Matrix Market (via scipy.io.mmread)
  .npz            SciPy sparse matrix (via scipy.sparse.load_npz) **or**
                  NumPy archive with keys 'data', 'indices', 'indptr',
                  'shape' matching a CSR layout.

Stable matrix_id assignment
----------------------------
IDs are derived from the source file's *stem* (filename without extension),
with special characters replaced by underscores and leading/trailing
underscores stripped.  The ID is NOT based on file ordering or directory
listing order; it is derived purely from the filename, so it is stable
across pipeline runs and machines.

If a collision is detected (two files produce the same id after normalisation),
a numeric suffix is appended and a warning is logged.  In practice, collisions
should be resolved by renaming source files before ingestion.

Exclusion log
-------------
Files that fail to load or validate are recorded in the exclusion log
dictionary returned by `load_directory`, with the reason for exclusion.
They are not silently skipped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import scipy.sparse as sp

from data_pipeline.validator import validate_csr, ValidationError

logger = logging.getLogger(__name__)

# Characters that are not alphanumeric, hyphen, or dot become underscores.
_ID_UNSAFE = re.compile(r"[^A-Za-z0-9.\-]+")


def _stem_to_id(stem: str) -> str:
    """Convert a filename stem to a normalized matrix_id."""
    normalized = _ID_UNSAFE.sub("_", stem).strip("_")
    if not normalized:
        normalized = "matrix"
    return normalized


def load_matrix(
    path: Path | str,
    matrix_id: Optional[str] = None,
) -> Tuple[sp.csr_matrix, str]:
    """Load a single sparse matrix file and return a validated CSR matrix.

    Parameters
    ----------
    path : Path or str
        Absolute or relative path to a .mtx or .npz file.
    matrix_id : str, optional
        Override the auto-derived matrix_id.  If None, the id is derived
        from the file stem.

    Returns
    -------
    (csr_matrix, matrix_id) : tuple
        The validated CSR matrix and its stable string identifier.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    ValidationError
        If the loaded matrix fails CSR structural validation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    derived_id = matrix_id if matrix_id is not None else _stem_to_id(path.stem)

    suffix = path.suffix.lower()

    if suffix == ".mtx":
        raw = _load_mtx(path, derived_id)
    elif suffix == ".npz":
        raw = _load_npz(path, derived_id)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}' for {path}. "
            "Supported formats: .mtx, .npz"
        )

    csr = validate_csr(raw, derived_id)
    return csr, derived_id


def _load_mtx(path: Path, matrix_id: str) -> sp.spmatrix:
    """Load a Matrix Market file."""
    logger.info("[%s] Loading Matrix Market file: %s", matrix_id, path)
    try:
        mat = scipy.io.mmread(str(path))
    except Exception as exc:
        raise ValidationError(
            f"[{matrix_id}] Failed to read Matrix Market file '{path}': {exc}"
        ) from exc
    if not sp.issparse(mat):
        # mmread may return a dense array for very small matrices
        mat = sp.csr_matrix(mat)
    return mat


def _load_npz(path: Path, matrix_id: str) -> sp.spmatrix:
    """Load a .npz file as a SciPy sparse matrix or a raw CSR archive."""
    logger.info("[%s] Loading .npz file: %s", matrix_id, path)
    # First try SciPy sparse format (saved with scipy.sparse.save_npz)
    try:
        mat = sp.load_npz(str(path))
        return mat
    except Exception:
        pass

    # Fallback: raw NumPy archive with CSR keys
    try:
        archive = np.load(str(path), allow_pickle=False)
        required = {"data", "indices", "indptr", "shape"}
        if not required.issubset(archive.files):
            raise ValidationError(
                f"[{matrix_id}] .npz archive is missing required CSR keys "
                f"(expected: {required}, found: {set(archive.files)})."
            )
        shape = tuple(int(x) for x in archive["shape"])
        mat = sp.csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]),
            shape=shape,
        )
        return mat
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(
            f"[{matrix_id}] Failed to load .npz file '{path}' as sparse matrix: {exc}"
        ) from exc


def load_directory(
    directory: Path | str,
    extensions: Tuple[str, ...] = (".mtx", ".npz"),
) -> Tuple[Dict[str, sp.csr_matrix], Dict[str, str]]:
    """Load all supported matrix files from a directory.

    Files are sorted by name before processing to ensure deterministic
    ordering, but the matrix_id is always derived from the filename, not
    from position in the sorted list.

    Parameters
    ----------
    directory : Path or str
        Directory to scan (non-recursive).
    extensions : tuple of str
        File extensions to include.

    Returns
    -------
    (matrices, exclusions) : tuple
        matrices   : dict mapping matrix_id → validated CSR matrix
        exclusions : dict mapping filename → reason string for any file
                     that could not be loaded/validated.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Collect candidate files, sorted for determinism
    candidates: List[Path] = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    matrices: Dict[str, sp.csr_matrix] = {}
    exclusions: Dict[str, str] = {}
    id_sources: Dict[str, str] = {}  # id → original filename (for collision detection)

    for path in candidates:
        derived_id = _stem_to_id(path.stem)

        # Collision resolution
        if derived_id in id_sources:
            original_path = id_sources[derived_id]
            suffix_num = 2
            candidate_id = f"{derived_id}_{suffix_num}"
            while candidate_id in id_sources:
                suffix_num += 1
                candidate_id = f"{derived_id}_{suffix_num}"
            logger.warning(
                "matrix_id collision: '%s' already assigned to '%s'. "
                "File '%s' will be assigned id '%s'. "
                "Consider renaming source files to avoid collisions.",
                derived_id,
                original_path,
                path.name,
                candidate_id,
            )
            derived_id = candidate_id

        try:
            csr, final_id = load_matrix(path, matrix_id=derived_id)
            matrices[final_id] = csr
            id_sources[final_id] = path.name
        except (ValidationError, ValueError, FileNotFoundError, Exception) as exc:
            reason = str(exc)
            logger.error("[%s] Excluded: %s", path.name, reason)
            exclusions[path.name] = reason

    logger.info(
        "load_directory: %d loaded, %d excluded from '%s'.",
        len(matrices),
        len(exclusions),
        directory,
    )
    return matrices, exclusions
