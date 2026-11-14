"""
pipeline.py
===========
Orchestration entry point for the Member 2 data pipeline.

This module ties together all sub-modules:
  loader      → load matrix files from a directory
  validator   → validate CSR structure (called inside loader)
  features    → extract structural features
  splits      → generate train/val/test manifest
  catalog     → write the three shared metadata CSV files

Usage (programmatic)
--------------------
    from data_pipeline.pipeline import run_pipeline
    run_pipeline(config_path="configs/dataset_config.yaml")

Usage (CLI)
-----------
    python scripts/run_pipeline.py [--config configs/dataset_config.yaml]
    python scripts/run_pipeline.py --pilot   # use synthetic pilot data

Configuration
-------------
All paths and parameters are read from ``configs/dataset_config.yaml``.
See that file for documentation of each option.

Exclusion log
-------------
Any matrix that fails loading or validation is recorded in::

    data/metadata/exclusion_log.csv

with columns: ``filename,reason``.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from data_pipeline.loader import load_directory
from data_pipeline.features import extract_features
from data_pipeline.splits import make_splits
from data_pipeline.catalog import (
    write_catalog,
    write_features,
    write_splits,
)

logger = logging.getLogger(__name__)

# Default path relative to the project root
_DEFAULT_CONFIG = Path("configs/dataset_config.yaml")


def run_pipeline(
    config_path: Path | str = _DEFAULT_CONFIG,
    raw_dir_override: Optional[Path | str] = None,
    groups_override: Optional[Dict[str, str]] = None,
    pilot: bool = False,
) -> Dict:
    """Run the full Member 2 data pipeline.

    Parameters
    ----------
    config_path : Path or str
        Path to the dataset configuration YAML file.
    raw_dir_override : Path or str, optional
        Override the raw_dir from config.
    groups_override : dict, optional
        Override / supplement matrix_id → matrix_group mapping.
        Useful when generating pilot data programmatically.
    pilot : bool
        If True, generate synthetic pilot matrices before running the
        pipeline.  The pilot matrices will be written to ``raw_dir`` and
        the pipeline will process them.

    Returns
    -------
    dict with keys:
        "matrices"        : dict[matrix_id, csr_matrix]
        "feature_records" : list[dict]
        "split_manifest"  : dict[matrix_id, str]
        "exclusions"      : dict[filename, reason]
    """
    # ------------------------------------------------------------------ #
    # Load configuration                                                   #
    # ------------------------------------------------------------------ #
    config = _load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    splits_cfg = config.get("splits", {})

    raw_dir = Path(raw_dir_override or dataset_cfg.get("raw_dir", "data/raw"))
    metadata_dir = Path(dataset_cfg.get("metadata_dir", "data/metadata"))

    logger.info("Pipeline start — raw_dir=%s  metadata_dir=%s", raw_dir, metadata_dir)

    # ------------------------------------------------------------------ #
    # Optional: generate pilot data                                        #
    # ------------------------------------------------------------------ #
    if pilot:
        from data_pipeline.pilot_data import generate_pilot_matrices
        pilot_groups = generate_pilot_matrices(raw_dir, overwrite=False)
        if groups_override is None:
            groups_override = pilot_groups
        else:
            groups_override = {**pilot_groups, **groups_override}
        logger.info("Pilot mode: synthetic matrices generated in %s", raw_dir)

    # ------------------------------------------------------------------ #
    # Load and validate matrices                                           #
    # ------------------------------------------------------------------ #
    matrices, exclusions = load_directory(raw_dir)

    if not matrices:
        logger.warning(
            "No matrices were loaded from '%s'. "
            "Use --pilot to generate synthetic data, or add matrix files "
            "(.mtx or .npz) to that directory.",
            raw_dir,
        )

    # ------------------------------------------------------------------ #
    # Optional curation: drop exact duplicate sparsity patterns             #
    # ------------------------------------------------------------------ #
    curation_cfg = config.get("curation") or {}
    redundancy: List[Dict[str, str]] = []
    if curation_cfg.get("drop_identical_patterns", False):
        matrices, redundancy = drop_identical_patterns(matrices)
        _write_redundancy_log(redundancy, metadata_dir / "redundancy_log.csv")

    # ------------------------------------------------------------------ #
    # Build group mapping                                                  #
    # ------------------------------------------------------------------ #
    groups: Dict[str, str] = {}
    manifest_path = dataset_cfg.get("manifest")
    if manifest_path:
        groups.update(_groups_from_manifest(Path(manifest_path)))
    # Explicit config mapping and programmatic overrides take precedence over the manifest.
    groups.update({str(k): str(v) for k, v in (config.get("matrix_groups") or {}).items()})
    if groups_override:
        groups.update(groups_override)
    # For any matrix without a group, label as "unknown"
    for mid in matrices:
        if mid not in groups:
            groups[mid] = "unknown"

    # ------------------------------------------------------------------ #
    # Extract features                                                     #
    # ------------------------------------------------------------------ #
    feature_records = []
    for mid, csr in sorted(matrices.items()):
        rec = extract_features(csr, mid)
        feature_records.append(rec)

    # Optional extended features (separate file; the 7-feature contract is unchanged)
    if (config.get("feature_extraction") or {}).get("extended", False):
        import pandas as pd
        from data_pipeline.features_extended import EXTENDED_COLUMNS, extract_extended_features

        ext = [extract_extended_features(csr, mid) for mid, csr in sorted(matrices.items())]
        out = metadata_dir / "features_extended.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ext, columns=["matrix_id"] + EXTENDED_COLUMNS).to_csv(out, index=False)
        logger.info("Extended features written: %s", out)

    # ------------------------------------------------------------------ #
    # Generate split manifest                                              #
    # ------------------------------------------------------------------ #
    matrix_ids = list(matrices.keys())
    split_manifest: Dict[str, str] = {}

    if matrix_ids:
        split_manifest = make_splits(
            matrix_ids=matrix_ids,
            groups=groups,
            train_ratio=float(splits_cfg.get("train_ratio", 0.70)),
            val_ratio=float(splits_cfg.get("val_ratio", 0.15)),
            test_ratio=float(splits_cfg.get("test_ratio", 0.15)),
            seed=int(splits_cfg.get("random_seed", 42)),
            strategy=str(splits_cfg.get("stratify_by", "matrix_group"))
                        .replace("matrix_group", "group"),
        )

    # ------------------------------------------------------------------ #
    # Write metadata CSVs                                                  #
    # ------------------------------------------------------------------ #
    write_catalog(matrices, groups, metadata_dir / "matrix_catalog.csv")
    write_features(feature_records, metadata_dir / "features.csv")
    write_splits(split_manifest, metadata_dir / "splits.csv")

    # ------------------------------------------------------------------ #
    # Write exclusion log                                                  #
    # ------------------------------------------------------------------ #
    exclusion_log_path = metadata_dir / "exclusion_log.csv"
    _write_exclusion_log(exclusions, exclusion_log_path)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    _print_summary(matrices, feature_records, split_manifest, exclusions)

    return {
        "matrices": matrices,
        "feature_records": feature_records,
        "split_manifest": split_manifest,
        "exclusions": exclusions,
    }


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _load_config(config_path: Path | str) -> dict:
    config_path = Path(config_path)
    if not config_path.is_file():
        logger.warning(
            "Config file not found at '%s'. Using built-in defaults.", config_path
        )
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded config from %s", config_path)
    return cfg


def drop_identical_patterns(matrices: Dict[str, "object"]):
    """Keep one matrix per exact sparsity pattern (hash of indptr + indices).

    Matrices that differ only in their values have identical structure, hence identical
    structural features, and (for CSR SpMV) the same performance behaviour; keeping all of
    them would inflate the sample size without adding structural information.  The kept
    representative is the smallest matrix_id; every drop is logged with the reason.
    Returns (kept_matrices, log_rows).
    """
    import hashlib

    seen: Dict[str, str] = {}
    kept: Dict[str, "object"] = {}
    log: List[Dict[str, str]] = []
    for mid in sorted(matrices):
        m = matrices[mid]
        h = hashlib.sha256(m.shape.__repr__().encode() + m.indptr.tobytes() + m.indices.tobytes()).hexdigest()
        if h in seen:
            log.append({"matrix_id": mid, "kept_representative": seen[h],
                        "reason": "identical_sparsity_pattern", "pattern_sha256": h})
        else:
            seen[h] = mid
            kept[mid] = m
    return kept, log


def _write_redundancy_log(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["matrix_id", "kept_representative", "reason", "pattern_sha256"])
        w.writeheader()
        w.writerows(rows)
    logger.info("%d duplicate-pattern matrix(es) dropped; see %s", len(rows), path)


def _groups_from_manifest(path: Path) -> Dict[str, str]:
    """matrix_id -> matrix_group from a dataset provenance manifest (see data_pipeline/acquire.py)."""
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    man = pd.read_csv(path)
    missing = {"matrix_id", "matrix_group"} - set(man.columns)
    if missing:
        raise ValueError(f"manifest {path} lacks column(s): {sorted(missing)}")
    return dict(zip(man["matrix_id"].astype(str), man["matrix_group"].astype(str)))


def _write_exclusion_log(exclusions: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "reason"])
        writer.writeheader()
        for filename, reason in sorted(exclusions.items()):
            writer.writerow({"filename": filename, "reason": reason})
    if exclusions:
        logger.warning(
            "%d file(s) excluded from pipeline. See %s.", len(exclusions), path
        )
    else:
        logger.info("No exclusions. Exclusion log written to %s.", path)


def _print_summary(matrices, feature_records, split_manifest, exclusions) -> None:
    total = len(matrices)
    n_train = sum(1 for v in split_manifest.values() if v == "train")
    n_val   = sum(1 for v in split_manifest.values() if v == "val")
    n_test  = sum(1 for v in split_manifest.values() if v == "test")

    print("\n" + "=" * 60)
    print("  Member 2 Pipeline — Run Summary")
    print("=" * 60)
    print(f"  Matrices loaded       : {total}")
    print(f"  Features extracted    : {len(feature_records)}")
    print(f"  Split manifest        : train={n_train}  val={n_val}  test={n_test}")
    print(f"  Excluded (errors)     : {len(exclusions)}")
    print("=" * 60 + "\n")
