#!/usr/bin/env python
"""
evaluate_generalization.py
==========================
Evaluate a FROZEN trained selector on the held-out TEST split and write the
per-matrix table, summary tables and figures into the run folder.

Safeguards
----------
* The input files must hash to exactly what the model was trained on.
* A run folder is evaluated once; use --overwrite to replace the report.
* Synthetic inputs need --allow-synthetic and are stamped everywhere.

Usage
-----
    python scripts/evaluate_generalization.py --run-dir results/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import evaluate  # noqa: E402
from evaluation.plots import make_all_plots  # noqa: E402
from evaluation.tables import (  # noqa: E402
    confusion_table, overall_table, write_manifest, write_report_tables,
)
from selector.config import (  # noqa: E402
    ConfigError, DEFAULT_CONFIG_PATH, apply_overrides, input_paths, is_synthetic,
    latest_run_dir, load_config, resolve_path,
)
from selector.data import DataContractError, build_dataset  # noqa: E402
from selector.model import KernelSelector, library_versions  # noqa: E402
from selector.train import file_sha256  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the frozen selector on the test split.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--run-dir", help="run folder from train_selector.py (default: newest)")
    p.add_argument("--features")
    p.add_argument("--splits")
    p.add_argument("--catalog")
    p.add_argument("--timings")
    p.add_argument("--results-dir")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="replace an existing test report")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_overrides(load_config(args.config), {
            "features": args.features, "splits": args.splits, "catalog": args.catalog,
            "timings": args.timings, "results_dir": args.results_dir})
        paths = input_paths(cfg)
        synthetic = is_synthetic(paths)
        if synthetic and not args.allow_synthetic:
            print("ERROR: an input file name contains 'synthetic'; pass --allow-synthetic to "
                  "proceed (outputs will be stamped).", file=sys.stderr)
            return 2

        run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(resolve_path(cfg["results_dir"]))
        run_dir = run_dir if run_dir.is_absolute() else (Path.cwd() / run_dir)
        card = json.loads((run_dir / "model_card.json").read_text(encoding="utf-8"))
        report = run_dir / "per_matrix_test.csv"
        if report.exists() and not args.overwrite:
            print(f"ERROR: {report} already exists. The test split is evaluated once per run; "
                  "train a new run or pass --overwrite.", file=sys.stderr)
            return 3

        now = {name: file_sha256(p) for name, p in paths.items()}
        if now != card["input_sha256"]:
            changed = sorted(k for k in now if now[k] != card["input_sha256"].get(k))
            print(f"ERROR: input file(s) changed since training: {changed}. Re-train, or point "
                  "to the original files.", file=sys.stderr)
            return 4

        dataset = build_dataset(paths["features"], paths["splits"], paths["catalog"], paths["timings"],
                                candidate_kernels=card["candidate_kernels"])
        selector = KernelSelector.load(run_dir / "model.joblib")
        tol = card["config"]["tie_tolerance"]
        ev = evaluate(selector, dataset, "test", tol)

        ecfg = cfg["evaluation"]
        n_boot, seed = ecfg.get("bootstrap_samples", 2000), ecfg.get("bootstrap_seed", 0)
        written = write_report_tables(run_dir, ev.per_matrix, ev.excluded, ev.best_fixed_kernel,
                                      ev.candidate_kernels, synthetic, n_boot, seed)
        confusion = confusion_table(ev.per_matrix, ev.candidate_kernels)
        figs = make_all_plots(run_dir / "figures", ev.per_matrix, confusion,
                              selector.feature_importances(), ev.best_fixed_kernel, synthetic, n_boot, seed)
        write_manifest(run_dir / "run_manifest.json", {
            "split_evaluated": "test",
            "n_matrices_evaluated": int(ev.per_matrix.loc[ev.per_matrix["method"] == "selector"].shape[0]),
            "n_matrices_excluded": int(len(ev.excluded)),
            "best_fixed_kernel_from_train": ev.best_fixed_kernel,
            "tie_tolerance": tol,
            "synthetic_data": synthetic,
            "input_sha256": now,
            "versions_at_evaluation": library_versions(),
            "model_trained_with_versions": card["versions"],
        })
    except (ConfigError, DataContractError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    overall = overall_table(ev.per_matrix, ev.best_fixed_kernel, n_boot, seed)
    show = ["method", "n_matrices", "accuracy", "mean_loss", "median_loss", "total_time_ratio"]
    if synthetic:
        print("*** SYNTHETIC DATA - NOT A RESULT ***")
    print(f"Evaluated {len(ev.per_matrix[ev.per_matrix['method'] == 'selector'])} test matrices "
          f"({len(ev.excluded)} excluded). Best fixed kernel (train): {ev.best_fixed_kernel}")
    print(overall[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Wrote {len(written)} tables and {len(figs)} figures to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
