#!/usr/bin/env python
"""
train_selector.py
=================
Train the kernel selector (train split), tune it (validation split), and save
the model + model card into ``results/<run_id>/``. Never reads the test split.

Usage
-----
    python scripts/train_selector.py --config configs/selector_config.yaml
    python scripts/train_selector.py --timings path/to/kernel_timings.csv --results-dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selector.config import (  # noqa: E402
    ConfigError, DEFAULT_CONFIG_PATH, apply_overrides, input_paths, is_synthetic,
    load_config, make_run_id, resolve_path, train_config,
)
from selector.data import DataContractError, build_dataset  # noqa: E402
from selector.train import train_selector, write_training_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the SpMV kernel selector.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config file")
    p.add_argument("--features", help="override paths.features")
    p.add_argument("--splits", help="override paths.splits")
    p.add_argument("--catalog", help="override paths.catalog")
    p.add_argument("--timings", help="override paths.timings")
    p.add_argument("--results-dir", help="override results_dir")
    p.add_argument("--run-id", help="run folder name (default: YYYYMMDD-HHMM-<git hash>)")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="permit inputs whose filename contains 'synthetic' (outputs are stamped)")
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
            print("ERROR: an input file name contains 'synthetic'. Synthetic timings are for tests "
                  "only; pass --allow-synthetic to proceed (outputs will be stamped).", file=sys.stderr)
            return 2

        dataset = build_dataset(paths["features"], paths["splits"], paths["catalog"], paths["timings"])
        result = train_selector(dataset, train_config(cfg), input_paths=paths)

        run_dir = resolve_path(cfg["results_dir"]) / (args.run_id or make_run_id())
        out = write_training_outputs(result, run_dir)
        dataset.excluded.to_csv(run_dir / "train_excluded_matrices.csv", index=False)
        if synthetic:
            (run_dir / "RESULTS_ARE_SYNTHETIC.txt").write_text(
                "Produced from SYNTHETIC timings. NOT a result.\n", encoding="utf-8")
    except (ConfigError, DataContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Run folder      : {run_dir}")
    print(f"Classes         : {result.selector.classes_}")
    print(f"Best parameters : {result.best_params}")
    print(f"Model card      : {out['model_card']}")
    if synthetic:
        print("WARNING: SYNTHETIC DATA - NOT A RESULT")
    extra = " --allow-synthetic" if synthetic else ""
    overrides = "".join(f" --{k} \"{getattr(args, k)}\"" for k in ("features", "splits", "catalog", "timings")
                        if getattr(args, k))
    print(f"Next: python scripts/evaluate_generalization.py --config \"{args.config}\"{overrides}"
          f"{extra} --run-dir \"{run_dir}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
