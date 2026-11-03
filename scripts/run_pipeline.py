"""
run_pipeline.py
===============
Command-line driver for the Member 2 data pipeline.

Usage
-----
    # Full run using real matrices already in data/raw/
    python scripts/run_pipeline.py

    # Use synthetic pilot data (for development / CI)
    python scripts/run_pipeline.py --pilot

    # Custom config file
    python scripts/run_pipeline.py --config path/to/dataset_config.yaml

    # Custom raw matrix directory
    python scripts/run_pipeline.py --raw-dir /path/to/matrices/

    # Verbose logging
    python scripts/run_pipeline.py --pilot --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path when run directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_pipeline.pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Member 2 data pipeline: load, validate, extract features, split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to the dataset configuration YAML (default: configs/dataset_config.yaml)",
    )
    parser.add_argument(
        "--raw-dir",
        dest="raw_dir",
        default=None,
        help="Override the raw matrix directory from config.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Generate and use synthetic pilot data (for development/testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_pipeline(
        config_path=args.config,
        raw_dir_override=args.raw_dir,
        pilot=args.pilot,
    )

    n_excluded = len(result.get("exclusions", {}))
    if n_excluded > 0:
        print(
            f"\nWARNING: {n_excluded} file(s) were excluded due to errors. "
            "See data/metadata/exclusion_log.csv for details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
