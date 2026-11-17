"""
run_experiment.py
=================
Run the complete real experiment into results/<run_id>/.

    python scripts/run_experiment.py --run-id run_real_A                # analysis of an existing benchmark
    python scripts/run_experiment.py --run-id run_real_B --benchmark    # build + validate + benchmark + analysis

Stages:  environment check -> dataset validation -> CSR pipeline (features/splits) -> [GPU benchmark]
         -> contract/coverage/correctness validation -> train/tune (train+val) -> ONE test pass
         -> statistics, overhead, figures, report.

Refuses to run when CUDA is unavailable (with --benchmark), matrix files or CSV contracts are missing or
invalid, the timing data is incomplete, a required kernel never passes correctness, inputs are synthetic,
or the test split of the run directory has already been evaluated.  A new run needs a new --run-id.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from experiment.run import RefusedToRun, run_experiment  # noqa: E402


def _call(*cmd: str) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run([sys.executable, *cmd], cwd=_root).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default="configs/dataset_real.yaml")
    ap.add_argument("--benchmark", action="store_true", help="(re)run the GPU benchmark into results/<run_id>/benchmark first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args(argv)

    run_dir = _root / "results" / args.run_id
    cfg = _root / args.config

    if args.benchmark:
        if (run_dir / "TEST_EVALUATED.json").exists():
            print(f"ERROR: run '{args.run_id}' already evaluated its test split; choose a new --run-id.", file=sys.stderr)
            return 2
        for step in (["scripts/check_environment.py"], ["scripts/validate_dataset.py"],
                     ["scripts/run_pipeline.py", "--config", args.config], ["scripts/build_gpu.py"]):
            if _call(*step) != 0:
                print(f"ERROR: stage failed: {' '.join(step)}", file=sys.stderr)
                return 2
        if _call("scripts/run_gpu_benchmark.py", "--config", args.config, "--out-dir", str(run_dir / "benchmark")) != 0:
            print("ERROR: benchmark reported failures; inspect benchmark_failures.csv. "
                  "The experiment will not proceed on an incomplete benchmark unless failures are within the policy.", file=sys.stderr)
            # continue: preflight decides, using the documented incompleteness policy

    try:
        res = run_experiment(_root, run_dir, cfg, run_dir / "benchmark", seed=args.seed)
    except RefusedToRun as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    if not args.skip_report:
        from experiment.report import write_report
        write_report(_root, res)
    print(f"Done. Results in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
