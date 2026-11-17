"""
Refusal conditions of the experiment driver (experiment.run.preflight / run_experiment).

They use the repository's real metadata (data/metadata/real) with doctored copies of a benchmark directory; they
need neither a GPU nor the matrix files, and they never evaluate a test split.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiment.run import RefusedToRun, preflight, run_experiment

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "dataset_real.yaml"
BENCH = ROOT / "results" / "run_real_001" / "benchmark"

pytestmark = pytest.mark.skipif(not (BENCH / "kernel_timings.csv").is_file() or not (ROOT / "data/metadata/real/features.csv").is_file(),
                                reason="real benchmark artifacts not present")


@pytest.fixture()
def cfg():
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


@pytest.fixture()
def bench(tmp_path):
    d = tmp_path / "benchmark"
    d.mkdir()
    for name in ("kernel_timings.csv", "benchmark_summary.csv", "overhead_measurements.csv", "benchmark_failures.csv",
                 "environment.json", "benchmark_manifest.json"):
        shutil.copy(BENCH / name, d / name)
    return d


def test_valid_inputs_pass_preflight(cfg, bench):
    paths = preflight(ROOT, cfg, bench)
    assert paths["timings"] == bench / "kernel_timings.csv"


def test_missing_timings_refused(cfg, bench):
    (bench / "kernel_timings.csv").unlink()
    with pytest.raises(RefusedToRun, match="required input missing"):
        preflight(ROOT, cfg, bench)


def test_missing_environment_record_refused(cfg, bench):
    (bench / "environment.json").unlink()
    with pytest.raises(RefusedToRun, match="environment record missing"):
        preflight(ROOT, cfg, bench)


def test_incomplete_benchmark_refused(cfg, bench):
    t = pd.read_csv(bench / "kernel_timings.csv")
    keep = sorted(t["matrix_id"].unique())[:20]               # 20 of 122 matrices benchmarked
    t[t["matrix_id"].isin(keep)].to_csv(bench / "kernel_timings.csv", index=False)
    with pytest.raises(RefusedToRun, match="incomplete"):
        preflight(ROOT, cfg, bench)


def test_kernel_that_never_passes_correctness_refused(cfg, bench):
    t = pd.read_csv(bench / "kernel_timings.csv")
    t.loc[t["kernel_name"] == "csr_vector", "correctness_pass"] = False
    t.to_csv(bench / "kernel_timings.csv", index=False)
    with pytest.raises(RefusedToRun, match="never passed correctness"):
        preflight(ROOT, cfg, bench)


def test_contract_violation_refused(cfg, bench):
    t = pd.read_csv(bench / "kernel_timings.csv")
    pd.concat([t, t.iloc[[0]]]).to_csv(bench / "kernel_timings.csv", index=False)   # duplicate (matrix, kernel)
    with pytest.raises(RefusedToRun, match="contract"):
        preflight(ROOT, cfg, bench)


def test_synthetic_manifest_refused(cfg, bench, tmp_path):
    """A manifest row whose source is synthetic must stop a real experiment."""
    root = tmp_path / "root"
    (root / "data/metadata").mkdir(parents=True)
    shutil.copytree(ROOT / "data/metadata/real", root / "data/metadata/real")
    pd.read_csv(ROOT / "data/metadata/dataset_manifest.csv").assign(source="synthetic generator").to_csv(
        root / "data/metadata/dataset_manifest.csv", index=False)
    with pytest.raises(RefusedToRun, match="synthetic"):
        preflight(root, cfg, bench)


def test_synthetic_named_file_refused(cfg, bench, tmp_path):
    """Input files whose name contains 'synthetic' are refused (mirrors the selector CLI guard)."""
    root = tmp_path / "root"
    (root / "data/metadata").mkdir(parents=True)
    shutil.copytree(ROOT / "data/metadata/real", root / "data/metadata/real")
    shutil.copy(ROOT / "data/metadata/dataset_manifest.csv", root / "data/metadata/synthetic_manifest.csv")
    cfg = dict(cfg)
    cfg["dataset"] = dict(cfg["dataset"], manifest="data/metadata/synthetic_manifest.csv")
    with pytest.raises(RefusedToRun, match="synthetic"):
        preflight(root, cfg, bench)


def test_second_test_evaluation_of_same_run_refused(tmp_path):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    (run_dir / "TEST_EVALUATED.json").write_text("{}")
    with pytest.raises(RefusedToRun, match="already evaluated"):
        run_experiment(ROOT, run_dir, CFG, run_dir / "benchmark")
