"""Regression tests: another package named ``tests`` must not break this project.

Google Colab ships a top-level ``tests`` package in dist-packages. Before
``tests/__init__.py`` existed, ``from tests.fixtures... import ...`` imported that package
instead of ours and failed. These tests plant a fake regular ``tests`` package on
PYTHONPATH (like Colab does) and check that our imports still work.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def shadow_env(tmp_path):
    shadow = tmp_path / "shadow"
    (shadow / "tests").mkdir(parents=True)
    (shadow / "tests" / "__init__.py").write_text("SHADOW = True\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(shadow)
    return env


def _run(args, env, cwd):
    return subprocess.run([sys.executable, *args], env=env, cwd=cwd, capture_output=True, text=True)


def test_project_tests_package_beats_a_shadowing_package(shadow_env):
    """`import tests.fixtures...` must resolve to OUR package even with a regular `tests` elsewhere."""
    code = (
        "import tests, tests.fixtures.synthetic_timings as m; "
        "print(tests.__file__); print(m.__file__)"
    )
    result = _run(["-c", code], shadow_env, ROOT)   # cwd=ROOT mirrors conftest/script putting the root first
    assert result.returncode == 0, result.stderr
    first, second = result.stdout.splitlines()
    assert Path(first).resolve() == ROOT / "tests" / "__init__.py"
    assert Path(second).resolve() == ROOT / "tests" / "fixtures" / "synthetic_timings.py"


def test_demo_script_works_with_a_shadowing_package(shadow_env, tmp_path):
    out = tmp_path / "demo"
    result = _run([str(ROOT / "scripts" / "make_synthetic_demo.py"), str(out)], shadow_env, tmp_path)
    assert result.returncode == 0, result.stderr
    for name in ("features.csv", "splits.csv", "matrix_catalog.csv",
                 "SYNTHETIC_kernel_timings.csv", "demo_config.yaml"):
        assert (out / name).is_file(), name
