"""
config.py
=========
Load ``configs/selector_config.yaml``, apply CLI overrides, and small run helpers
shared by ``scripts/train_selector.py`` and ``scripts/evaluate_generalization.py``.
"""

from __future__ import annotations

import copy
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from selector.train import PROJECT_ROOT, TrainConfig

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "selector_config.yaml"
PATH_KEYS = ("features", "splits", "catalog", "timings")

_ALLOWED = {
    "paths": set(PATH_KEYS),
    "selector": {"feature_columns", "seed", "tie_tolerance", "refit_on_train_val", "grid"},
    "evaluation": {"bootstrap_samples", "bootstrap_seed"},
}
_TOP_LEVEL = set(_ALLOWED) | {"results_dir"}


class ConfigError(ValueError):
    """Raised for an invalid or incomplete configuration file."""


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Read and validate the YAML config. Unknown keys are errors (typos must not pass)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a YAML mapping")
    extra = set(cfg) - _TOP_LEVEL
    if extra:
        raise ConfigError(f"unknown top-level config key(s): {sorted(extra)}")
    for section, allowed in _ALLOWED.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ConfigError(f"config section '{section}' is missing")
        extra = set(cfg[section]) - allowed
        if extra:
            raise ConfigError(f"unknown key(s) in '{section}': {sorted(extra)}")
    missing = [k for k in PATH_KEYS if k not in cfg["paths"]]
    if missing:
        raise ConfigError(f"config 'paths' is missing: {missing}")
    sel = cfg["selector"]
    if "tie_tolerance" in sel and not (isinstance(sel["tie_tolerance"], (int, float)) and sel["tie_tolerance"] >= 0):
        raise ConfigError("selector.tie_tolerance must be a number >= 0")
    if "refit_on_train_val" in sel and not isinstance(sel["refit_on_train_val"], bool):
        raise ConfigError("selector.refit_on_train_val must be true or false")
    cfg.setdefault("results_dir", "results")
    return cfg


def apply_overrides(cfg: Mapping[str, Any], overrides: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    """Return a copy with non-None CLI path overrides (and ``results_dir``) applied."""
    out = copy.deepcopy(dict(cfg))
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "results_dir":
            out["results_dir"] = value
        elif key in PATH_KEYS:
            out["paths"][key] = value
        else:
            raise ConfigError(f"unknown override: {key}")
    return out


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def input_paths(cfg: Mapping[str, Any]) -> Dict[str, Path]:
    return {k: resolve_path(cfg["paths"][k]) for k in PATH_KEYS}


def train_config(cfg: Mapping[str, Any]) -> TrainConfig:
    s = cfg["selector"]
    kwargs: Dict[str, Any] = {}
    for key in ("feature_columns", "seed", "tie_tolerance", "grid", "refit_on_train_val"):
        if key in s:
            kwargs[key] = tuple(s[key]) if key == "feature_columns" else s[key]
    return TrainConfig(**kwargs)


def is_synthetic(paths: Mapping[str, Path | str]) -> bool:
    """True if any input file name contains 'synthetic' (case-insensitive)."""
    return any("synthetic" in Path(p).name.lower() for p in paths.values())


def make_run_id(now: Optional[datetime] = None) -> str:
    """``YYYYMMDD-HHMM-<short git hash>`` (``nogit`` if git is unavailable)."""
    now = now or datetime.now()
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        short = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "nogit"
    except (OSError, subprocess.SubprocessError):
        short = "nogit"
    return f"{now:%Y%m%d-%H%M}-{short}"


def latest_run_dir(results_dir: Path) -> Path:
    """Newest run folder (by name) that contains a trained model."""
    candidates = sorted(p for p in results_dir.glob("*") if (p / "model.joblib").is_file())
    if not candidates:
        raise ConfigError(f"no trained run found in {results_dir}; run scripts/train_selector.py first")
    return candidates[-1]
