# Scripts Module

**Owner:** Shared / Collaborative

## Purpose

Executable entry points. Scripts are thin: parse arguments, call the library modules, print a summary.

## Available now

| Script | Owner | What it does |
|---|---|---|
| `run_pipeline.py` | Member 2 | Validation, CPU reference and feature extraction; writes the metadata CSVs |
| `train_selector.py` | Member 3 | Train on `train`, tune on `val`, write `results/<run_id>/model.joblib` + `model_card.json` |
| `evaluate_generalization.py` | Member 3 | Evaluate the frozen model on `test`; write tables and figures into the run folder |
| `make_synthetic_demo.py` | Member 3 | Write SYNTHETIC demo CSVs + config so the pipeline can be tried without real timings |

### Selector workflow

```bash
python scripts/train_selector.py --config configs/selector_config.yaml
python scripts/evaluate_generalization.py --config configs/selector_config.yaml --run-dir results/<run_id>
```

Common flags: `--features --splits --catalog --timings --results-dir` (override config paths),
`--allow-synthetic` (needed when an input filename contains "synthetic"),
`--run-id` (train), `--run-dir` and `--overwrite` (evaluate).

Exit codes: `0` ok, `1` data/config error, `2` synthetic input without `--allow-synthetic`,
`3` test report already exists, `4` input files changed since training.

## Planned (not yet written)

- `download_matrices.*`: pull matrices specified in configs.
- `run_benchmarks.*`: GPU benchmark driver (Member 1).
