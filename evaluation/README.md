# Evaluation Module

**Owner:** Member 3 (Kernel selector, generalization evaluation, metrics, and calibration)

## Purpose

Scores how well the trained selector generalizes to held-out matrices, against fixed-kernel
baselines and the measured oracle. Metric definitions and caveats:
[docs/selector_evaluation.md](../docs/selector_evaluation.md).

## Modules

| File | Role |
|---|---|
| `metrics.py` | Pure functions: oracle-relative loss, selection accuracy, total-time ratio, bootstrap CI, `score_selections` |
| `evaluate.py` | Builds the tidy per-matrix table (`method` = `selector`, `fixed_<kernel>`, `oracle`) |
| `tables.py` | Overall / per-group / confusion tables as CSV and Markdown; synthetic stamping |
| `plots.py` | Six PNG figures (matplotlib, headless) |

## Headline metrics

- **Oracle-relative loss** = `selected_time / oracle_time - 1` (0 = optimal). Stored as a fraction.
- **Selection accuracy** (ties within a tolerance count as correct). Always read it next to loss.
- Mean / median / p90 / max loss, 95 % bootstrap CI, share of matrices within 0/5/10/25 % of the oracle.
- End-to-end time (kernel + preprocessing) alongside kernel-only time.
- A pick of a kernel that failed correctness is a failure (NaN loss, counted separately).

## Use

```bash
python scripts/evaluate_generalization.py --config configs/selector_config.yaml --run-dir results/<run_id>
```

The report is for the **test** split only, from a frozen model, once per run folder.
Synthetic inputs need `--allow-synthetic` and every output is stamped
`SYNTHETIC DATA - NOT A RESULT`.

## Not implemented yet

Calibration (Milestone 6, optional). The planned design is a measurement-based replay study; see
section 11 of the protocol document. Note the terminology difference with the root README.
