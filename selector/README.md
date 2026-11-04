# Kernel Selector Module

**Owner:** Member 3 (Kernel selector, generalization evaluation, metrics, and calibration)

## Purpose

Predicts which GPU CSR SpMV kernel is fastest for a matrix from its structural features, and
provides everything that produces a *decision*: data loading and validation, oracle labels, the
model, baselines and training. Everything that *scores* decisions lives in
[`evaluation/`](../evaluation/README.md). Full protocol: [docs/selector_evaluation.md](../docs/selector_evaluation.md).

## Modules

| File | Role |
|---|---|
| `data.py` | Load + strictly validate `features.csv`, `splits.csv`, `matrix_catalog.csv`, `kernel_timings.csv`; join by `matrix_id`; apply the candidate-coverage policy; `Dataset` (timings only reachable per split) |
| `labels.py` | Oracle per matrix (fastest **correct** kernel), tie tolerance, end-to-end oracle, training labels (train split only) |
| `model.py` | `KernelSelector`: Random Forest wrapper (features by name, deterministic, save/load with library versions) |
| `baselines.py` | Fixed-kernel baselines decided on the train split |
| `train.py` | Fit on train, tune on validation, write `model.joblib`, `model_card.json`, `tuning_results.csv` |
| `config.py` | Load/validate `configs/selector_config.yaml`, CLI overrides, run IDs, synthetic-input check |

## Use

```bash
python scripts/train_selector.py --config configs/selector_config.yaml
```

```python
from selector.data import build_dataset
from selector.train import train_selector

ds = build_dataset(features_csv, splits_csv, catalog_csv, timings_csv)
result = train_selector(ds)                    # uses only the train and val splits
picks = result.selector.predict(ds.features_for("test"))
```

## Rules this module enforces

- The shared contracts are the only interface; split label is `val`, not `validation`.
- Kernel names come from the timings file, never from code.
- A kernel that failed correctness is never an oracle or a label.
- Training never touches the test split (tested).
- Synthetic timings are for tests only and are never written to `benchmarks/kernel_timings.csv`.
