# Tests Module

**Owner:** Shared / Collaborative

## Purpose

Automated tests for code quality and contract compliance. Run all of them from the project root:

```bash
pytest tests/ -q
```

All tests are CPU-only and build their inputs programmatically (`tmp_path`); no raw matrices or GPU are needed.

## Member 2 (data pipeline)

- `test_data_pipeline.py`: CSR loading, validation, feature extraction, splits.
- `test_reference.py`: CPU reference SpMV against `scipy.sparse`.
- `test_contracts.py`: generated metadata CSVs conform to `docs/data_dictionary.md`.

## Member 3 (selector and evaluation)

- `test_selector_data.py`: every input-validation rule has a failing-case test; shuffled rows give an identical dataset.
- `test_labels.py`: oracle ignores incorrect kernels, tie rule, end-to-end oracle, train-only labels.
- `test_metrics.py`: hand-computed metric values, incorrect-selection policy, bootstrap, edge cases.
- `test_selector_model.py`: model determinism, save/load, feature-order checks, baselines, training, **no-test-leakage** tests.
- `test_evaluation_e2e.py`: evaluate, tables, plots, config and both CLI scripts end to end.

## Synthetic fixtures

`fixtures/synthetic_timings.py` generates **synthetic** timings from an invented formula. They exist
only to exercise code paths. They are not GPU measurements, must never be written to
`benchmarks/kernel_timings.csv`, and any output derived from them is stamped as synthetic.
`scripts/make_synthetic_demo.py` writes the same data to disk for manual trials.

## Speed note

The whole suite takes about a minute, mostly library import time and many small forest fits.
