# Evaluation Module

**Owner:** Member 3 (Kernel selector, generalization evaluation, metrics, and calibration)

## Purpose

The `evaluation` module assesses how well the trained kernel selector generalizes across diverse and unseen sparse matrix distributions.

## Planned Capabilities

- Evaluation metrics:
  - Top-1 prediction accuracy (how often the optimal kernel is chosen).
  - Performance gap / speedup relative to default heuristic or vendor baseline (cuSPARSE).
  - Achieved fraction of oracle speedup.
- Generalization testing across held-out matrix groups and synthetic distributions.
- Selector confidence calibration analysis (e.g., expected calibration error, reliability diagrams).

