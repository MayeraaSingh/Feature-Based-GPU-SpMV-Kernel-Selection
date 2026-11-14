"""Regression tests for the group-aware statistics and the group-split fix (audit findings)."""

import numpy as np
import pytest

from data_pipeline.splits import make_splits
from evaluation.group_stats import (
    cluster_bootstrap_ci,
    group_mean,
    holm_adjust,
    paired_group_signflip_test,
)
from evaluation.metrics import bootstrap_ci


def test_cluster_ci_wider_than_iid_ci_for_correlated_groups():
    # 5 groups of 20 identical values: the iid bootstrap pretends n=100, the cluster one knows n=5.
    vals = np.repeat([0.0, 0.1, 0.2, 0.3, 0.4], 20)
    groups = np.repeat(list("abcde"), 20)
    iid = bootstrap_ci(vals, n_boot=3000, seed=1)
    cl = cluster_bootstrap_ci(vals, groups, n_boot=3000, seed=1)
    assert (cl[1] - cl[0]) > 1.5 * (iid[1] - iid[0])


def test_cluster_ci_single_group_is_nan():
    lo, hi = cluster_bootstrap_ci([1.0, 2.0, 3.0], ["g", "g", "g"])
    assert np.isnan(lo) and np.isnan(hi)


def test_group_mean_weights_groups_equally():
    assert group_mean([0, 0, 0, 0, 10], list("aaaab")) == pytest.approx(5.0)


def test_signflip_detects_consistent_difference_and_reports_floor():
    diff = np.repeat([0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.52], 3)
    groups = np.repeat(list("abcdefgh"), 3)
    r = paired_group_signflip_test(diff, groups)
    assert r["n_groups"] == 8 and r["p_value"] == pytest.approx(r["p_floor"])
    assert r["mean_group_diff"] > 0.4


def test_signflip_no_difference_not_significant():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 1, 40)
    groups = np.repeat(np.arange(10), 4)
    assert paired_group_signflip_test(diff, groups)["p_value"] > 0.05


def test_holm_adjust_known_values_and_nan_passthrough():
    adj = holm_adjust([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        group_mean([1.0, 2.0], ["a"])


# ---- regression: group split sizing (audit finding) ------------------------------------------
def _dominant_group_case():
    sizes = [30, 6, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1]
    ids, groups = [], {}
    for i, s in enumerate(sizes):
        for j in range(s):
            mid = f"m{i}_{j}"
            ids.append(mid)
            groups[mid] = f"G{i}"
    return ids, groups


@pytest.mark.parametrize("seed", range(12))
def test_group_split_keeps_train_majority_with_dominant_group(seed):
    ids, groups = _dominant_group_case()
    m = make_splits(ids, groups, seed=seed)
    counts = {k: sum(v == k for v in m.values()) for k in ("train", "val", "test")}
    assert counts["train"] >= 0.6 * len(ids), counts  # was as low as 9 of 59 before the fix
    assert counts["val"] > 0 and counts["test"] > 0


@pytest.mark.parametrize("seed", range(6))
def test_group_split_never_splits_a_group(seed):
    ids, groups = _dominant_group_case()
    m = make_splits(ids, groups, seed=seed)
    for g in set(groups.values()):
        assert len({m[i] for i in ids if groups[i] == g}) == 1


def test_group_split_is_deterministic():
    ids, groups = _dominant_group_case()
    assert make_splits(ids, groups, seed=3) == make_splits(ids, groups, seed=3)
