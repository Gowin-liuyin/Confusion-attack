"""Acceptance tests for W2 micro/macro intervals (F12, protocol §8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.metrics.intervals import macro_bootstrap, micro_cluster_bootstrap

pytestmark = pytest.mark.unit


def test_micro_point_and_ci_share_one_estimand():
    result = micro_cluster_bootstrap(
        np.array([1, 0, 1, 0]), np.array([1, 9, 1, 9]), np.array([0, 1, 2, 3]),
        n_boot=500, seed=1,
    )
    assert result["micro"] == pytest.approx(0.1)
    assert result["denominator"] == 20
    assert result["n_units"] == 4
    assert result["n_clusters"] == 4
    assert result["lo"] <= 0.1 <= result["hi"]
    midpoint = (result["lo"] + result["hi"]) / 2.0
    # The old bug paired this micro estimate with a macro interval centred on 0.5.
    assert abs(midpoint - 0.1) < abs(midpoint - 0.5)


def test_cluster_structure_is_respected():
    # One independent cluster identifies no resampling variance: the point is
    # reported, but there is no 2000-replicate interval (n_boot=0, no bounds).
    result = micro_cluster_bootstrap(
        np.array([1, 0]), np.array([1, 9]), np.array([0, 0]), n_boot=300, seed=2
    )
    assert result["micro"] == pytest.approx(0.1)
    assert result["n_boot"] == 0
    assert result["lo"] is None and result["hi"] is None


def test_macro_is_reported_separately():
    result = macro_bootstrap(np.array([1.0, 0.0]), np.array([0, 1]), n_boot=300, seed=3)
    assert result["macro"] == pytest.approx(0.5)
    assert result["lo"] <= 0.5 <= result["hi"]


def test_empty_input_is_null_not_zero():
    result = micro_cluster_bootstrap(np.array([]), np.array([]), np.array([]))
    assert result["micro"] is None and result["denominator"] == 0


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError):
        micro_cluster_bootstrap(np.array([2]), np.array([1]), np.array([0]))


def test_micro_n_boot_counts_only_finite_replicates():
    result = micro_cluster_bootstrap(
        np.array([1, 0]), np.array([1, 0]), np.array([0, 1]),
        n_boot=2000, seed=0,
    )
    assert result["micro"] == pytest.approx(1.0)
    # Zero-denominator draws are undefined and dropped; the reported count is
    # the number of replicates actually used, never the nominal request.
    assert 0 < result["n_boot"] < 2000


def test_macro_n_boot_is_the_used_replicate_count():
    result = macro_bootstrap(
        np.array([1.0, 0.0]), np.array([0, 1]), n_boot=2000, seed=0,
    )
    assert result["n_boot"] == 2000
    assert result["lo"] <= 0.5 <= result["hi"]
