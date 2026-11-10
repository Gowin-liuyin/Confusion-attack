"""Custom-scheme interface + information isolation (Goal section 9)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.nn_knn.euclidean_knn import EuclideanKNN
from obfuscation_attacks.attacks.vma.weight_rowsort import WeightRowSortVMA
from obfuscation_attacks.data.contracts import AttackStatus, Capability
from obfuscation_attacks.evaluation.scoring import compute_metrics
from obfuscation_attacks.schemes.base import build_context
from obfuscation_attacks.schemes.custom_example import CustomNullSpaceScheme

pytestmark = pytest.mark.integration


def _asr1(epsilon: float) -> float:
    rng = np.random.default_rng(0)
    E = rng.standard_normal((60, 16))
    ids = rng.integers(0, 60, size=20)
    scheme = CustomNullSpaceScheme(epsilon=epsilon, seed=0)
    sample = scheme.build_sample(ids, E)
    ctx = build_context(scheme, sample)
    res = EuclideanKNN(top_k=10).run(ctx)
    return compute_metrics(res, sample)["token_recovery"]["asr@1"]


def test_custom_scheme_runs_knn_and_epsilon_is_a_defender_knob():
    small = _asr1(0.05)
    large = _asr1(5.0)
    assert small >= 0.8          # weak perturbation leaks
    assert small >= large        # stronger perturbation reduces leakage
    assert large < small or large < 1.0


def test_custom_scheme_withholds_weights_so_weight_vma_not_applicable():
    rng = np.random.default_rng(1)
    E = rng.standard_normal((20, 8))
    ids = rng.integers(0, 20, size=5)
    scheme = CustomNullSpaceScheme(epsilon=0.1, seed=0)
    sample = scheme.build_sample(ids, E)
    ctx = build_context(scheme, sample)

    # the scheme exposes only EMBEDDINGS
    assert ctx.has(Capability.EMBEDDINGS)
    assert not ctx.has(Capability.WHITE_BOX_WEIGHTS)

    res = WeightRowSortVMA().run(ctx)
    assert res.status == AttackStatus.NOT_APPLICABLE
