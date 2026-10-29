"""Mathematical tests for NN/KNN (FORMULA_LEDGER NN-1, NN-2)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.nn_knn.core import (
    _cosine_distances,
    _euclidean_distances,
    run_knn,
)
from obfuscation_attacks.data.contracts import AttackContext, Capability

pytestmark = pytest.mark.mathematical


def _ctx(H, E):
    return AttackContext(
        granted=frozenset({Capability.EMBEDDINGS}),
        descriptor={"vocab_size": E.shape[0]},
        _obfuscated_embeddings=H,
        reference_embeddings=E,
    )


def test_euclidean_distance_matches_hand_computation():
    H = np.array([[0.0, 0.0]])
    E = np.array([[3.0, 4.0], [1.0, 0.0]])
    d = _euclidean_distances(H, E)
    assert d[0, 0] == pytest.approx(5.0)  # sqrt(9+16)
    assert d[0, 1] == pytest.approx(1.0)


def test_euclidean_ranking_and_topk_truth():
    E = np.array([[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    H = np.array([[0.9, 0.1]])  # closest to token 0
    res = run_knn(_ctx(H, E), "euclidean", top_k=3)
    assert res.recovered_token_ids[0] == 0
    assert res.candidates[0][0] == 0  # nearest first


def test_cosine_distance_hand_and_zero_vector():
    H = np.array([[1.0, 0.0], [0.0, 0.0]])  # second is the zero vector
    E = np.array([[2.0, 0.0], [0.0, 3.0]])
    d = _cosine_distances(H, E)
    assert d[0, 0] == pytest.approx(0.0)  # same direction -> distance 0
    assert d[0, 1] == pytest.approx(1.0)  # orthogonal -> distance 1
    # zero vector has undefined direction -> maximal distance 1 to everything
    assert d[1, 0] == pytest.approx(1.0)
    assert d[1, 1] == pytest.approx(1.0)


def test_distance_tie_keeps_both_in_topk():
    E = np.array([[1.0, 0.0], [-1.0, 0.0]])  # both at distance 1 from origin
    H = np.array([[0.0, 0.0]])
    res = run_knn(_ctx(H, E), "euclidean", top_k=2)
    assert set(res.candidates[0]) == {0, 1}
    assert res.diagnostics["first_second_margin"][0] == pytest.approx(0.0)
