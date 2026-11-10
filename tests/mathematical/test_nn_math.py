"""Mathematical tests for NN/KNN (FORMULA_LEDGER NN-1, NN-2).

The retrieval returns the FULL per-position vocabulary ranking (every id in
ascending-distance order); the scorer slices the Top-k it needs. These tests pin
the distance maths (L2 / cosine, including zero-vector handling) and the
full-ranking contract, so ASR@k is never capped by the attack's requested k.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.nn_knn.core import (
    _cosine_distances,
    _euclidean_distances,
    run_knn,
)
from obfuscation_attacks.data.contracts import AttackContext, Capability
from obfuscation_attacks.metrics import asr_at_k

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


def test_cosine_distance_hand_and_zero_vector():
    H = np.array([[1.0, 0.0], [0.0, 0.0]])  # second is the zero vector
    E = np.array([[2.0, 0.0], [0.0, 3.0]])
    d = _cosine_distances(H, E)
    assert d[0, 0] == pytest.approx(0.0)  # same direction -> distance 0
    assert d[0, 1] == pytest.approx(1.0)  # orthogonal -> distance 1
    # zero vector has undefined direction -> maximal distance 1 to everything
    assert d[1, 0] == pytest.approx(1.0)
    assert d[1, 1] == pytest.approx(1.0)


def test_euclidean_returns_full_ranking_and_uncaps_asr():
    # 5 tokens on the x-axis; H sits right next to token 0.
    E = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    H = np.array([[0.1, 0.0]])
    vocab = E.shape[0]
    # Even with top_k=1 requested, the candidate pool must NOT be truncated.
    res = run_knn(_ctx(H, E), "euclidean", top_k=1)
    assert res.recovered_token_ids[0] == 0                   # nearest token
    assert res.candidates[0][0] == 0                         # nearest first
    assert len(res.candidates[0]) == vocab                   # full ranking length
    assert sorted(res.candidates[0]) == list(range(vocab))   # a permutation of all ids
    assert res.diagnostics["ranking_length"] == vocab
    # Ground-truth token 3 is NOT the nearest, so a capped ASR@1 misses it; the
    # full ranking guarantees ASR@vocab == 1.0 (the cap is removed).
    truth = [3]
    assert asr_at_k(res.candidates, truth, 1) == 0.0
    assert asr_at_k(res.candidates, truth, vocab) == 1.0


def test_cosine_ranking_through_run_knn():
    # Direction matters, not magnitude: tokens 0 and 1 share H's direction.
    E = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    H = np.array([[5.0, 0.0]])
    vocab = E.shape[0]
    res = run_knn(_ctx(H, E), "cosine", top_k=2)
    assert len(res.candidates[0]) == vocab
    assert sorted(res.candidates[0]) == list(range(vocab))
    # aligned tokens (cosine distance 0) rank first; opposite token (distance 2) last
    assert set(res.candidates[0][:2]) == {0, 1}
    assert res.candidates[0][-1] == 3
    assert res.recovered_token_ids[0] in (0, 1)
    # orthogonal token 2 is rank 3 of 4 -> only captured by a deep enough k
    assert asr_at_k(res.candidates, [2], 2) == 0.0
    assert asr_at_k(res.candidates, [2], vocab) == 1.0


def test_distance_tie_keeps_both_in_ranking():
    E = np.array([[1.0, 0.0], [-1.0, 0.0]])  # both at distance 1 from the origin
    H = np.array([[0.0, 0.0]])
    res = run_knn(_ctx(H, E), "euclidean", top_k=2)
    assert set(res.candidates[0]) == {0, 1}
    assert res.diagnostics["first_second_margin"][0] == pytest.approx(0.0)
