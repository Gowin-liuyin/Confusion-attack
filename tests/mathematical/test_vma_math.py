"""Mathematical tests for the three VMA paths (FORMULA_LEDGER VMA-B1/B2/B3)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.vma.autoregressive import _l1, _sorted_l1
from obfuscation_attacks.attacks.vma.weight_rowsort import recover_permutations
from obfuscation_attacks.data.contracts import AttackContext, Capability

pytestmark = pytest.mark.mathematical


# --- B2: weight RowSort (Y = Z1 X Z2) -------------------------------------
def _reconstruct(X, z1, z2):
    return X[np.ix_(z1, z2)]


def test_rowsort_recovers_distinct_rows_exactly():
    rng = np.random.default_rng(0)
    n, m = 9, 6
    X = rng.standard_normal((n, m))
    z1 = rng.permutation(n)
    z2 = rng.permutation(m)
    Y = _reconstruct(X, z1, z2)
    res = recover_permutations(X, Y)
    assert res["status"] == "ok"
    assert np.array_equal(res["Z1"], z1)
    # reconstruct with recovered perms reproduces Y exactly
    recon = X[np.ix_(res["Z1"], res["Z2_layers"][0])]
    assert np.allclose(recon, Y)


def test_rowsort_duplicate_rows_resolved_by_verification():
    rng = np.random.default_rng(1)
    n, m = 6, 5
    X = rng.standard_normal((n, m))
    X[1] = X[0]  # duplicate row -> fingerprint collision
    z1 = rng.permutation(n)
    z2 = rng.permutation(m)
    Y = X[np.ix_(z1, z2)]
    res = recover_permutations(X, Y)
    # ambiguity is bounded and resolved by a verified search; Z1 may be
    # non-unique for the identical rows, but the recovered (Z1, Z2) MUST
    # reproduce Y exactly (never an arbitrary unverified match).
    assert res["status"] == "ok"
    assert res["collisions"] is True
    recon = X[np.ix_(res["Z1"], res["Z2_layers"][0])]
    assert np.allclose(recon, Y)


def test_rowsort_multilayer_voting_breaks_a_single_layer_tie():
    rng = np.random.default_rng(2)
    n, m = 6, 5
    X = rng.standard_normal((2, n, m))
    X[0, 1] = X[0, 0]  # layer 0 has a duplicate row; layer 1 is distinct
    z1 = rng.permutation(n)
    Y = np.empty_like(X)
    z2s = []
    for l in range(2):
        z2 = rng.permutation(m)
        z2s.append(z2)
        Y[l] = X[l][np.ix_(z1, z2)]
    res = recover_permutations(X, Y)
    assert res["status"] == "ok"
    # concatenated multi-layer fingerprints are unique -> Z1 recovered exactly
    assert res["collisions"] is False
    assert np.array_equal(res["Z1"], z1)


# --- B1: autoregressive L1 / sorted-L1 ------------------------------------
def test_sorted_l1_is_permutation_invariant():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(10)
    perm = rng.permutation(10)
    assert _sorted_l1(a, a[perm]) == pytest.approx(0.0)
    # plain L1 is NOT permutation invariant (sanity: the metrics differ)
    assert _l1(a, a[perm]) > 1e-6


# --- B3: OSNIP projection sorted-L1 ---------------------------------------
def test_osnip_projection_sorted_l1_recovers_under_hidden_permutation():
    from obfuscation_attacks.attacks.vma.osnip_projection import OSNIPProjectionVMA

    rng = np.random.default_rng(4)
    vocab, d, n = 20, 8, 5
    pool = rng.standard_normal((vocab, d))
    ids = rng.integers(0, vocab, size=n)
    perm = rng.permutation(d)  # unknown hidden-dim permutation on the observations
    target = pool[ids][:, perm]
    ctx = AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor={"vocab_size": vocab},
        _obfuscated_hidden_states=target,
        reference_embeddings=pool,
    )
    res = OSNIPProjectionVMA(sorted_l1=True, top_k=5)._run(ctx)
    # sorted-L1 cancels the permutation -> exact recovery
    assert np.array_equal(res.recovered_token_ids, ids)
