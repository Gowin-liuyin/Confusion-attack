"""Mathematical tests for the three VMA paths (FORMULA_LEDGER VMA-B1/B2/B3).

Papers:
* B1 Hidden No More (Thomas et al. 2025) - Algorithm 1 (epsilon/proposal),
  Algorithm 4 (sequence-dim), Algorithm 5 (sorted-L1 hidden-dim), Algorithm 6
  (factorized-2D).
* B2 AloePri App. D.1 / Table 9 - Y = Z1 X Z2, RowSort + neighbor matching,
  multi-layer voting, duplicate handling, additive weight noise.
* B3 OSNIP App. C.2 - autoregressive L1-replay vocabulary matching.

A tiny causal model is built INLINE here (no synthetic-scenario import) so the
causal L1-replay attacks can be exercised end to end on the CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.vma.autoregressive import (
    AutoregressiveVMA,
    autoregressive_reconstruct,
    _l1,
    _sorted_l1,
)
from obfuscation_attacks.attacks.vma.osnip_projection import OSNIPProjectionVMA
from obfuscation_attacks.attacks.vma.weight_rowsort import (
    WeightRowSortVMA,
    recover_permutations,
)
from obfuscation_attacks.data.contracts import (
    AttackContext,
    AttackStatus,
    Capability,
)

pytestmark = pytest.mark.mathematical


# ===========================================================================
# Inline tiny causal model (a 1-layer tanh RNN): the hidden state at position i
# depends only on tokens[: i + 1], so replaying [xhat_<i, v] reproduces the
# per-position state exactly - the causal property VMA exploits. Distinct random
# embeddings/weights make the per-position state injective in the last token.
# ===========================================================================
class _ToyCausalModel:
    def __init__(self, vocab: int, d: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.d = d
        self.E = rng.standard_normal((vocab, d))
        self.W = rng.standard_normal((d, d)) / np.sqrt(d)
        self.U = rng.standard_normal((d, d)) / np.sqrt(d)

    def hidden(self, token_ids) -> np.ndarray:
        h = np.zeros(self.d)
        out = []
        for t in token_ids:
            h = np.tanh(self.E[int(t)] @ self.W + h @ self.U)
            out.append(h.copy())
        return np.asarray(out)


def _replay_ctx(model: _ToyCausalModel, target: np.ndarray, vocab: int) -> AttackContext:
    return AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES, Capability.MODEL_REPLAY}),
        descriptor={"vocab_size": vocab},
        _obfuscated_hidden_states=np.asarray(target, dtype=float),
        _replay_fn=model.hidden,
    )


# ===========================================================================
# B1 - autoregressive L1 / sorted-L1, permutation variants, epsilon/proposal
# ===========================================================================
def test_sorted_l1_is_permutation_invariant():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(10)
    perm = rng.permutation(10)
    assert _sorted_l1(a, a[perm]) == pytest.approx(0.0)
    # plain L1 is NOT permutation invariant (sanity: the metrics differ)
    assert _l1(a, a[perm]) > 1e-6


def test_algorithm1_plain_l1_recovers_tokens():
    vocab, d, n = 14, 8, 6
    model = _ToyCausalModel(vocab, d, seed=0)
    ids = np.random.default_rng(1).integers(0, vocab, size=n)
    target = model.hidden(ids)
    res = autoregressive_reconstruct(model.hidden, target, vocab)
    assert np.array_equal(res["recovered"], ids)
    assert res["epsilon_fallback"] is None


def test_algorithm5_sorted_l1_reverses_hidden_dimension_permutation():
    vocab, d, n = 14, 8, 6
    model = _ToyCausalModel(vocab, d, seed=0)
    rng = np.random.default_rng(2)
    ids = rng.integers(0, vocab, size=n)
    perm = rng.permutation(d)  # unknown per-position column permutation
    target = model.hidden(ids)[:, perm]
    res = autoregressive_reconstruct(model.hidden, target, vocab, sorted_l1=True)
    assert np.array_equal(res["recovered"], ids)
    # plain L1 cannot reverse the hidden-dim permutation
    res_plain = autoregressive_reconstruct(model.hidden, target, vocab, sorted_l1=False)
    assert not np.array_equal(res_plain["recovered"], ids)


def test_algorithm4_reverses_sequence_dimension_permutation():
    vocab, d, n = 14, 8, 7
    model = _ToyCausalModel(vocab, d, seed=0)
    rng = np.random.default_rng(4)
    ids = rng.integers(0, vocab, size=n)
    sigma = rng.permutation(n)  # the N rows are permuted
    target = model.hidden(ids)[sigma]
    res = autoregressive_reconstruct(model.hidden, target, vocab, sequence_permuted=True)
    # recovered sequence is in TRUE (causal) order, matched against the
    # not-yet-used observed rows (Algorithm 4)
    assert np.array_equal(res["recovered"], ids)


def test_algorithm6_reverses_factorized_2d_permutation():
    vocab, d, n = 14, 8, 7
    model = _ToyCausalModel(vocab, d, seed=0)
    rng = np.random.default_rng(5)
    ids = rng.integers(0, vocab, size=n)
    sigma = rng.permutation(n)
    perm = rng.permutation(d)
    target = model.hidden(ids)[sigma][:, perm]  # rows + columns permuted
    res = autoregressive_reconstruct(
        model.hidden, target, vocab, sorted_l1=True, sequence_permuted=True
    )
    assert np.array_equal(res["recovered"], ids)


def test_epsilon_without_proposal_falls_back_to_global_argmin():
    # Algorithm 1 line 3 iterates argsort(P(xhat)); with NO proposal ordering we
    # must NOT iterate in token-id order for the early-accept. A huge epsilon
    # would make an id-order early-accept commit to token 0 everywhere; the
    # fallback to pure argmin must still recover the true (non-zero) tokens.
    vocab, d, n = 14, 8, 6
    model = _ToyCausalModel(vocab, d, seed=0)
    ids = np.array([3, 5, 7, 2, 9, 11])  # deliberately contains no zeros
    target = model.hidden(ids)
    res = autoregressive_reconstruct(model.hidden, target, vocab, epsilon=1e9)
    assert np.array_equal(res["recovered"], ids)
    assert not np.array_equal(res["recovered"], np.zeros(n, dtype=int))
    assert res["epsilon_enabled"] is False
    assert res["epsilon_fallback"] is not None


def test_epsilon_with_proposal_order_early_accepts():
    vocab, d, n = 14, 8, 6
    model = _ToyCausalModel(vocab, d, seed=0)
    rng = np.random.default_rng(6)
    ids = rng.integers(0, vocab, size=n)
    target = model.hidden(ids)

    def proposal_fn(prefix):
        # proposal model proposes the correct next token first
        i = len(prefix)
        correct = int(ids[i])
        return [correct] + [v for v in range(vocab) if v != correct]

    res = autoregressive_reconstruct(
        model.hidden, target, vocab, epsilon=1e-6, proposal_fn=proposal_fn
    )
    assert np.array_equal(res["recovered"], ids)
    assert all(res["accepted_by_epsilon"])  # every position cleared epsilon early
    assert res["epsilon_enabled"] is True


def test_negative_epsilon_is_pure_argmin():
    vocab, d, n = 12, 8, 5
    model = _ToyCausalModel(vocab, d, seed=0)
    ids = np.random.default_rng(8).integers(0, vocab, size=n)
    target = model.hidden(ids)
    res = autoregressive_reconstruct(model.hidden, target, vocab, epsilon=-1.0)
    assert np.array_equal(res["recovered"], ids)
    assert res["epsilon_enabled"] is False
    assert res["epsilon_fallback"] is None  # <0 is "disabled", not a fallback


def test_autoregressive_attack_class_permutation_param():
    vocab, d, n = 14, 8, 7
    model = _ToyCausalModel(vocab, d, seed=0)
    rng = np.random.default_rng(9)
    ids = rng.integers(0, vocab, size=n)
    sigma = rng.permutation(n)
    perm = rng.permutation(d)
    target = model.hidden(ids)[sigma][:, perm]
    ctx = _replay_ctx(model, target, vocab)
    res = AutoregressiveVMA(permutation="factorized_2d")._run(ctx)
    assert res.status == AttackStatus.OK
    assert np.array_equal(res.recovered_token_ids, ids)
    assert res.diagnostics["distance"] == "sorted_l1"
    assert res.diagnostics["permutation"] == "factorized_2d"


# ===========================================================================
# B2 - weight RowSort (Y = Z1 X Z2): noise, voting, duplicates, identity
# ===========================================================================
def _reconstruct(X, z1, z2):
    return X[np.ix_(z1, z2)]


def test_rowsort_identity_is_z1_times_rowsort_x():
    # Fix #7: the exact identity is RowSort(Y) = Z1 . RowSort(X); the paper's
    # RowSort(Y) = RowSort(X Z2) holds only as a SET of rows (up to order Z1).
    rng = np.random.default_rng(0)
    n, m = 7, 6
    X = rng.standard_normal((n, m))
    z1 = np.array([2, 0, 5, 1, 6, 3, 4])  # a non-identity row permutation
    z2 = rng.permutation(m)
    Y = _reconstruct(X, z1, z2)
    rs = lambda M: np.sort(M, axis=1)
    # exact per-row identity
    assert np.allclose(rs(Y), rs(X)[z1])
    # loose "RowSort(Y) = RowSort(X Z2)" is NOT exact row-for-row (order differs)
    assert not np.allclose(rs(Y), rs(X @ np.eye(m)[:, z2]))
    # but the multiset of sorted rows is identical (column-sorted the same)
    assert np.allclose(np.sort(rs(Y), axis=0), np.sort(rs(X), axis=0))


def test_rowsort_recovers_distinct_rows_exactly():
    rng = np.random.default_rng(0)
    n, m = 9, 6
    X = rng.standard_normal((n, m))
    z1 = rng.permutation(n)
    z2 = rng.permutation(m)
    Y = _reconstruct(X, z1, z2)
    res = recover_permutations(X, Y)
    assert res["status"] == "ok"
    assert res["verified"] is True
    assert np.array_equal(res["Z1"], z1)
    recon = X[np.ix_(res["Z1"], res["Z2_layers"][0])]
    assert np.allclose(recon, Y)
    assert res["reconstruction_residual"] == pytest.approx(0.0, abs=1e-9)


def test_rowsort_tolerates_additive_weight_noise():
    # C2: AloePri weights are noisy (W* = W_e + alpha_e E). RowSort + L1 nearest
    # neighbor recovers the permutation under small noise; exact fingerprinting
    # could not.
    rng = np.random.default_rng(11)
    n, m = 10, 7
    X = rng.standard_normal((n, m))
    z1 = rng.permutation(n)
    z2 = rng.permutation(m)
    noise = 0.01 * rng.standard_normal((n, m))
    Y = _reconstruct(X, z1, z2) + noise
    res = recover_permutations(X, Y)
    assert res["status"] == "ok"
    assert np.array_equal(res["Z1"], z1)
    assert 0.0 < res["reconstruction_residual"] < 1.0  # residual reflects the noise


def test_rowsort_multilayer_majority_vote_with_different_shapes():
    # M1: recover Pi per layer, then per-row majority vote across layers (not a
    # concatenated fingerprint). Layers may have different column counts.
    rng = np.random.default_rng(7)
    n = 6
    ms = [4, 5, 6]  # different shape per layer
    z1 = rng.permutation(n)
    Xl = [rng.standard_normal((n, mm)) for mm in ms]
    Xl[0][1] = Xl[0][0]  # layer 0 has a per-layer row tie; layers 1,2 are distinct
    Yl = [Xl[l][np.ix_(z1, rng.permutation(ms[l]))] for l in range(3)]
    res = recover_permutations(Xl, Yl)
    assert res["status"] == "ok"
    # globally the rows are distinct (layers 1,2 break the layer-0 tie) -> voting
    # recovers the shared Z1 exactly
    assert res["collisions"] is False
    assert np.array_equal(res["Z1"], z1)


def test_rowsort_duplicate_rows_reported_partial_with_candidates():
    # M2: genuine duplicate rows (identical in ALL layers) cannot be ordered by
    # RowSort. Unique rows are assigned + verified; the duplicate group is
    # reported as a partial result with candidate sets - never a fake verified.
    rng = np.random.default_rng(1)
    n, m = 6, 5
    X = rng.standard_normal((n, m))
    X[1] = X[0]  # exact duplicate row -> unresolvable fingerprint group {0, 1}
    z1 = rng.permutation(n)
    z2 = rng.permutation(m)
    Y = _reconstruct(X, z1, z2)
    res = recover_permutations(X, Y)
    assert res["status"] == "partial"
    assert res["partial"] is True
    assert res["verified"] is False
    assert res["collisions"] is True
    # the ambiguous output rows are exactly those mapping to inputs 0 or 1
    expected_ambiguous = {int(i) for i in range(n) if z1[i] in (0, 1)}
    assert set(res["candidates"].keys()) == expected_ambiguous
    for i in expected_ambiguous:
        assert sorted(res["candidates"][i]) == [0, 1]
    # the UNIQUE rows are still recovered correctly
    for i in range(n):
        if int(i) not in expected_ambiguous:
            assert res["Z1"][i] == z1[i]


def test_weight_attack_wrapper_reports_partial_without_faking_verified():
    rng = np.random.default_rng(1)
    n, m = 6, 5
    X = rng.standard_normal((n, m))
    X[1] = X[0]
    z1 = rng.permutation(n)
    z2 = rng.permutation(m)
    Y = _reconstruct(X, z1, z2)
    ctx = AttackContext(
        granted=frozenset({Capability.WHITE_BOX_WEIGHTS}),
        descriptor={},
        _known_weights={"X": X, "Y": Y},
    )
    res = WeightRowSortVMA()._run(ctx)
    assert res.status == AttackStatus.OK          # it ran and partially recovered
    assert res.diagnostics["partial"] is True     # distinct partial signal
    assert res.diagnostics["verified"] is False   # never a fake verified
    assert res.mapping_candidates is not None
    assert "candidates" in res.diagnostics


def test_weight_attack_wrapper_verifies_clean_case():
    rng = np.random.default_rng(0)
    n, m = 8, 5
    X = rng.standard_normal((n, m))
    z1, z2 = rng.permutation(n), rng.permutation(m)
    Y = _reconstruct(X, z1, z2)
    ctx = AttackContext(
        granted=frozenset({Capability.WHITE_BOX_WEIGHTS}),
        descriptor={},
        _known_weights={"X": X, "Y": Y},
    )
    res = WeightRowSortVMA()._run(ctx)
    assert res.status == AttackStatus.OK
    assert res.diagnostics["verified"] is True
    assert res.diagnostics["partial"] is False
    assert np.array_equal(res.recovered_permutations["Z1"], z1)


# ===========================================================================
# B3 - OSNIP autoregressive L1-replay vocabulary matching
# ===========================================================================
def test_osnip_autoregressive_l1_recovers_tokens():
    vocab, d, n = 20, 8, 6
    model = _ToyCausalModel(vocab, d, seed=0)
    ids = np.random.default_rng(4).integers(0, vocab, size=n)
    target = model.hidden(ids)
    ctx = _replay_ctx(model, target, vocab)
    res = OSNIPProjectionVMA(top_k=5)._run(ctx)
    assert res.status == AttackStatus.OK
    assert np.array_equal(res.recovered_token_ids, ids)
    assert res.diagnostics["distance"] == "l1"
    assert res.diagnostics["mode"] == "autoregressive_replay"


def test_osnip_sorted_l1_recovers_under_hidden_permutation():
    vocab, d, n = 20, 8, 5
    model = _ToyCausalModel(vocab, d, seed=0)
    rng = np.random.default_rng(4)
    ids = rng.integers(0, vocab, size=n)
    perm = rng.permutation(d)  # unknown hidden-dim permutation on the observations
    target = model.hidden(ids)[:, perm]
    ctx = _replay_ctx(model, target, vocab)
    res = OSNIPProjectionVMA(sorted_l1=True, top_k=5)._run(ctx)
    assert np.array_equal(res.recovered_token_ids, ids)


def test_osnip_requires_model_replay_capability():
    # C1: OSNIP is autoregressive replay, so it genuinely needs MODEL_REPLAY;
    # on a scheme that exposes only hidden states it is honestly not_applicable.
    vocab, d, n = 20, 8, 5
    model = _ToyCausalModel(vocab, d, seed=0)
    ids = np.random.default_rng(4).integers(0, vocab, size=n)
    target = model.hidden(ids)
    ctx = AttackContext(
        granted=frozenset({Capability.HIDDEN_STATES}),
        descriptor={"vocab_size": vocab},
        _obfuscated_hidden_states=target,
    )
    res = OSNIPProjectionVMA(top_k=5).run(ctx)  # base.run applies the capability gate
    assert res.status == AttackStatus.NOT_APPLICABLE
    assert Capability.MODEL_REPLAY.value in res.status_reason
