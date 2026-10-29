"""Mathematical tests for TFMA (FORMULA_LEDGER TFMA-1/2/3)."""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.tfma.frequency_matching import (
    GlobalAssignmentTFMA,
    IndependentTFMA,
    _cipher_frequencies,
    _frequencies_from_counts,
)
from obfuscation_attacks.data.contracts import AttackContext, Capability
from obfuscation_attacks.metrics import mapping_asr_at_k

pytestmark = pytest.mark.mathematical


def _ctx(cipher, ref_counts, vocab):
    return AttackContext(
        granted=frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}),
        descriptor={"vocab_size": vocab},
        _ciphertext_token_ids=np.asarray(cipher),
        reference_token_counts=ref_counts,
    )


def test_frequencies_sum_to_one():
    f = _frequencies_from_counts({0: 3, 1: 1}, vocab_size=4)
    assert f.sum() == pytest.approx(1.0)
    assert f[0] == pytest.approx(0.75)
    fc, observed = _cipher_frequencies(np.array([2, 2, 3]), vocab_size=4)
    assert fc.sum() == pytest.approx(1.0)
    assert set(observed.tolist()) == {2, 3}


def test_permutation_preserves_counts():
    # deterministic permutation keeps per-symbol counts (only the labels change)
    rng = np.random.default_rng(0)
    plain = rng.integers(0, 5, size=200)
    perm = rng.permutation(5)
    cipher = perm[plain]
    for t in range(5):
        assert int((plain == t).sum()) == int((cipher == perm[t]).sum())


def test_independent_nearest_frequency_hand_computed():
    # cipher freqs: symbol 10 -> 3/4, symbol 11 -> 1/4
    cipher = [10, 10, 10, 11]
    # reference freqs: token 2 -> 0.8, token 3 -> 0.2 (others 0)
    ref = {2: 80, 3: 20}
    ctx = _ctx(cipher, ref, vocab=12)
    res = IndependentTFMA(top_k=2)._run(ctx)
    # 0.75 is closest to token 2 (0.8); 0.25 closest to token 3 (0.2)
    assert res.recovered_mapping[10] == 2
    assert res.recovered_mapping[11] == 3
    # per-symbol method: Top-1 candidate is exactly the argmin|df| winner
    assert res.mapping_candidates[10][0] == 2
    assert res.mapping_candidates[11][0] == 3


def test_global_assignment_is_a_bijection():
    cipher = [10, 10, 10, 11, 11, 12]
    ref = {0: 50, 1: 30, 2: 20}
    ctx = _ctx(cipher, ref, vocab=13)
    res = GlobalAssignmentTFMA(top_k=3)._run(ctx)
    assigned = list(res.recovered_mapping.values())
    assert len(assigned) == len(set(assigned))  # one-to-one assignment


def test_global_assignment_top1_candidate_is_the_assigned_token():
    # 3 cipher symbols with frequencies 0.5 / 0.3 / 0.2 over a length-10 text,
    # matching 3 prior tokens with the same frequencies.
    cipher = [10, 10, 10, 10, 10, 11, 11, 11, 12, 12]
    ref = {0: 50, 1: 30, 2: 20}  # f_R: 0.5 / 0.3 / 0.2
    ctx = _ctx(cipher, ref, vocab=13)
    res = GlobalAssignmentTFMA(top_k=13)._run(ctx)

    # (1) The Top-1 candidate of every symbol equals its Hungarian assignment.
    for c, assigned in res.recovered_mapping.items():
        assert res.mapping_candidates[c][0] == assigned

    # (2) mapping_asr@1 (candidate-based, as the scorer computes it) matches the
    #     assignment top-1 accuracy (recovered_mapping-based) for ANY ground
    #     truth. Use a partly wrong truth so the shared value is strictly in
    #     (0, 1) and the equality is actually exercised.
    true_map = {10: 1, 11: 0, 12: 2}  # 12 correct; 10/11 wrong vs the freq assignment
    observed = [c for c in res.recovered_mapping if c in true_map]
    top1_acc = sum(res.recovered_mapping[c] == true_map[c] for c in observed) / len(observed)
    asr1 = mapping_asr_at_k(res.mapping_candidates, true_map, 1, eval_keys=observed)
    assert asr1 == pytest.approx(top1_acc)
    assert 0.0 < asr1 < 1.0


def test_short_text_frequency_ties_are_flagged():
    cipher = [10, 11]  # both frequency 0.5 -> tie
    ref = {2: 1, 3: 1}
    ctx = _ctx(cipher, ref, vocab=12)
    res = IndependentTFMA(top_k=2)._run(ctx)
    assert res.diagnostics["symbols_with_frequency_ties"] >= 1
    # the tie rule is documented in diagnostics (lowest token id wins)
    assert "lowest token id" in res.diagnostics["tie_rule"]
