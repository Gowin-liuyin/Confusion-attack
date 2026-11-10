"""Acceptance tests for real TFMA top-k candidates (F13)."""

from __future__ import annotations

import pytest

from obfuscation_attacks.evaluation.tfma_ranking import max_candidates, rank_candidates

pytestmark = pytest.mark.unit

REF = {token: 1000 - token for token in range(200)}
SYMBOLS = {100 + i: 20 - i for i in range(20)}


def test_top100_returns_100_distinct_valid_candidates():
    ranking = rank_candidates(SYMBOLS, REF, k=100)
    assert len(ranking) == len(SYMBOLS)
    for candidates in ranking.values():
        assert len(candidates) == 100
        assert len(set(candidates)) == 100
        assert all(0 <= token < 200 for token in candidates)
    assert max_candidates(ranking) == 100


def test_first_candidate_is_the_global_rank_mapping():
    ref = {5: 100, 7: 80, 9: 60}
    symbols = {100: 10, 200: 5}
    ranking = rank_candidates(symbols, ref, k=3)
    assert ranking[100][0] == 5
    assert ranking[200][0] == 7


def test_k_larger_than_the_reference_vocabulary_is_distinct_not_padded():
    ref = {1: 3, 2: 2, 3: 1}
    ranking = rank_candidates({10: 1}, ref, k=100)
    assert ranking[10] == [1, 2, 3]
    assert -1 not in ranking[10]


def test_context_reorders_candidates_but_keeps_them_valid():
    def context(symbol: int, candidate: int) -> float:
        return {9: 0.0, 5: 1.0, 7: 2.0}.get(candidate, 99.0)

    ranking = rank_candidates({100: 10, 200: 5}, {5: 100, 7: 80, 9: 60}, k=3, context=context)
    assert ranking[100][0] == 9
    assert set(ranking[100]) == {5, 7, 9}


def test_unknown_symbols_are_not_invented():
    ranking = rank_candidates({}, REF, k=10)
    assert ranking == {}
    assert max_candidates(ranking) == 0


def test_frequency_distance_not_rank_distance_decides_the_first_candidate():
    ref = {5: 100, 7: 60, 9: 10}
    ranking = rank_candidates({100: 3, 200: 4}, ref, k=3)
    # f_C(200)=4/7=0.571; |0.571-f_R(5)=0.588|=0.017 beats |0.571-0.353|=0.218,
    # although symbol 200 has rank 1 and token 7 has ref rank 1.
    assert ranking[200][0] == 5


def test_registered_matcher_and_stream_match_ties_and_unseen_tokens():
    import numpy as np
    from obfuscation_attacks.attacks.tfma.frequency_matching import IndependentTFMA
    from obfuscation_attacks.data.contracts import AttackContext, Capability

    counts = {0: 3, 3: 1}
    cipher = np.array([5, 5, 5, 7])
    ctx = AttackContext(granted=frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}),
                        descriptor={"vocab_size": 8}, _ciphertext_token_ids=cipher,
                        reference_token_counts=counts)
    result = IndependentTFMA(top_k=8).run(ctx)
    stream = rank_candidates({5: 3, 7: 1}, counts, k=8, vocab_size=8)
    assert result.mapping_candidates == stream
    assert stream[7][1:] == [1, 2, 4, 5, 6, 7, 0]


def test_cipher_symbol_renaming_does_not_change_nearest_plaintext_candidates():
    first = rank_candidates({8: 3, 9: 1}, {0: 3, 2: 1}, 6, vocab_size=6)
    second = rank_candidates({100: 3, 20: 1}, {0: 3, 2: 1}, 6, vocab_size=6)
    assert first[8] == second[100] and first[9] == second[20]


def test_global_assignment_matches_bruteforce_and_respects_memory_budget():
    import itertools
    import numpy as np
    from obfuscation_attacks.attacks.tfma.frequency_matching import GlobalAssignmentTFMA
    from obfuscation_attacks.data.contracts import AttackContext, AttackStatus, Capability

    ctx = AttackContext(granted=frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}),
                        descriptor={"vocab_size": 4}, _ciphertext_token_ids=np.array([1, 1, 1, 2]),
                        reference_token_counts={0: 3, 1: 2, 2: 1})
    result = GlobalAssignmentTFMA(top_k=4).run(ctx)
    assert result.status is AttackStatus.OK
    frequencies = np.array([.5, 1/3, 1/6, 0.])
    objective = lambda assignment: sum((x - frequencies[y]) ** 2 for x, y in zip((.75, .25), assignment))
    chosen = [result.recovered_mapping[1], result.recovered_mapping[2]]
    assert objective(chosen) == pytest.approx(min(map(objective, itertools.permutations(range(4), 2))))
    assert result.diagnostics["objective"] == "sum_squared_frequency_difference"
    blocked = GlobalAssignmentTFMA(memory_budget_bytes=1).run(ctx)
    assert blocked.status is AttackStatus.FAILED and "resource_blocked" in blocked.status_reason
