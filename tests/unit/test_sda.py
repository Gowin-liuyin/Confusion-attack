"""Unit tests for the SDA encodings and substitution-synthesis helpers.

These pin the two DISTINCT encodings (recurrence vs frequency-rank), the
explicit tie handling of the 2021 frequency ranking, and the homophonic
substitution helper used to synthesise 2023 training data. They touch no neural
model, so they are pure/fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.sda.encodings import (
    allocate_homophones,
    apply_substitution,
    frequency_rank_encode,
    frequency_rank_info,
    make_substitution_key,
    recurrence_encode,
)

pytestmark = pytest.mark.unit


def _ids(text: str, mapping: dict) -> list[int]:
    """Render a string to integer token ids via an explicit symbol map."""
    return [mapping[c] for c in text]


# ---------------------------------------------------------------------------
# (a) recurrence encoding exactness
# ---------------------------------------------------------------------------
def test_recurrence_encode_abacad():
    # Use non-trivial, non-sequential ids to prove the encoding depends only on
    # first-appearance order, not on the id values themselves.
    mapping = {"A": 5, "B": 9, "C": 2, "D": 7}
    tokens = _ids("ABACAD", mapping)  # [5, 9, 5, 2, 5, 7]
    assert recurrence_encode(tokens) == [1, 2, 1, 3, 1, 4]


def test_recurrence_encode_is_1_indexed_and_reuses_first_number():
    assert recurrence_encode([42]) == [1]
    assert recurrence_encode([7, 7, 7]) == [1, 1, 1]
    assert recurrence_encode([]) == []
    # numpy array input is accepted
    assert recurrence_encode(np.array([3, 3, 8, 3, 8])) == [1, 1, 2, 1, 2]


# ---------------------------------------------------------------------------
# (b) frequency-rank differs from recurrence on a clear (tie-free) example
# ---------------------------------------------------------------------------
def test_frequency_rank_differs_from_recurrence():
    mapping = {"A": 0, "B": 1, "C": 2}
    tokens = _ids("AABBBC", mapping)  # counts: A=2, B=3, C=1  (all distinct)
    rec = recurrence_encode(tokens)
    freq = frequency_rank_encode(tokens)
    assert rec == [1, 1, 2, 2, 2, 3]
    # B is most frequent -> rank 1; A -> rank 2; C -> rank 3
    assert freq == [2, 2, 1, 1, 1, 3]
    assert rec != freq


def test_frequency_rank_one_is_most_frequent():
    tokens = [9, 9, 9, 4, 4, 1]  # 9 most frequent
    info = frequency_rank_info(tokens)
    assert info.symbol_to_rank[9] == 1
    assert info.symbol_to_rank[4] == 2
    assert info.symbol_to_rank[1] == 3
    assert not info.has_ties


# ---------------------------------------------------------------------------
# (c) ties are flagged, never resolved by an invented order
# ---------------------------------------------------------------------------
def test_frequency_rank_ties_are_flagged_not_broken():
    mapping = {"A": 0, "B": 1}
    tokens = _ids("AABB", mapping)  # A=2, B=2 -> a genuine tie
    info = frequency_rank_info(tokens)
    # both tied symbols share the SAME rank (no arbitrary A<B ordering)
    assert info.symbol_to_rank[0] == info.symbol_to_rank[1] == 1
    assert info.ranks == [1, 1, 1, 1]
    assert info.has_ties
    assert info.tied_ranks == [1]
    assert [0, 1] in info.tie_groups


def test_frequency_rank_competition_ranking_skips_after_ties():
    # A=2, B=2 (tie at rank 1), C=1 -> competition ranking gives C rank 3, not 2.
    tokens = [0, 0, 1, 1, 2]
    info = frequency_rank_info(tokens)
    assert info.symbol_to_rank[0] == 1
    assert info.symbol_to_rank[1] == 1
    assert info.symbol_to_rank[2] == 3  # rank 2 is skipped (shared by the tie)
    assert 2 not in info.ranks
    assert info.tied_ranks == [1]


# ---------------------------------------------------------------------------
# substitution key properties (1:1)
# ---------------------------------------------------------------------------
def test_one_to_one_key_is_a_bijection_and_inverts():
    rng = np.random.default_rng(0)
    V = 12
    key = make_substitution_key(range(V), rng=rng, homophonic=False, cipher_alphabet_size=V)
    assert key.is_one_to_one()
    plain = np.array([3, 1, 4, 1, 5, 9, 2, 6, 3], dtype=int) % V
    cipher = apply_substitution(plain, key)
    # invert with the key's own inverse map
    back = np.array([key.cipher_to_plain[int(c)] for c in cipher])
    assert np.array_equal(back, plain)
    # 1:1 encoding is key-invariant: recurrence(cipher) == recurrence(plain)
    assert recurrence_encode(cipher) == recurrence_encode(plain)
    assert frequency_rank_encode(cipher) == frequency_rank_encode(plain)


# ---------------------------------------------------------------------------
# (d) homophonic: one plaintext symbol -> several cipher symbols, recurrence
#     still captures repetition structure of the cipher stream
# ---------------------------------------------------------------------------
def test_homophonic_allocation_favours_frequent_symbols():
    # symbol 0 is far more frequent than 1 and 2
    plain_counts = {0: 80, 1: 15, 2: 5}
    sym_per_char = allocate_homophones(plain_counts, approx_num_symbols=20)
    assert sym_per_char[0] >= sym_per_char[1] >= sym_per_char[2]
    assert sym_per_char[0] >= 2  # frequent symbol gets several homophones
    assert all(v >= 1 for v in sym_per_char.values())


def test_homophonic_substitution_and_recurrence_structure():
    rng = np.random.default_rng(3)
    V = 4
    # strongly skewed plaintext so Algorithm 1 assigns symbol 0 several homophones
    plain = np.array([0, 1, 0, 2, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0], dtype=int)
    counts = {int(t): int(c) for t, c in zip(*np.unique(plain, return_counts=True))}
    key = make_substitution_key(
        range(V),
        rng=rng,
        homophonic=True,
        plain_counts={s: counts.get(s, 1) for s in range(V)},
        approx_num_symbols=12,
    )
    # one plaintext symbol maps to MULTIPLE cipher symbols
    assert not key.is_one_to_one()
    assert max(len(v) for v in key.plain_to_cipher.values()) >= 2
    assert len(key.plain_to_cipher[0]) >= 2  # the frequent symbol is homophonic

    cipher = apply_substitution(plain, key, rng=rng)
    assert cipher.shape == plain.shape

    # every cipher symbol is a legal homophone of its plaintext symbol
    for p, c in zip(plain.tolist(), cipher.tolist()):
        assert c in key.plain_to_cipher[p]

    codes = recurrence_encode(cipher)
    assert len(codes) == len(cipher)
    # recurrence captures EXACT repetition structure: two positions share a code
    # iff they carry the same cipher symbol (independent of homophony).
    for i in range(len(cipher)):
        for j in range(len(cipher)):
            assert (codes[i] == codes[j]) == (int(cipher[i]) == int(cipher[j]))

    # the frequent plaintext symbol 0 actually used >= 2 distinct cipher symbols
    # here, so its occurrences split across >= 2 distinct recurrence codes.
    zero_cipher_symbols = {int(cipher[i]) for i in range(len(plain)) if plain[i] == 0}
    assert len(zero_cipher_symbols) >= 2
    zero_codes = {codes[i] for i in range(len(plain)) if plain[i] == 0}
    assert len(zero_codes) == len(zero_cipher_symbols)
