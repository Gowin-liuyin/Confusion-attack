"""Unit tests for the SDA encodings and substitution-synthesis helpers.

These pin the two DISTINCT encodings (recurrence vs frequency-rank), the
INJECTIVE frequency encoding (distinct symbols -> distinct codes, Fig 5) with its
explicit tie *reporting*, the homophonic substitution helper used to synthesise
2023 training data, the 2021 experimental-condition generators (cipher length,
space handling, transcription noise) scored with SER/TER, and the structural
consistency helper. They touch no neural model, so they are pure/fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from obfuscation_attacks.attacks.sda.encodings import (
    allocate_homophones,
    apply_substitution,
    apply_transcription_noise,
    crop_to_length,
    enforce_substitution_consistency,
    frequency_rank_encode,
    frequency_rank_info,
    handle_spaces,
    make_substitution_key,
    recurrence_encode,
    remove_spaces,
)
from obfuscation_attacks.metrics import symbol_error_rate

pytestmark = pytest.mark.unit


def _ids(text: str, mapping: dict) -> list[int]:
    """Render a string to integer token ids via an explicit symbol map."""
    return [mapping[c] for c in text]


def _char_ter(pred, ref) -> float:
    """Character-level Translation Edit Rate = edit_distance / len(reference).

    Length-aware edit rate (insert/delete/substitute), the metric the 2021 paper
    uses for the length-CHANGING conditions (space restoration, insertion and
    deletion noise). ``symbol_error_rate`` is the position-aligned special case
    for equal-length outputs.
    """
    p = list(pred)
    r = list(ref)
    m, n = len(p), len(r)
    if n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if p[i - 1] == r[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n] / n


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
    # B is most frequent -> code 1; A -> code 2; C -> code 3
    assert freq == [2, 2, 1, 1, 1, 3]
    assert rec != freq


def test_frequency_rank_one_is_most_frequent():
    tokens = [9, 9, 9, 4, 4, 1]  # 9 most frequent
    info = frequency_rank_info(tokens)
    assert info.symbol_to_code[9] == 1
    assert info.symbol_to_code[4] == 2
    assert info.symbol_to_code[1] == 3
    assert not info.has_ties


# ---------------------------------------------------------------------------
# (c) INJECTIVE frequency encoding: distinct symbols -> distinct codes (Fig 5).
#     The raw ranking still REPORTS ties, but the encoding never collapses them.
# ---------------------------------------------------------------------------
def test_frequency_rank_encode_assigns_distinct_codes_per_symbol():
    # A=2, B=2 (a tie in the RAW rank) and C=1. The competition rank collapses A
    # and B to one rank, which is reported by `info`; but the ENCODING must stay
    # injective: A and B get DISTINCT dense codes (count desc, first-appearance).
    tokens = [0, 0, 1, 1, 2]
    codes = frequency_rank_encode(tokens)
    assert codes == [1, 1, 2, 2, 3]          # dense + distinct; no skipped code
    assert len(set(codes)) == 3              # three symbols -> three codes
    # 'aa' vs 'ab' must NOT collapse to one code sequence (the original bug)
    assert frequency_rank_encode([0, 0]) == [1, 1]
    assert frequency_rank_encode([0, 1]) == [1, 2]
    assert frequency_rank_encode([0, 1]) != frequency_rank_encode([0, 0])
    # the raw ranking still REPORTS the tie (competition "1224" ranking) without
    # letting it collapse the injective encoding the model consumes
    info = frequency_rank_info(tokens)
    assert info.has_ties and info.tied_ranks == [1]
    assert info.symbol_to_rank[0] == info.symbol_to_rank[1] == 1  # tie exposed
    assert info.symbol_to_rank[2] == 3                            # competition skip
    assert info.codes == codes                                    # injective codes carried too
    assert info.symbol_to_code[0] != info.symbol_to_code[1]       # but encoding is injective


def test_frequency_rank_encode_matches_fig5_equal_count_symbols_get_distinct_codes():
    # Paper Fig 5: j, m, r each occur with the SAME count yet receive DISTINCT
    # codes, ordered by first appearance. Non-sequential ids prove the codes
    # depend on count+first-appearance, not on the id values.
    j, m, r = 10, 11, 12
    text = [j, m, r, j, m, r, j, m, r, j, m, r]  # each appears 4x (a 3-way tie)
    codes = frequency_rank_encode(text)
    assert codes[:3] == [1, 2, 3]       # j -> 1, m -> 2, r -> 3 (first appearance)
    assert len(set(codes)) == 3         # the three tied symbols stay distinct
    info = frequency_rank_info(text)
    assert info.has_ties                # the equal counts are still reported
    assert len(info.tie_groups[0]) == 3  # j, m, r share a raw rank


def test_frequency_rank_ties_are_reported_by_info():
    mapping = {"A": 0, "B": 1}
    tokens = _ids("AABB", mapping)  # A=2, B=2 -> a genuine tie in the raw rank
    info = frequency_rank_info(tokens)
    # the raw competition rank exposes the tie (both share rank 1) ...
    assert info.symbol_to_rank[0] == info.symbol_to_rank[1] == 1
    assert info.ranks == [1, 1, 1, 1]
    assert info.has_ties
    assert info.tied_ranks == [1]
    assert [0, 1] in info.tie_groups
    # ... while the injective encoding still gives A and B DISTINCT codes
    assert info.codes == [1, 1, 2, 2]
    assert frequency_rank_encode(tokens) == [1, 1, 2, 2]


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
    # the INJECTIVE 1:1 encoding is key-invariant: distinct plaintext symbols map
    # to distinct cipher symbols with the same counts and first-appearance order,
    # so recurrence(cipher) == recurrence(plain) and freq(cipher) == freq(plain).
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


# ---------------------------------------------------------------------------
# (e) 2021 experimental conditions (Sec 5.1 length / 5.2 spaces / 5.4 noise)
#     scored with Symbol Error Rate; length-changing ones noted with TER.
# ---------------------------------------------------------------------------
def test_crop_to_length_takes_a_window_of_exact_length():
    seq = np.arange(100, dtype=int)
    # the paper studies lengths 16/32/64/128/256; a window has EXACTLY that length
    assert np.array_equal(crop_to_length(seq, 16, from_start=True), np.arange(16))
    rng = np.random.default_rng(0)
    w = crop_to_length(seq, 32, rng=rng)
    assert len(w) == 32 and np.array_equal(w, np.arange(int(w[0]), int(w[0]) + 32))
    # sequences shorter than the target length are returned unchanged
    short = np.array([1, 2, 3], dtype=int)
    assert np.array_equal(crop_to_length(short, 16), short)


def test_handle_spaces_enciphered_keeps_separator_removed_drops_it():
    seq = np.array([0, 1, 2, 0, 3, 0], dtype=int)  # 0 == the space/separator symbol
    # enciphered spaces: keep the separator so it is frequency-encoded like any
    # other symbol (here it is the MOST frequent -> code 1). Length preserved.
    assert np.array_equal(handle_spaces(seq, mode="enciphered", space_id=0), seq)
    assert frequency_rank_encode(seq)[0] == 1
    # no-space cipher: the separator is dropped on both sides (length shrinks).
    removed = handle_spaces(seq, mode="removed", space_id=0)
    assert np.array_equal(removed, np.array([1, 2, 3]))
    assert np.array_equal(remove_spaces(seq, 0), removed)
    with pytest.raises(ValueError):
        handle_spaces(seq, mode="bogus")


def test_transcription_substitution_noise_preserves_length_and_is_ser_scored():
    rng = np.random.default_rng(0)
    clean = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=int)
    # substitution-only noise keeps the length, so it is scored with SER
    full = apply_transcription_noise(clean, rate=1.0, rng=rng, kinds=("sub",), alphabet=range(10))
    assert len(full) == len(clean)
    assert symbol_error_rate(list(full), list(clean)) == 1.0  # every position substituted
    # rate 0 is a no-op
    none = apply_transcription_noise(clean, rate=0.0, rng=rng, kinds=("sub",), alphabet=range(10))
    assert np.array_equal(none, clean)
    assert symbol_error_rate(list(none), list(clean)) == 0.0
    # a partial rate corrupts only some positions
    partial = apply_transcription_noise(
        clean, rate=0.5, rng=np.random.default_rng(1), kinds=("sub",), alphabet=range(10)
    )
    assert len(partial) == len(clean)
    assert 0.0 < symbol_error_rate(list(partial), list(clean)) <= 1.0


def test_transcription_insert_delete_noise_change_length_scored_with_ter():
    clean = np.array([1, 2, 3, 4, 5], dtype=int)
    # insertion grows the sequence (each symbol kept, one inserted after it)
    ins = apply_transcription_noise(
        clean, rate=1.0, rng=np.random.default_rng(0), kinds=("ins",), alphabet=range(10)
    )
    assert len(ins) == 2 * len(clean)         # length CHANGES -> score with TER, not SER
    assert _char_ter(ins, clean) > 0.0
    # deletion shrinks the sequence (rate 1.0 drops every symbol)
    dele = apply_transcription_noise(
        clean, rate=1.0, rng=np.random.default_rng(0), kinds=("del",), alphabet=range(10)
    )
    assert len(dele) == 0
    assert _char_ter(dele, clean) == 1.0      # all reference chars deleted
    # mixed noise: both SER (position-aligned) and TER (edit) are defined
    mixed = apply_transcription_noise(
        clean, rate=0.4, rng=np.random.default_rng(2), kinds=("sub", "ins", "del"), alphabet=range(10)
    )
    assert _char_ter(mixed, clean) >= 0.0


# ---------------------------------------------------------------------------
# (f) structural consistency: a deterministic cipher->plain map (1:1 or
#     homophonic) is a FUNCTION of the cipher symbol.
# ---------------------------------------------------------------------------
def test_enforce_substitution_consistency_is_a_function_of_cipher_symbol():
    cipher = [5, 7, 5, 7, 9, 5]
    decoded = [1, 2, 3, 2, 8, 1]  # symbol 5 decoded as 1, 3, 1 -> majority vote 1
    consistent, mapping = enforce_substitution_consistency(cipher, decoded)
    assert mapping == {5: 1, 7: 2, 9: 8}
    assert consistent == [1, 2, 1, 2, 8, 1]
    # equal cipher symbols ALWAYS decode to equal plaintext
    for i in range(len(cipher)):
        for j in range(len(cipher)):
            if cipher[i] == cipher[j]:
                assert consistent[i] == consistent[j]
    with pytest.raises(ValueError):
        enforce_substitution_consistency([1, 2, 3], [1, 2])  # length mismatch
