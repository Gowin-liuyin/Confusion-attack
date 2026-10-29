"""SDA symbol encodings and substitution-cipher synthesis helpers.

This module holds the two DISTINCT per-symbol encodings used by the two SDA
paper lineages, plus the key-generation / enciphering helpers an attacker uses
to synthesise training pairs. The two encodings must never be conflated.

Recurrence encoding (Kambhatla et al. 2023, *Decipherment as Regression*;
FORMULA_LEDGER SDA-2023-ENC):

    r(c_i) = k    <=>    c_i is the k-th DISTINCT symbol by FIRST appearance

Repeats reuse the number assigned on first appearance, so the encoding is a
function of *first-appearance order*, NOT of frequency. Example on the symbol
stream ``A B A C A D``::

    recurrence_encode("ABACAD") == [1, 2, 1, 3, 1, 4]

Frequency-rank encoding (Aldarrab & May 2021, *Can Seq2Seq Models Crack
Substitution Ciphers?*; FORMULA_LEDGER SDA-2021-ENC):

    if n_C(c_1) >= n_C(c_2) >= ...  then  r_F(c_j) = j        (no-tie case)

rank 1 == most frequent symbol. The no-tie formula is unambiguous; ties are NOT
broken by an arbitrary order. :func:`frequency_rank_encode` uses *competition
ranking* (standard "1224" ranking): every symbol with the same count receives
the same rank, equal to ``1 + (number of symbols with a strictly greater
count)``. :func:`frequency_rank_info` additionally reports exactly which ranks
are shared, so the ambiguity is exposed rather than silently resolved.

Both encodings are 1-indexed (smallest code is 1). This shared 1-indexing is an
implementation choice recorded for reproducibility (the 2021 paper illustrates
0-indexed ranks); it keeps the two lineages on one integer convention without
changing any relative structure.

Substitution synthesis
-----------------------
The attacker owns only *public* plaintext (``reference_corpus``). It fabricates
training pairs by drawing random substitution keys and enciphering that public
text. :func:`make_substitution_key` builds either a 1:1 key (a deterministic
permutation, matching the AloePri token-permutation target and the 2021 setting)
or a homophonic key (one plaintext symbol -> several cipher symbols, allocated
by the 2023 paper's Algorithm 1). :func:`apply_substitution` enciphers a
plaintext id sequence under such a key.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# small normalisation helpers
# ---------------------------------------------------------------------------
def _norm(tok: Hashable) -> Hashable:
    """Normalise a numpy scalar to a plain Python scalar; leave others as-is."""
    if isinstance(tok, np.generic):
        return tok.item()
    return tok


def _as_symbol_list(tokens: Iterable[Hashable]) -> list:
    if isinstance(tokens, np.ndarray):
        return tokens.tolist()
    return list(tokens)


def _sort_key(sym: Hashable):
    """Deterministic, type-safe ordering key for reporting tie groups."""
    if isinstance(sym, (int, np.integer)):
        return (0, float(sym), "")
    if isinstance(sym, float):
        return (0, sym, "")
    return (1, 0.0, repr(sym))


# ---------------------------------------------------------------------------
# 2023: recurrence encoding
# ---------------------------------------------------------------------------
def recurrence_encode(tokens: Iterable[Hashable]) -> list[int]:
    """Return the 2023 recurrence encoding of ``tokens``.

    ``r(c_i) = k`` iff ``c_i`` is the k-th distinct symbol by first appearance;
    repeats reuse the first assigned number. 1-indexed.
    """
    seen: dict[Hashable, int] = {}
    out: list[int] = []
    for tok in _as_symbol_list(tokens):
        key = _norm(tok)
        if key not in seen:
            seen[key] = len(seen) + 1
        out.append(seen[key])
    return out


# ---------------------------------------------------------------------------
# 2021: frequency-rank encoding (+ explicit tie reporting)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrequencyRankInfo:
    """Full frequency-rank result with explicit tie information.

    Attributes
    ----------
    ranks:
        Per-position competition rank (1 == most frequent).
    symbol_to_rank:
        Mapping from distinct symbol to its competition rank.
    symbol_to_count:
        Mapping from distinct symbol to its raw count.
    tie_groups:
        Groups of >= 2 symbols sharing the same count (hence the same rank),
        each group sorted deterministically, groups ordered most-frequent first.
    tied_ranks:
        Sorted list of rank values shared by more than one symbol.
    """

    ranks: list[int]
    symbol_to_rank: dict
    symbol_to_count: dict
    tie_groups: list[list]
    tied_ranks: list[int]

    @property
    def has_ties(self) -> bool:
        return len(self.tied_ranks) > 0


def frequency_rank_info(tokens: Iterable[Hashable]) -> FrequencyRankInfo:
    """Compute competition frequency ranks and report tie ambiguity.

    Competition ranking: ``rank(s) = 1 + |{s' : count[s'] > count[s]}|``. Equal
    counts therefore yield an *equal* rank (no invented order), and the ranks
    shared by multiple symbols are returned in ``tied_ranks`` / ``tie_groups``.
    """
    syms = [_norm(t) for t in _as_symbol_list(tokens)]
    counts = Counter(syms)

    symbol_to_rank: dict = {}
    for s, c in counts.items():
        greater = sum(1 for other_c in counts.values() if other_c > c)
        symbol_to_rank[s] = greater + 1

    ranks = [symbol_to_rank[s] for s in syms]

    by_count: dict[int, list] = {}
    for s, c in counts.items():
        by_count.setdefault(c, []).append(s)

    tie_groups: list[list] = []
    tied_ranks: set[int] = set()
    for c, group in by_count.items():
        if len(group) >= 2:
            tie_groups.append(sorted(group, key=_sort_key))
            tied_ranks.add(symbol_to_rank[group[0]])
    tie_groups.sort(key=lambda g: -counts[g[0]])

    return FrequencyRankInfo(
        ranks=ranks,
        symbol_to_rank=dict(symbol_to_rank),
        symbol_to_count=dict(counts),
        tie_groups=tie_groups,
        tied_ranks=sorted(tied_ranks),
    )


def frequency_rank_encode(tokens: Iterable[Hashable]) -> list[int]:
    """Return the 2021 competition frequency-rank encoding (1 == most frequent).

    Equally-frequent symbols share a rank; see :func:`frequency_rank_info` to
    recover which ranks are tied instead of assuming an arbitrary tiebreak.
    """
    return frequency_rank_info(tokens).ranks


# ---------------------------------------------------------------------------
# substitution key synthesis (1:1 and homophonic)
# ---------------------------------------------------------------------------
@dataclass
class SubstitutionKey:
    """A substitution cipher key over integer token ids.

    ``plain_to_cipher`` maps every covered plaintext id to one (1:1) or several
    (homophonic) cipher ids. ``cipher_to_plain`` is the inverse (well defined
    because distinct plaintext symbols receive disjoint cipher ids).
    """

    homophonic: bool
    plain_to_cipher: dict[int, list[int]]
    cipher_to_plain: dict[int, int]

    def is_one_to_one(self) -> bool:
        return all(len(v) == 1 for v in self.plain_to_cipher.values())

    @property
    def num_cipher_symbols(self) -> int:
        return sum(len(v) for v in self.plain_to_cipher.values())


def allocate_homophones(
    plain_counts: dict[int, int], approx_num_symbols: int
) -> dict[int, int]:
    """2023 Algorithm 1: number of cipher symbols to allocate per plaintext char.

    ``char_weight = freq / n``; ``wsym = int(char_weight * #sym)``;
    ``num_sym = 1 if wsym == 0 else wsym``. Frequent characters thus receive
    more homophones, flattening the cipher-symbol frequency distribution.
    """
    n = sum(plain_counts.values())
    if n <= 0:
        raise ValueError("plain_counts must contain at least one symbol")
    sym_per_char: dict[int, int] = {}
    for char, freq in plain_counts.items():
        char_weight = freq / n
        wsym = int(char_weight * approx_num_symbols)
        sym_per_char[int(char)] = 1 if wsym == 0 else wsym
    return sym_per_char


def make_substitution_key(
    plain_symbols: Iterable[int],
    *,
    rng: np.random.Generator,
    homophonic: bool = False,
    cipher_alphabet_size: Optional[int] = None,
    plain_counts: Optional[dict[int, int]] = None,
    approx_num_symbols: Optional[int] = None,
) -> SubstitutionKey:
    """Build a random substitution key over the given plaintext symbol ids.

    1:1 (``homophonic=False``): sample a random injection into a cipher alphabet
    (a deterministic permutation). Homophonic (``homophonic=True``): allocate a
    disjoint set of cipher ids per plaintext symbol according to
    :func:`allocate_homophones`, requiring ``plain_counts``.
    """
    symbols = sorted({int(s) for s in plain_symbols})
    if not symbols:
        raise ValueError("plain_symbols is empty")

    if not homophonic:
        size = cipher_alphabet_size if cipher_alphabet_size is not None else (symbols[-1] + 1)
        if size < len(symbols):
            raise ValueError(
                f"cipher_alphabet_size={size} too small for {len(symbols)} symbols"
            )
        perm = rng.permutation(size)
        plain_to_cipher: dict[int, list[int]] = {}
        cipher_to_plain: dict[int, int] = {}
        for i, p in enumerate(symbols):
            c = int(perm[i])
            plain_to_cipher[p] = [c]
            cipher_to_plain[c] = p
        return SubstitutionKey(False, plain_to_cipher, cipher_to_plain)

    if plain_counts is None:
        raise ValueError("homophonic key requires plain_counts for Algorithm 1")
    # ensure every symbol has a count entry (unseen -> 1 so it is still covered)
    counts = {int(s): int(plain_counts.get(s, 1)) for s in symbols}
    if approx_num_symbols is None:
        approx_num_symbols = 2 * len(symbols)
    sym_per_char = allocate_homophones(counts, approx_num_symbols)
    total = sum(sym_per_char[s] for s in symbols)
    pool_size = cipher_alphabet_size if cipher_alphabet_size is not None else (total + len(symbols))
    if pool_size < total:
        raise ValueError(
            f"cipher_alphabet_size={pool_size} too small for {total} homophones"
        )
    pool = rng.permutation(pool_size).tolist()
    plain_to_cipher = {}
    cipher_to_plain = {}
    idx = 0
    for p in symbols:
        k = sym_per_char[p]
        ciphers = [int(pool[idx + j]) for j in range(k)]
        idx += k
        plain_to_cipher[p] = ciphers
        for c in ciphers:
            cipher_to_plain[c] = p
    return SubstitutionKey(True, plain_to_cipher, cipher_to_plain)


def apply_substitution(
    tokens: Iterable[int],
    key: SubstitutionKey,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Encipher a plaintext id sequence under ``key``.

    For a 1:1 key the mapping is deterministic. For a homophonic key each
    occurrence is replaced by one of the plaintext symbol's cipher ids, drawn
    uniformly at random (which flattens the cipher-symbol frequencies, as the
    2023 paper intends); ``rng`` is then required.
    """
    out: list[int] = []
    for tok in _as_symbol_list(tokens):
        p = int(_norm(tok))
        ciphers = key.plain_to_cipher.get(p)
        if ciphers is None:
            raise KeyError(f"plaintext symbol {p!r} is not covered by the key")
        if len(ciphers) == 1:
            out.append(int(ciphers[0]))
        else:
            if rng is None:
                raise ValueError("homophonic substitution requires an rng")
            out.append(int(ciphers[int(rng.integers(len(ciphers)))]))
    return np.asarray(out, dtype=int)


def counts_from_corpus(corpus: Iterable[Sequence[int]], vocab_size: int) -> dict[int, int]:
    """Aggregate per-symbol counts over a corpus of id sequences."""
    counts: dict[int, int] = {i: 0 for i in range(vocab_size)}
    for seq in corpus:
        for tok in _as_symbol_list(seq):
            t = int(_norm(tok))
            if 0 <= t < vocab_size:
                counts[t] += 1
    return counts


__all__ = [
    "recurrence_encode",
    "frequency_rank_encode",
    "frequency_rank_info",
    "FrequencyRankInfo",
    "SubstitutionKey",
    "make_substitution_key",
    "apply_substitution",
    "allocate_homophones",
    "counts_from_corpus",
]
