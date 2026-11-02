"""SDA symbol encodings and substitution-cipher synthesis helpers.

This module holds the two DISTINCT per-symbol encodings used by the two SDA
paper lineages, plus the key-generation / enciphering helpers an attacker uses
to synthesise training pairs, and the pure-Python helpers for the four 2021
experimental conditions (cipher length, space handling, transcription noise).
The two encodings must never be conflated.

Recurrence encoding (Kambhatla et al. 2023, *Decipherment as Regression*;
FORMULA_LEDGER SDA-2023-ENC):

    r(c_i) = k    <=>    c_i is the k-th DISTINCT symbol by FIRST appearance

Repeats reuse the number assigned on first appearance, so the encoding is a
function of *first-appearance order*, NOT of frequency. Example on the symbol
stream ``A B A C A D``::

    recurrence_encode("ABACAD") == [1, 2, 1, 3, 1, 4]

Frequency-rank encoding (Aldarrab & May 2021, *Can Seq2Seq Models Crack
Substitution Ciphers?*; FORMULA_LEDGER SDA-2021-ENC):

    sort the DISTINCT symbols by COUNT descending; the k-th symbol in that order
    gets code k. Equal counts are ordered by FIRST APPEARANCE in the text - a
    key-independent tiebreak, exactly ``Counter(x).most_common()`` order.

The encoding is therefore INJECTIVE on symbols: distinct symbols always receive
distinct codes. Paper Figure 5 maps 14 distinct cipher symbols to 14 distinct
integers (``j -> 0, m -> 1, r -> 2, ...`` even though j, m and r share the same
count), which preserves the substitution structure. Crucially ``[a, b]`` and
``[a, a]`` never collapse to the same code sequence. :func:`frequency_rank_encode`
returns these injective codes. :func:`frequency_rank_info` additionally reports
the RAW *competition* ranks (standard "1224" ranking, equal counts share a rank)
and exactly which symbols share a count, so equal-frequency ambiguity is exposed
for diagnostics -- but it never collapses the encoding the model consumes.

Both encodings are 1-indexed (smallest code is 1). The 2021 paper illustrates
0-indexed ranks; the 2021 seq2seq model reserves source id 0 for padding, so the
codes start at 1. This shared 1-indexing keeps the two lineages on one integer
convention without changing any relative structure.

Substitution synthesis
-----------------------
The attacker owns only *public* plaintext (``reference_corpus``). It fabricates
training pairs by drawing random substitution keys and enciphering that public
text. :func:`make_substitution_key` builds either a 1:1 key (a deterministic
permutation, matching the AloePri token-permutation target and the 2021 setting)
or a homophonic key (one plaintext symbol -> several cipher symbols, allocated
by the 2023 paper's Algorithm 1). :func:`apply_substitution` enciphers a
plaintext id sequence under such a key.

Experimental-condition helpers
------------------------------
:func:`crop_to_length` (2021 Sec 5.1 cipher-length study), :func:`handle_spaces`
(2021 Sec 5.2 enciphered vs removed spaces) and :func:`apply_transcription_noise`
(2021 Sec 5.4 substitution/insertion/deletion noise) transform a symbol stream
into the condition under test. Substitution noise and enciphered/kept spaces
preserve length (score with Symbol Error Rate); insertion/deletion noise and
removed spaces change length (score with Translation Edit Rate).

Structural consistency
----------------------
Under a deterministic substitution (1:1 or homophonic) the cipher->plaintext
relation is a FUNCTION of the cipher symbol. :func:`enforce_substitution_consistency`
collapses a per-position decode into that function by majority vote, so equal
cipher symbols always decipher to equal plaintext.
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
# 2021: frequency-rank encoding (injective) + explicit tie reporting
# ---------------------------------------------------------------------------
def _dense_frequency_codes(syms: Sequence[Hashable], counts: Counter) -> tuple[dict, list[int]]:
    """Injective 2021 codes: count desc, first-appearance tiebreak, 1-indexed.

    ``Counter.most_common()`` sorts by count descending and, for equal counts,
    preserves first-appearance order (the insertion order of ``Counter(syms)``),
    which is a key-independent tiebreak. The k-th symbol in that order gets code
    ``k`` (1 == most frequent), so distinct symbols get distinct codes.
    """
    symbol_to_code = {s: i + 1 for i, (s, _) in enumerate(counts.most_common())}
    codes = [symbol_to_code[s] for s in syms]
    return symbol_to_code, codes


@dataclass(frozen=True)
class FrequencyRankInfo:
    """Full frequency-rank result: the injective encoding plus tie diagnostics.

    Attributes
    ----------
    ranks:
        Per-position RAW competition rank (1 == most frequent; equal counts share
        a rank). Kept so ties are *visible*, not to feed the model.
    symbol_to_rank:
        Mapping from distinct symbol to its competition rank.
    symbol_to_count:
        Mapping from distinct symbol to its raw count.
    tie_groups:
        Groups of >= 2 symbols sharing the same count (hence the same competition
        rank), each group sorted deterministically, most-frequent group first.
    tied_ranks:
        Sorted list of competition-rank values shared by more than one symbol.
    codes:
        Per-position INJECTIVE encoding (:func:`frequency_rank_encode`); distinct
        symbols get distinct codes (count desc, first-appearance tiebreak).
    symbol_to_code:
        Mapping from distinct symbol to its injective code.
    """

    ranks: list[int]
    symbol_to_rank: dict
    symbol_to_count: dict
    tie_groups: list[list]
    tied_ranks: list[int]
    codes: list[int]
    symbol_to_code: dict

    @property
    def has_ties(self) -> bool:
        return len(self.tied_ranks) > 0


def frequency_rank_info(tokens: Iterable[Hashable]) -> FrequencyRankInfo:
    """Compute the injective frequency encoding and report tie ambiguity.

    The ``codes`` field is the injective encoding the 2021 model consumes
    (distinct symbols -> distinct codes). The ``ranks`` / ``tie_groups`` /
    ``tied_ranks`` fields expose the RAW competition ranking
    (``rank(s) = 1 + |{s' : count[s'] > count[s]}|``) so equal-count ambiguity is
    reported rather than silently assumed away.
    """
    syms = [_norm(t) for t in _as_symbol_list(tokens)]
    counts = Counter(syms)

    symbol_to_rank: dict = {}
    for s, c in counts.items():
        greater = sum(1 for other_c in counts.values() if other_c > c)
        symbol_to_rank[s] = greater + 1

    ranks = [symbol_to_rank[s] for s in syms]
    symbol_to_code, codes = _dense_frequency_codes(syms, counts)

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
        codes=codes,
        symbol_to_code=dict(symbol_to_code),
    )


def frequency_rank_encode(tokens: Iterable[Hashable]) -> list[int]:
    """Return the 2021 INJECTIVE frequency-rank encoding (1 == most frequent).

    Distinct symbols always receive distinct codes (Fig 5); equal counts are
    broken by first appearance (key-independent). Use :func:`frequency_rank_info`
    to recover which raw ranks were tied.
    """
    syms = [_norm(t) for t in _as_symbol_list(tokens)]
    _symbol_to_code, codes = _dense_frequency_codes(syms, Counter(syms))
    return codes


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

    Algorithm 1 (paper) reads ``char_weight = int(freq / n)``;
    ``wsym = int(char_weight * #sym)``; ``num_sym = 1 if wsym == 0 else wsym``.

    EXPERIMENTAL_COMPLETION: taken literally, ``freq <= n`` always, so
    ``int(freq / n)`` is 0 for every character except one that fills the entire
    sample, which would force ``num_sym = 1`` everywhere and allocate NO
    homophones at all -- contradicting the paper's own text and Figure 2, where
    frequent characters receive MANY homophones to flatten the distribution. We
    therefore read ``char_weight`` as the FRACTION ``freq / n`` (not truncated to
    int), which is the only reading under which Algorithm 1 behaves as described.
    The remaining ``int(...)`` and the ``wsym == 0 -> 1`` floor are kept verbatim.
    """
    n = sum(plain_counts.values())
    if n <= 0:
        raise ValueError("plain_counts must contain at least one symbol")
    sym_per_char: dict[int, int] = {}
    for char, freq in plain_counts.items():
        char_weight = freq / n  # EXPERIMENTAL_COMPLETION: fraction, not int(freq/n)
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


# ---------------------------------------------------------------------------
# 2021 experimental conditions (Aldarrab & May, Sec 5.1 / 5.2 / 5.4)
# ---------------------------------------------------------------------------
def crop_to_length(
    seq: Iterable[int],
    length: int,
    *,
    rng: Optional[np.random.Generator] = None,
    from_start: bool = False,
) -> np.ndarray:
    """2021 Sec 5.1 cipher-length condition: take a window of exactly ``length``.

    The paper studies ciphers of length 16/32/64/128/256. This returns a
    contiguous window of ``length`` symbols (random start when ``rng`` is given
    and ``from_start`` is False, else from the start). Sequences shorter than
    ``length`` are returned unchanged.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    s = [int(_norm(t)) for t in _as_symbol_list(seq)]
    n = len(s)
    if n <= length:
        return np.asarray(s, dtype=int)
    if from_start or rng is None:
        start = 0
    else:
        start = int(rng.integers(0, n - length + 1))
    return np.asarray(s[start : start + length], dtype=int)


def handle_spaces(
    tokens: Iterable[int],
    *,
    mode: str = "enciphered",
    space_id: int = 0,
) -> np.ndarray:
    """2021 Sec 5.2 space conditions.

    ``mode="enciphered"``: keep the space symbol so it is frequency-encoded like
    any other character (scenario 2). Length is unchanged -> SER-comparable.

    ``mode="removed"``: drop the space symbol from the stream (scenario 3,
    no-space cipher). Length shrinks; when the goal is to restore spaces on the
    target side (scenario 4) the output length differs from the input, so score
    with TER rather than SER.
    """
    s = [int(_norm(t)) for t in _as_symbol_list(tokens)]
    if mode == "enciphered":
        return np.asarray(s, dtype=int)
    if mode == "removed":
        return np.asarray([t for t in s if t != int(space_id)], dtype=int)
    raise ValueError(f"unknown space mode {mode!r}; use 'enciphered' or 'removed'")


def remove_spaces(tokens: Iterable[int], space_id: int = 0) -> np.ndarray:
    """Convenience wrapper for :func:`handle_spaces` with ``mode='removed'``."""
    return handle_spaces(tokens, mode="removed", space_id=space_id)


def apply_transcription_noise(
    symbols: Iterable[int],
    *,
    rate: float,
    rng: np.random.Generator,
    kinds: Sequence[str] = ("sub", "ins", "del"),
    alphabet: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """2021 Sec 5.4 transcription noise: random substitution/insertion/deletion.

    Each original symbol is independently corrupted with probability ``rate``;
    the corruption type is drawn uniformly from ``kinds`` (``"sub"``, ``"ins"``,
    ``"del"``). ``"sub"`` replaces a symbol with a DIFFERENT random symbol from
    ``alphabet`` and preserves length (score with SER). ``"ins"`` and ``"del"``
    change the length (score with TER, the character-level edit rate). When
    ``alphabet`` is omitted the distinct observed symbols are used.
    """
    if not (0.0 <= float(rate) <= 1.0):
        raise ValueError("rate must be in [0, 1]")
    s = [int(_norm(t)) for t in _as_symbol_list(symbols)]
    kinds = tuple(kinds)
    if not kinds:
        return np.asarray(s, dtype=int)
    alpha = [int(a) for a in (alphabet if alphabet is not None else sorted(set(s)))]
    if not alpha:
        return np.asarray(s, dtype=int)
    out: list[int] = []
    for sym in s:
        if float(rng.random()) < float(rate):
            kind = kinds[int(rng.integers(len(kinds)))]
            if kind == "del":
                continue  # drop this symbol (length shrinks)
            if kind == "ins":
                out.append(sym)  # keep the original ...
                out.append(int(alpha[int(rng.integers(len(alpha)))]))  # ... and insert
                continue
            if kind == "sub":
                choices = [a for a in alpha if a != sym] or [sym]
                out.append(int(choices[int(rng.integers(len(choices)))]))
                continue
            raise ValueError(f"unknown noise kind {kind!r}")
        out.append(sym)
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# structural consistency (deterministic cipher -> plaintext key)
# ---------------------------------------------------------------------------
def enforce_substitution_consistency(
    cipher_symbols: Iterable[int],
    decoded_ids: Iterable[int],
) -> tuple[list[int], dict[int, int]]:
    """Collapse a per-position decode into a deterministic cipher->plain map.

    Under any deterministic substitution (1:1 OR homophonic) the cipher->plain
    relation is a FUNCTION of the cipher symbol - every occurrence of a cipher
    symbol must decipher to one plaintext id. We recover that function by
    majority vote over the model's per-position outputs (ties broken by first
    occurrence) and re-emit the plaintext by applying it, which guarantees that
    equal cipher symbols decode to equal plaintext. Returns
    ``(consistent_sequence, cipher_id -> plain_id)``.
    """
    cipher = [int(_norm(t)) for t in _as_symbol_list(cipher_symbols)]
    dec = [int(_norm(t)) for t in _as_symbol_list(decoded_ids)]
    if len(cipher) != len(dec):
        raise ValueError(
            f"cipher length {len(cipher)} != decoded length {len(dec)}"
        )
    votes: dict[int, Counter] = {}
    for c, d in zip(cipher, dec):
        votes.setdefault(c, Counter())[d] += 1
    mapping = {c: ctr.most_common(1)[0][0] for c, ctr in votes.items()}
    consistent = [mapping[c] for c in cipher]
    return consistent, mapping


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
    "crop_to_length",
    "handle_spaces",
    "remove_spaces",
    "apply_transcription_noise",
    "enforce_substitution_consistency",
]
