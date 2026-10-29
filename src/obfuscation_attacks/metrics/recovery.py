"""Recovery metrics with explicit counting rules (Goal section 7).

Counting conventions (fixed here so every attack reports comparably):

* **Statistical unit.** ``asr_at_k`` counts per *token position*;
  ``mapping_asr_at_k`` counts per *mapping entry* (cipher symbol). The unit is
  named in each function so a caller never mixes them.
* **Special tokens.** Positions flagged by ``special_mask`` (BOS/EOS/pad/...) are
  excluded from every metric before counting.
* **Clean vs Total.** ``total_and_clean_asr`` reports Total over all non-special
  positions and Clean over positions additionally flagged as *content*
  (non stop-word), matching the OSNIP Total/Clean ASR split.
* **Duplicate tokens.** ``token_prf1`` is multiset (bag) based: a token type that
  appears ``a`` times in the prediction and ``b`` times in the truth contributes
  ``min(a, b)`` true positives. Position is ignored (set/bag overlap).
* **Position alignment.** ``ttrsr`` and ``exact_sequence_match`` are positional:
  they compare index by index, so recovering the right tokens in the wrong order
  does not count.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def _counted_positions(n: int, special_mask: Optional[Sequence[bool]]) -> np.ndarray:
    idx = np.arange(n)
    if special_mask is None:
        return idx
    mask = np.asarray(special_mask, dtype=bool)
    if mask.shape[0] != n:
        raise ValueError(f"special_mask length {mask.shape[0]} != sequence length {n}")
    return idx[~mask]


def asr_at_k(
    candidates: Sequence[Sequence[int]],
    truth: Sequence[int],
    k: int,
    *,
    special_mask: Optional[Sequence[bool]] = None,
) -> float:
    r"""Top-k attack success rate, counted per token position.

    ``ASR@k = (1/N) * sum_i 1[ truth_i in candidates_i[:k] ]`` over counted
    positions N (all non-special positions).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    truth = list(truth)
    if len(candidates) != len(truth):
        raise ValueError(
            f"candidates ({len(candidates)}) and truth ({len(truth)}) length mismatch"
        )
    positions = _counted_positions(len(truth), special_mask)
    if positions.size == 0:
        return 0.0
    hits = 0
    for i in positions:
        topk = list(candidates[i])[:k]
        if truth[i] in topk:
            hits += 1
    return hits / positions.size


def mapping_asr_at_k(
    mapping_candidates: Mapping[int, Sequence[int]],
    true_mapping: Mapping[int, int],
    k: int,
    *,
    eval_keys: Optional[Iterable[int]] = None,
) -> float:
    r"""Top-k success rate for a recovered substitution map, per mapping entry.

    ``ASR@k = (1/|C|) * sum_c 1[ pi*(c) in S_k(c) ]`` where ``c`` ranges over the
    evaluated cipher symbols and ``S_k(c)`` is the attacker's top-k plaintext
    candidates for ``c``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    keys = list(eval_keys) if eval_keys is not None else list(true_mapping.keys())
    if not keys:
        return 0.0
    hits = 0
    for c in keys:
        if c not in true_mapping:
            continue
        cand = list(mapping_candidates.get(c, []))[:k]
        if true_mapping[c] in cand:
            hits += 1
    return hits / len(keys)


def token_prf1(
    pred_ids: Sequence[int],
    truth_ids: Sequence[int],
    *,
    special_ids: Iterable[int] = (),
) -> PRF1:
    """Multiset token precision/recall/F1 (position independent).

    Special token ids are removed from both bags before counting. Duplicate
    tokens contribute ``min(pred_count, truth_count)`` true positives.
    """
    special = set(special_ids)
    pred_bag = Counter(t for t in pred_ids if t not in special)
    truth_bag = Counter(t for t in truth_ids if t not in special)
    tp = int(sum((pred_bag & truth_bag).values()))
    fp = int(sum(pred_bag.values()) - tp)
    fn = int(sum(truth_bag.values()) - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return PRF1(precision, recall, f1, tp, fp, fn)


def exact_sequence_match(
    pred_ids: Sequence[int],
    truth_ids: Sequence[int],
    *,
    special_ids: Iterable[int] = (),
) -> bool:
    """True iff the two id sequences are identical after dropping special ids.

    Positional and order-sensitive: same tokens in a different order do not match.
    """
    special = set(special_ids)
    p = [t for t in pred_ids if t not in special]
    g = [t for t in truth_ids if t not in special]
    return p == g


def ttrsr(
    pred_ids: Sequence[int],
    truth_ids: Sequence[int],
    *,
    special_mask: Optional[Sequence[bool]] = None,
) -> float:
    """Text Token Recovery Success Ratio - positional token recovery.

    Proportion of original (non-special) token positions where the prediction
    equals the truth. If the prediction is shorter than the truth, the missing
    tail counts as unrecovered.
    """
    truth = list(truth_ids)
    pred = list(pred_ids)
    positions = _counted_positions(len(truth), special_mask)
    if positions.size == 0:
        return 0.0
    hits = 0
    for i in positions:
        if i < len(pred) and pred[i] == truth[i]:
            hits += 1
    return hits / positions.size


def total_and_clean_asr(
    candidates: Sequence[Sequence[int]],
    truth: Sequence[int],
    k: int,
    *,
    special_mask: Optional[Sequence[bool]] = None,
    content_mask: Optional[Sequence[bool]] = None,
) -> tuple[float, float]:
    """Return (Total ASR@k, Clean ASR@k).

    Total is counted over all non-special positions. Clean is counted over
    positions that are additionally flagged as content (non stop-word). When no
    ``content_mask`` is given, Clean equals Total.
    """
    total = asr_at_k(candidates, truth, k, special_mask=special_mask)
    if content_mask is None:
        return total, total
    content = np.asarray(content_mask, dtype=bool)
    # Clean ASR counts only positions that are content AND non-special. Build the
    # exclusion mask (positions to skip) accordingly.
    if special_mask is not None:
        exclude = ~(content & ~np.asarray(special_mask, dtype=bool))
    else:
        exclude = ~content
    clean = asr_at_k(candidates, truth, k, special_mask=exclude)
    return total, clean
