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


# ---------------------------------------------------------------------------
# Additional paper metrics (SDA: SER; AloePri: CosSim, BLEU-4)
# ---------------------------------------------------------------------------
def symbol_error_rate(
    pred_ids: Sequence[int],
    truth_ids: Sequence[int],
    *,
    special_mask: Optional[Sequence[bool]] = None,
) -> float:
    """Symbol Error Rate (SDA 2021/2023): fraction of positions decoded wrong.

    Positional and length-aware: a position counts as an error when the
    prediction is missing (shorter output) or differs from the truth. With equal
    lengths ``SER == 1 - ttrsr``. Special positions are excluded before counting.
    """
    truth = list(truth_ids)
    pred = list(pred_ids)
    positions = _counted_positions(len(truth), special_mask)
    if positions.size == 0:
        return 0.0
    errors = 0
    for i in positions:
        if i >= len(pred) or pred[i] != truth[i]:
            errors += 1
    return errors / positions.size


def recovered_cosine_similarity(
    pred_ids: Sequence[int],
    truth_ids: Sequence[int],
    embeddings: np.ndarray,
    *,
    special_mask: Optional[Sequence[bool]] = None,
) -> float:
    """Mean per-position cosine between recovered and true token embeddings.

    A token-level proxy for AloePri's Recovered-Text Cosine Similarity (CosSim).
    It is NOT a sentence-encoder similarity (recorded in docs/DEVIATIONS.md); it
    averages ``cos(E[pred_i], E[truth_i])`` over counted positions. A missing or
    out-of-range prediction contributes 0.
    """
    E = np.asarray(embeddings, dtype=float)
    vocab = E.shape[0]
    truth = list(truth_ids)
    pred = list(pred_ids)
    positions = _counted_positions(len(truth), special_mask)
    if positions.size == 0:
        return 0.0
    sims = []
    for i in positions:
        if i >= len(pred):
            sims.append(0.0)
            continue
        ti, pi = int(truth[i]), int(pred[i])
        if not (0 <= ti < vocab and 0 <= pi < vocab):
            sims.append(0.0)
            continue
        a, b = E[ti], E[pi]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        sims.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0)
    return float(np.mean(sims)) if sims else 0.0


def bleu4(pred_ids: Sequence[int], truth_ids: Sequence[int], *, max_n: int = 4) -> float:
    """Token-level sentence BLEU-4 with brevity penalty (AloePri BLEU-4).

    Computes clipped n-gram precisions for ``n = 1..max_n`` over integer token
    ids, their geometric mean, and the standard brevity penalty. Order-sensitive
    (n-grams); if any n-gram precision is 0 the score is 0 (no smoothing), as in
    the unsmoothed corpus-free definition.
    """
    import math

    pred = [int(t) for t in pred_ids]
    ref = [int(t) for t in truth_ids]
    if not pred or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ng = Counter(tuple(pred[i : i + n]) for i in range(len(pred) - n + 1))
        ref_ng = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        denom = sum(pred_ng.values())
        if denom == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(c, ref_ng.get(g, 0)) for g, c in pred_ng.items())
        precisions.append(overlap / denom)
    if min(precisions) <= 0.0:
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    bp = 1.0 if len(pred) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(pred)))
    return float(bp * geo)


# ---------------------------------------------------------------------------
# Plan P0-9 linkage, P0-10 PII, P1-8 output-side reconstruction
# (privacy_test_plan_20261106 §5.4 formulas (3), (5), (6))
# delta buckets follow §6.7: 1 / 2-3 / 4-15 / 16-63 / 64-127 / 128-511 / >=512
# ---------------------------------------------------------------------------
DELTA_BUCKETS: tuple[tuple[int, Optional[int]], ...] = (
    (1, 1),
    (2, 3),
    (4, 15),
    (16, 63),
    (64, 127),
    (128, 511),
    (512, None),
)


def delta_bucket(delta: int) -> str:
    """Bucket label for a request-index gap, e.g. ``"16-63"`` / ``"512+"``."""
    d = abs(int(delta))
    for lo, hi in DELTA_BUCKETS:
        if hi is None and d >= lo:
            return f"{lo}+"
        if hi is not None and lo <= d <= hi:
            return f"{lo}-{hi}" if lo != hi else f"{lo}"
    return f"{DELTA_BUCKETS[0][0]}"


def linkage_exact_rate(same_token: Sequence[bool], same_symbol: Sequence[bool]) -> Optional[float]:
    r"""Exact link rate over cross-request *same-token* pairs.

    ``LinkRate = #(same-token pairs that share a symbol) / #(cross-request
    same-token pairs)`` - plan §5.4 formula (6), exact variant. Returns ``None``
    when there is no pair to count (never 0%, which would read as "protected").
    """
    if len(same_token) != len(same_symbol):
        raise ValueError("same_token and same_symbol must align")
    n = sum(1 for flag in same_token if flag)
    if n == 0:
        return None
    hits = sum(1 for tok, sym in zip(same_token, same_symbol) if tok and sym)
    return hits / n


def linkage_tpr_fpr(scores: Sequence[float], same_token: Sequence[bool], threshold: float) -> dict:
    """TPR/FPR of the cosine-threshold linker at ``threshold`` (plan §5.4 (6)).

    TPR: fraction of same-token pairs scored ``>= threshold``.
    FPR: fraction of different-token pairs scored ``>= threshold``.
    Both are reported - reporting TPR alone overstates the attack.
    """
    if len(scores) != len(same_token):
        raise ValueError("scores and same_token must align")
    import math

    for raw in scores:
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"linkage_tpr_fpr: score {raw!r} is not numeric"
            ) from None
        if not math.isfinite(value):
            raise ValueError(
                "linkage_tpr_fpr: non-finite score (NaN/inf); refusing to count "
                "an undefined similarity as below threshold"
            )
    tp = fn = fp = tn = 0
    for score, is_same in zip(scores, same_token):
        hit = float(score) >= float(threshold)
        if is_same:
            tp += int(hit)
            fn += int(not hit)
        else:
            fp += int(hit)
            tn += int(not hit)
    tpr = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {"threshold": float(threshold), "tpr": tpr, "fpr": fpr, "tp": tp, "fp": fp,
            "fn": fn, "tn": tn}


def entity_recovered(
    recovered_ids: Sequence[int],
    truth_ids: Sequence[int],
    span: Sequence[int],
    *,
    special_ids: Iterable[int] = (),
) -> bool:
    """True iff every position of ``span`` is recovered verbatim (PIIRSR rule).

    Plan §5.4 formula (3): *all* tokens of an entity must be correct at their
    original positions; a partially recovered entity counts as a failure.
    """
    special = set(special_ids)
    for i in span:
        idx = int(i)
        if idx < 0 or idx >= len(truth_ids):
            return False
        if idx >= len(recovered_ids):
            return False
        if truth_ids[idx] in special:
            continue
        if int(recovered_ids[idx]) != int(truth_ids[idx]):
            return False
    return True


def pii_recovery_success_ratio(flags: Sequence[bool]) -> Optional[float]:
    """``PIIRSR = #fully recovered entities / #entities`` (plan §5.4 formula (3)).

    Returns ``None`` when the sample has no annotated entity: the plan forbids
    writing ``0%`` for "no entity", because that reads as a failure rate.
    """
    n = len(flags)
    if n == 0:
        return None
    return sum(1 for f in flags if f) / n


def ngram_overlap_ratio(
    pred_ids: Sequence[int], truth_ids: Sequence[int], n: int = 2, *, max_order: Optional[int] = None
) -> float:
    """Clipped n-gram overlap of ``pred`` against ``truth`` (InferDPT-style metric).

    ``overlap_n = sum_g min(count_pred(g), count_truth(g)) / max(1, #pred n-grams)``
    i.e. the BLEU precision of order ``n`` without smoothing. Averaged over
    ``n = 1..max_order`` (default ``n``), so a single number summarises 1..N-gram
    overlap exactly as the InferDPT/RANTEXT evaluations report it.
    """
    pred = [int(t) for t in pred_ids]
    ref = [int(t) for t in truth_ids]
    order = int(max_order or n)
    if order < 1:
        raise ValueError("n must be >= 1")
    vals: list[float] = []
    for k in range(1, order + 1):
        pred_ng = Counter(tuple(pred[i : i + k]) for i in range(len(pred) - k + 1))
        ref_ng = Counter(tuple(ref[i : i + k]) for i in range(len(ref) - k + 1))
        denom = sum(pred_ng.values())
        if denom == 0:
            vals.append(0.0)
            continue
        overlap = sum(min(c, ref_ng.get(g, 0)) for g, c in pred_ng.items())
        vals.append(overlap / denom)
    return float(np.mean(vals)) if vals else 0.0


def rouge_l(pred_ids: Sequence[int], truth_ids: Sequence[int]) -> float:
    """Token-level ROUGE-L F1 (longest common subsequence) - C02's text metric.

    ``P = LCS/len(pred)``, ``R = LCS/len(truth)``, ``F1 = 2PR/(P+R)``.
    """
    pred = [int(t) for t in pred_ids]
    ref = [int(t) for t in truth_ids]
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for a in pred:
        cur = [0]
        for j, b in enumerate(ref, start=1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p = lcs / len(pred)
    r = lcs / len(ref)
    return float(2 * p * r / (p + r))
