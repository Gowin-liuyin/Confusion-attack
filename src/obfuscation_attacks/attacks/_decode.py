"""Shared decode step for the output-side families (C13 PII / C14 LM recon).

Both the PII-extraction attack and the output-side reconstruction attack observe
the same thing - a token-id stream the server emits - and both need to turn it
back into plaintext ids before anything entity- or text-level can be measured.
The decode is the frequency-matching step the plan assigns to TFMA (C10): the
attacker ranks plaintext candidates per observed cipher symbol by
``|f_C(c) - f_R(t)|`` over a public reference corpus. Keeping one implementation
here means C13/C14 cannot quietly diverge into two different symbol tables (the
plan forbids two symbol tables in §8; same rule as C10/C12).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from .tfma.matcher import nearest_frequency_order


def frequency_decode(
    ctx, *, top_k: int = 100, reference_counts: Optional[dict] = None
) -> dict:
    """Decode a cipher token stream by nearest-frequency matching.

    Parameters come only from attacker-visible resources: the ciphertext stream
    (``TOKEN_CIPHERTEXT``) and the public reference frequency table. Returns the
    recovered ids, the ranked candidate list per position and the recovered
    mapping and reference support size. Truth coverage is a scoring-side fact.
    """
    cipher = np.asarray(ctx.token_ciphertext(), dtype=int).ravel()
    vocab = int(ctx.descriptor.get("vocab_size") or 0)
    if vocab <= 0:
        raise ValueError("vocab_size unknown; cannot rank plaintext candidates")
    counts = dict(reference_counts or ctx.reference_token_counts or {})
    if not counts:
        raise ValueError("frequency decode needs a public reference token-frequency table")

    f_R = np.zeros(vocab, dtype=float)
    for tid, c in counts.items():
        if not 0 <= int(tid) < vocab or int(c) < 0:
            raise ValueError("invalid reference frequency table")
        f_R[int(tid)] = float(c)
    if f_R.sum() <= 0:
        raise ValueError("reference frequency table is empty over the vocabulary")
    f_R /= f_R.sum()

    observed, obs_counts = np.unique(cipher, return_counts=True)
    if observed.size and (int(observed.min()) < 0 or int(observed.max()) >= vocab):
        raise ValueError("cipher symbol outside the declared vocabulary")
    f_C = {}
    total = float(cipher.size) if cipher.size else 1.0
    for sym, c in zip(observed.tolist(), obs_counts.tolist()):
        f_C[int(sym)] = c / total

    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    k = min(int(top_k), vocab)
    mapping: dict[int, int] = {}
    ranked: dict[int, list[int]] = {}
    for sym, freq in f_C.items():
        order = nearest_frequency_order(freq, f_R, np.arange(vocab))
        ranked[sym] = order[:k].tolist()
        mapping[sym] = int(order[0])

    recovered = np.array([mapping.get(int(s), -1) for s in cipher], dtype=int)
    candidates = [ranked[int(s)] for s in cipher]
    return {
        "recovered": recovered,
        "candidates": candidates,
        "mapping": mapping,
        "coverage": None,
        "coverage_definition": "unknown without scoring-side plaintext truth",
        "reference_supported_types": int(np.count_nonzero(f_R)),
        "num_observed_symbols": int(len(observed)),
        "candidate_top_k": k,
    }


def detect_entities(ids: np.ndarray, lexicon: Optional[list[list[int]]]) -> list[dict]:
    """Scan a decoded id sequence for public gazetteer entries (attacker-side NER).

    ``lexicon`` is the attacker's public list of entity token-id sequences (e.g.
    name / phone / e-mail / address patterns). Returned spans are what the
    attacker *claims*; whether a claimed span is a true annotated entity is a
    scoring question, so the scorer keeps the annotation.
    """
    if not lexicon:
        return []
    ids = [int(t) for t in np.asarray(ids).ravel()]
    spans: list[dict] = []
    for entry_idx, pattern in enumerate(lexicon):
        pat = [int(t) for t in np.asarray(pattern).ravel()]
        if not pat:
            continue
        n = len(pat)
        for start in range(0, len(ids) - n + 1):
            if ids[start : start + n] == pat:
                spans.append({"lexicon_entry": entry_idx, "start": start, "end": start + n})
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans
