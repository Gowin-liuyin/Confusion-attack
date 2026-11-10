"""Retrieval from the public deployment table using wire-vector norms.

Returned IDs refer to deployment rows. A separately recovered public mapping is
needed before these IDs can be interpreted as plaintext IDs.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


def norm_topk(queries, table, *, k: int, direction_weight: float = 0.0,
              radius_scale: float = 1.0) -> np.ndarray:
    """Exhaustive deterministic ranking; hyperparameters must be chosen on validation data.

    Score is absolute radius error / radius_scale plus direction_weight times
    cosine distance. Ties are resolved by deployment row ID.
    """
    q = np.asarray(queries, dtype=np.float64)
    e = np.asarray(table, dtype=np.float64)
    if q.ndim != 2 or e.ndim != 2 or q.shape[1] != e.shape[1] or not len(e):
        raise ValueError("queries and nonempty table must have matching dimensions")
    if not np.isfinite(q).all() or not np.isfinite(e).all():
        raise ValueError("vectors must be finite")
    if not 1 <= k <= len(e) or not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError("invalid k or radius_scale")
    if not np.isfinite(direction_weight) or direction_weight < 0:
        raise ValueError("direction_weight must be nonnegative and finite")
    radii = np.linalg.norm(e, axis=1)
    ids = np.arange(len(e))
    result = np.empty((len(q), k), dtype=np.int64)
    for i, vector in enumerate(q):
        radius = np.linalg.norm(vector)
        score = np.abs(radii - radius) / radius_scale
        if direction_weight:
            denominator = radii * radius
            cosine = np.divide(e @ vector, denominator, out=np.zeros_like(radii),
                               where=denominator > 0)
            score += direction_weight * (1 - np.clip(cosine, -1, 1))
        result[i] = np.lexsort((ids, score))[:k]
    return result


def compose_candidates(candidates, mapping: Mapping[int, int]) -> list[list[int]]:
    """Translate only deployment IDs covered by an attacker-recovered mapping."""
    rows = []
    for candidates_row in np.asarray(candidates):
        seen = set()
        translated = []
        for item in candidates_row:
            token = mapping.get(int(item))
            if token is not None and int(token) not in seen:
                seen.add(int(token))
                translated.append(int(token))
        rows.append(translated)
    return rows


def request_radius_statistics(requests: Sequence[np.ndarray], table, *, k: int = 1) -> dict:
    """Summarize public radius retrieval per request without assuming alignment."""
    counts: Counter[int] = Counter()
    per_request = []
    for vectors in requests:
        candidates = norm_topk(vectors, table, k=k)
        top1 = candidates[:, 0].tolist()
        counts.update(top1)
        per_request.append({"positions": len(top1), "top1_counts": dict(Counter(top1))})
    return {"requests": len(per_request), "positions": sum(counts.values()),
            "top1_counts": dict(counts), "per_request": per_request}
