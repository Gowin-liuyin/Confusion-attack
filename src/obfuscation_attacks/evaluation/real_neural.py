"""Neural attacks on the packaged artifact: VMA-H (S2), ISA (S2), BeamClean (S1/S2).

These all need forward passes, so they are built around one batching helper. The point of the
module is to make the *paper's* protocols reachable on a CPU: a VMA-H database is just the
layer-l hidden state of each candidate token, and the whole database can be produced with a few
batched forward passes instead of one pass per token.

Deviations from the paper's protocol are recorded in the returned dicts (candidate coverage,
prompt count, layer set) rather than silently absorbed: the plan's rule is that a number below
the strength floor may be reported as protocol evidence, never as a paper-strength result.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..schemes.ab_deployment import ABDeployment


def _token_batches(ids: np.ndarray, batch: int) -> Iterable[np.ndarray]:
    for start in range(0, len(ids), batch):
        yield ids[start:start + batch]


def hidden_state_database(
    dep: ABDeployment, candidate_ids: np.ndarray, layer: int, *, batch: int = 2048,
) -> np.ndarray:
    """Layer-``layer`` hidden state of each candidate token fed in isolation -> ``[V, d]``.

    This is the VMA database: the attacker knows the model, so it can push every candidate
    through it. Batching keeps this to a handful of passes instead of one per token.
    """
    import torch

    model = dep.model
    out_rows: list[np.ndarray] = []
    for chunk in _token_batches(np.asarray(candidate_ids, dtype=np.int64), batch):
        ids = torch.from_numpy(chunk).long().unsqueeze(1)          # [B, 1]
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        out_rows.append(out.hidden_states[layer][:, 0, :].float().numpy())
    return np.concatenate(out_rows, axis=0)


def match_hidden_to_database(
    queries: np.ndarray, database: np.ndarray, candidate_ids: np.ndarray, *, k: int = 100,
    chunk: int = 512, database_chunk: int | None = None,
) -> np.ndarray:
    """Nearest database row per query, in L1, returned as *token ids* ``[n, k]``.

    OSNIP App. C.2 specifies the L1 metric for VMA; the database is small enough here that a
    brute-force scan is exact (no ANN approximation).
    """
    q = np.asarray(queries, dtype=np.float32)
    db = np.asarray(database, dtype=np.float32)
    cands = np.asarray(candidate_ids, dtype=np.int64)
    if (q.ndim != 2 or db.ndim != 2 or q.shape[1] != db.shape[1]
            or cands.ndim != 1 or len(cands) != len(db)):
        raise ValueError("queries, database and candidate IDs have incompatible shapes")
    if len(np.unique(cands)) != len(cands):
        raise ValueError("candidate IDs must be unique")
    if (k < 1 or chunk < 1 or (database_chunk is not None and database_chunk < 1)
            or not np.isfinite(q).all() or not np.isfinite(db).all()):
        raise ValueError("k and chunk must be positive and vectors finite")
    k = min(k, len(cands))
    out = np.empty((len(q), k), dtype=np.int64)
    if k == 0 or len(q) == 0:
        return out
    # Exact L1 top-k with both the query block and the database block bounded:
    # a single [512, V, d] temporary is hundreds of GiB at the real vocabulary.
    budget = 256 * 1024 ** 2
    query_block = max(1, min(int(chunk), len(q)))
    db_block = max(1, min(len(db), int(budget / max(1, query_block * db.shape[1] * 4))))
    if database_chunk is not None:
        db_block = min(db_block, int(database_chunk))
    for start in range(0, len(q), query_block):
        block = q[start:start + query_block]
        best_d = np.full((len(block), 0), np.inf, dtype=np.float64)
        best_ids = np.full((len(block), 0), -1, dtype=np.int64)
        for db_start in range(0, len(db), db_block):
            db_stop = min(db_start + db_block, len(db))
            d = np.abs(
                block[:, None, :] - db[db_start:db_stop][None, :, :]
            ).sum(axis=2, dtype=np.float64)
            ids = np.broadcast_to(cands[db_start:db_stop], d.shape)
            combined_d = np.concatenate([best_d, d], axis=1)
            combined_ids = np.concatenate([best_ids, ids], axis=1)
            take = np.lexsort((combined_ids, combined_d), axis=1)[:, :min(k, combined_d.shape[1])]
            best_d = np.take_along_axis(combined_d, take, axis=1)
            best_ids = np.take_along_axis(combined_ids, take, axis=1)
        out[start:start + len(block)] = best_ids
    return out


def stopword_mask(dep: ABDeployment, token_ids: Sequence[int]) -> np.ndarray:
    """True where the token is an NLTK stop word (plan P0-3's ``Clean`` denominator).

    ``Clean ASR`` excludes these positions, so the mask is computed from the tokenizer surface
    form, not from a hand-written list. Using the NLTK English list matches the plan's wording
    ("NLTK 整词停用词"); non-English tokens simply never match a stop word.
    """
    from nltk.corpus import stopwords  # imported lazily; nltk is an optional dependency

    try:
        words = set(stopwords.words("english"))
    except LookupError:  # pragma: no cover - offline corpus
        words = {"a", "the", "of", "and", "is", "in", "to", "it", "that", "for", "on", "with"}
    mask = np.zeros(len(token_ids), dtype=bool)
    for i, tid in enumerate(token_ids):
        surface = dep.tokenizer.decode([int(tid)]).strip().lower()
        mask[i] = surface in words
    return mask


def stopword_definition() -> str:
    """Human-readable source of the Clean-ASR stop-word mask used here."""
    try:
        from nltk.corpus import stopwords

        stopwords.words("english")
        return "NLTK English stop words excluded (whole surface form)"
    except Exception:
        return (
            "built-in 12-word fallback stop words excluded "
            "(NLTK corpus unavailable; Clean-ASR is not the NLTK definition)"
        )


def total_clean_asr(candidates: np.ndarray, truth: np.ndarray, clean_mask: np.ndarray,
                    ks: Sequence[int] = (1, 5, 10, 100)) -> dict:
    """Occurrence-level ASR over all positions (Total) and over content positions (Clean)."""
    truth = np.asarray(truth, dtype=np.int64)
    out: dict = {"total": {}, "clean": {}}
    for k in ks:
        kk = min(k, candidates.shape[1])
        hit = (candidates[:, :kk] == truth[:, None]).any(axis=1)
        out["total"][f"top{k}"] = {
            "value": float(hit.mean()) if hit.size else None,
            "numerator": int(hit.sum()), "denominator": int(hit.size)}
        content = ~np.asarray(clean_mask, dtype=bool)
        out["clean"][f"top{k}"] = {
            "value": float(hit[content].mean()) if content.any() else None,
            "numerator": int(hit[content].sum()), "denominator": int(content.sum())}
    return out
