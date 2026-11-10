"""Multi-observation aggregation (protocol §8.2; W4).

Repeated observations of the same request are grouped only by public metadata
(session id / epoch / request group). The attack never receives the true repeated
fragment labels; an oracle grouping is a separately named diagnostic.

Two public aggregators are implemented:

* candidate voting: count how often each deployment id appears across the m
  observations' ranked candidates;
* normalized representation aggregation: mean of L2-normalized wire vectors,
  then one nearest-neighbour lookup against the server table.

A KNN-aligned-then-aggregate variant needs a stage-2 mapping and is reported
``not_run`` when none is supplied, never with a fabricated number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import time
from typing import Mapping, Sequence

import numpy as np

from ..schemes.ab_deployment import WirePublic
from ..schemes.public_view import PublicDeployment, assert_public


def vote_candidates(candidate_lists: Sequence[np.ndarray], k: int = 100) -> np.ndarray:
    """Aggregate ranked candidates across observations by positional vote count."""
    arrays = [np.asarray(c, dtype=np.int64) for c in candidate_lists]
    if not arrays:
        raise ValueError("candidate_lists is empty")
    if any(array.ndim != 2 for array in arrays) or k < 1:
        raise ValueError("candidates must be 2-D and k must be positive")
    n_positions = min(array.shape[0] for array in arrays)
    rankings = []
    for position in range(n_positions):
        votes: dict[int, float] = {}
        for array in arrays:
            seen = set()
            for rank, token in enumerate(array[position]):
                if token < 0 or int(token) in seen:
                    continue
                seen.add(int(token))
                votes[int(token)] = votes.get(int(token), 0.0) + 1.0 / (rank + 1)
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        rankings.append(ranked)
    width = min(k, max((len(row) for row in rankings), default=0))
    aggregated = np.full((n_positions, width), -1, dtype=np.int64)
    for position, ranked in enumerate(rankings):
        for index, (token, _) in enumerate(ranked[:width]):
            aggregated[position, index] = token
    return aggregated


def aggregate_representations(
    public: PublicDeployment,
    observations: Sequence[np.ndarray],
    *,
    k: int = 100,
    arm: str | None = None,
) -> np.ndarray:
    """Normalize each observation, average, then one full-table KNN lookup."""
    vectors = [np.asarray(vector, dtype=np.float32) for vector in observations]
    n_positions = min(vector.shape[0] for vector in vectors)
    normalized = []
    for vector in vectors:
        block = vector[:n_positions]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        normalized.append(block / np.maximum(norms, 1e-12))
    mean = np.mean(normalized, axis=0)
    mean = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    return _topk_queries(public, mean, k=k, arm=arm)


def _prepare_cosine_table(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = np.asarray(table, dtype=np.float32)
    if table.ndim != 2 or not len(table) or not np.isfinite(table).all():
        raise ValueError("table must be a finite nonempty matrix")
    tnorm = np.linalg.norm(table, axis=1, keepdims=True)
    if not np.isfinite(tnorm).all():
        raise ValueError("table norm is non-finite")
    return table / np.maximum(tnorm, 1e-12), tnorm[:, 0] > 0


def _topk_queries(public: PublicDeployment, vectors: np.ndarray, *, k: int,
                  arm: str | None = None, prepared: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    tn, valid_table = prepared if prepared is not None else _prepare_cosine_table(public.table(arm or public.arm))
    q = np.asarray(vectors, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != tn.shape[1] or not np.isfinite(q).all():
        raise ValueError("queries must be finite and compatible with the table")
    qnorm = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(qnorm).all():
        raise ValueError("query norm is non-finite")
    qn = q / np.maximum(qnorm, 1e-12)
    kk = max(1, min(int(k), tn.shape[0]))
    out = np.full((len(q), kk), -1, dtype=np.int64)
    chunk = max(1, (1 << 22) // len(tn))
    n_valid = min(kk, int(valid_table.sum()))
    for start in range(0, len(q), chunk):
        stop = min(start + chunk, len(q))
        scores = qn[start:stop] @ tn.T
        scores[:, ~valid_table] = -np.inf
        for row, values in enumerate(scores):
            if not n_valid or qnorm[start + row, 0] <= 0:
                continue
            boundary = np.partition(values, len(values) - n_valid)[len(values) - n_valid]
            better = np.flatnonzero(values > boundary)
            tied = np.flatnonzero(values == boundary)[:n_valid - len(better)]
            selected = np.concatenate((better, tied))
            order = np.lexsort((selected, -values[selected]))
            out[start + row, :n_valid] = selected[order]
    return out


def _align_ids(left: Sequence[int], right: Sequence[int]) -> tuple[int, list[tuple[int, int]]]:
    """Global edit alignment, with deterministic diagonal-first tie handling."""
    n, m = len(left), len(right)
    costs = np.empty((n + 1, m + 1), dtype=np.int32)
    costs[:, 0] = np.arange(n + 1)
    costs[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = left[i - 1] >= 0 and left[i - 1] == right[j - 1]
            costs[i, j] = min(costs[i - 1, j - 1] + (not same),
                              costs[i - 1, j] + 1, costs[i, j - 1] + 1)
    distance = int(costs[n, m])
    pairs = []
    i, j = n, m
    while i or j:
        same = i and j and left[i - 1] >= 0 and left[i - 1] == right[j - 1]
        if i and j and costs[i, j] == costs[i - 1, j - 1] + (not same):
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and costs[i, j] == costs[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return distance, list(reversed(pairs))


def aggregate_public_observations(public: PublicDeployment, wires: Sequence[WirePublic], *,
                                  similarity_threshold: float, k: int = 100,
                                  same_session_only: bool = False,
                                  max_pairs: int = 100_000,
                                  max_pair_visits: int = 2_000_000,
                                  max_alignment_cells: int = 2_000_000) -> dict:
    """Infer request pairs and position alignment from public KNN predictions only.

    The threshold must be chosen outside the scored test stream. Limits refuse
    excess work rather than silently selecting a more favorable subset.
    """
    started = time.perf_counter()
    assert_public(public)
    if not 0 <= float(similarity_threshold) <= 1 or int(k) < 1:
        raise ValueError("similarity_threshold must be in [0, 1] and k must be positive")
    if int(max_pairs) < 1 or int(max_pair_visits) < 1 or int(max_alignment_cells) < 1:
        raise ValueError("pair and alignment budgets must be positive")
    wires = list(wires)
    if any(not isinstance(wire, WirePublic) for wire in wires):
        raise TypeError("aggregation accepts WirePublic records only")
    request_ids = [wire.request_id for wire in wires]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request IDs must be unique")
    table = np.asarray(public.table(public.arm), dtype=np.float32)
    if table.ndim != 2 or not table.shape[0] or not np.isfinite(table).all():
        raise ValueError("public table must be a nonempty finite matrix")
    prepared = _prepare_cosine_table(table)
    candidates = []
    for wire in wires:
        vectors = np.asarray(wire.wire_vectors, dtype=np.float32)
        if (vectors.ndim != 2 or vectors.shape != (wire.seq_len, table.shape[1])
                or not np.isfinite(vectors).all()):
            raise ValueError("wire vectors must be finite and match the public sequence length")
        candidates.append(_topk_queries(public, vectors, k=min(int(k), len(table)), prepared=prepared))
    inverted: dict[object, list[int]] = defaultdict(list)
    for index, rows in enumerate(candidates):
        if same_session_only and wires[index].session_id is None:
            continue
        for token in set(int(token) for token in rows[:, 0] if token >= 0):
            group = (token, wires[index].session_id) if same_session_only else token
            inverted[group].append(index)
    pair_set: set[tuple[int, int]] = set()
    pair_visits = 0
    for members in inverted.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                pair_visits += 1
                if pair_visits > max_pair_visits:
                    raise ValueError("public pair enumeration budget exceeded; no partial result produced")
                pair_set.add((left, right))
                if len(pair_set) > max_pairs:
                    raise ValueError("public pair budget exceeded; no partial result produced")
    votes = [[[(int(token), rank) for rank, token in enumerate(row)] for row in rows]
             for rows in candidates]
    pairs = []
    cells = 0
    for left, right in sorted(pair_set):
        l_ids, r_ids = candidates[left][:, 0], candidates[right][:, 0]
        cells += (len(l_ids) + 1) * (len(r_ids) + 1)
        if cells > max_alignment_cells:
            raise ValueError("public alignment budget exceeded; no partial result produced")
        distance, aligned = _align_ids(l_ids, r_ids)
        similarity = 1.0 - distance / max(len(l_ids), len(r_ids), 1)
        if similarity < similarity_threshold:
            continue
        pairs.append({"left": left, "right": right, "similarity": similarity,
                      "distance": distance, "aligned_positions": aligned})
        for i, j in aligned:
            votes[left][i].extend((int(token), rank) for rank, token in enumerate(candidates[right][j]))
            votes[right][j].extend((int(token), rank) for rank, token in enumerate(candidates[left][i]))
    aggregated = []
    for positions in votes:
        rows = np.full((len(positions), min(int(k), len(table))), -1, dtype=np.int64)
        for index, entries in enumerate(positions):
            scores: dict[int, float] = defaultdict(float)
            for token, rank in entries:
                if token >= 0:
                    scores[token] += 1.0 / (rank + 1)
            ranked = sorted(scores, key=lambda token: (-scores[token], token))[:rows.shape[1]]
            rows[index, :len(ranked)] = ranked
        aggregated.append(rows)
    return {"request_ids": request_ids, "single_candidates": candidates,
            "aggregated_candidates": aggregated, "pairs": pairs,
            "candidate_pairs": len(pair_set), "pair_visits": pair_visits, "alignment_cells": cells,
            "similarity_threshold": float(similarity_threshold),
            "threshold_provenance": "caller supplied; validation selection not verified",
            "same_session_only": bool(same_session_only),
            "seconds": time.perf_counter() - started,
            "method": "public top-1 edit alignment and reciprocal-rank candidate voting"}


def aggregate_public_stream(public: PublicDeployment, requests: Sequence[Mapping], **kwargs) -> dict:
    """Apply public aggregation directly to build_stream's server-visible records."""
    wires = []
    for request in requests:
        vectors = np.asarray(request["vectors"])
        if vectors.ndim != 2:
            raise ValueError("stream vectors must be a 2-D matrix")
        wires.append(WirePublic(arm=public.arm, request_id=request["request_id"],
                                seq_len=len(vectors), wire_vectors=vectors,
                                session_id=request.get("session_id"),
                                group_id=request.get("group_id"), epoch=request.get("epoch")))
    return aggregate_public_observations(public, wires, **kwargs)


def score_public_aggregation(result: dict, truth_ids: Sequence[Sequence[int]],
                             repeat_labels: Sequence[object], *,
                             max_alignment_cells: int = 2_000_000) -> dict:
    """Score a frozen public attack result; truth is never passed into the attack."""
    n = len(result["request_ids"])
    if len(truth_ids) != n or len(repeat_labels) != n:
        raise ValueError("one truth sequence and repeat label are required per request")
    if any(label is None for label in repeat_labels):
        raise ValueError("repeat labels must be explicit; unknown labels cannot be pooled")
    if int(max_alignment_cells) < 1:
        raise ValueError("scoring alignment budget must be positive")
    hits = {"single": 0, "aggregated": 0}
    denominator = 0
    for index, truth in enumerate(truth_ids):
        truth = [int(token) for token in truth]
        denominator += len(truth)
        for name, field in (("single", "single_candidates"), ("aggregated", "aggregated_candidates")):
            predictions = result[field][index]
            hits[name] += sum(int(i < len(predictions) and predictions[i, 0] == token)
                              for i, token in enumerate(truth))
    predicted_pairs = {(int(pair["left"]), int(pair["right"])) for pair in result["pairs"]}
    if (len(predicted_pairs) != len(result["pairs"]) or any(
            left < 0 or right >= n or left >= right for left, right in predicted_pairs)):
        raise ValueError("invalid or duplicate public pair records")
    label_counts = Counter(repeat_labels)
    n_true_pairs = sum(count * (count - 1) // 2 for count in label_counts.values())
    correct_pairs = sum(repeat_labels[left] == repeat_labels[right] for left, right in predicted_pairs)
    aligned_total = aligned_correct = cells = 0
    for pair in result["pairs"]:
        left, right = pair["left"], pair["right"]
        if repeat_labels[left] != repeat_labels[right]:
            continue
        cells += (len(truth_ids[left]) + 1) * (len(truth_ids[right]) + 1)
        if cells > max_alignment_cells:
            raise ValueError("scoring alignment budget exceeded; no partial result produced")
        _, oracle = _align_ids(truth_ids[left], truth_ids[right])
        actual = {tuple(item) for item in pair["aligned_positions"]}
        aligned_total += len(actual)
        aligned_correct += len(actual & set(oracle))
    return {
        name: {"numerator": count, "denominator": denominator,
               "value": count / denominator if denominator else None}
        for name, count in hits.items()
    } | {
        "pairing": {"predicted_pairs": len(predicted_pairs), "true_pairs": n_true_pairs,
                    "correct_pairs": correct_pairs, "false_pairs": len(predicted_pairs) - correct_pairs,
                    "missed_pairs": n_true_pairs - correct_pairs},
        "alignment": {"matched_pairs": aligned_correct, "predicted_pairs": aligned_total,
                      "reference": "deterministic truth-side edit alignment; ambiguous alignments are not resolved"},
        "prediction_space": "deployment token IDs; not end-to-end plaintext recovery",
    }


def run_multi_observation(
    dep,
    texts: Sequence[str],
    *,
    m_values: Sequence[int] = (1, 2, 4, 8, 16),
    k: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Compare single, vote and normalized-representation aggregation over m repeats."""
    import time

    t0 = time.time()
    public = dep.public_view("ab", "S-public")
    results: dict = {}
    for m in m_values:
        observations: list[list[WirePublic]] = [[] for _ in texts]
        truths: list[np.ndarray] = []
        for index, text in enumerate(texts):
            for repeat in range(int(m)):
                wire = dep.observe(
                    text, arm="ab", request_id=index * m + repeat,
                    key_index=index * 1000 + repeat, epoch=0,
                    session_id=f"session{index}",
                )
                observations[index].append(wire.public)
                if repeat == 0:
                    truths.append(np.asarray(wire.truth.permuted_ids, dtype=np.int64))
        single_rows, vote_rows, repr_rows = [], [], []
        for index, wires in enumerate(observations):
            vectors = [np.asarray(wire.wire_vectors, dtype=np.float32) for wire in wires]
            n_positions = min(vector.shape[0] for vector in vectors)
            single = _topk_queries(public, vectors[0], k=k)
            candidates = [
                _topk_queries(public, vector[:n_positions], k=k) for vector in vectors
            ]
            voted = vote_candidates(candidates, k=k)
            represented = aggregate_representations(public, vectors, k=k)
            full_positions = int(truths[index].size)
            for name, rows, candidates_array in (
                ("single", single_rows, single),
                ("vote", vote_rows, voted),
                ("representation", repr_rows, represented),
            ):
                comparable = min(len(candidates_array), full_positions)
                hit = (candidates_array[:comparable, :1]
                       == truths[index][:comparable, None]).any(axis=1)
                # Positions the shorter repeats could not cover stay in the
                # denominator as misses.
                rows.append((int(hit.sum()), full_positions))
        def _merge(rows):
            hits = sum(row[0] for row in rows)
            total = sum(row[1] for row in rows)
            return {"value": hits / total if total else None,
                    "numerator": hits, "denominator": total}
        results[f"m={int(m)}"] = {
            "single": _merge(single_rows),
            "candidate_vote": _merge(vote_rows),
            "representation_aggregate": _merge(repr_rows),
            "knn_aligned_then_aggregate": {
                "status": "not_run",
                "reason": "requires a full stage-2 mapping",
            },
        }
        if verbose:
            print(f"  aggregation m={m}: vote={results[f'm={int(m)}']['candidate_vote']['value']}",
                  flush=True)
    results["seconds"] = round(time.time() - t0, 1)
    results["grouping"] = "controlled repeats of the same text with known positional alignment; not inferred from session metadata"
    return results
