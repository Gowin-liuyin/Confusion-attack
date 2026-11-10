"""Cross-request linkage attack (plan P0-9 / class C12).

**Provenance.** The plan is explicit that linkage is *this project's extension*
of AloePri's frequency-leakage discussion - there is no source paper for the
protocol (see ``docs/ATTACK_PROVENANCE_14.md``), so nothing here is attributed to
a fabricated reference. What the plan does pin down is the protocol (§4 P0-9,
§6.7) and the metrics (§5.4 formula 6):

* exact linker: two positions link when they carry the *same symbol*;
* approximate linker: two positions link when ``cos(z_i, z_j) >= T`` for
  ``T in {0.3, ..., 0.9}``, reporting **TPR and FPR** (TPR alone overstates);
* ``LinkRate = #(same-symbol same-token pairs) / #(cross-request same-token
  pairs)``;
* localisation: given a position known to share a token with another request, is
  the highest-cosine cross-request position a true partner (>= 95% is "strong");
* results are bucketed by request gap ``Delta`` into 1 / 2-3 / 4-15 / 16-63 /
  64-127 / 128-511 / >=512, because rotation boundaries show up as a cliff.

**Who sees what.** The attack sees the symbol stream and - in the SH-P setting -
the request metadata (session / epoch / arrival) that lets it group requests and
reason about key families. It never sees the plaintext ids: the pair *labels*
("do these two positions carry the same token?") live on the stream and are read
only by :func:`score_linkage`, which the scorer calls. So an attack cannot tune
itself to the answer, and the SH-B variant is produced by simply withholding
``REQUEST_METADATA`` (the same code path, a weaker attacker).
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ...data.streams import RequestRecord, RequestStream, StreamContext
from ...metrics import (
    DELTA_BUCKETS,
    delta_bucket,
    linkage_exact_rate,
    linkage_tpr_fpr,
)
from ..base import Attack, register

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# pair enumeration (attacker side: no ground truth)
# ---------------------------------------------------------------------------
def enumerate_pairs(
    records: list[RequestRecord],
    *,
    within_epoch: bool,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict:
    """Sample cross-request position pairs uniformly without enumerating positions.

    ``within_epoch`` is a legacy name for same visible key-group filtering;
    unknown groups remain eligible and are recorded separately. Epoch is never
    substituted for key identity. Sampling is without replacement, with equal
    inclusion probability across all eligible position pairs.
    """
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    lengths = np.asarray([int(record.symbols.size) for record in records], dtype=np.int64)
    n = len(records)
    groups = [record.key_id for record in records]
    epochs = [record.epoch for record in records]
    # Index weighted request lengths once. For a known group, eligible right
    # requests are that group plus unknown groups; unknown left groups see all.
    if sum(int(x) for x in lengths) > np.iinfo(np.int64).max:
        raise ValueError("linkage position population exceeds int64")
    group_indices: dict[str | None, list[int]] = {}
    for i, group in enumerate(groups):
        group_indices.setdefault(group, []).append(i)

    def make_index(indices):
        indices = np.asarray(indices, dtype=np.int64)
        return indices, np.concatenate(([0], np.cumsum(lengths[indices], dtype=np.int64)))

    all_index = make_index(range(n))
    indexes = {group: make_index(indices) for group, indices in group_indices.items()}
    empty_index = make_index([])

    def weight_through(index, positions):
        indices, prefix = index
        return prefix[np.searchsorted(indices, positions, side="right")]

    def eligible_weight(i, positions):
        if not within_epoch or groups[i] is None:
            return weight_through(all_index, positions)
        return (weight_through(indexes[groups[i]], positions)
                + weight_through(indexes.get(None, empty_index), positions))

    counts = []
    for i in range(n):
        available = int(eligible_weight(i, n - 1) - eligible_weight(i, i))
        counts.append(int(lengths[i]) * available)
    if sum(counts) > np.iinfo(np.int64).max:
        raise ValueError("linkage pair population exceeds int64")
    cumulative = np.cumsum(np.asarray(counts, dtype=np.int64), dtype=np.int64)
    total = int(cumulative[-1]) if n else 0
    take = min(int(max_pairs), total)
    if take == total:
        selected = np.arange(total, dtype=np.int64)
    else:
        selected = np.sort(np.random.default_rng(seed).choice(total, size=take, replace=False, shuffle=False))
    result = {name: np.empty(take, dtype=np.int64) for name in
              ("left_req", "left_pos", "right_req", "right_pos", "delta")}
    result.update({name: np.empty(take, dtype=bool) for name in ("same_epoch", "epoch_known", "same_key", "key_known")})
    if take:
        left = np.searchsorted(cumulative, selected, side="right")
        previous = np.concatenate(([0], cumulative[:-1]))
        for i in np.unique(left):
            start = int(np.searchsorted(left, i, side="left"))
            end = int(np.searchsorted(left, i, side="right"))
            output_positions = slice(start, end)
            local = selected[output_positions] - previous[i]
            initial = int(eligible_weight(int(i), int(i)))
            target = initial + local // lengths[i]
            low = np.full(local.shape, i + 1, dtype=np.int64)
            high = np.full(local.shape, n - 1, dtype=np.int64)
            while np.any(low < high):
                middle = (low + high) // 2
                advance = eligible_weight(int(i), middle) <= target
                low = np.where(advance, middle + 1, low)
                high = np.where(advance, high, middle)
            right = low
            before = eligible_weight(int(i), right - 1) - initial
            offset = local - before * lengths[i]
            result["left_req"][output_positions] = i
            result["right_req"][output_positions] = right
            result["left_pos"][output_positions] = offset // lengths[right]
            result["right_pos"][output_positions] = offset % lengths[right]
            result["delta"][output_positions] = right - i
            result["same_epoch"][output_positions] = [epochs[i] is not None and epochs[i] == epochs[j] for j in right]
            result["epoch_known"][output_positions] = [epochs[i] is not None and epochs[j] is not None for j in right]
            known = np.asarray([groups[i] is not None and groups[j] is not None for j in right], dtype=bool)
            result["key_known"][output_positions] = known
            result["same_key"][output_positions] = [groups[i] is not None and groups[j] is not None and groups[i] == groups[j] for j in right]
    result.update({"total_pairs": total, "sampled": take < total,
                   "sample_stride": None if take < total else 1,
                   "inclusion_probability": take / total if total else None,
                   "sample_rule": "uniform without replacement over eligible position pairs"})
    return result


def symbol_equality(records: list[RequestRecord], pairs: dict) -> np.ndarray:
    """``1[symbol(left) == symbol(right)]`` for every enumerated pair."""
    eq = np.empty(pairs["delta"].size, dtype=bool)
    for k in range(pairs["delta"].size):
        a = records[int(pairs["left_req"][k])]
        b = records[int(pairs["right_req"][k])]
        eq[k] = a.symbols[int(pairs["left_pos"][k])] == b.symbols[int(pairs["right_pos"][k])]
    return eq


def cosine_scores(records: list[RequestRecord], pairs: dict) -> np.ndarray:
    """Cosine between the two wire vectors of every pair (approximate linker).

    Non-finite vectors or similarities are undefined, not "not similar": they
    raise instead of being coerced to 0.0 / NaN and silently counted below the
    threshold by :func:`linkage_tpr_fpr`.
    """
    out = np.empty(pairs["delta"].size, dtype=float)
    for k in range(pairs["delta"].size):
        a = records[int(pairs["left_req"][k])]
        b = records[int(pairs["right_req"][k])]
        if a.vectors is None or b.vectors is None:
            raise ValueError("cosine linkage needs per-position wire vectors on every request")
        va = np.asarray(a.vectors[int(pairs["left_pos"][k])], dtype=np.float64)
        vb = np.asarray(b.vectors[int(pairs["right_pos"][k])], dtype=np.float64)
        if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vb))):
            raise ValueError(
                "cosine linkage: wire vector contains non-finite values "
                "(NaN/inf); refusing to score an undefined similarity"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if not (np.isfinite(na) and np.isfinite(nb)):
            raise ValueError("cosine linkage: non-finite wire-vector norm")
        if na <= 0 or nb <= 0:
            raise ValueError(
                "cosine linkage: zero-norm wire vector (cosine undefined); "
                "refusing to score it as 0.0"
            )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            score = float(va @ vb / (na * nb))
        if not np.isfinite(score):
            raise ValueError(
                "cosine linkage: non-finite similarity; refusing to score"
            )
        out[k] = score
    return out


def nearest_partner(
    records: list[RequestRecord], *, max_queries: int = 500, seed: int = 0
) -> dict:
    """For sampled query positions, the highest-cosine cross-request position.

    Localisation metric of plan §6.7: the attacker knows a query shares a token
    with *some* other request; the linker is strong when the nearest cross-request
    position is a true partner. Returns indices only - the scorer decides whether
    the partner is correct, because that needs the plaintext.
    """
    rng = np.random.default_rng(seed)
    queries: list[tuple[int, int]] = []
    for ri, rec in enumerate(records):
        for pos in range(int(rec.symbols.size)):
            queries.append((ri, pos))
    if not queries:
        return {"n_queries": 0, "query_req": [], "query_pos": [], "best_req": [],
                "best_pos": [], "best_cos": [], "second_cos": []}
    if len(queries) > int(max_queries):
        idx = rng.choice(len(queries), size=int(max_queries), replace=False)
        idx.sort()
        queries = [queries[i] for i in idx]

    query_req, query_pos, best_req, best_pos, best_cos, second_cos = ([] for _ in range(6))
    for ri, pos in queries:
        if records[ri].vectors is None:
            raise ValueError("nearest_partner needs per-position wire vectors")
        v = np.asarray(records[ri].vectors[pos], dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise ValueError(
                "nearest_partner: query wire vector contains non-finite values"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            nv = np.linalg.norm(v)
        if not (np.isfinite(nv) and nv > 0):
            raise ValueError(
                "nearest_partner: zero/non-finite query-vector norm "
                "(cosine undefined)"
            )
        sims: list[tuple[float, int, int]] = []
        for rj, other in enumerate(records):
            if rj == ri or other.vectors is None:
                continue
            matrix = np.asarray(other.vectors, dtype=np.float64)
            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    "nearest_partner: wire vector contains non-finite values"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                no = np.linalg.norm(matrix, axis=1)
            if not np.all(np.isfinite(no)) or np.any(no <= 0):
                raise ValueError(
                    "nearest_partner: zero/non-finite wire-vector norm "
                    "(cosine undefined)"
                )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                sim = (matrix @ v) / (no * nv)
            if not np.all(np.isfinite(sim)):
                # NaN comparisons sort arbitrarily; never let an undefined
                # similarity become the best/second partner.
                raise ValueError(
                    "nearest_partner: non-finite similarity; refusing to rank it"
                )
            for pj in range(sim.size):
                sims.append((float(sim[pj]), rj, pj))
        if not sims:
            continue
        sims.sort(key=lambda t: t[0], reverse=True)
        query_req.append(ri)
        query_pos.append(pos)
        best_cos.append(sims[0][0])
        best_req.append(sims[0][1])
        best_pos.append(sims[0][2])
        second_cos.append(sims[1][0] if len(sims) > 1 else float("-inf"))
    return {
        "n_queries": len(query_req),
        "query_req": query_req,
        "query_pos": query_pos,
        "best_req": best_req,
        "best_pos": best_pos,
        "best_cos": best_cos,
        "second_cos": second_cos,
    }


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------
def _stream_ctx(ctx) -> StreamContext:
    if not isinstance(ctx, StreamContext):
        raise TypeError("linkage attacks run on a StreamContext")
    return ctx


def _pair_meta(records: list[RequestRecord], pairs: dict, rotation_aware: bool) -> dict:
    return {
        "n_requests": len(records),
        "n_pairs": int(pairs["delta"].size),
        "total_pairs": int(pairs["total_pairs"]),
        "pairs_sampled": bool(pairs["sampled"]),
        # Visibility is the granted knowledge level, not whatever fields the
        # record objects happen to carry: in SH-B the rotation metadata exists for
        # scoring but is not available to the attacker.
        "metadata_visible": bool(rotation_aware),
    }


@register
class ExactLinkage(Attack):
    """Exact same-symbol linker (plan P0-9, exact variant)."""

    name = "linkage.exact"
    family = "linkage"
    variant = "protocol_extension"
    required_capabilities = frozenset({Capability.TOKEN_CIPHERTEXT})
    stream_capable = True

    def _run(self, ctx: AttackContext) -> AttackResult:  # pragma: no cover - guarded
        return AttackResult.not_applicable("linkage.exact needs a request stream (S-C)")

    def run_stream(self, ctx) -> AttackResult:
        sctx = _stream_ctx(ctx)
        try:
            records = sctx.stream()
            sh_p = sctx.knows_rotation()
        except Exception as exc:  # CapabilityError or missing stream
            return AttackResult.not_applicable(str(exc))
        pairs = enumerate_pairs(
            records,
            within_epoch=False,
            max_pairs=int(self.params.get("max_pairs", 200_000)),
            seed=int(self.params.get("seed", 0)),
        )
        if pairs["total_pairs"] == 0:
            return AttackResult.not_applicable("no cross-request pairs in the stream")
        eq = symbol_equality(records, pairs)
        return AttackResult(
            stream_metrics={
                "kind": "linkage_exact",
                "pairs": pairs,
                "same_symbol": eq,
                "thresholds": [],
                "attacker_knowledge": "SH-P" if sh_p else "SH-B",
                **_pair_meta(records, pairs, sh_p),
            },
            diagnostics={
                "method": "exact symbol equality across requests",
                "attacker_knowledge": "SH-P" if sh_p else "SH-B",
                "epoch_grouping": "all cross-request pairs; visible groups reported separately" if sh_p else "all cross-request pairs; rotation metadata unavailable",
                "rotation_aware": sh_p,
            },
        )


@register
class ApproximateLinkage(Attack):
    """Cosine-threshold linker with TPR/FPR sweep (plan P0-9, approximate)."""

    name = "linkage.approximate"
    family = "linkage"
    variant = "protocol_extension"
    required_capabilities = frozenset({Capability.EMBEDDINGS})
    stream_capable = True

    def _run(self, ctx: AttackContext) -> AttackResult:  # pragma: no cover - guarded
        return AttackResult.not_applicable(
            "linkage.approximate needs a request stream with wire vectors (S-C)"
        )

    def run_stream(self, ctx) -> AttackResult:
        sctx = _stream_ctx(ctx)
        try:
            records = sctx.stream(require_vectors=True)
            sh_p = sctx.knows_rotation()
        except Exception as exc:
            return AttackResult.not_applicable(str(exc))
        pairs = enumerate_pairs(
            records,
            within_epoch=False,
            max_pairs=int(self.params.get("max_pairs", 200_000)),
            seed=int(self.params.get("seed", 0)),
        )
        if pairs["total_pairs"] == 0:
            return AttackResult.not_applicable("no cross-request pairs in the stream")
        cos = cosine_scores(records, pairs)
        nearest = nearest_partner(
            records,
            max_queries=int(self.params.get("max_queries", 500)),
            seed=int(self.params.get("seed", 0)),
        )
        thresholds = self.params.get("thresholds") or list(DEFAULT_THRESHOLDS)
        return AttackResult(
            stream_metrics={
                "kind": "linkage_approx",
                "pairs": pairs,
                "cosine": cos,
                "nearest": nearest,
                "thresholds": [float(t) for t in thresholds],
                "attacker_knowledge": "SH-P" if sh_p else "SH-B",
                **_pair_meta(records, pairs, sh_p),
            },
            diagnostics={
                "method": "cosine threshold on wire vectors",
                "thresholds": [float(t) for t in thresholds],
                "attacker_knowledge": "SH-P" if sh_p else "SH-B",
                "epoch_grouping": "all cross-request pairs; visible groups reported separately" if sh_p else "all cross-request pairs; rotation metadata unavailable",
                "rotation_aware": sh_p,
            },
        )


# ---------------------------------------------------------------------------
# scoring (runs on the stream; the only place plaintext labels are read)
# ---------------------------------------------------------------------------
def score_linkage(result: AttackResult, stream: RequestStream) -> dict:
    """Compute the plan's linkage metrics from the attacker's pair evidence."""
    sm = result.stream_metrics or {}
    pairs = sm["pairs"]
    kind = sm["kind"]
    truth = stream.plaintext_ids

    same_token = np.empty(pairs["delta"].size, dtype=bool)
    for k in range(pairs["delta"].size):
        a = truth[int(pairs["left_req"][k])][int(pairs["left_pos"][k])]
        b = truth[int(pairs["right_req"][k])][int(pairs["right_pos"][k])]
        same_token[k] = int(a) == int(b)

    out: dict = {
        "kind": kind,
        "n_pairs": int(pairs["delta"].size),
        "n_same_token_pairs": int(same_token.sum()),
        "pairs_sampled": bool(sm.get("pairs_sampled", False)),
        "total_pairs_available": int(sm.get("total_pairs", 0)),
        "attacker_knowledge": sm.get("attacker_knowledge"),
    }

    if kind == "linkage_exact":
        same_symbol = np.asarray(sm["same_symbol"], dtype=bool)
        out["link_rate"] = linkage_exact_rate(same_token.tolist(), same_symbol.tolist())
        by_delta: dict[str, dict] = {}
        for lo, hi in DELTA_BUCKETS:
            label = f"{lo}-{hi}" if (hi is not None and hi != lo) else (f"{lo}+" if hi is None else f"{lo}")
            mask = np.array([delta_bucket(d) == label for d in pairs["delta"]], dtype=bool)
            if not mask.any():
                continue
            by_delta[label] = {
                "n_pairs": int(mask.sum()),
                "link_rate": linkage_exact_rate(
                    same_token[mask].tolist(), same_symbol[mask].tolist()
                ),
            }
        out["by_delta"] = by_delta
        if sm.get("attacker_knowledge") == "SH-P":
            ep = np.asarray(pairs["same_epoch"], dtype=bool)
            ep_known = np.asarray(pairs.get("epoch_known", np.ones_like(ep)), dtype=bool)
            out["same_epoch_link_rate"] = linkage_exact_rate(
                same_token[ep & ep_known].tolist(), same_symbol[ep & ep_known].tolist()
            )
            out["cross_epoch_link_rate"] = linkage_exact_rate(
                same_token[(~ep) & ep_known].tolist(), same_symbol[(~ep) & ep_known].tolist()
            )
            if "same_key" in pairs:
                same_key = np.asarray(pairs["same_key"], dtype=bool)
                known = np.asarray(pairs.get("key_known", np.ones_like(same_key)), dtype=bool)
                out["same_key_link_rate"] = linkage_exact_rate(
                    same_token[same_key & known].tolist(), same_symbol[same_key & known].tolist()
                )
                out["cross_key_link_rate"] = linkage_exact_rate(
                    same_token[(~same_key) & known].tolist(), same_symbol[(~same_key) & known].tolist()
                )
                out["unknown_key_pairs"] = int((~known).sum())
        return out

    # approximate
    cos = np.asarray(sm["cosine"], dtype=float)
    sweep = [linkage_tpr_fpr(cos.tolist(), same_token.tolist(), t) for t in sm["thresholds"]]
    out["threshold_sweep"] = sweep
    t_ref = float(self_default_threshold(sm["thresholds"]))
    ref = [row for row in sweep if abs(row["threshold"] - t_ref) < 1e-9]
    out["tpr_at_default"] = ref[0]["tpr"] if ref else None
    out["fpr_at_default"] = ref[0]["fpr"] if ref else None
    out["default_threshold"] = t_ref

    nearest = sm.get("nearest") or {}
    if nearest.get("n_queries"):
        correct = 0
        considered = 0
        for qr, qp, br, bp, bcos, scos in zip(
            nearest["query_req"], nearest["query_pos"], nearest["best_req"],
            nearest["best_pos"], nearest["best_cos"], nearest["second_cos"],
        ):
            truth_token = int(truth[int(qr)][int(qp)])
            # Only queries that really do share a token with another request are
            # counted (plan §6.7: "known to share a token").
            shares = any(
                truth_token in set(int(t) for t in truth[rj])
                for rj in range(len(truth)) if rj != int(qr)
            )
            if not shares:
                continue
            considered += 1
            correct += int(int(truth[int(br)][int(bp)]) == truth_token)
        out["localization"] = {
            "n_queries": int(nearest["n_queries"]),
            "n_queries_sharing_token": considered,
            "accuracy": (correct / considered) if considered else None,
        }
    return out


def self_default_threshold(thresholds) -> float:
    """The reference threshold used for the headline TPR/FPR (plan wants T=0.5)."""
    for t in thresholds:
        if abs(float(t) - 0.5) < 1e-9:
            return 0.5
    return float(thresholds[len(thresholds) // 2]) if thresholds else 0.5
