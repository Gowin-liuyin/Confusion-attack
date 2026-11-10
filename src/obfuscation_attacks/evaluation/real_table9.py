"""Public weight matching with the feature combinations of AloePri Table 9."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..data.contracts import CapabilityError
from ..schemes.ab_deployment import knn_topk
from .real_weight_attacks import _topk_against_full_table


def _feature(public, side, group, layer):
    """Return a block producer with vocabulary on its row axis."""
    table = public.table("plaintext" if side == "clean" else public.arm)

    def weight(suffix):
        name = f"model.layers.{layer}.{suffix}.weight"
        return public.weight_tensor(name, side)

    def head():
        return public.weight_tensor("lm_head.weight", side)

    if group == "embedding_head":
        right = head().T
    elif group == "attention_qk":
        q = weight("self_attn.q_proj")
        k = weight("self_attn.k_proj")
        heads, kv_heads = public.n_heads, public.n_kv_heads
        if heads < 1 or kv_heads < 1 or heads % kv_heads or k.shape[0] % kv_heads:
            raise ValueError("incompatible grouped-query heads")
        k = np.repeat(k.reshape(kv_heads, -1, k.shape[1]), heads // kv_heads, axis=0)
        k = k.reshape(-1, k.shape[-1])
        if q.shape != k.shape:
            raise ValueError("query and repeated key projections have different shapes")
        keys = table @ k.T
        return lambda rows: (table[rows] @ q.T) @ keys.T
    elif group == "down_head":
        # Table 9 puts the vocabulary on the columns of W_down W_h.
        left = head()
        right = weight("mlp.down_proj")
        return lambda rows: left[rows] @ right
    elif group == "router":
        router = weight("mlp.gate")
        if side == "clean":
            norms = np.linalg.norm(router, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("cannot normalize zero router weights")
            router = router / norms
        right = router.T
    else:
        right = weight(f"mlp.{group}_proj").T
    return lambda rows: table[rows] @ right


def table9_candidates(public, deployment_rows, *, top_k=10, chunk=128):
    """Recover clean IDs from public deployment IDs via exact per-feature RowSort.

    Each group/layer votes for its best match. Ties use the remaining ranks in
    order, then the clean ID. Up/down retain the paper's raw features, whose
    permutation premise is approximate when FFN scales differ.
    """
    clean_size = len(public.table("plaintext"))
    deploy_size = len(public.table(public.arm))
    rows = np.asarray(deployment_rows, dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or np.any(rows < 0)
            or np.any(rows >= deploy_size) or chunk < 1 or top_k < 1):
        raise ValueError("expected valid deployment rows, positive top_k and positive chunk")
    rows = np.unique(rows)
    specs = [("embedding_head", None)] + [
        (group, layer)
        for layer in range(public.n_layers)
        for group in ("attention_qk", "gate", "up", "down_head", "router")
    ]
    votes = [Counter() for _ in rows]
    used, missing = [], []
    k = min(top_k, clean_size)
    for group, layer in specs:
        try:
            clean = _feature(public, "clean", group, layer)
            deployed = _feature(public, "deployment", group, layer)
        except KeyError as exc:
            missing.append({"group": group, "layer": layer, "reason": str(exc)})
            continue
        # Compute only row blocks. This avoids materializing V x V products.
        for start in range(0, len(rows), chunk):
            query = np.sort(deployed(rows[start:start + chunk]), axis=1)
            ranked = _topk_against_full_table(
                query,
                ((i, np.sort(clean(slice(i, i + chunk)), axis=1))
                 for i in range(0, clean_size, chunk)),
                k, clean_size,
            )
            for index, candidates in enumerate(ranked, start):
                votes[index].update((rank, int(token)) for rank, token in enumerate(candidates))
        used.append({"group": group, "layer": layer})
        del clean, deployed
    if not used:
        raise CapabilityError(f"Table 9 has no usable weight groups: {missing}")

    candidates = {}
    for row, counts in zip(rows, votes):
        ids = {token for _, token in counts}
        candidates[int(row)] = sorted(
            ids, key=lambda token: tuple(-counts[rank, token] for rank in range(k)) + (token,)
        )[:k]
    return {"candidates": candidates, "n_candidates": clean_size,
            "used_weight_groups": used, "missing_weight_groups": missing}


def compose_table9_candidates(public, stage1, *, top_k=10, chunk=128):
    """Compose KNN deployment IDs with Table 9 votes, using no scoring labels."""
    stage1 = np.asarray(stage1, dtype=np.int64)
    if stage1.ndim != 2 or not stage1.size:
        raise ValueError("stage1 must contain ranked deployment IDs per position")
    result = table9_candidates(public, stage1.ravel(), top_k=top_k, chunk=chunk)
    composed = np.full((len(stage1), top_k), -1, dtype=np.int64)
    for index, ids in enumerate(stage1):
        ranked = dict.fromkeys(
            token
            for rank in range(top_k)
            for row in ids
            for token in result["candidates"][int(row)][rank:rank + 1]
        )
        selected = list(ranked)[:top_k]
        composed[index, :len(selected)] = selected
    return composed, result


def recover_table9_from_wire(public, wire_vectors, *, top_k=10, chunk=128):
    """Run the composed attack from public vectors and weights."""
    table = public.table(public.arm)
    stage1 = knn_topk(wire_vectors, table, k=min(top_k, len(table)),
                      query_chunk=chunk, table_norm=public.table(public.arm, normalize=True))
    composed, evidence = compose_table9_candidates(public, stage1, top_k=top_k, chunk=chunk)
    return {"stage1_candidates": stage1, "composed_candidates": composed, **evidence}
