"""VMA-H on the public wire: recovered-prefix and observed-prefix variants (F04).

Both variants evaluate candidates inside a prefix, but the prefix differs:

* ``recovered``: per layer, the attack maintains its own decoded prefix from the
  first position and never reads the observed plaintext or the secret.
* ``observed``: the attack reuses the observed public prefix for each position.
  This is a legitimate, stronger server variant, but it is *not* the paper's
  autoregressive recovered-prefix attack and is reported under its own id.

Scoring is separate: :func:`score_vma` takes the truth record and the clean mask.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..schemes.public_view import PublicDeployment
from ..schemes.ab_deployment import WirePublic


def prefix_embeddings(public: PublicDeployment, wire: WirePublic, *, arm: str) -> np.ndarray:
    """The public input sequence the attacker may replay for this arm."""
    if arm == "plaintext":
        return np.asarray(public.table("plaintext")[np.asarray(wire.wire_token_ids)], dtype=np.float32)
    if arm == "ab":
        return np.asarray(wire.wire_vectors, dtype=np.float32)
    if arm == "aloepri":
        return np.asarray(public.table(public.arm)[np.asarray(wire.wire_token_ids)], dtype=np.float32)
    raise ValueError(f"unsupported arm for a VMA prefix: {arm}")


def propose_from_logits(prev_logits: np.ndarray, *, n_propose: int = 200) -> np.ndarray:
    """Top-N next-token proposals from the logits the server already observed."""
    logits = np.asarray(prev_logits, dtype=np.float32).ravel()
    n = min(n_propose, logits.size)
    idx = np.argpartition(-logits, n - 1)[:n]
    return idx[np.argsort(-logits[idx])]


def _forward_hidden(public: PublicDeployment, embeds: np.ndarray, layer: int) -> np.ndarray:
    out = public.forward(embeds)
    hidden = out.hidden_states[layer]
    hidden = np.asarray(hidden, dtype=np.float32)
    return hidden[:, -1, :] if hidden.ndim == 3 else hidden[-1]


def score_candidates_in_prefix(
    public: PublicDeployment,
    prefix: np.ndarray,
    candidates: np.ndarray,
    layer: int,
    *,
    table: np.ndarray | None = None,
    chunk: int = 64,
) -> np.ndarray:
    """Layer-``layer`` hidden state at the candidate position, one row per candidate."""
    table = np.asarray(table if table is not None else public.table(public.arm), dtype=np.float32)
    out: list[np.ndarray] = []
    for start in range(0, len(candidates), chunk):
        block = np.asarray(candidates[start:start + chunk], dtype=np.int64)
        if prefix.size:
            repeated = np.repeat(prefix[None, :, :], len(block), axis=0)
            emb = np.concatenate([repeated, table[block][:, None, :]], axis=1)
        else:
            emb = table[block][:, None, :]
        out.append(_forward_hidden(public, emb, layer))
    return np.concatenate(out, axis=0)


def _score_positions(
    public: PublicDeployment,
    wire: WirePublic,
    layer: int,
    *,
    arm: str,
    max_positions: int,
    full_vocab: bool,
    n_propose: int,
    top_k: int,
    chunk: int,
    prefix_mode: str,
    pose: str = "empty",
    bos_id: int | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    observed = np.asarray(wire.hidden_states[public.layers.index(layer)], dtype=np.float32)
    if observed.ndim == 3:
        observed = observed[0]
    elif observed.ndim == 1:
        observed = observed[None, :]
    n_pos = min(int(max_positions), len(observed))
    table = np.asarray(public.table(public.arm), dtype=np.float32)
    vocab = int(table.shape[0])
    recovered: list[int] = []
    candidates_out: list[np.ndarray] = []
    public_prefix = prefix_embeddings(public, wire, arm=arm)

    if pose not in {"empty", "bos"}:
        raise ValueError(f"pose must be 'empty' or 'bos'; got {pose!r}")
    if pose == "bos" and bos_id is None:
        raise ValueError("pose='bos' requires the public bos_id")
    for t in range(n_pos):
        if prefix_mode == "recovered":
            prefix = (
                table[np.asarray(recovered, dtype=np.int64)]
                if recovered else np.empty((0, table.shape[1]), dtype=np.float32)
            )
            if pose == "bos":
                prefix = np.concatenate(
                    [table[int(bos_id)][None, :], prefix], axis=0
                )
        else:
            prefix = public_prefix[:t]
        if full_vocab:
            candidates = np.arange(vocab, dtype=np.int64)
        else:
            if t == 0:
                candidates = np.arange(min(n_propose, vocab), dtype=np.int64)
            else:
                logits = np.asarray(wire.logits_perm[t - 1], dtype=np.float32)
                candidates = propose_from_logits(logits, n_propose=n_propose)
        hidden = score_candidates_in_prefix(
            public, prefix, candidates, layer, table=table, chunk=chunk
        )
        distance = np.abs(hidden - observed[t][None, :]).sum(axis=1)
        order = np.argsort(distance, kind="stable")
        ranked = np.asarray(candidates, dtype=np.int64)[order][:top_k]
        candidates_out.append(ranked)
        recovered.append(int(ranked[0]))
    return recovered, candidates_out


def _run_variant(
    public: PublicDeployment,
    wires: Sequence[WirePublic],
    *,
    layers: Sequence[int],
    arm: str,
    max_positions: int,
    full_vocab: bool,
    n_propose: int,
    top_k: int,
    chunk: int,
    prefix_mode: str,
    attack_id: str,
    verbose: bool,
    pose: str = "empty",
    bos_id: int | None = None,
) -> dict:
    t0 = time.time()
    per_layer: dict[str, list] = {f"layer{int(layer)}": [] for layer in layers}
    recovered_all: dict[str, list] = {f"layer{int(layer)}": [] for layer in layers}
    for index, wire in enumerate(wires):
        for layer in layers:
            recovered, candidates = _score_positions(
                public, wire, int(layer), arm=arm, max_positions=max_positions,
                full_vocab=full_vocab, n_propose=n_propose, top_k=top_k,
                chunk=chunk, prefix_mode=prefix_mode,
                pose=pose, bos_id=bos_id,
            )
            key = f"layer{int(layer)}"
            per_layer[key].append(candidates)
            recovered_all[key].append(recovered)
        if verbose:
            print(f"  [{prefix_mode}] text {index + 1}/{len(wires)}", flush=True)

    out_layers = {}
    for layer in layers:
        key = f"layer{int(layer)}"
        out_layers[key] = {
            "candidates": [
                np.asarray(cands, dtype=np.int64).tolist() for cands in per_layer[key]
            ],
            "recovered": recovered_all[key],
        }
    return {
        "attack_id": attack_id,
        "prefix_mode": prefix_mode,
        "arm": arm,
        "candidate_mode": "full" if full_vocab else "proposal",
        "pose": pose,
        "bos_id": int(bos_id) if bos_id is not None else None,
        "first_token_included": True,
        "max_positions": int(max_positions),
        "n_propose": None if full_vocab else int(n_propose),
        "layers": out_layers,
        "seconds": round(time.time() - t0, 1),
    }


def vma_h_recovered(
    public: PublicDeployment,
    wires: Sequence[WirePublic],
    *,
    layers: Sequence[int] = (1,),
    full_vocab: bool = True,
    n_propose: int = 200,
    max_positions: int = 16,
    top_k: int = 100,
    chunk: int = 64,
    pose: str = "empty",
    bos_id: int | None = None,
    verbose: bool = True,
) -> dict:
    """Autoregressive recovered-prefix VMA-H; every layer keeps its own prefix."""
    if pose not in {"empty", "bos"}:
        raise ValueError(f"pose must be 'empty' or 'bos'; got {pose!r}")
    if pose == "bos" and bos_id is None:
        raise ValueError("pose='bos' requires the public bos_id")
    return _run_variant(
        public, wires, layers=layers, arm=public.arm, max_positions=max_positions,
        full_vocab=full_vocab, n_propose=n_propose, top_k=top_k, chunk=chunk,
        prefix_mode="recovered", attack_id="vma_h_recovered_prefix", verbose=verbose,
        pose=pose, bos_id=bos_id,
    )


def vma_h_observed(
    public: PublicDeployment,
    wires: Sequence[WirePublic],
    *,
    layers: Sequence[int] = (1,),
    full_vocab: bool = True,
    n_propose: int = 200,
    max_positions: int = 16,
    top_k: int = 100,
    chunk: int = 64,
    verbose: bool = True,
) -> dict:
    """Observed-prefix variant (legitimate stronger observation, not the paper attack)."""
    return _run_variant(
        public, wires, layers=layers, arm=public.arm, max_positions=max_positions,
        full_vocab=full_vocab, n_propose=n_propose, top_k=top_k, chunk=chunk,
        prefix_mode="observed", attack_id="vma_h_observed_prefix", verbose=verbose,
    )


def score_vma(
    result: dict,
    truth: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
    ks: Sequence[int] = (1, 5, 10, 100),
    clean_mask: Sequence[bool] | None = None,
) -> dict:
    """Score a VMA result against the truth; every position stays in the denominator."""
    per_text_truth = (
        isinstance(truth, (list, tuple))
        and bool(truth)
        and all(np.ndim(item) > 0 for item in truth)
    )
    target = None if per_text_truth else np.asarray(truth, dtype=np.int64)
    per_text_mask = (
        isinstance(clean_mask, (list, tuple))
        and bool(clean_mask)
        and all(np.ndim(item) > 0 for item in clean_mask)
    )
    mask = None if clean_mask is None else (
        [np.asarray(item, dtype=bool) for item in clean_mask]
        if per_text_mask else np.asarray(clean_mask, dtype=bool)
    )
    out: dict = {}
    for key, layer in result["layers"].items():
        if layers is not None and int(key.replace("layer", "")) not in set(layers):
            continue
        recovered = [np.asarray(item, dtype=np.int64) for item in layer["recovered"]]
        candidates = [np.asarray(c, dtype=np.int64) for c in layer["candidates"]]
        if per_text_truth:
            targets = [np.asarray(item, dtype=np.int64) for item in truth]
            if len(targets) != len(candidates):
                raise ValueError(
                    f"{len(targets)} truth rows but {len(candidates)} candidate lists"
                )
        elif target.ndim == 1:
            if len(candidates) != 1:
                raise ValueError(
                    "a 1-D truth cannot score multiple per-text candidate lists; "
                    "pass one truth row per text"
                )
            targets = [target]
        else:
            targets = [np.asarray(row, dtype=np.int64) for row in target]
            if len(targets) != len(candidates):
                raise ValueError(
                    f"{len(targets)} truth rows but {len(candidates)} candidate lists"
                )
        masks = None
        if mask is not None:
            if isinstance(mask, list):
                masks = list(mask)
                if len(masks) != len(candidates):
                    raise ValueError(
                        f"{len(masks)} clean masks but {len(candidates)} candidate lists"
                    )
            elif mask.ndim == 1:
                if len(candidates) != 1:
                    raise ValueError(
                        "a 1-D clean mask cannot score multiple per-text candidate lists"
                    )
                masks = [mask]
            else:
                masks = list(mask)
        metrics = {}
        for k in ks:
            hit_blocks: list[np.ndarray] = []
            total = 0
            for i, c in enumerate(candidates):
                truth_i = targets[i]
                n_i = int(truth_i.size)
                total += n_i
                if n_i == 0:
                    continue
                cand = np.asarray(c, dtype=np.int64)
                if cand.ndim == 1:
                    cand = cand[:, None]
                kk = min(int(k), cand.shape[1]) if cand.size else 0
                row = np.full((n_i, kk), -1, dtype=np.int64)
                take = min(n_i, cand.shape[0])
                if take and kk:
                    row[:take] = cand[:take, :kk]
                hit_blocks.append(
                    (row == truth_i[:n_i, None]).any(axis=1) if kk else np.zeros(n_i, bool)
                )
            hit = np.concatenate(hit_blocks) if hit_blocks else np.zeros(0, bool)
            metrics[f"total_top{k}"] = {
                "value": float(hit.mean()) if hit.size else None,
                "numerator": int(hit.sum()),
                "denominator": int(total),
            }
            if mask is not None:
                clean_blocks: list[np.ndarray] = []
                clean_total = 0
                offset = 0
                for i, c in enumerate(candidates):
                    n_i = int(targets[i].size)
                    if n_i == 0:
                        continue
                    m_i = np.asarray(masks[i], dtype=bool)[:n_i]
                    if m_i.size < n_i:
                        m_i = np.pad(m_i, (0, n_i - m_i.size), constant_values=True)
                    block = hit[offset:offset + n_i]
                    offset += n_i
                    clean_blocks.append(block[m_i])
                    clean_total += int(m_i.sum())
                clean_hit = (
                    np.concatenate(clean_blocks) if clean_blocks else np.zeros(0, bool)
                )
                metrics[f"clean_top{k}"] = {
                    "value": float(clean_hit.mean()) if clean_hit.size else None,
                    "numerator": int(clean_hit.sum()),
                    "denominator": int(clean_total),
                }
        first_error = None
        for i, rec in enumerate(recovered):
            truth_i = targets[i]
            rec_i = np.asarray(rec, dtype=np.int64)
            shared = min(int(rec_i.size), int(truth_i.size))
            match = np.flatnonzero(rec_i[:shared] != truth_i[:shared])
            if match.size:
                first_error = int(match[0])
                break
            if rec_i.size < truth_i.size:
                # The first position the attack never produced is an error, not
                # a position that disappears from the denominator.
                first_error = int(rec_i.size)
                break
        metrics["first_error_position"] = first_error
        metrics["recovered"] = [item.tolist() for item in recovered]
        out[key] = metrics
    return out
