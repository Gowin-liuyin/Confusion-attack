"""Monomial diagnostic: recover the hidden-axis permutation from attention weights.

The Table 9 feature-combination attack lives in evaluation.real_table9.

Why this exists. ``vma.weight_rowsort`` (Thomas et al. RowSort) assumes the
obfuscation is a pure row x column permutation, so it scores 0.5-1.0% on the packaged
artifact. That number is *not* AloePri's Table 9 attack strength: the artifact's
transform is a monomial **with per-coordinate scaling**, and for such a family the
right invariant lives in the attention weights, not in the embedding table:

    embedding   E' = E Q                      (Q monomial: column permutation + scales)
    q_proj      W'_q = W_q Q^-T               (HF layout: h' W'_q^T = h W_q^T)

``W'_q = W_q Q^-T`` means every **column** of ``W'_q`` is a scaled copy of exactly one
column of ``W_q`` -- and the output (q) dimension is shared, so the column vectors are
comparable *without knowing the token permutation*. Recovering the column map recovers Q,
and Q is what un-mixes the embedding table; once un-mixed, the token permutation is a
nearest-neighbour lookup.

This module only recovers geometry (Q and the un-mixed table). Attack orchestration,
scoring and the plan's App.-A payload stay in :mod:`real_battery`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

#: Weight groups the plan's Table 9 lists. Only the q/k/v projections consume the
#: residual hidden state on their *input* (column) axis, which is the axis Q mixes.
#: ``o_proj`` is ``[hidden_out, head_in]``: its hidden axis is the output (row) axis,
#: so its columns live in the attention/head basis and must not vote for Q here.
HIDDEN_AXIS_GROUPS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
)
INTERMEDIATE_AXIS_GROUPS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


@dataclass
class AxisRecovery:
    """One recovered axis permutation with its diagnostics."""

    col_map: np.ndarray          # deployment column -> clean column
    corr_best: float
    corr_second: float
    bijection: float
    n_columns: int

    def as_dict(self) -> dict:
        return {
            "n_columns": int(self.n_columns),
            "bijection_rate": float(self.bijection),
            "corr_best_mean": float(self.corr_best),
            "corr_second_mean": float(self.corr_second),
            "margin": float(self.corr_best - self.corr_second),
        }


def _column_correlation(clean: np.ndarray, deployment: np.ndarray) -> np.ndarray:
    """Absolute column correlation, invariant to signed column scaling."""
    c = np.asarray(clean, dtype=np.float32)
    d = np.asarray(deployment, dtype=np.float32)
    if c.ndim != 2 or c.shape != d.shape or c.shape[0] < 2 or c.shape[1] < 1:
        raise ValueError(f"incompatible column matrices: {c.shape} vs {d.shape}")
    if not np.isfinite(c).all() or not np.isfinite(d).all():
        raise ValueError("column matrices must be finite")
    c = c - c.mean(axis=0, keepdims=True)
    d = d - d.mean(axis=0, keepdims=True)
    c /= np.linalg.norm(c, axis=0, keepdims=True) + 1e-12
    d /= np.linalg.norm(d, axis=0, keepdims=True) + 1e-12
    return np.abs(c.T @ d)


def _column_map(clean: np.ndarray, deployment: np.ndarray) -> AxisRecovery:
    """Independently match columns of ``deployment`` to ``clean`` for diagnostics.

    Both matrices are ``[out_dim, in_dim]``; matching runs over ``in_dim`` and compares
    the ``out_dim``-length column vectors, which is exactly the axis the hidden-space
    transform touches. Columns are mean-centred and unit-normalised so a per-column
    scale (the monomial's diagonal) cannot bias the match.
    """
    corr = _column_correlation(clean, deployment)  # [clean_in, dep_in]
    col_map = corr.argmax(axis=0).astype(np.int64)  # dep column -> clean column
    best = corr.max(axis=0)
    second = np.partition(corr, -2, axis=0)[-2] if corr.shape[0] > 1 else np.zeros(1)
    return AxisRecovery(
        col_map=col_map,
        corr_best=float(best.mean()),
        corr_second=float(second.mean()),
        bijection=float(len(set(col_map.tolist())) / max(1, col_map.size)),
        n_columns=int(col_map.size),
    )


def recover_hidden_axis(dep, *, layers: Optional[Sequence[int]] = None,
                        groups: Sequence[str] = HIDDEN_AXIS_GROUPS) -> dict:
    """Recover the hidden-axis column map from attention weights, with layer voting.

    Returns ``{"voted": AxisRecovery, "per_layer": {name: dict}, "agreement": float}``.
    Sum absolute column correlations across available groups and layers, then
    maximize their total subject to a one-to-one mapping. This is a monomial
    engineering diagnostic, not a full AloePri Table 9 implementation.
    """
    w_clean = dep.weights("clean")
    w_dep = dep.weights("deployment")
    layer_ids = list(range(dep.n_layers)) if layers is None else list(layers)
    proposals: list[np.ndarray] = []
    per_layer: dict[str, dict] = {}
    aggregate = None
    for layer in layer_ids:
        for group in groups:
            name = f"model.layers.{layer}.{group}.weight"
            if name not in w_clean or name not in w_dep:
                continue
            corr = _column_correlation(w_clean[name], w_dep[name])
            if aggregate is None:
                aggregate = np.zeros_like(corr, dtype=np.float64)
            if corr.shape != aggregate.shape:
                raise ValueError(f"incompatible hidden axis in {name}: {corr.shape} != {aggregate.shape}")
            aggregate += corr
            best = corr.argmax(axis=0).astype(np.int64)
            second = np.partition(corr, -2, axis=0)[-2] if corr.shape[0] > 1 else np.zeros(1)
            rec = AxisRecovery(best, float(corr.max(axis=0).mean()), float(second.mean()),
                               float(np.unique(best).size / best.size), int(best.size))
            proposals.append(rec.col_map)
            per_layer[f"layer{layer}.{group}"] = rec.as_dict()
    if not proposals:
        raise RuntimeError("no attention weights available for the hidden-axis recovery")
    stack = np.stack(proposals)
    rows, columns = linear_sum_assignment(aggregate, maximize=True)
    voted = np.empty(stack.shape[1], dtype=np.int64)
    voted[columns] = rows
    agreement = float(np.mean(stack == voted[None, :]))
    return {
        "voted": AxisRecovery(
            col_map=voted,
            corr_best=float(aggregate[voted, np.arange(voted.size)].mean() / len(proposals)),
            corr_second=float(np.mean([v["corr_second_mean"] for v in per_layer.values()])),
            bijection=float(len(set(voted.tolist())) / max(1, voted.size)),
            n_columns=int(voted.size),
        ),
        "per_layer": per_layer,
        "agreement": agreement,
        "n_proposals": len(proposals),
        "algorithm": "signed_scale_invariant_correlation_global_assignment_engineering_variant",
    }


def unmix_embedding_table(dep, col_map: np.ndarray) -> dict:
    """Undo the hidden-axis mixing on the deployment embedding table.

    ``E'[:, j] = s_j * E[:, col_map[j]]`` up to noise, so un-mixing is the inverse
    column reordering followed by a permutation-invariant per-column scale:
    column ``c`` of the reordered table is ``s_j * E[:, c]`` with
    ``j = col_map^-1[c]``, and the coefficient of ``E[:, c]`` in that column is
    ``s_j``. Returns the un-mixed table (float32), the per-clean-column scale
    vector, and the relative residual of the linear model.
    """
    E = np.asarray(dep.E_clean, dtype=np.float32)
    D = np.asarray(dep.E_dep, dtype=np.float32)
    q = np.asarray(col_map, dtype=np.int64).ravel()
    if q.size != E.shape[1] or not np.array_equal(np.sort(q), np.arange(q.size)):
        raise ValueError(
            f"col_map must be a permutation of {E.shape[1]} clean hidden columns; "
            f"got {q.size} entries"
        )
    inverse = np.empty_like(q)                       # clean column -> deployment column
    inverse[q] = np.arange(q.size)
    reordered = D[:, inverse]                        # [V, d] deployment columns in clean order
    # Sorting eliminates the unknown token-row permutation. Test both signs:
    # negative scaling reverses the sorted order.
    clean_sorted = np.sort(E, axis=0)
    observed_sorted = np.sort(reordered, axis=0)
    den = np.sum(clean_sorted * clean_sorted, axis=0) + 1e-12
    positive = np.maximum(0, np.sum(clean_sorted * observed_sorted, axis=0) / den)
    negative = np.minimum(0, np.sum(clean_sorted[::-1] * observed_sorted, axis=0) / den)
    err_positive = np.sum((observed_sorted - clean_sorted * positive) ** 2, axis=0)
    err_negative = np.sum((observed_sorted - clean_sorted[::-1] * negative) ** 2, axis=0)
    scale = np.where(err_positive <= err_negative, positive, negative)
    degenerate = int(np.sum(np.abs(scale) < 1e-8))
    unmixed = reordered / np.where(
        np.abs(scale) < 1e-8, 1.0, scale
    ).astype(np.float32)
    resid = float(np.linalg.norm(np.sort(unmixed, axis=0) - clean_sorted) / (np.linalg.norm(E) + 1e-12))
    return {
        "unmixed": unmixed,
        "scale": scale,
        "relative_residual": resid,
        "degenerate_columns": degenerate,
        "scale_summary": {"mean": float(scale.mean()), "min": float(scale.min()),
                          "max": float(scale.max())},
    }


def token_permutation_from_embedding(dep, unmixed: np.ndarray,
                                    ref_types: Sequence[int]) -> dict:
    """Nearest-neighbour the un-mixed deployment rows against the clean table.

    Returns the recovered ``dep_row -> clean_id`` map for ``ref_types``. Accuracy
    against the secret permutation is a scoring-side quantity and is deliberately
    not computed here: this function is on the attack side and never reads
    ``dep.perm``.
    """
    ref = np.asarray(list(ref_types), dtype=np.int64)
    queries = np.asarray(dep.E_clean, dtype=np.float32)[ref]
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    table = unmixed / (np.linalg.norm(unmixed, axis=1, keepdims=True) + 1e-12)
    recovered: dict[int, int] = {}
    for start in range(0, len(ref), 256):
        block = qn[start:start + 256]
        nn = (block @ table.T).argmax(axis=1)
        for offset, dep_row in enumerate(nn):
            recovered[int(dep_row)] = int(ref[start + offset])
    return {
        "mapping": recovered,
        "n_types": int(len(ref)),
    }
