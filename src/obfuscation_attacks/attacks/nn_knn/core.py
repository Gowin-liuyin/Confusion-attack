"""Shared retrieval core for NN / KNN.

Formulae (docs/FORMULA_LEDGER.md: NN-1, NN-2):

    euclidean:  xhat_i = argmin_v || h_i - E_v ||_2
    cosine:     d_cos(h, E_v) = 1 - (h . E_v) / (||h|| ||E_v||)

The attacker observes the perturbed representations ``h_i`` (EMBEDDINGS) and
owns a clean reference embedding matrix ``E`` (vocab x d).
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult


def _euclidean_distances(H: np.ndarray, E: np.ndarray) -> np.ndarray:
    # ||h||^2 + ||E||^2 - 2 h.E ; clipped to >=0 for numerical safety.
    h_sq = np.sum(H * H, axis=1, keepdims=True)
    e_sq = np.sum(E * E, axis=1, keepdims=True).T
    cross = H @ E.T
    d2 = np.maximum(h_sq + e_sq - 2.0 * cross, 0.0)
    return np.sqrt(d2)


def _cosine_distances(H: np.ndarray, E: np.ndarray) -> np.ndarray:
    h_norm = np.linalg.norm(H, axis=1, keepdims=True)
    e_norm = np.linalg.norm(E, axis=1, keepdims=True)
    # Zero vectors have undefined direction: assign maximal cosine distance 1.0.
    h_safe = np.where(h_norm == 0, 1.0, h_norm)
    e_safe = np.where(e_norm == 0, 1.0, e_norm)
    cos = (H @ E.T) / (h_safe @ e_safe.T)
    cos = np.clip(cos, -1.0, 1.0)
    dist = 1.0 - cos
    zero_rows = (h_norm == 0).ravel()
    zero_cols = (e_norm == 0).ravel()
    if zero_rows.any():
        dist[zero_rows, :] = 1.0
    if zero_cols.any():
        dist[:, zero_cols] = 1.0
    return dist


def run_knn(ctx: AttackContext, metric: str, top_k: int) -> AttackResult:
    E = ctx.reference_embeddings
    if E is None:
        return AttackResult.not_applicable("NN/KNN needs a reference embedding matrix")
    H = ctx.obfuscated_embeddings()
    E = np.asarray(E, dtype=float)
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or E.ndim != 2 or H.shape[1] != E.shape[1]:
        return AttackResult.failed(
            f"shape mismatch: H={H.shape}, E={E.shape} (hidden dims must match)"
        )

    if metric == "euclidean":
        dist = _euclidean_distances(H, E)
    elif metric == "cosine":
        dist = _cosine_distances(H, E)
    else:
        raise ValueError(f"unknown metric {metric!r}")

    n, vocab = dist.shape
    # Return the FULL per-position ranking: every vocabulary id in ascending
    # distance order. Both Song & Raghunathan (2020) and OSNIP (App. C.1) report
    # whether the ground-truth token appears in the Top-k of the *entire*
    # vocabulary, and the scorer (metrics.asr_at_k) slices the k it needs.
    # Truncating here would cap ASR@k at the attack's requested k, so we never
    # shrink the candidate pool in code (small pools live only in smoke configs).
    order = np.argsort(dist, axis=1, kind="stable")
    candidates = order.tolist()  # n lists, each of length == vocab
    recovered = order[:, 0].astype(int)

    # First/second candidate margin per position (diagnostic for ambiguity).
    margins = []
    for i in range(n):
        if vocab >= 2:
            d1 = dist[i, order[i, 0]]
            d2 = dist[i, order[i, 1]]
            margins.append(float(d2 - d1))
        else:
            margins.append(float("inf"))

    return AttackResult(
        recovered_token_ids=recovered,
        candidates=candidates,
        diagnostics={
            "metric": metric,
            # requested_top_k is informational only; the ranking is NOT truncated
            # to it (the scorer slices Top-k itself).
            "requested_top_k": int(top_k),
            "ranking_length": int(vocab),
            "vocab_size": int(vocab),
            "num_positions": int(n),
            "first_second_margin": margins,
            "mean_margin": float(np.mean(margins)) if margins else 0.0,
        },
    )
