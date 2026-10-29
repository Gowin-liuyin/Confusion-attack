"""B3 - OSNIP projection VMA (OSNIP App. C.2; reuses Thomas et al. 2025).

OSNIP assesses lexical leakage by projecting perturbed hidden states to the
vocabulary space and matching against a candidate pool with L1 distance
(App. D.2 states the vocabulary-matching adversary uses L1). The candidate pool
is the per-token layer-``l`` reference representation (attacker-owned, computed
from the clean model). For robustness to an unknown hidden-dim permutation the
sorted-L1 matcher of Thomas et al. (5.2.2) is available.

This path is kept distinct from B1 (autoregressive prefix replay) and from NN
(embedding space, L2/cosine): here the space is hidden states, the metric is L1
over a vocabulary candidate pool, and Total/Clean ASR is reported downstream.
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register


@register
class OSNIPProjectionVMA(Attack):
    name = "vma.osnip_projection"
    family = "vma"
    variant = "osnip"
    required_capabilities = frozenset({Capability.HIDDEN_STATES})

    def _run(self, ctx: AttackContext) -> AttackResult:
        pool = ctx.reference_embeddings  # (P, d) per-token layer-l reference reps
        if pool is None:
            return AttackResult.not_applicable(
                "OSNIP VMA needs a vocabulary candidate pool of reference representations"
            )
        H = np.asarray(ctx.obfuscated_hidden_states(), dtype=float)
        R = np.asarray(pool, dtype=float)
        if H.shape[1] != R.shape[1]:
            return AttackResult.failed(f"dim mismatch: H={H.shape}, pool={R.shape}")

        top_k = int(self.params.get("top_k", 10))
        use_sorted = bool(self.params.get("sorted_l1", False))
        n, p = H.shape[0], R.shape[0]
        k = min(top_k, p)

        R_cmp = np.sort(R, axis=1) if use_sorted else R
        candidates: list[list[int]] = []
        recovered = np.empty(n, dtype=int)
        margins = []
        for i in range(n):
            h = np.sort(H[i]) if use_sorted else H[i]
            dists = np.sum(np.abs(R_cmp - h), axis=1)  # L1 / sorted-L1
            order = np.argsort(dists, kind="stable")
            candidates.append(order[:k].tolist())
            recovered[i] = int(order[0])
            if p >= 2:
                margins.append(float(dists[order[1]] - dists[order[0]]))
            else:
                margins.append(float("inf"))

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={
                "distance": "sorted_l1" if use_sorted else "l1",
                "candidate_pool_size": int(p),
                "num_positions": int(n),
                "mean_margin": float(np.mean(margins)) if margins else 0.0,
            },
        )
