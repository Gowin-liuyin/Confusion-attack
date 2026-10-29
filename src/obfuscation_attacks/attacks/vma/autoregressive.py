"""B1 - original autoregressive VMA (Thomas et al. 2025, *Hidden No More*).

Formula (FORMULA_LEDGER: VMA-B1-1):

    xhat_i = argmin_{v in V} || M_{<=l}([xhat_{<i}, v])_i - h_i ||_1

The causal property of a decoder-only model means position ``i`` of the replayed
hidden state depends only on the prefix ``[xhat_{<i}, v]``, so the exponential
sequence search collapses to O(|V| * N) per-position matches.

Epsilon early-accept (paper Algorithm 1): iterate candidates and accept the
first with L1 error < epsilon; if none clears the threshold, take the global
argmin. Setting ``epsilon < 0`` disables early-accept and always takes the
argmin (the paper's deterministic analysis setting). We keep both.

Hidden-dim permutation (paper 5.2.2): when the hidden dimension is permuted by
an unknown column permutation, compare with sorted-L1

    d_sorted(a, b) = || sort(a) - sort(b) ||_1
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _sorted_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.sort(a) - np.sort(b))))


@register
class AutoregressiveVMA(Attack):
    name = "vma.autoregressive"
    family = "vma"
    variant = "original"
    required_capabilities = frozenset({Capability.MODEL_REPLAY, Capability.HIDDEN_STATES})

    def _run(self, ctx: AttackContext) -> AttackResult:
        target = ctx.obfuscated_hidden_states()  # (N, d) observed states to match
        target = np.asarray(target, dtype=float)
        n = target.shape[0]

        vocab = int(ctx.descriptor.get("vocab_size") or self.params.get("vocab_size", 0))
        if vocab <= 0:
            return AttackResult.failed("vocab_size unknown; cannot enumerate candidates")

        epsilon = self.params.get("epsilon", None)  # None => pure argmin
        use_sorted = bool(self.params.get("sorted_l1", False))
        prefix_ids = list(self.params.get("prefix", []))  # optional BOS prefix
        dist_fn = _sorted_l1 if use_sorted else _l1

        recovered: list[int] = []
        per_pos_best: list[float] = []
        per_pos_accepted_by_epsilon: list[bool] = []

        prefix = list(prefix_ids)
        for i in range(n):
            h_i = target[i]
            best_v, best_d = -1, float("inf")
            accepted = False
            for v in range(vocab):
                g = np.asarray(ctx.replay(prefix + [v]), dtype=float)[-1]
                d = dist_fn(g, h_i)
                if d < best_d:
                    best_d, best_v = d, v
                if epsilon is not None and epsilon >= 0 and d < float(epsilon):
                    best_v, best_d, accepted = v, d, True
                    break
            recovered.append(best_v)
            per_pos_best.append(best_d)
            per_pos_accepted_by_epsilon.append(accepted)
            prefix.append(best_v)

        return AttackResult(
            recovered_token_ids=np.asarray(recovered, dtype=int),
            candidates=[[tid] for tid in recovered],
            diagnostics={
                "distance": "sorted_l1" if use_sorted else "l1",
                "epsilon": epsilon,
                "per_position_best_distance": per_pos_best,
                "accepted_by_epsilon": per_pos_accepted_by_epsilon,
                "num_positions": int(n),
            },
        )
