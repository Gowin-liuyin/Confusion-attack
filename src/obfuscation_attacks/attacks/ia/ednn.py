"""Original EDNN invariant attack (Lin et al. 2024, EMNLP).

Glide-reflection transform (FORMULA_LEDGER: IA-EDNN-1):

    e'_i = e_i - 2 (e_i . l_i / l_i . l_i) l_i + t_i,  l_i = 1*a_i,  t_i = 1*b_i

Because every coordinate of ``l_i`` equals ``a_i`` and every coordinate of
``t_i`` equals ``b_i``, the coordinate *difference* is preserved:

    e'_i[k1] - e'_i[k2] = e_i[k1] - e_i[k2]     (IA-EDNN-2)

This is NOT a general mirror/translation/affine invariant - it relies on the
all-equal-coordinate structure of ``l_i`` and ``t_i``.

EDNN distance (IA-EDNN-3):

    D_ij = || (ebar_i - lshift(ebar_i)) - (e_j - lshift(e_j)) ||

where ``lshift`` is a circular left shift by one position (the last coordinate
wraps around; it is never dropped).
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register


def lshift(x: np.ndarray) -> np.ndarray:
    """Circular left shift by one position (wrap-around, keep all coordinates)."""
    return np.roll(x, -1, axis=-1)


def difference_representation(E: np.ndarray) -> np.ndarray:
    """Return E - lshift(E), the coordinate-difference fingerprint."""
    return E - lshift(E)


@register
class EDNN(Attack):
    name = "ia.ednn"
    family = "ia"
    variant = "original"
    required_capabilities = frozenset({Capability.EMBEDDINGS})

    def _run(self, ctx: AttackContext) -> AttackResult:
        ref = ctx.reference_embeddings
        if ref is None:
            return AttackResult.not_applicable("EDNN needs a reference embedding matrix")
        Ebar = np.asarray(ctx.obfuscated_embeddings(), dtype=float)  # transformed observations
        Eref = np.asarray(ref, dtype=float)
        if Ebar.shape[1] != Eref.shape[1]:
            return AttackResult.failed(f"dim mismatch obs={Ebar.shape} ref={Eref.shape}")

        top_k = int(self.params.get("top_k", 10))
        obs_diff = difference_representation(Ebar)  # (N, d)
        ref_diff = difference_representation(Eref)  # (V, d)

        n, v = Ebar.shape[0], Eref.shape[0]
        k = min(top_k, v)
        candidates: list[list[int]] = []
        recovered = np.empty(n, dtype=int)
        margins = []
        for i in range(n):
            d = np.linalg.norm(ref_diff - obs_diff[i], axis=1)  # D_ij
            order = np.argsort(d, kind="stable")
            candidates.append(order[:k].tolist())
            recovered[i] = int(order[0])
            margins.append(float(d[order[1]] - d[order[0]]) if v >= 2 else float("inf"))

        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={"num_positions": int(n), "vocab": int(v),
                         "mean_margin": float(np.mean(margins)) if margins else 0.0},
        )
