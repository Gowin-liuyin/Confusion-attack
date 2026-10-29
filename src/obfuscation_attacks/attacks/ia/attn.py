"""AloePri Attn-IA (AloePri App. D.1).

Invariant (FORMULA_LEDGER: IA-ATTN-1):

    e (Q_{:,j}^T Q_{:,j})^{-1} e^T  =  e~ (Q~_{:,j}^T Q~_{:,j})^{-1} e~^T

We read this as a whitening / Mahalanobis quadratic form of the per-token query
block: with ``Q = W_e W_query`` the all-token query matrix, ``Q_{:,j}`` the j-th
column block (aligned with RoPE 2x2 blocks), and ``e`` the token's own query
block, the quantity ``q_block G_j^{-1} q_block^T`` with ``G_j = Q_{:,j}^T Q_{:,j}``
is invariant under the invertible block transforms the obfuscation applies to
the query space (A^T G A and qA cancel). Stacking valid blocks gives a per-token
invariant vector that reverses the mapping Pi.

Faithfulness guard: when a Gram block ``G_j`` is singular/ill-conditioned the
block is SKIPPED (recorded), never replaced by a pseudo-inverse. If no block is
invertible the attack reports that it cannot run.
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register
from .core import match_by_invariant

_COND_LIMIT = 1e12


def whitening_invariant(
    Q: np.ndarray, block_size: int, cond_limit: float = _COND_LIMIT
) -> tuple[np.ndarray, list[int]]:
    """Return (per-token invariant matrix, list of used block start columns)."""
    Q = np.asarray(Q, dtype=float)
    n, dq = Q.shape
    cols = []
    used_blocks: list[int] = []
    for start in range(0, dq, block_size):
        block = Q[:, start : start + block_size]  # (n, b)
        G = block.T @ block  # (b, b) Gram over tokens
        if block.shape[1] == 0:
            continue
        cond = np.linalg.cond(G) if G.shape[0] > 0 else np.inf
        if not np.isfinite(cond) or cond > cond_limit:
            continue  # singular / ill-conditioned block: skip, do not pseudo-invert
        Ginv = np.linalg.inv(G)
        vals = np.einsum("nb,bc,nc->n", block, Ginv, block)  # q G^-1 q^T per token
        cols.append(vals)
        used_blocks.append(start)
    if not cols:
        return np.empty((n, 0)), used_blocks
    return np.stack(cols, axis=1), used_blocks


@register
class AttnIA(Attack):
    name = "ia.attn"
    family = "ia"
    variant = "aloepri"
    required_capabilities = frozenset({Capability.WHITE_BOX_WEIGHTS})

    def _run(self, ctx: AttackContext) -> AttackResult:
        names = ctx.weight_names()
        required = {"Q", "Q_obf"}
        if not required.issubset(names):
            return AttackResult.not_applicable(f"Attn-IA needs {sorted(required)}; have {names}")
        block_size = int(self.params.get("block_size", 2))
        plain_inv, plain_blocks = whitening_invariant(ctx.weight("Q"), block_size)
        obf_inv, obf_blocks = whitening_invariant(ctx.weight("Q_obf"), block_size)
        if plain_inv.shape[1] == 0 or obf_inv.shape[1] == 0:
            return AttackResult.not_applicable(
                "no invertible query Gram block; invariant unavailable (no pseudo-inverse used)"
            )
        # Use the blocks that are valid on both sides.
        common = [b for b in plain_blocks if b in set(obf_blocks)]
        if not common:
            return AttackResult.not_applicable("no block invertible on both plaintext and obf sides")
        p_idx = [plain_blocks.index(b) for b in common]
        o_idx = [obf_blocks.index(b) for b in common]
        mapping, candidates = match_by_invariant(
            plain_inv[:, p_idx], obf_inv[:, o_idx], top_k=int(self.params.get("top_k", 10))
        )
        return AttackResult(
            recovered_mapping=mapping,
            mapping_candidates=candidates,
            diagnostics={
                "block_size": block_size,
                "used_blocks": common,
                "skipped_blocks_plain": sorted(set(range(0, ctx.weight("Q").shape[1], block_size)) - set(plain_blocks)),
            },
        )
