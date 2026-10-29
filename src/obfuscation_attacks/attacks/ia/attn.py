"""AloePri Attn-IA (AloePri App. D.1).

Invariant (FORMULA_LEDGER: IA-ATTN-1), quoting the paper's Attn-IA paragraph:
with ``q = e W_query^(i)`` and ``Q = W_e W_query^(i)`` the all-token query
matrix of head ``i``, for any RoPE block index ``j``

    e (Q[:,j]^T Q[:,j])^{-1} e^T  =  e~ (Q~[:,j]^T Q~[:,j])^{-1} e~^T

Read as a whitening / Mahalanobis quadratic form of the per-token query block:
``q_j G_j^{-1} q_j^T`` with ``G_j = Q[:,j]^T Q[:,j]`` the 2x2 Gram of the j-th
RoPE block over the vocabulary. AloePri obfuscates each head's query as
``W~_query^(i) = Q^_q W_query^(i) R^_qk H^_qk Z^_block`` (Algorithm 2) and the
embedding as ``W~_embed = Pi W*_embed P^_embed`` with ``P^_embed Q^_q = I``, so
on a block ``Q~[:,j] = Pi Q[:,j] A_j`` for an invertible 2x2 ``A_j = R_j H_j``.
Because ``G~_j = A_j^T G_j A_j`` and ``q~_j = q_j A_j``, the ``A_j`` cancel and
the quadratic form is preserved. The same cancellation holds for the key side
(``W~_key^(i) = Q^_k W_key^(i) R^_qk H^_qk^{-1} Z^_block^T`` with
``P^_embed Q^_k = I``), so both query and key blocks yield invariants.

Scope vs. the paper's analysis: the invariant is derived "ignoring the head
permutation and block permutation techniques" and holds exactly only at
``alpha_e = 0`` (no embedding noise, so ``W*_embed = W_e``). AloePri's defense
is exactly to break those assumptions: additive noise (``alpha_e = 1.0`` in
Table 10 for Tab. 2/6/7) plus permutation over attention heads and RoPE blocks.

Generalization implemented here (the paper treats a single head and aligns
blocks by index):

* ``Q`` / ``Q_obf`` may be 2-D ``(tokens, d_head)`` for a single head/layer or
  3-D ``(heads_or_layers, tokens, d_head)``; every head/layer is processed and
  its valid blocks are concatenated into one per-token invariant vector.
* Key blocks ``K`` / ``K_obf`` are used additionally when exposed.
* ``pairing`` selects how the two columns of a RoPE block are chosen:
  ``adjacent`` (columns ``2j, 2j+1``; the default) or ``half_split`` (columns
  ``i`` and ``i + d_head/2``; the HuggingFace ``rotate_half`` layout).
* A partial final block (width not a multiple of ``block_size``, i.e. not a
  whole 2-column RoPE block) is REJECTED with a clear error, never silently
  used.

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


def rope_block_columns(
    d_head: int, block_size: int = 2, pairing: str = "adjacent"
) -> list[tuple[int, ...]]:
    """Return the column-index tuple of each RoPE block of a ``d_head`` space.

    * ``adjacent`` - contiguous groups of ``block_size`` columns:
      ``(0, 1), (2, 3), ...`` (the standard RoPE 2x2 block for ``block_size=2``).
    * ``half_split`` - the HuggingFace ``rotate_half`` pairing ``(i, i + d/2)``;
      this is the 2-column RoPE block, so ``block_size`` must be 2.

    A partial final block - ``d_head`` not a multiple of ``block_size`` for
    ``adjacent``, or an odd ``d_head`` for ``half_split`` - is rejected with a
    clear :class:`ValueError` rather than silently used as a narrower block.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")
    if pairing == "adjacent":
        if d_head % block_size != 0:
            raise ValueError(
                f"query/key width {d_head} is not a multiple of block_size "
                f"{block_size}; refusing to use a partial last RoPE block"
            )
        return [tuple(range(s, s + block_size)) for s in range(0, d_head, block_size)]
    if pairing == "half_split":
        if block_size != 2:
            raise ValueError(
                "half_split is the rotate_half 2-column RoPE block; set "
                f"block_size=2 (got {block_size})"
            )
        if d_head % 2 != 0:
            raise ValueError(
                f"half_split needs an even width; width {d_head} is odd "
                "(partial RoPE block)"
            )
        half = d_head // 2
        return [(i, i + half) for i in range(half)]
    raise ValueError(f"unknown pairing {pairing!r}; use 'adjacent' or 'half_split'")


def whitening_invariant(
    Q: np.ndarray,
    block_size: int = 2,
    pairing: str = "adjacent",
    cond_limit: float = _COND_LIMIT,
) -> tuple[np.ndarray, list[int]]:
    """Per-token whitening invariant ``q_j G_j^{-1} q_j^T`` for each RoPE block.

    ``Q`` is one 2-D ``(tokens, d_head)`` query/key block. Returns
    ``(invariant (tokens, num_valid_blocks), used_block_starts)`` where each
    used block is identified by its first column. A singular / ill-conditioned
    Gram block is SKIPPED (never pseudo-inverted).
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2:
        raise ValueError(
            f"whitening_invariant expects a 2-D (tokens, d_head) block; got {Q.shape}"
        )
    n, dq = Q.shape
    blocks = rope_block_columns(dq, block_size, pairing)
    cols: list[np.ndarray] = []
    used_blocks: list[int] = []
    for cidx in blocks:
        block = Q[:, list(cidx)]  # (n, b)
        if block.shape[1] == 0:
            continue
        G = block.T @ block  # (b, b) Gram over tokens
        cond = np.linalg.cond(G) if G.shape[0] > 0 else np.inf
        if not np.isfinite(cond) or cond > cond_limit:
            continue  # singular / ill-conditioned block: skip, do not pseudo-invert
        Ginv = np.linalg.inv(G)
        vals = np.einsum("nb,bc,nc->n", block, Ginv, block)  # q G^-1 q^T per token
        cols.append(vals)
        used_blocks.append(int(cidx[0]))
    if not cols:
        return np.empty((n, 0)), used_blocks
    return np.stack(cols, axis=1), used_blocks


def _as_head_stack(w: np.ndarray) -> np.ndarray:
    """Return a 3-D ``(heads/layers, tokens, d_head)`` view of a 2-D/3-D weight."""
    w = np.asarray(w, dtype=float)
    if w.ndim == 2:
        return w[None]
    if w.ndim == 3:
        return w
    raise ValueError(
        "query/key weight must be 2-D (tokens, d_head) or 3-D "
        f"(heads/layers, tokens, d_head); got {w.shape}"
    )


def stacked_invariant(
    groups: dict[str, np.ndarray],
    block_size: int = 2,
    pairing: str = "adjacent",
    cond_limit: float = _COND_LIMIT,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Concatenate per-block whitening invariants across weight groups/heads.

    ``groups`` maps a label (e.g. ``"Q"`` or ``"K"``) to a 2-D or 3-D weight.
    Each head/layer of each group contributes its valid blocks. Returns
    ``(invariant (tokens, num_valid), keys)`` where every key is
    ``(group, head_index, block_start)`` so plaintext and obfuscated sides can
    be aligned block-for-block.
    """
    cols: list[np.ndarray] = []
    keys: list[tuple[str, int, int]] = []
    n_tokens: int | None = None
    for group, w in groups.items():
        stack = _as_head_stack(w)
        for h in range(stack.shape[0]):
            inv_h, used_h = whitening_invariant(stack[h], block_size, pairing, cond_limit)
            if n_tokens is None:
                n_tokens = inv_h.shape[0]
            elif inv_h.shape[0] != n_tokens:
                raise ValueError(
                    "inconsistent token count across heads/groups: "
                    f"{inv_h.shape[0]} != {n_tokens}"
                )
            for c, start in enumerate(used_h):
                cols.append(inv_h[:, c])
                keys.append((group, h, int(start)))
    if not cols:
        return np.empty((n_tokens or 0, 0)), keys
    return np.stack(cols, axis=1), keys


@register
class AttnIA(Attack):
    name = "ia.attn"
    family = "ia"
    variant = "aloepri"
    required_capabilities = frozenset({Capability.WHITE_BOX_WEIGHTS})

    def _run(self, ctx: AttackContext) -> AttackResult:
        names = ctx.weight_names()
        if not {"Q", "Q_obf"}.issubset(names):
            return AttackResult.not_applicable(
                f"Attn-IA needs ['Q', 'Q_obf']; have {names}"
            )
        block_size = int(self.params.get("block_size", 2))
        pairing = str(self.params.get("pairing", "adjacent"))
        use_key = {"K", "K_obf"}.issubset(names)

        plain_groups: dict[str, np.ndarray] = {"Q": ctx.weight("Q")}
        obf_groups: dict[str, np.ndarray] = {"Q": ctx.weight("Q_obf")}
        if use_key:
            plain_groups["K"] = ctx.weight("K")
            obf_groups["K"] = ctx.weight("K_obf")

        try:
            plain_inv, plain_keys = stacked_invariant(plain_groups, block_size, pairing)
            obf_inv, obf_keys = stacked_invariant(obf_groups, block_size, pairing)
        except ValueError as exc:
            # Shape / layout error (e.g. partial RoPE block): report, do not crash.
            return AttackResult.failed(str(exc))

        if plain_inv.shape[1] == 0 or obf_inv.shape[1] == 0:
            return AttackResult.not_applicable(
                "no invertible query/key Gram block; invariant unavailable "
                "(no pseudo-inverse used)"
            )
        obf_key_set = set(obf_keys)
        common = [k for k in plain_keys if k in obf_key_set]
        if not common:
            return AttackResult.not_applicable(
                "no block invertible on both plaintext and obfuscated sides"
            )
        p_idx = [plain_keys.index(k) for k in common]
        o_idx = [obf_keys.index(k) for k in common]
        mapping, candidates = match_by_invariant(
            plain_inv[:, p_idx], obf_inv[:, o_idx], top_k=int(self.params.get("top_k", 10))
        )

        # Blocks skipped on the plaintext side (per group/head), for diagnostics.
        skipped: list[tuple[str, int, int]] = []
        for group, w in plain_groups.items():
            stack = _as_head_stack(w)
            for h in range(stack.shape[0]):
                try:
                    all_blocks = [c[0] for c in rope_block_columns(stack[h].shape[1], block_size, pairing)]
                except ValueError:
                    all_blocks = []
                used_here = {b for (g, hh, b) in plain_keys if g == group and hh == h}
                skipped.extend((group, h, int(b)) for b in all_blocks if b not in used_here)

        return AttackResult(
            recovered_mapping=mapping,
            mapping_candidates=candidates,
            diagnostics={
                "block_size": block_size,
                "pairing": pairing,
                "used_key_side": use_key,
                "num_heads_q": int(_as_head_stack(plain_groups["Q"]).shape[0]),
                "num_invariant_dims": len(common),
                "used_blocks": [list(k) for k in common],
                "skipped_blocks_plain": [list(s) for s in skipped],
            },
        )
