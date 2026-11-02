"""AloePri Gate-IA (AloePri App. D.1).

Invariant (FORMULA_LEDGER: IA-GATE-1):

    Avg(e W_gate) = Avg(e~ W~_gate)

The gate-weighted average of a token embedding is preserved by the obfuscation
when no embedding noise is added (alpha_e = 0): the FFN gate obfuscation
``W~_gate = Q_gate W_gate Z_ffn`` permutes columns (Z_ffn) and the key matrix
cancels covariantly, and the mean over the FFN dimension is invariant to the
column permutation. Stacking the per-layer averages into a vector makes the
invariant discriminative enough to reverse the token mapping Pi.

Precondition recorded in config/tests: the invariant is exact only for
alpha_e = 0; additive embedding noise breaks it.
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register
from .core import match_by_invariant


def _as_stack(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.ndim == 2:
        return w[None]
    if w.ndim == 3:
        return w
    raise ValueError(f"gate weight must be 2-D or 3-D (L,d,hf); got {w.shape}")


def gate_invariant(E: np.ndarray, W_gate: np.ndarray) -> np.ndarray:
    """Return the (num_tokens, num_layers) stacked gate-average invariant."""
    E = np.asarray(E, dtype=float)
    Ws = _as_stack(W_gate)
    cols = []
    for l in range(Ws.shape[0]):
        activ = E @ Ws[l]  # (tokens, hf)
        cols.append(activ.mean(axis=1))  # Avg over FFN dimension
    return np.stack(cols, axis=1)  # (tokens, L)


@register
class GateIA(Attack):
    name = "ia.gate"
    family = "ia"
    variant = "aloepri"
    required_capabilities = frozenset({Capability.WHITE_BOX_WEIGHTS})

    def _run(self, ctx: AttackContext) -> AttackResult:
        names = ctx.weight_names()
        required = {"E", "E_obf", "W_gate", "W_gate_obf"}
        if not required.issubset(names):
            return AttackResult.not_applicable(
                f"Gate-IA needs {sorted(required)}; have {names}"
            )
        plain_inv = gate_invariant(ctx.weight("E"), ctx.weight("W_gate"))
        obf_inv = gate_invariant(ctx.weight("E_obf"), ctx.weight("W_gate_obf"))
        mapping, candidates = match_by_invariant(
            plain_inv, obf_inv, top_k=int(self.params.get("top_k", 10))
        )
        return AttackResult(
            recovered_mapping=mapping,
            mapping_candidates=candidates,
            diagnostics={
                "num_layers": int(plain_inv.shape[1]),
                "num_obf_tokens": int(obf_inv.shape[0]),
                "alpha_e_precondition": "invariant exact only for alpha_e = 0",
            },
        )
