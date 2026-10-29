"""B1 - original autoregressive VMA (Thomas et al. 2025, *Hidden No More*).

Formula (FORMULA_LEDGER: VMA-B1-1):

    xhat_i = argmin_{v in V} || M_{<=l}([xhat_{<i}, v])_i - h_i ||_1

The causal property of a decoder-only model means position ``i`` of the replayed
hidden state depends only on the prefix ``[xhat_{<i}, v]``, so the exponential
sequence search collapses to O(|V| * N) per-position matches (paper Sec 4.2.2 /
Algorithm 1 lines 6-8).

Epsilon early-accept + proposal model (paper Algorithm 1):

* line 3  ``V_ordered <- argsort(P(xhat))`` iterates the vocabulary in the order
  given by a *proposal model* P (the attacked model itself in the paper, Sec
  4.4). The proposal order is conditioned on the already-decoded prefix so the
  epsilon-stopping condition is met early (paper Sec 4.2.3 / App E lines
  864-868).
* lines 13-16  accept the first candidate whose L1 error is ``< epsilon`` and
  break; lines 18-19 fall back to the global argmin (``best_match``) when no
  candidate cleared the threshold.

Because the early-accept is driven by the proposal order, epsilon is only
meaningful together with a proposal ordering. If epsilon is requested but no
proposal ordering is supplied we DO NOT silently iterate in token-id order
(that is not the paper's algorithm and biases early-accept toward low ids);
instead we disable early-accept and take the pure global argmin, recording the
fallback in the diagnostics. Setting ``epsilon = None`` or ``epsilon < 0``
disables early-accept and always takes the global argmin (the deterministic
analysis setting). Both are kept.

Permutation types (paper Section 5, Algorithms 4-6) selected by ``permutation``:

* ``none``                Algorithm 1  - plain L1, position i matched to h_i.
* ``hidden_dimension``    Algorithm 5  - sorted-L1 (Sec 5.2.2) cancels an
  unknown per-position column permutation; position i still matched to h_i.
* ``sequence_dimension``  Algorithm 4  - the N rows are permuted; each replayed
  candidate row is matched to the best remaining (not-yet-used) observed row,
  which is then removed from the pool (plain L1).
* ``factorized_2d``       Algorithm 6  - rows + columns: sorted-L1 against the
  not-yet-used observed rows with removal.

Sorted-L1 (paper 5.2.2, FORMULA_LEDGER VMA-B1-2):

    d_sorted(a, b) = || sort(a) - sort(b) ||_1
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _sorted_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.sort(a) - np.sort(b))))


def _proposal_order(
    vocab: int,
    proposal_fn: Optional[Callable[[Sequence[int]], Sequence[int]]],
    proposal_order: Optional[Sequence[int]],
    prefix: Sequence[int],
) -> Optional[list[int]]:
    """Return the proposal-model vocabulary order for the current prefix.

    ``proposal_fn(prefix) -> ordered ids`` (paper Algorithm 1 line 3,
    ``argsort(P(xhat))``) takes precedence; a precomputed ``proposal_order`` is
    used otherwise. ``None`` means no proposal ordering is available.
    """
    if proposal_fn is not None:
        return [int(v) for v in proposal_fn(list(prefix))]
    if proposal_order is not None:
        return [int(v) for v in proposal_order]
    return None


def autoregressive_reconstruct(
    replay: Callable[[Sequence[int]], np.ndarray],
    target: np.ndarray,
    vocab: int,
    *,
    epsilon: Optional[float] = None,
    proposal_fn: Optional[Callable[[Sequence[int]], Sequence[int]]] = None,
    proposal_order: Optional[Sequence[int]] = None,
    prefix: Sequence[int] = (),
    sorted_l1: bool = False,
    sequence_permuted: bool = False,
    top_k: Optional[int] = None,
) -> dict:
    """Shared causal L1-replay reconstruction (Thomas et al. Algorithms 1/4/5/6).

    This is the common core reused by B1 (original VMA, with epsilon/proposal and
    the permutation variants) and by B3 (OSNIP, pure argmin, no epsilon/proposal).
    It never changes the L1 / sorted-L1 maths.

    Parameters
    ----------
    replay:
        ``token_ids -> (len, d)`` layer-``l`` hidden states of the clean model.
    target:
        ``(N, d)`` observed states to decode (possibly permuted).
    epsilon:
        early-accept L1 threshold; ``None`` / ``< 0`` disables it (global argmin).
    proposal_fn / proposal_order:
        proposal-model ordering hook (Algorithm 1 line 3). Required for the
        epsilon early-accept; absent, epsilon falls back to global argmin.
    sorted_l1:
        use sorted-L1 (Algorithms 5/6, hidden-dimension permutation).
    sequence_permuted:
        match against the not-yet-used observed rows with removal (Algorithms
        4/6, sequence-dimension permutation).
    """
    target = np.asarray(target, dtype=float)
    n = int(target.shape[0])
    dist_fn = _sorted_l1 if sorted_l1 else _l1

    epsilon_enabled = epsilon is not None and float(epsilon) >= 0.0
    has_proposal = proposal_fn is not None or proposal_order is not None
    epsilon_fallback: Optional[str] = None
    if epsilon_enabled and not has_proposal:
        # Algorithm 1 iterates argsort(P(xhat)); without a proposal order we must
        # not substitute token-id order for the early-accept (it would bias the
        # break toward low ids). Fall back to the pure global argmin and record.
        epsilon_enabled = False
        epsilon_fallback = (
            "epsilon requested but no proposal_fn/proposal_order supplied; "
            "early-accept disabled, used pure global argmin (not token-id order)"
        )

    recovered: list[int] = []
    candidates: list[list[int]] = []
    per_pos_best: list[float] = []
    accepted_flags: list[bool] = []

    prefix_ids = [int(v) for v in prefix]
    remaining = list(range(n))  # observed-row pool (sequence-dimension permutation)

    for i in range(n):
        order: Optional[list[int]] = None
        if epsilon_enabled:
            order = _proposal_order(vocab, proposal_fn, proposal_order, prefix_ids)
        if order is None:
            order = list(range(vocab))

        rows = remaining if sequence_permuted else [i]

        best_v, best_d, best_row = -1, float("inf"), -1
        accepted = False
        dist_by_v: dict[int, float] = {}
        for v in order:
            g = np.asarray(replay(prefix_ids + [int(v)]), dtype=float)[-1]
            local_d, local_row = float("inf"), -1
            for r in rows:
                d = dist_fn(g, target[r])
                if d < local_d:
                    local_d, local_row = d, r
            dist_by_v[int(v)] = local_d
            if local_d < best_d:
                best_d, best_v, best_row = local_d, int(v), local_row
            if epsilon_enabled and local_d < float(epsilon):
                best_d, best_v, best_row, accepted = local_d, int(v), local_row, True
                break

        recovered.append(best_v)
        per_pos_best.append(float(best_d))
        accepted_flags.append(accepted)
        if accepted:
            # Committed by the epsilon rule; the remaining order was not scored.
            candidates.append([best_v])
        else:
            ranked = sorted(dist_by_v, key=lambda vv: (dist_by_v[vv], vv))
            candidates.append(ranked[:top_k] if top_k else ranked)
        prefix_ids.append(best_v)
        if sequence_permuted and best_row >= 0:
            remaining.remove(best_row)

    return {
        "recovered": np.asarray(recovered, dtype=int),
        "candidates": candidates,
        "per_position_best_distance": per_pos_best,
        "accepted_by_epsilon": accepted_flags,
        "epsilon_enabled": epsilon_enabled,
        "epsilon_fallback": epsilon_fallback,
    }


_PERMUTATIONS = {
    "none": (False, False),               # Algorithm 1
    "hidden_dimension": (True, False),    # Algorithm 5
    "sequence_dimension": (False, True),  # Algorithm 4
    "factorized_2d": (True, True),        # Algorithm 6
}


@register
class AutoregressiveVMA(Attack):
    name = "vma.autoregressive"
    family = "vma"
    variant = "original"
    required_capabilities = frozenset({Capability.MODEL_REPLAY, Capability.HIDDEN_STATES})

    def _run(self, ctx: AttackContext) -> AttackResult:
        target = np.asarray(ctx.obfuscated_hidden_states(), dtype=float)  # (N, d)
        n = target.shape[0]

        vocab = int(ctx.descriptor.get("vocab_size") or self.params.get("vocab_size", 0))
        if vocab <= 0:
            return AttackResult.failed("vocab_size unknown; cannot enumerate candidates")

        permutation = str(self.params.get("permutation", "none")).lower()
        # Backwards-compatible alias: ``sorted_l1`` selects the hidden-dimension
        # variant (Algorithm 5) when no explicit permutation type is given.
        if permutation == "none" and bool(self.params.get("sorted_l1", False)):
            permutation = "hidden_dimension"
        if permutation not in _PERMUTATIONS:
            return AttackResult.failed(
                f"unknown permutation type {permutation!r}; "
                f"expected one of {sorted(_PERMUTATIONS)}"
            )
        use_sorted, sequence_permuted = _PERMUTATIONS[permutation]

        epsilon = self.params.get("epsilon", None)
        prefix = self.params.get("prefix", [])
        proposal_fn = self.params.get("proposal_fn", None)
        proposal_order = self.params.get("proposal_order", None)
        top_k = self.params.get("top_k", None)

        res = autoregressive_reconstruct(
            ctx.replay,
            target,
            vocab,
            epsilon=epsilon,
            proposal_fn=proposal_fn,
            proposal_order=proposal_order,
            prefix=prefix,
            sorted_l1=use_sorted,
            sequence_permuted=sequence_permuted,
            top_k=top_k,
        )

        proposal_kind = (
            "fn" if proposal_fn is not None
            else "precomputed" if proposal_order is not None
            else "none"
        )
        return AttackResult(
            recovered_token_ids=res["recovered"],
            candidates=res["candidates"],
            diagnostics={
                "distance": "sorted_l1" if use_sorted else "l1",
                "permutation": permutation,
                "epsilon": epsilon,
                "epsilon_enabled": res["epsilon_enabled"],
                "epsilon_fallback": res["epsilon_fallback"],
                "proposal": proposal_kind,
                "per_position_best_distance": res["per_position_best_distance"],
                "accepted_by_epsilon": res["accepted_by_epsilon"],
                "num_positions": int(n),
            },
        )
