"""Token-Frequency Matching Attack.

Frequencies (FORMULA_LEDGER: TFMA-1):

    f_C(c) = n_C(c) / sum_c' n_C(c')          (ciphertext frequency)
    f_R(t) = n_R(t) / sum_t' n_R(t')          (prior/reference frequency)

Independent nearest-frequency matching (TFMA-2, the method AloePri uses):

    pi_hat(c) = argmin_{t in V} | f_C(c) - f_R(t) |

Global one-to-one assignment (TFMA-3, a separately named EXTENSION):

    pi_hat = argmin_{pi in Pi} sum_c ( f_C(c) - f_R(pi(c)) )^2

solved exactly with the Hungarian algorithm. The deterministic-permutation
precondition is required: the same plaintext token must always map to the same
ciphertext symbol, otherwise frequency is not preserved.
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register


def _frequencies_from_counts(counts: dict[int, int], vocab_size: int) -> np.ndarray:
    f = np.zeros(vocab_size, dtype=float)
    for tid, c in counts.items():
        if 0 <= int(tid) < vocab_size:
            f[int(tid)] = float(c)
    total = f.sum()
    if total > 0:
        f /= total
    return f


def _cipher_frequencies(cipher_ids: np.ndarray, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    ids, counts = np.unique(np.asarray(cipher_ids, dtype=int), return_counts=True)
    f = np.zeros(vocab_size, dtype=float)
    f[ids] = counts
    total = f.sum()
    if total > 0:
        f = f / total
    return f, ids  # f over full vocab, ids = observed cipher symbols


def _prepare(ctx: AttackContext):
    if ctx.reference_token_counts is None:
        return None, "TFMA needs a prior reference token-frequency table"
    cipher = ctx.token_ciphertext()
    vocab = int(ctx.descriptor.get("vocab_size") or 0)
    if vocab <= 0:
        return None, "vocab_size unknown"
    f_R = _frequencies_from_counts(dict(ctx.reference_token_counts), vocab)
    f_C, observed = _cipher_frequencies(cipher, vocab)
    return (f_C, f_R, observed, vocab), None


@register
class IndependentTFMA(Attack):
    name = "tfma.independent"
    family = "tfma"
    variant = "aloepri"
    required_capabilities = frozenset(
        {Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}
    )

    def _run(self, ctx: AttackContext) -> AttackResult:
        prepared, err = _prepare(ctx)
        if err:
            return AttackResult.not_applicable(err)
        f_C, f_R, observed, vocab = prepared
        top_k = int(self.params.get("top_k", 100))
        k = min(top_k, vocab)

        mapping: dict[int, int] = {}
        candidates: dict[int, list[int]] = {}
        ties: dict[int, int] = {}
        for c in observed.tolist():
            cost = np.abs(f_R - f_C[c])  # |f_C(c) - f_R(t)| for all t
            order = np.argsort(cost, kind="stable")
            candidates[c] = order[:k].tolist()
            mapping[c] = int(order[0])
            # Tie rule (documented, not arbitrary): np.argsort(kind="stable")
            # preserves the natural ascending-id order among equal costs, so the
            # LOWEST token id wins an exact frequency tie. We also count how many
            # tokens share the winner's cost so the ambiguity stays visible in
            # diagnostics rather than being hidden.
            ties[c] = int(np.sum(cost == cost[order[0]]))

        return AttackResult(
            recovered_mapping=mapping,
            mapping_candidates=candidates,
            diagnostics={
                "method": "independent_nearest_frequency",
                "num_observed_symbols": int(observed.size),
                "symbols_with_frequency_ties": int(sum(1 for v in ties.values() if v > 1)),
                "tie_rule": "stable argsort by |df|: lowest token id wins equal-frequency ties",
            },
        )


@register
class GlobalAssignmentTFMA(Attack):
    name = "tfma.global_assignment"
    family = "tfma"
    variant = "experimental_completion"  # NOT the paper method; named/reported apart
    required_capabilities = frozenset(
        {Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}
    )

    def _run(self, ctx: AttackContext) -> AttackResult:
        from scipy.optimize import linear_sum_assignment

        prepared, err = _prepare(ctx)
        if err:
            return AttackResult.not_applicable(err)
        f_C, f_R, observed, vocab = prepared
        obs = observed.astype(int)
        # Cost matrix over observed cipher symbols x full vocab: (f_C(c) - f_R(t))^2
        cost = (f_C[obs][:, None] - f_R[None, :]) ** 2
        rows, cols = linear_sum_assignment(cost)
        row_to_assigned = {int(r): int(c) for r, c in zip(rows.tolist(), cols.tolist())}
        mapping = {int(obs[r]): row_to_assigned[int(r)] for r in rows.tolist()}
        top_k = int(self.params.get("top_k", 100))
        k = min(top_k, vocab)
        candidates: dict[int, list[int]] = {}
        for i, c in enumerate(obs.tolist()):
            # Rank per symbol by the same (df)^2 cost, BUT place the globally
            # ASSIGNED token first. Ranking by the per-symbol cost alone (the old
            # behaviour) could put a different token at rank 1, so mapping_asr@1
            # disagreed with the Hungarian top-1 accuracy this variant is meant
            # to report. Putting the assignment first makes candidates[c][0] ==
            # mapping[c]; the remaining tokens follow in ascending (df)^2 order.
            order = np.argsort(cost[i], kind="stable").tolist()
            assigned_tok = row_to_assigned.get(i)
            if assigned_tok is None:
                ranked = order
            else:
                ranked = [assigned_tok] + [t for t in order if t != assigned_tok]
            candidates[c] = ranked[:k]
        return AttackResult(
            recovered_mapping=mapping,
            mapping_candidates=candidates,
            diagnostics={
                "method": "hungarian_global_assignment",
                "num_observed_symbols": int(obs.size),
                "candidate_order": "assigned token first, then ascending (delta f)^2",
            },
        )
