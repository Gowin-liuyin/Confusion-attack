"""Score an :class:`AttackResult` against a sample's ground truth.

Dispatches on which result fields are populated, so one scorer serves every
family: token recovery (NN/VMA-autoregressive/VMA-osnip/EDNN/ISA/IMA/SDA),
mapping recovery (TFMA/Gate-IA/Attn-IA), and permutation recovery (weight VMA).
An honest ``not_applicable``/``failed`` status is reported as-is, never as a 0%
success rate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.contracts import AttackResult, AttackSample, AttackStatus
from ..metrics import (
    asr_at_k,
    exact_sequence_match,
    mapping_asr_at_k,
    token_prf1,
    total_and_clean_asr,
    ttrsr,
)


def compute_metrics(
    result: AttackResult, sample: AttackSample, ks: Sequence[int] = (1, 5, 10)
) -> dict:
    out: dict = {"status": result.status.value, "status_reason": result.status_reason}
    if result.status != AttackStatus.OK:
        return out

    ks = list(ks)

    # --- per-position token recovery ----------------------------------------
    if result.recovered_token_ids is not None and sample.plaintext_token_ids is not None:
        truth = [int(t) for t in sample.plaintext_token_ids.tolist()]
        recovered = result.recovered_token_ids
        cands = (
            result.candidates
            if result.candidates is not None
            else [[int(t)] for t in recovered]
        )
        prf = token_prf1(recovered.tolist(), truth)
        tok = {
            "ttrsr": ttrsr(recovered.tolist(), truth, special_mask=sample.special_token_mask),
            "token_precision": prf.precision,
            "token_recall": prf.recall,
            "token_f1": prf.f1,
            "exact_sequence_match": exact_sequence_match(recovered.tolist(), truth),
        }
        for k in ks:
            tok[f"asr@{k}"] = asr_at_k(cands, truth, k, special_mask=sample.special_token_mask)
        if sample.content_mask is not None:
            for k in ks:
                total, clean = total_and_clean_asr(
                    cands, truth, k,
                    special_mask=sample.special_token_mask,
                    content_mask=sample.content_mask,
                )
                tok[f"total_asr@{k}"] = total
                tok[f"clean_asr@{k}"] = clean
        out["token_recovery"] = tok

    # --- substitution-map recovery ------------------------------------------
    true_map = (sample.meta or {}).get("true_mapping")
    if (result.recovered_mapping is not None or result.mapping_candidates is not None) and true_map:
        true_map = {int(k): int(v) for k, v in true_map.items()}
        mp: dict = {}
        if result.recovered_mapping is not None:
            keys = list(true_map)
            correct = sum(1 for c in keys if result.recovered_mapping.get(c) == true_map[c])
            mp["mapping_top1_accuracy"] = correct / len(keys) if keys else 0.0
        if result.mapping_candidates is not None:
            for k in ks:
                mp[f"mapping_asr@{k}"] = mapping_asr_at_k(result.mapping_candidates, true_map, k)
        out["mapping_recovery"] = mp

    # --- permutation / matrix recovery (weight VMA) --------------------------
    if result.recovered_permutations is not None and sample.secret_matrices:
        perm: dict = {"verified": bool(result.diagnostics.get("verified", False))}
        if "Z1" in result.recovered_permutations and "Z1" in sample.secret_matrices:
            perm["Z1_exact"] = bool(
                np.array_equal(
                    np.asarray(result.recovered_permutations["Z1"]),
                    np.asarray(sample.secret_matrices["Z1"]),
                )
            )
        out["permutation_recovery"] = perm

    out["diagnostics"] = result.diagnostics
    return out
