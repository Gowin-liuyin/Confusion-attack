"""Score an :class:`AttackResult` against a sample's ground truth.

Dispatches on which result fields are populated, so one scorer serves every
family: token recovery (NN/VMA-autoregressive/VMA-osnip/EDNN/ISA/IMA/SDA),
mapping recovery (TFMA/Gate-IA/Attn-IA), and permutation recovery (weight VMA).
An honest ``not_applicable``/``failed`` status is reported as-is, never as a 0%
success rate.

Counting rules (Goal section 7):

* Special positions (``special_token_mask``) are excluded from *every* token
  metric - including token P/R/F1 and exact-sequence match, which are computed
  on the special-filtered positions so the rule is applied uniformly.
* Mapping metrics are counted over the cipher symbols the attack actually
  observed (the keys of the recovered mapping / candidates), never over vocab
  entries that never appeared in the ciphertext.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.contracts import AttackResult, AttackSample, AttackStatus
from ..metrics import (
    asr_at_k,
    bleu4,
    exact_sequence_match,
    mapping_asr_at_k,
    recovered_cosine_similarity,
    symbol_error_rate,
    token_prf1,
    total_and_clean_asr,
    ttrsr,
)


def _counted_positions(n: int, special_mask) -> list[int]:
    if special_mask is None:
        return list(range(n))
    mask = np.asarray(special_mask, dtype=bool)
    return [i for i in range(n) if not mask[i]]


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
        rec_list = recovered.tolist()
        cands = (
            result.candidates
            if result.candidates is not None
            else [[int(t)] for t in rec_list]
        )
        special = sample.special_token_mask
        # Special positions are filtered out uniformly for P/R/F1 and exact match.
        kept = _counted_positions(len(truth), special)
        truth_kept = [truth[i] for i in kept]
        pred_kept = [rec_list[i] if i < len(rec_list) else -1 for i in kept]
        prf = token_prf1(pred_kept, truth_kept)
        tok = {
            "ttrsr": ttrsr(rec_list, truth, special_mask=special),
            "symbol_error_rate": symbol_error_rate(rec_list, truth, special_mask=special),
            "token_precision": prf.precision,
            "token_recall": prf.recall,
            "token_f1": prf.f1,
            "exact_sequence_match": exact_sequence_match(pred_kept, truth_kept),
        }
        for k in ks:
            tok[f"asr@{k}"] = asr_at_k(cands, truth, k, special_mask=special)
        if sample.content_mask is not None:
            for k in ks:
                total, clean = total_and_clean_asr(
                    cands, truth, k, special_mask=special, content_mask=sample.content_mask,
                )
                tok[f"total_asr@{k}"] = total
                tok[f"clean_asr@{k}"] = clean
        # CosSim (token-level proxy) and BLEU-4 when a clean embedding table is known.
        if sample.reference_embeddings is not None:
            tok["cosine_similarity"] = recovered_cosine_similarity(
                rec_list, truth, sample.reference_embeddings, special_mask=special
            )
        tok["bleu4"] = bleu4(rec_list, truth)
        out["token_recovery"] = tok

    # --- substitution-map recovery ------------------------------------------
    true_map = (sample.meta or {}).get("true_mapping")
    if (result.recovered_mapping is not None or result.mapping_candidates is not None) and true_map:
        true_map = {int(k): int(v) for k, v in true_map.items()}
        # Count only over cipher symbols the attack actually observed (the keys it
        # produced), never over vocab symbols absent from the ciphertext.
        if result.recovered_mapping is not None:
            observed = [int(c) for c in result.recovered_mapping]
        else:
            observed = [int(c) for c in result.mapping_candidates]
        observed = [c for c in observed if c in true_map]
        mp: dict = {"num_observed_symbols": len(observed)}
        if result.recovered_mapping is not None:
            correct = sum(1 for c in observed if result.recovered_mapping.get(c) == true_map[c])
            mp["mapping_top1_accuracy"] = correct / len(observed) if observed else 0.0
        if result.mapping_candidates is not None:
            for k in ks:
                mp[f"mapping_asr@{k}"] = mapping_asr_at_k(
                    result.mapping_candidates, true_map, k, eval_keys=observed
                )
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
