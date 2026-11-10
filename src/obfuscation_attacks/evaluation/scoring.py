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
        cands = list(cands)
        if len(cands) != len(truth):
            # A shorter/truncated prediction keeps every truth position in the
            # denominator: positions without a candidate are misses, never dropped.
            cands = [
                list(cands[index]) if index < len(cands) else []
                for index in range(len(truth))
            ]
        special = sample.special_token_mask
        # Special positions are filtered out uniformly for P/R/F1 and exact match.
        kept = _counted_positions(len(truth), special)
        truth_kept = [truth[i] for i in kept]
        pred_kept = [rec_list[i] for i in kept if i < len(rec_list)]
        pred_kept.extend(rec_list[len(truth):])
        prf = token_prf1(pred_kept, truth_kept)
        tok = {
            "ttrsr": ttrsr(rec_list, truth, special_mask=special),
            "symbol_error_rate": symbol_error_rate(rec_list, truth, special_mask=special),
            "token_precision": prf.precision,
            "token_recall": prf.recall,
            "token_f1": prf.f1,
            "exact_sequence_match": exact_sequence_match(pred_kept, truth_kept),
        }
        single_hypothesis = (result.diagnostics or {}).get("candidate_semantics") == "one selected hypothesis per position; not top-k"
        scored_ks = [k for k in ks if not single_hypothesis or k == 1]
        if single_hypothesis:
            tok["candidate_coverage"] = {"status": "single_hypothesis", "available_k": 1,
                                         "unavailable_requested_k": [k for k in ks if k != 1]}
        for k in scored_ks:
            tok[f"asr@{k}"] = asr_at_k(cands, truth, k, special_mask=special)
        if sample.content_mask is not None:
            for k in scored_ks:
                total, clean = total_and_clean_asr(
                    cands, truth, k, special_mask=special, content_mask=sample.content_mask,
                )
                tok[f"total_asr@{k}"] = total
                tok[f"clean_asr@{k}"] = clean
        # CosSim (token-level proxy) and BLEU-4 when a clean embedding table is known.
        if sample.reference_embeddings is not None:
            tok["token_embedding_cosine_proxy"] = recovered_cosine_similarity(
                rec_list, truth, sample.reference_embeddings, special_mask=special
            )
        # BLEU-4 uses the same special-filtered positions as every other token
        # metric (DEVIATIONS: special positions are excluded uniformly).
        tok["bleu4"] = bleu4(pred_kept, truth_kept)
        out["token_recovery"] = tok

    # --- substitution-map recovery ------------------------------------------
    true_map = (sample.meta or {}).get("true_mapping")
    if (result.recovered_mapping is not None or result.mapping_candidates is not None) and true_map:
        true_map = {int(k): int(v) for k, v in true_map.items()}
        # Count only over cipher symbols the attack actually observed (the keys it
        # produced), never over vocab symbols absent from the ciphertext.
        if sample.ciphertext_token_ids is not None:
            # Denominator = the cipher symbols actually present in the observed
            # ciphertext, not only the keys the attack managed to emit.
            observed = sorted({
                int(c) for c in np.asarray(sample.ciphertext_token_ids).ravel()
                if int(c) in true_map
            })
        elif result.recovered_mapping is not None:
            observed = [int(c) for c in result.recovered_mapping if int(c) in true_map]
        else:
            observed = [int(c) for c in result.mapping_candidates if int(c) in true_map]
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

    # --- output-side families (C13 PII / C14 reconstruction) ----------------
    # The attack marks its own scoring口径 in ``diagnostics["score_as"]``; the
    # scorer dispatches on it so a new family cannot reuse the token-recovery
    # branch by accident (plan §5.4 keeps the denominators apart).
    score_as = (result.diagnostics or {}).get("score_as")
    if score_as == "pii":
        from ..attacks.pii import score_pii

        out["pii"] = score_pii(result, sample)
    elif score_as == "lm_recon":
        from ..attacks.lm_recon import score_output_recon

        out["output_reconstruction"] = score_output_recon(result, sample)

    out["diagnostics"] = result.diagnostics
    return out


def compute_stream_metrics(result: AttackResult, stream) -> dict:
    """Score a stream (S-C) attack result against the logged request stream.

    Only this function reads ``stream.plaintext_ids`` - the attack's evidence
    (pair table / cosine scores / nearest-partner indices) never contains the
    labels, so the SH-P linker cannot tune itself to the answer.
    """
    out: dict = {"status": result.status.value, "status_reason": result.status_reason}
    if result.status != AttackStatus.OK:
        return out
    kind = (result.stream_metrics or {}).get("kind")
    if kind in {"linkage_exact", "linkage_approx"}:
        from ..attacks.linkage import score_linkage

        out["linkage"] = score_linkage(result, stream)
    out["diagnostics"] = result.diagnostics
    return out
