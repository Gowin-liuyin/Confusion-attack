"""Output-side LM / n-gram reconstruction (plan class C14, P1).

**Provenance.** InferDPT (TDSC 2025; official repo ``mengtong0110/InferDPT``,
"Privacy-Preserving Inference for Closed-box Large Language Model") and RANTEXT
evaluate reconstruction of the *text* from what the server side can see, with
n-gram overlap, embedding cosine similarity and token-level top-k recovery; the
GPT-4-based reconstruction in that line of work is the strongest variant.

**What this module implements.** Two variants, both on the plan's S5 output
surface:

* ``lm_recon.ngram`` - a real, runnable attack: decode the observed output token
  stream (frequency matcher, shared symbol table with C10/C12), then report the
  recovered sequence so the scorer can compute the paper's text metrics
  (1..4-gram overlap, CosSim, BLEU-4, exact match, TTRSR).
* ``lm_recon.local_lm`` - the GPT-4 reconstruction *substitute*. It needs a
  generative model handle. When the harness has none (the case on this machine:
  no API key, no local generative backend wired into the runner) the attack
  returns an honest ``not_applicable`` carrying the reason - the plan forbids
  inventing a number for an attack that did not run (§3.2-7 / §7.5).

**D11 discipline.** Both variants are output-side: they must never be reported as
the input-side scheme's protection. The attack-model spec marks them
``main_table=False`` and the status doc keeps them in Limitations + appendix.
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, AttackSample, Capability
from ...metrics import bleu4, ngram_overlap_ratio, recovered_cosine_similarity
from .._decode import frequency_decode
from ..base import Attack, register


def _decode(ctx: AttackContext, top_k: int) -> tuple[dict, str]:
    """Decode the observed output stream; plaintext arm needs no substitution decode.

    ``DETERMINISTIC_PERMUTATION`` decides the path, exactly as in C13. The
    plan's output-side control is "plaintext output must be recovered well"
    (§7.1 / §4 P0-10), so the identity arm must not be forced through a
    frequency matcher that cannot recover an identity map from a short stream.
    """
    if ctx.descriptor.get("arm") == "plaintext":
        stream = np.asarray(ctx.token_ciphertext(), dtype=int).ravel()
        return (
            {
                "recovered": stream.copy(),
                "candidates": [[int(s)] for s in stream],
                "mapping": {int(s): int(s) for s in np.unique(stream)},
                "coverage": 1.0,
                "num_observed_symbols": int(np.unique(stream).size),
                "candidate_top_k": 1,
            },
            "plaintext wire (scenario-declared positive control)",
        )
    if not ctx.is_deterministic_permutation():
        raise ValueError(
            "obfuscated wire exposes neither a declared plaintext control nor a "
            "deterministic-permutation decode path"
        )
    decoded = frequency_decode(ctx, top_k=top_k)
    return decoded, "frequency_decode; metrics: 1..4-gram overlap / CosSim / BLEU-4"


@register
class NgramReconstruction(Attack):
    """n-gram / CosSim reconstruction from the decoded output stream."""

    name = "lm_recon.ngram"
    family = "lm_recon"
    variant = "inferdpt_metrics"
    required_capabilities = frozenset({Capability.TOKEN_CIPHERTEXT})

    def _run(self, ctx: AttackContext) -> AttackResult:
        try:
            decoded, method = _decode(ctx, top_k=int(self.params.get("top_k", 100)))
        except ValueError as exc:
            return AttackResult.not_applicable(str(exc))
        return AttackResult(
            recovered_token_ids=decoded["recovered"],
            candidates=decoded["candidates"],
            recovered_mapping=decoded["mapping"],
            diagnostics={
                "score_as": "lm_recon",
                "method": method,
                "candidate_coverage": decoded["coverage"],
                "reference_supported_types": decoded.get("reference_supported_types"),
                "candidate_top_k": decoded["candidate_top_k"],
                "num_observed_symbols": decoded["num_observed_symbols"],
            },
        )


@register
class LocalLMReconstruction(Attack):
    """GPT-4 reconstruction substitute: requires a generative LM handle."""

    name = "lm_recon.local_lm"
    family = "lm_recon"
    variant = "local_lm_substitute"
    required_capabilities = frozenset({Capability.TOKEN_CIPHERTEXT})

    def _run(self, ctx: AttackContext) -> AttackResult:
        model = getattr(ctx, "attacker_model", None)
        if model is None or not hasattr(model, "generate_from_token_ids"):
            return AttackResult.not_applicable(
                "GPT-4 reconstruction is unavailable in this harness (no API access), and no "
                "local generative handle exposing generate_from_token_ids was supplied. "
                "Plan rule: an attack that did not run reports no number (never 0%)."
            )
        try:  # pragma: no cover - needs a generative backend
            decoded, _ = _decode(ctx, top_k=int(self.params.get("top_k", 100)))
        except ValueError as exc:
            return AttackResult.not_applicable(str(exc))
        prompt = np.asarray(decoded["recovered"], dtype=int)
        regenerated = model.generate_from_token_ids(
            prompt.tolist(),
            max_new_tokens=int(self.params.get("max_new_tokens", 32)),
            temperature=float(self.params.get("temperature", 0.0)),
        )
        return AttackResult(
            recovered_token_ids=np.asarray(regenerated, dtype=int),
            diagnostics={
                "score_as": "lm_recon",
                "method": "local generative LM over the decoded stream",
                "model": getattr(model, "name", type(model).__name__),
                "temperature": float(self.params.get("temperature", 0.0)),
            },
        )


def score_output_recon(result: AttackResult, sample: AttackSample) -> dict:
    """Paper text metrics on the recovered output (InferDPT / RANTEXT style)."""
    recovered = result.recovered_token_ids
    truth = sample.plaintext_token_ids
    if recovered is None or truth is None:
        return {"status": "no_recovery"}
    pred = np.asarray(recovered).tolist()
    gold = np.asarray(truth).tolist()
    out = {
        "ngram_overlap": {
            f"{n}-gram": ngram_overlap_ratio(pred, gold, n)
            for n in (1, 2, 3, 4)
        },
        "bleu4": bleu4(pred, gold),
        "token_embedding_cosine_proxy": (
            recovered_cosine_similarity(pred, gold, sample.reference_embeddings)
            if sample.reference_embeddings is not None
            else None
        ),
        "exact_sequence_match": int(pred == gold),
        "denominator_rule": (
            "n-gram overlap is clipped-precision over the reconstructed sequence; "
            "CosSim is the token-level proxy documented in docs/DEVIATIONS.md"
        ),
    }
    return out
