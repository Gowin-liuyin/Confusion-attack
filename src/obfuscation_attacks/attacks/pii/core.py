"""PII extraction attack (plan class C13, P1; Limitations + appendix only).

**Provenance.** The plan's "PUPA" is AloePri's reference [14], which the AloePri
reference list resolves to *Papillon: Privacy preservation from internet-based
and local language model ensembles* (Li et al., NAACL 2025) - i.e. **PUPA is a
dataset, not an attack paper**. AloePri then scores entity recovery with its own
entity scorer (``evaluate_entities.py`` in the official code drop, whose
convention matches plan §5.4 formula 3 exactly: an entity counts as recovered
only when *all* of its tokens are correct at their original positions, and "no
entity" is null rather than 0%). This module implements that protocol; the metric
is therefore an *entity-level* extension of token recovery, and the plan's D11
rule applies: these numbers go to Limitations / appendix and may never be written
as the input-side scheme's protection.

**What the attack does.** It decodes the observed output-token stream with the
frequency matcher (same symbol table as C10/C12), then runs the attacker's public
gazetteer scan over the decoded ids to claim entity spans. The scorer - which
alone holds the annotation - computes PIIRSR, so a claimed span cannot be scored
against itself.
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, AttackSample, Capability
from ...metrics import entity_recovered, pii_recovery_success_ratio
from .._decode import detect_entities, frequency_decode
from ..base import Attack, register


@register
class PIIExtraction(Attack):
    """Entity recovery from a decoded output stream (PIIRSR)."""

    name = "pii.extract"
    family = "pii"
    variant = "aloepri"
    # TOKEN_CIPHERTEXT is required; DETERMINISTIC_PERMUTATION only selects the
    # decode path. Requiring it would make the plaintext control arm (no
    # permutation applied at all) unreachable - and the plan's P0-10 control is
    # exactly "plaintext output must be recovered well" (§7.1 / §4 P0-10), while
    # the permuted arm must run the substitution decode.
    required_capabilities = frozenset({Capability.TOKEN_CIPHERTEXT})

    def _run(self, ctx: AttackContext) -> AttackResult:
        if ctx.descriptor.get("arm") == "plaintext":
            # Plaintext control arm: nothing was obfuscated, so the observed
            # symbols *are* the output tokens. The scenario declares this arm
            # explicitly; absence of the permutation capability alone is not
            # evidence that the wire is plaintext.
            stream = np.asarray(ctx.token_ciphertext(), dtype=int).ravel()
            decoded = {
                "recovered": stream.copy(),
                "candidates": [[int(s)] for s in stream],
                "mapping": {int(s): int(s) for s in np.unique(stream)},
                "coverage": 1.0,
                "num_observed_symbols": int(np.unique(stream).size),
                "candidate_top_k": 1,
            }
            method = "plaintext wire (scenario-declared positive control)"
        elif ctx.is_deterministic_permutation():
            try:
                decoded = frequency_decode(ctx, top_k=int(self.params.get("top_k", 100)))
            except ValueError as exc:
                return AttackResult.not_applicable(str(exc))
            method = "frequency_decode + public gazetteer scan"
        else:
            return AttackResult.not_applicable(
                "obfuscated wire exposes neither a declared plaintext control nor a "
                "deterministic-permutation decode path"
            )
        lexicon = self.params.get("lexicon")
        claims = detect_entities(decoded["recovered"], lexicon)
        return AttackResult(
            recovered_token_ids=decoded["recovered"],
            candidates=decoded["candidates"],
            recovered_mapping=decoded["mapping"],
            diagnostics={
                "score_as": "pii",
                "method": method,
                "candidate_coverage": decoded["coverage"],
                "reference_supported_types": decoded.get("reference_supported_types"),
                "num_observed_symbols": decoded["num_observed_symbols"],
                "candidate_top_k": decoded["candidate_top_k"],
                "lexicon_size": len(lexicon) if lexicon else 0,
                "claimed_entity_spans": claims,
                "num_claimed_entities": len(claims),
            },
        )


def score_pii(result: AttackResult, sample: AttackSample) -> dict:
    """Entity-level PIIRSR (plan §5.4 formula 3) plus per-type breakdown.

    The annotation lives in ``sample.meta["pii_entities"]`` as
    ``[{"type", "start", "end", "value"}...]`` with token positions. An entity is
    recovered iff every position in its span matches; partial recovery is a
    failure. ``pii_rsr`` is ``None`` when the sample has no annotation.
    """
    entities = list((sample.meta or {}).get("pii_entities") or [])
    recovered = result.recovered_token_ids
    truth = sample.plaintext_token_ids
    if recovered is None or truth is None:
        return {"status": "no_recovery"}
    recovered = np.asarray(recovered).tolist()
    truth = np.asarray(truth).tolist()

    flags: list[bool] = []
    by_type: dict[str, list[bool]] = {}
    for ent in entities:
        span = list(range(int(ent["start"]), int(ent["end"])))
        ok = entity_recovered(recovered, truth, span)
        flags.append(ok)
        by_type.setdefault(str(ent.get("type", "unknown")), []).append(ok)

    claims = (result.diagnostics or {}).get("claimed_entity_spans") or []
    claimed_hits = 0
    for claim in claims:
        span = list(range(int(claim["start"]), int(claim["end"])))
        if entity_recovered(recovered, truth, span):
            claimed_hits += 1

    return {
        "pii_rsr": pii_recovery_success_ratio(flags),
        "n_entities": len(entities),
        "n_entities_recovered": int(sum(flags)),
        "pii_rsr_by_type": {
            t: (sum(v) / len(v) if v else None) for t, v in by_type.items()
        },
        "claimed_entities": len(claims),
        "claimed_precision": (claimed_hits / len(claims)) if claims else None,
        "denominator_rule": "entities (not tokens); null when the sample has none",
    }
