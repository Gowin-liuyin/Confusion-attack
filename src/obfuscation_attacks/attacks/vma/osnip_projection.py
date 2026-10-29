"""B3 - OSNIP vocabulary-matching attack (OSNIP App. C.2; reuses Thomas 2025).

OSNIP evaluates lexical leakage with the vocabulary-matching attack of Thomas
et al. 2025. The paper (Sec 4.1, lines 284-288) states it exactly:

    "the attacker reconstructs the sequence autoregressively by selecting, at
     each step, the vocabulary token whose hidden state is closest (in L1
     distance) to the perturbed representation, and feeds the output back for
     the next step."

So B3 is the *autoregressive* L1-replay attack, identical in form to B1
Algorithm 1 but with NO epsilon early-accept and NO proposal model: at step i it
replays the clean model on ``[xhat_{<i}, v]`` for every vocabulary token ``v``,
reads the layer-``l`` hidden state, picks the L1-nearest to the observed
(perturbed) state ``h_i``, appends it, and continues. It therefore shares the
causal L1-replay helper with B1 but stays a separate class/variant.

This is distinct from OSNIP's *KNN* attack (App. C / Sec 4.1 lines 280-283),
which ranks the vocabulary by *Euclidean* distance in the *embedding* space -
that is the NN family, not this one. Here the space is hidden states and the
metric is L1 (App. D "the vocabulary-matching adversary uses L1").

The App. C.2 "candidate pool capped at 300,000" is the pool of *text samples*
from which the 200 test instances (x 5 keys = 1,000 perturbed representations)
are drawn; it is NOT a pool of candidate token representations to match against.
Candidate tokens come from replaying the vocabulary, as above.

For an unknown hidden-dimension permutation the sorted-L1 matcher of Thomas et
al. (5.2.2) is available via ``sorted_l1`` (paper Algorithm 5).
"""

from __future__ import annotations

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register
from .autoregressive import autoregressive_reconstruct


@register
class OSNIPProjectionVMA(Attack):
    name = "vma.osnip_projection"
    family = "vma"
    variant = "osnip"
    # Autoregressive replay of the clean model is required (App. C.2 / Sec 4.1).
    required_capabilities = frozenset({Capability.HIDDEN_STATES, Capability.MODEL_REPLAY})

    def _run(self, ctx: AttackContext) -> AttackResult:
        target = np.asarray(ctx.obfuscated_hidden_states(), dtype=float)  # (N, d)
        n = target.shape[0]

        vocab = int(ctx.descriptor.get("vocab_size") or self.params.get("vocab_size", 0))
        if vocab <= 0:
            return AttackResult.failed("vocab_size unknown; cannot enumerate the vocabulary")

        use_sorted = bool(self.params.get("sorted_l1", False))
        top_k = int(self.params.get("top_k", 10))
        prefix = self.params.get("prefix", [])

        # Pure argmin, feed-back autoregression: no epsilon, no proposal model.
        res = autoregressive_reconstruct(
            ctx.replay,
            target,
            vocab,
            epsilon=None,
            proposal_fn=None,
            proposal_order=None,
            prefix=prefix,
            sorted_l1=use_sorted,
            sequence_permuted=False,
            top_k=top_k,
        )

        return AttackResult(
            recovered_token_ids=res["recovered"],
            candidates=res["candidates"],
            diagnostics={
                "distance": "sorted_l1" if use_sorted else "l1",
                "mode": "autoregressive_replay",
                "per_position_best_distance": res["per_position_best_distance"],
                "num_positions": int(n),
                "note": "300k App. C.2 pool is a text-sample pool, not candidate reps",
            },
        )
