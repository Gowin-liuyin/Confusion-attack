"""Worked example of a *custom* obfuscation scheme (Goal section 9).

This shows how a user plugs a new defense into the attack harness through the
controlled interface. ``CustomNullSpaceScheme`` is an OSNIP-flavoured toy: it
perturbs token embeddings along random directions scaled by ``epsilon`` and
exposes ONLY the perturbed embeddings (``EMBEDDINGS``). It deliberately does not
expose weights, hidden states, token ciphertext, replay or the algorithm, so:

* geometric attacks that need only ``EMBEDDINGS`` (NN/KNN, EDNN) can run, and
  their success degrades as ``epsilon`` grows - the defender's knob;
* attacks that need capabilities the scheme withholds (e.g. the weight VMA,
  which needs ``WHITE_BOX_WEIGHTS``) are reported *not applicable* rather than
  being fed the defender's secret or scored as 0% success.

The example is for demonstrating the interface only; it is never part of the
paper-reproduction results.
"""

from __future__ import annotations

import numpy as np

from ..data.contracts import AttackSample, Capability
from .base import ObfuscationScheme


class CustomNullSpaceScheme(ObfuscationScheme):
    name = "custom_nullspace_example"

    def __init__(self, epsilon: float = 0.25, seed: int = 0) -> None:
        self.epsilon = float(epsilon)
        self.seed = int(seed)

    def exposed_capabilities(self) -> frozenset[Capability]:
        # Only the perturbed embeddings are revealed.
        return frozenset({Capability.EMBEDDINGS})

    def build_sample(
        self,
        plaintext_token_ids: np.ndarray,
        reference_embeddings: np.ndarray,
        *,
        tokenizer_id: str = "custom-demo-tok",
        model_id: str = "custom-demo-model",
    ) -> AttackSample:
        """Perturb the clean embeddings of ``plaintext_token_ids`` and expose them."""
        ids = np.asarray(plaintext_token_ids, dtype=int)
        E = np.asarray(reference_embeddings, dtype=float)
        rng = np.random.default_rng(self.seed)
        clean = E[ids]
        # Random unit-direction perturbation scaled by epsilon * ||clean||.
        noise = rng.standard_normal(clean.shape)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12
        scale = self.epsilon * np.linalg.norm(clean, axis=1, keepdims=True)
        obf = clean + scale * noise
        return AttackSample(
            tokenizer_id=tokenizer_id,
            model_id=model_id,
            seq_len=int(ids.size),
            vocab_size=int(E.shape[0]),
            obfuscated_embeddings=obf,
            reference_embeddings=E,
            plaintext_token_ids=ids,
            text_source="custom_scheme_example",
            data_split="demo",
            meta={"epsilon": self.epsilon, "scheme": self.name},
        )
