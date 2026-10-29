"""Obfuscation scheme base class and the capability-gated context builder.

A scheme is the *defense*. It holds a secret (permutation / key matrices),
produces :class:`AttackSample` objects, and declares which capabilities it
exposes to an attacker. :func:`build_context` is the single choke point that
turns a sample into the attacker-visible :class:`AttackContext`: it copies a
gated payload *only* when the scheme exposes the matching capability. This is
how the custom-scheme interface guarantees an attacker can use only the
information the scheme really reveals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..data.contracts import AttackContext, AttackSample, Capability


class ObfuscationScheme(ABC):
    """Base class for obfuscation schemes (defenses)."""

    name: str = ""

    @abstractmethod
    def exposed_capabilities(self) -> frozenset[Capability]:
        """Return the capabilities this scheme reveals to an attacker."""

    def replay_fn(self) -> Optional[Callable[[Sequence[int]], np.ndarray]]:
        """Return a model-replay callable if the scheme exposes MODEL_REPLAY."""
        return None

    def obfuscation_fn(self) -> Optional[Callable[..., Any]]:
        """Return the obfuscation routine if KNOWN_OBFUSCATION_ALGORITHM is exposed.

        Attackers with knowledge of the algorithm use this to build training
        pairs (IMA / SDA). It must *not* close over the per-sample secret key in
        a way that leaks the ground-truth mapping of the evaluated sample.
        """
        return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "exposed_capabilities": sorted(c.value for c in self.exposed_capabilities()),
        }


def _safe_descriptor(sample: AttackSample) -> dict:
    return {
        "tokenizer_id": sample.tokenizer_id,
        "model_id": sample.model_id,
        "seq_len": sample.seq_len,
        "vocab_size": sample.vocab_size,
        "layer": sample.layer,
        "key_id": sample.key_id,
        "seed": sample.seed,
        "text_source": sample.text_source,
        "data_split": sample.data_split,
        "special_token_mask": sample.special_token_mask,
        "content_mask": sample.content_mask,
    }


def build_context(
    scheme: ObfuscationScheme,
    sample: AttackSample,
    *,
    attacker_model: Optional[Any] = None,
    reference_embeddings: Optional[np.ndarray] = None,
    reference_token_counts: Optional[Mapping[int, int]] = None,
    reference_corpus: Optional[list] = None,
) -> AttackContext:
    """Construct the gated :class:`AttackContext` for one sample.

    Only payloads whose capability is in ``scheme.exposed_capabilities()`` are
    attached. Ground-truth fields on the sample (plaintext ids, secret
    permutation/matrices) are deliberately not copied.
    """
    granted = scheme.exposed_capabilities()

    ctx = AttackContext(
        granted=granted,
        descriptor=_safe_descriptor(sample),
        reference_embeddings=(
            reference_embeddings if reference_embeddings is not None else sample.reference_embeddings
        ),
        reference_token_counts=(
            reference_token_counts
            if reference_token_counts is not None
            else sample.reference_token_counts
        ),
        reference_corpus=(
            reference_corpus if reference_corpus is not None else sample.reference_corpus
        ),
        attacker_model=attacker_model,
    )

    if Capability.TOKEN_CIPHERTEXT in granted:
        ctx._ciphertext_token_ids = sample.ciphertext_token_ids
    if Capability.EMBEDDINGS in granted:
        ctx._obfuscated_embeddings = sample.obfuscated_embeddings
    if Capability.HIDDEN_STATES in granted:
        ctx._obfuscated_hidden_states = sample.obfuscated_hidden_states
    if Capability.WHITE_BOX_WEIGHTS in granted:
        ctx._known_weights = sample.known_weights
    if Capability.MODEL_REPLAY in granted:
        ctx._replay_fn = scheme.replay_fn()
    if Capability.KNOWN_OBFUSCATION_ALGORITHM in granted:
        ctx.obfuscation_fn = scheme.obfuscation_fn()

    return ctx
