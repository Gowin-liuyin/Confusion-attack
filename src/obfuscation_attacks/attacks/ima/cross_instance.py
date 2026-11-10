"""Self-built-instance inversion (plan class C02) and cross-checkpoint transfer (C04).

**Why this is not the same number as C09.** The plan separates two pair
provenances (§1.4, D8):

* ``KP`` - the attacker holds plaintext/observation pairs **under the deployment
  key**. In this harness that is what ``ima.classify`` / ``ima.seq2seq`` /
  ``ima.qwen2_aloepri`` do, because the ``obfuscation_fn`` they query closes over
  the deployment secret. KP is a client-leakage/collusion口径 and lives in the
  appendix.
* ``KI`` - the attacker uses the *public algorithm* plus **its own** secret to
  build an instance of the mechanism, mints its own pairs, trains there, and is
  then pointed at the deployment instance. That is the paper's threat model
  (AloePri §3.2 / Table 2) and the only inversion口径 allowed in the main table.

This module implements exactly the KI path. The attacker's instance arrives as
``ctx.attacker_obfuscator`` (built by the scenario from the public algorithm and
a seed the attacker owns); the deployment key is never read. Two entry points are
registered:

* ``ima.cross_instance`` - C02: the attacker's instance is a fresh secret of the
  same mechanism;
* ``ima.cross_checkpoint`` - C04: the attacker's instance is a *different
  checkpoint* of the same architecture (transfer attack).

Both share one trainer (InvBERT-GB style per-position classification, which is
the strongest form under a known tokenizer) and both report the two numbers the
plan requires before any cross-instance claim: the attacker-side in-instance
accuracy (positive control, must be >= 95%, §6.5) and the cross-instance recovery
on the deployment stream.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register
from .classify import (
    PerPositionClassifier,
    classifier_accuracy,
    classifier_predict,
    train_classifier,
)


def _build_attacker_pairs(
    obfuscator, corpus: Sequence[Sequence[int]], *, vocab: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mint ``(own-instance observation, plaintext id)`` pairs from the corpus."""
    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for seq in corpus:
        ids = np.asarray(seq, dtype=int).ravel()
        if ids.size == 0:
            continue
        if int(ids.min()) < 0 or int(ids.max()) >= int(vocab):
            raise ValueError(
                f"reference corpus token id outside [0, {vocab}); "
                "the classifier head cannot represent it"
            )
        obs = np.asarray(obfuscator(ids), dtype=float)
        if obs.ndim != 2 or obs.shape[0] != ids.size:
            raise ValueError(
                f"attacker_obfuscator must map (l,) -> (l, d); got {obs.shape} for l={ids.size}"
            )
        X.append(obs)
        y.append(ids)
    if not X:
        raise ValueError("no attacker-side pairs could be minted from the reference corpus")
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0).astype(int)


class _SelfInstanceInverterBase(Attack):
    required_capabilities = frozenset({Capability.EMBEDDINGS})
    variant = "ki"
    instance_kind = "ki_same_mechanism"

    def _run(self, ctx: AttackContext) -> AttackResult:
        obfuscator = getattr(ctx, "attacker_obfuscator", None)
        if obfuscator is None:
            return AttackResult.not_applicable(
                "KI inversion needs ctx.attacker_obfuscator (the attacker's own instance of "
                "the mechanism). Without it the only available pairs would be deployment-key "
                "pairs, i.e. the KP appendix口径 - not this attack."
            )
        if not ctx.reference_corpus:
            return AttackResult.not_applicable(
                "KI inversion needs a public reference corpus to mint training pairs"
            )
        vocab = int(ctx.descriptor.get("vocab_size") or 0)
        if vocab <= 0:
            return AttackResult.failed("vocab_size unknown; cannot size the classifier head")
        H = np.asarray(ctx.obfuscated_embeddings(), dtype=float)
        if H.ndim != 2:
            return AttackResult.failed(f"obfuscated embeddings must be 2-D; got {H.shape}")

        corpus = [np.asarray(s).astype(int).ravel() for s in ctx.reference_corpus]
        heldout_frac = float(self.params.get("heldout_frac", 0.2))
        n_all = len(corpus)
        if n_all < 2 or not 0 < heldout_frac < 1:
            return AttackResult.failed("insufficient_data: independent training and held-out documents are required")
        n_held = max(1, int(round(heldout_frac * n_all))) if n_all > 1 else 0
        n_held = min(n_held, n_all - 1)
        train_corpus, held_corpus = corpus[: n_all - n_held], corpus[n_all - n_held :]

        try:
            X, y = _build_attacker_pairs(obfuscator, train_corpus, vocab=vocab)
            X_held, y_held = (
                _build_attacker_pairs(obfuscator, held_corpus, vocab=vocab)
                if held_corpus
                else (np.zeros((0, 0)), np.zeros(0, dtype=int))
            )
        except ValueError as exc:
            return AttackResult.failed(str(exc))
        if X.shape[1] != H.shape[1]:
            return AttackResult.failed(
                f"attacker instance dim d={X.shape[1]} does not match the deployment wire "
                f"d={H.shape[1]}; a cross-instance attack needs the same mechanism geometry"
            )

        seed = int(self.params.get("seed", 0))
        hidden_dims = tuple(self.params.get("hidden_dims", (64,)))
        epochs = int(self.params.get("epochs", 120))
        lr = float(self.params.get("lr", 1e-2))
        top_k = int(self.params.get("top_k", 10))

        model = PerPositionClassifier(X.shape[1], vocab, hidden_dims=hidden_dims, seed=seed)
        history = train_classifier(model, X, y, epochs=epochs, lr=lr, seed=seed)
        in_instance = classifier_accuracy(model, X, y)
        heldout = (
            classifier_accuracy(model, X_held, y_held) if y_held.size else None
        )
        recovered, candidates = classifier_predict(model, H, top_k=top_k)

        positive_control = {
            "name": "in_instance_recovery",
            "rate": float(heldout),
            "passed": bool(heldout >= 0.95),
            "note": "plan §6.5: <95% on the attacker's own instance means underfitting, "
                    "and then a low cross-instance number must NOT be read as protection",
        }
        return AttackResult(
            recovered_token_ids=recovered,
            candidates=candidates,
            diagnostics={
                "score_as": "token_recovery",
                "instance_kind": self.instance_kind,
                "training_pair_source": "attacker_own_instance",
                "uses_deployment_key": False,
                "num_train_pairs": int(y.size),
                "num_heldout_pairs": int(y_held.size),
                "in_instance_accuracy": float(in_instance),
                "heldout_instance_accuracy": heldout,
                "train_final_loss": float(history[-1]) if history else None,
                "epochs": epochs,
                "vocab_size": vocab,
                "embed_dim": int(X.shape[1]),
                "num_positions": int(H.shape[0]),
                "positive_control": positive_control,
            },
        )


@register
class CrossInstanceInverter(_SelfInstanceInverterBase):
    """C02: train on the attacker's own fresh-secret instance, attack the deployment."""

    name = "ima.cross_instance"
    family = "ima"
    variant = "ki"
    instance_kind = "ki_fresh_secret"


@register
class CrossCheckpointInverter(_SelfInstanceInverterBase):
    """C04: train on a different checkpoint of the same architecture (transfer)."""

    name = "ima.cross_checkpoint"
    family = "ima"
    variant = "ki_transfer"
    instance_kind = "ki_other_checkpoint"
