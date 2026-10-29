"""Cosine-distance KNN variant."""

from __future__ import annotations

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register
from .core import run_knn


@register
class CosineKNN(Attack):
    name = "nn_knn.cosine"
    family = "nn_knn"
    variant = "original"
    required_capabilities = frozenset({Capability.EMBEDDINGS})

    def _run(self, ctx: AttackContext) -> AttackResult:
        top_k = int(self.params.get("top_k", 10))
        return run_knn(ctx, metric="cosine", top_k=top_k)
