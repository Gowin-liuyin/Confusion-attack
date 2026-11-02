"""Euclidean-distance KNN (OSNIP App. C.1 / Song & Raghunathan 2020 lineage)."""

from __future__ import annotations

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register
from .core import run_knn


@register
class EuclideanKNN(Attack):
    name = "nn_knn.euclidean"
    family = "nn_knn"
    variant = "original"
    required_capabilities = frozenset({Capability.EMBEDDINGS})

    def _run(self, ctx: AttackContext) -> AttackResult:
        top_k = int(self.params.get("top_k", 10))
        return run_knn(ctx, metric="euclidean", top_k=top_k)
