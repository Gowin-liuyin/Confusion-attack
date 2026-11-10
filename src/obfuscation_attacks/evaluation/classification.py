"""Classification-task harness for any classification defense.

The protocol's classification arms (SentinelLMs, DP-Forward, ObfusLM+) run on
BERT-family encoders. This module supplies:

* :class:`HFClassificationBackend` — a real HuggingFace sequence classifier
  implementing the adapter's ``ClassificationBackend`` protocol, including the
  client transform and the server classifier.
* :func:`run_ednn_classifier` — the EDNN/KNN retrieval attack on the
  classifier's representation surface (embedding or hidden layer), with the
  same exact full-vocabulary candidate search and denominator handling as C01.

The external arms plug their author encoder/classifier into the same backend
protocol; nothing here assumes a decoder.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .real_battery import asr_from_ranks, truth_rank
from ..schemes.ab_deployment import knn_topk


class HFClassificationBackend:
    """Wrap a HuggingFace sequence-classification model + tokenizer."""

    def __init__(self, model: Any, tokenizer: Any, *, max_length: int = 128) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def transform(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text, add_special_tokens=True, truncation=True,
            max_length=self.max_length, return_tensors="np",
        )["input_ids"][0]
        return np.asarray(encoded, dtype=np.int64)

    def classify_ids(self, ids: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.model(
                input_ids=torch.from_numpy(np.asarray(ids)).long().unsqueeze(0)
            ).logits[0].float()
            return torch.softmax(logits, dim=-1).numpy()

    def embedding_table(self) -> np.ndarray:
        return (
            self.model.get_input_embeddings().weight.detach().float().cpu().numpy()
        )

    def hidden_states(self, ids: np.ndarray, layer: int = 1) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.model(
                input_ids=torch.from_numpy(np.asarray(ids)).long().unsqueeze(0),
                output_hidden_states=True,
            )
        return out.hidden_states[int(layer)][0].float().numpy()


def run_ednn_classifier(
    backend: Any,
    texts: Sequence[str],
    *,
    layer: int = 1,
    metric: str = "l2",
    top_k: int = 100,
    surface: str = "hidden",
) -> dict:
    """EDNN/KNN token recovery on a classifier's representation.

    ``surface='hidden'`` uses the server-computed hidden state (the stronger
    observation); ``surface='embedding'`` uses the server's input embedding of
    the received ids (exact-solvable positive control). The candidate table is
    the classifier's public input embedding table; every position stays in the
    denominator.
    """
    uploads = [np.asarray(backend.transform(text), dtype=np.int64) for text in texts]
    truth = np.concatenate(uploads)
    table = np.asarray(backend.embedding_table(), dtype=np.float32)
    if surface == "hidden":
        queries = np.concatenate(
            [np.asarray(backend.hidden_states(ids, layer=layer), dtype=np.float32)
             for ids in uploads],
            axis=0,
        )
    elif surface == "embedding":
        queries = table[truth]
    else:
        raise ValueError(f"unknown surface {surface!r}; use 'hidden' or 'embedding'")
    candidates = knn_topk(queries, table, k=int(top_k), metric=metric)
    ranks = truth_rank(candidates, truth)
    metrics = asr_from_ranks(ranks, (1, 5, 10, int(top_k)))
    lengths = [len(ids) for ids in uploads]
    per_unit = []
    start = 0
    for index, n in enumerate(lengths):
        block = ranks[start:start + n]
        per_unit.append({
            "unit": index,
            "hits": int((block <= 1).sum()),
            "count": int(n),
            "rank": block.tolist(),
            "predicted": candidates[start:start + n, 0].tolist(),
            "truth": truth[start:start + n].tolist(),
        })
        start += n
    return {
        "status": "ok",
        "surface": surface,
        "layer": int(layer) if surface == "hidden" else None,
        "metric": metric,
        "top_k": int(top_k),
        "n_prompts": len(texts),
        "n_tokens": int(truth.size),
        "metrics": {"stage1_permuted_id": metrics},
        "per_unit_predictions": per_unit,
        "candidates": candidates,
        "ranks": ranks,
    }
