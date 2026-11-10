"""Candidate-lattice prefix search; an unverified ObfusLM-inspired probe."""
from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence


def candidate_prefix_beam_search(candidates: Sequence[Sequence[int]], score_prefixes: Callable,
                                 *, beam_width: int, batch_size: int = 128) -> dict:
    """Keep the highest-scoring prefixes; scores are not summed across prefixes.

    The scorer receives only candidate token IDs, never a reference or label.
    A bidirectional sequence classifier can score each prefix, but this search
    order is an engineering choice until checked against the author's code.
    """
    for name, value in (("beam_width", beam_width), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not callable(score_prefixes) or not candidates:
        raise ValueError("a scorer and non-empty candidate lattice are required")
    lattice = []
    for choices in candidates:
        values = list(choices)
        if (not values or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**63 for x in values)
                or len(set(values)) != len(values)):
            raise ValueError("each candidate set must contain distinct non-negative integer IDs")
        lattice.append(values)
    beam = [([], 0.0)]
    steps = []
    for position, choices in enumerate(lattice):
        heap, chunk, order = [], [], 0

        def consume(batch):
            nonlocal order
            scores = list(score_prefixes([tokens for tokens in batch]))
            if len(scores) != len(batch):
                raise ValueError("scorer must return one score per candidate prefix")
            for tokens, value in zip(batch, scores):
                if isinstance(value, bool):
                    raise ValueError("candidate scores must be finite numbers")
                try:
                    value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("candidate scores must be finite numbers") from exc
                if not math.isfinite(value):
                    raise ValueError("candidate scores must be finite numbers")
                entry = (value, -order, tokens)
                if len(heap) < beam_width:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
                order += 1

        for prefix, _ in beam:
            for token in choices:
                chunk.append(prefix + [token])
                if len(chunk) == batch_size:
                    consume(chunk)
                    chunk = []
        if chunk:
            consume(chunk)
        beam = [(tokens, score) for score, _, tokens in sorted(heap, key=lambda row: (-row[0], -row[1]))]
        steps.append({"position": position, "candidate_count": len(choices),
                      "evaluated_prefixes": order, "retained_prefixes": len(beam)})
    return {"tokens": beam[0][0], "score": beam[0][1],
            "beams": [{"tokens": tokens, "score": score} for tokens, score in beam],
            "beam_width": beam_width, "steps": steps,
            "score_rule": "sequence-classifier score of each complete current prefix; not a sum of prefix scores",
            "variant": "candidate_prefix_scoring_probe"}


class BertCandidateScorer:
    """Adapter for a caller-supplied binary sequence classifier and tokenizers."""

    def __init__(self, model, tokenizer, decode_candidate_ids: Callable, *, device: str = "cpu",
                 positive_label: int = 1, negative_label: int = 0):
        if not callable(decode_candidate_ids) or positive_label == negative_label:
            raise ValueError("supply a token decoder and two distinct classifier labels")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (positive_label, negative_label)):
            raise ValueError("classifier label indices must be non-negative integers")
        self.model = model.to(device).eval()
        self.tokenizer, self.decode_candidate_ids = tokenizer, decode_candidate_ids
        self.device, self.positive_label, self.negative_label = device, positive_label, negative_label

    def __call__(self, candidates: Sequence[Sequence[int]]) -> list[float]:
        import torch
        texts = [self.decode_candidate_ids(ids) for ids in candidates]
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("candidate token decoder must return strings")
        batch = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and batch["input_ids"].shape[1] > int(limit):
            raise ValueError("candidate exceeds classifier positional capacity; no truncation was performed")
        with torch.inference_mode():
            logits = self.model(**batch).logits
        if logits.ndim != 2 or logits.shape[0] != len(texts) or logits.shape[1] <= max(self.positive_label, self.negative_label):
            raise ValueError("classifier must return both label logits for every candidate")
        return (logits[:, self.positive_label] - logits[:, self.negative_label]).detach().cpu().tolist()
