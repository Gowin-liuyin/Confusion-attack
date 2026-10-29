"""Method-agnostic torch plumbing shared by the two SDA neural attacks.

This module holds ONLY generic infrastructure that carries no decipherment
semantics: a sinusoidal positional encoding and a length-matched beam-search
driver. The actual SDA formulations live in, and are never merged across, their
own modules:

* :mod:`..sda.sda2021_seq2seq`         - encoder-decoder, L_Seq2Seq = -log P(Y|X)
* :mod:`..sda.sda2023_recurrence_clm`  - causal LM, L_CLM = -log P(X) - log P(Y|X)

Importing this module imports torch; the pure-Python encodings in
:mod:`..sda.encodings` deliberately do not depend on it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Return an additive ``(size, size)`` mask with -inf above the diagonal."""
    return torch.triu(
        torch.full((size, size), float("-inf"), device=device), diagonal=1
    )


@torch.no_grad()
def length_matched_beam_search(
    next_logprob_fn: Callable[[torch.Tensor], torch.Tensor],
    num_steps: int,
    beam_width: int,
    *,
    allowed_ids: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[list[int], float, list[tuple[list[int], float]]]:
    """Generate exactly ``num_steps`` tokens with beam search.

    ``next_logprob_fn`` maps a batch of already-generated token sequences
    (``LongTensor`` of shape ``[B, g]``, possibly ``g == 0``) to next-token log
    probabilities (``FloatTensor`` of shape ``[B, vocab]``). The driver knows
    nothing about the model behind the closure, so it is reused by both the
    encoder-decoder and the causal LM.

    ``allowed_ids`` optionally restricts generation to a subset of the vocab
    (e.g. plaintext-only ids), which both enforces valid output and, with a
    fixed ``num_steps``, yields an output whose length matches the ciphertext.
    """
    if num_steps <= 0:
        return [], 0.0, [([], 0.0)]
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    for _ in range(num_steps):
        g = len(beams[0][0])
        if g == 0:
            batch = torch.zeros((len(beams), 0), dtype=torch.long, device=device)
        else:
            batch = torch.tensor([b[0] for b in beams], dtype=torch.long, device=device)
        logp = next_logprob_fn(batch)  # [B, vocab]
        if allowed_ids is not None:
            masked = torch.full_like(logp, float("-inf"))
            masked[:, allowed_ids] = logp[:, allowed_ids]
            logp = masked
        width = min(beam_width, logp.shape[-1])
        topv, topi = torch.topk(logp, width, dim=-1)
        candidates: list[tuple[list[int], float]] = []
        for i, (toks, score) in enumerate(beams):
            for v, idx in zip(topv[i].tolist(), topi[i].tolist()):
                candidates.append((toks + [int(idx)], score + float(v)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
    best = max(beams, key=lambda x: x[1])
    return best[0], best[1], beams


__all__ = [
    "SinusoidalPositionalEncoding",
    "causal_mask",
    "length_matched_beam_search",
]
