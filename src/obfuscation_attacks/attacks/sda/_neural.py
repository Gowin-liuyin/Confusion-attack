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
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn


def validate_sda_device(device: str | torch.device) -> torch.device:
    """Reject unavailable CUDA before creating datasets or checkpoint artifacts."""
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"SDA requested {selected}, but CUDA is unavailable in this PyTorch runtime "
                f"(torch {torch.__version__}, CUDA build {torch.version.cuda})"
            )
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise RuntimeError(f"SDA requested {selected}, but only {torch.cuda.device_count()} CUDA devices are available")
    return selected


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
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds positional capacity {self.pe.size(1)}")
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
    cipher_symbols: Optional[Sequence[int]] = None,
    injective: bool = False,
    device: torch.device = torch.device("cpu"),
    sequence_lengths: Optional[Sequence[int]] = None,
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

    With ``sequence_lengths``, the callback also receives the global position,
    and its prefix contains only the current sequence. Full candidates are
    reconstructed from the constrained substitution mapping after search.
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if cipher_symbols is not None and len(cipher_symbols) != num_steps:
        raise ValueError("cipher_symbols length must equal num_steps")
    if injective and cipher_symbols is None:
        raise ValueError("injective search requires cipher_symbols")
    boundaries = set()
    if sequence_lengths is not None:
        lengths = list(sequence_lengths)
        if (cipher_symbols is None or not lengths or
                any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in lengths) or
                sum(lengths) != num_steps):
            raise ValueError("sequence_lengths require constrained search and must cover all steps")
        offset = 0
        for length in lengths[:-1]:
            offset += length
            boundaries.add(offset)
    if num_steps == 0:
        return [], 0.0, [([], 0.0)]
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    mappings: list[dict[int, int]] = [{}] if cipher_symbols is not None else []
    for position in range(num_steps):
        if position in boundaries:
            beams = [([], score) for _, score in beams]
        g = len(beams[0][0])
        if g == 0:
            batch = torch.zeros((len(beams), 0), dtype=torch.long, device=device)
        else:
            batch = torch.tensor([b[0] for b in beams], dtype=torch.long, device=device)
        logp = next_logprob_fn(batch, position) if sequence_lengths is not None else next_logprob_fn(batch)
        if allowed_ids is not None:
            masked = torch.full_like(logp, float("-inf"))
            masked[:, allowed_ids] = logp[:, allowed_ids]
            logp = masked
        if cipher_symbols is not None:
            logp = logp.clone()
            symbol = int(cipher_symbols[position])
            for i, mapping in enumerate(mappings):
                if symbol in mapping:
                    value = logp[i, mapping[symbol]].clone()
                    logp[i].fill_(float("-inf"))
                    logp[i, mapping[symbol]] = value
                elif injective and mapping:
                    logp[i, list(mapping.values())] = float("-inf")
        width = min(beam_width, logp.shape[-1])
        topv, topi = torch.topk(logp, width, dim=-1)
        # The previous implementation built and sorted width**2 Python tuples.
        # Keep scores in float64 to match Python-float addition and stable ties.
        prior = torch.tensor([score for _, score in beams], dtype=torch.float64, device=device)
        combined = topv.to(torch.float64) + prior[:, None]
        order = torch.argsort(combined.reshape(-1), descending=True, stable=True)
        flat_scores = combined.reshape(-1)
        selected = order[torch.isfinite(flat_scores[order])][:beam_width].tolist()
        if not selected:
            raise ValueError("beam search has no finite candidates")
        if cipher_symbols is not None:
            # Mapping keys are identical across beams; copy only when a new symbol appears.
            is_new = symbol not in mappings[0]
            next_mappings = []
            for index in selected:
                parent = index // width
                mapping = mappings[parent]
                if is_new:
                    mapping = {**mapping, symbol: int(topi[parent, index % width])}
                next_mappings.append(mapping)
            mappings = next_mappings
        beams = [
            (beams[index // width][0] + [int(topi[index // width, index % width])],
             float(flat_scores[index]))
            for index in selected
        ]
    if sequence_lengths is not None:
        beams = [([mapping[int(symbol)] for symbol in cipher_symbols], score)
                 for (_, score), mapping in zip(beams, mappings)]
    best = max(beams, key=lambda x: x[1])
    return best[0], best[1], beams


@torch.no_grad()
def variable_length_beam_search(
    next_logprob_fn: Callable[[torch.Tensor], torch.Tensor],
    beam_width: int, *, eos_id: int, allowed_ids: torch.Tensor,
    max_output_len: int, device: torch.device = torch.device("cpu"),
) -> tuple[list[int], float, list[tuple[list[int], float]]]:
    """EOS-terminated beam; scores include EOS and never silently truncate.

    Active and completed hypotheses each retain their own top ``beam_width``.
    The maximum bounds plaintext tokens, with one additional EOS scoring step.
    """
    if beam_width <= 0 or max_output_len < 0:
        raise ValueError("beam_width must be positive and max_output_len non-negative")
    active: list[tuple[list[int], float]] = [([], 0.0)]
    completed: list[tuple[list[int], float]] = []
    for position in range(max_output_len + 1):
        if not active:
            break
        batch = torch.tensor([tokens for tokens, _ in active], dtype=torch.long, device=device)
        logp = next_logprob_fn(batch)
        if logp.ndim != 2 or logp.shape[0] != len(active):
            raise ValueError("beam scorer must return one vocabulary row per hypothesis")
        if eos_id < 0 or eos_id >= logp.shape[1]:
            raise ValueError("EOS token is outside scorer vocabulary")
        next_active: list[tuple[list[int], float]] = []
        for i, (tokens, score) in enumerate(active):
            eos_score = float(logp[i, eos_id])
            if math.isfinite(eos_score):
                completed.append((tokens, score + eos_score))
            if position == max_output_len:
                continue
            candidate_scores = logp[i, allowed_ids]
            width = min(beam_width, candidate_scores.numel())
            if width:
                values, indices = torch.topk(candidate_scores, width)
                for value, index in zip(values.tolist(), indices.tolist()):
                    token = int(allowed_ids[index])
                    if token != eos_id and math.isfinite(value):
                        next_active.append((tokens + [token], score + float(value)))
        completed.sort(key=lambda item: item[1], reverse=True)
        completed = completed[:beam_width]
        next_active.sort(key=lambda item: item[1], reverse=True)
        active = next_active[:beam_width]
    if not completed:
        raise ValueError("no finite EOS-terminated beam within max_output_len")
    return completed[0][0], completed[0][1], completed


__all__ = [
    "SinusoidalPositionalEncoding",
    "causal_mask",
    "length_matched_beam_search",
    "variable_length_beam_search",
]
