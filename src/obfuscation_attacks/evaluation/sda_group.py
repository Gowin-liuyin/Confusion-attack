"""Experimental shared-key beam search with per-request model contexts."""
from __future__ import annotations

from bisect import bisect_right

import numpy as np
import torch
import torch.nn.functional as F

from ..attacks.sda import encodings as enc
from ..attacks.sda._neural import length_matched_beam_search
from ..attacks.sda.sda2023_recurrence_clm import CachedCausalStep


@torch.no_grad()
def decode_sda_group(bundle: dict, cipher_sequences, *, beam_width: int, use_cache: bool = False) -> dict:
    """Constrain one mapping across requests while resetting each model context.

    Beam scores are sums of the selected models' per-request conditional log
    probabilities. This is a group-search engineering extension, not the
    published single-sequence SDA algorithm.
    """
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if not isinstance(use_cache, bool):
        raise ValueError("use_cache must be a boolean")
    method = bundle["method"]
    if method not in {"sda2021", "sda2023"}:
        raise ValueError(f"unknown SDA method {method!r}")
    if method == "sda2021" and (use_cache or bundle.get("length_mode", "fixed") != "fixed"):
        raise ValueError("group decoding requires fixed-length SDA2021 without KV cache")
    sequences = []
    for seq in cipher_sequences:
        raw = np.asarray(seq)
        if (raw.ndim != 1 or not raw.size or len(raw) > bundle["max_cipher_len"] or
                not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_) or
                np.any(raw < 0) or (np.issubdtype(raw.dtype, np.unsignedinteger) and
                                    int(raw.max()) > np.iinfo(np.int64).max)):
            raise ValueError("each group request must contain non-negative int64 IDs and be within max_cipher_len")
        sequences.append(raw.astype(np.int64, copy=False))
    if not sequences:
        raise ValueError("each group request must contain non-negative int64 IDs and be within max_cipher_len")
    cipher = [int(token) for seq in sequences for token in seq]
    cfg, model = bundle["cfg"], bundle["model"]
    injective = method == "sda2021" or not bundle.get("homophonic", False)
    if injective and len(set(cipher)) > cfg.vocab_size:
        return {"status": "infeasible", "injective": True}
    model.eval()
    device = next(model.parameters()).device
    lengths = [len(seq) for seq in sequences]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    active_index = -1
    step_for_request = None

    def step(generated, position):
        nonlocal active_index, step_for_request
        index = bisect_right(starts, position) - 1
        if index != active_index:
            sequence = sequences[index]
            if method == "sda2021":
                source = torch.tensor([enc.frequency_rank_encode(sequence)], dtype=torch.long, device=device)
                padding = source == cfg.src_pad
                memory = model.encode(source, padding)

                def step_for_request(local):
                    bos = torch.full((len(local), 1), cfg.bos, dtype=torch.long, device=device)
                    logits = model.decode(torch.cat((bos, local), dim=1), memory.expand(len(local), -1, -1),
                                          memory_kpm=padding.expand(len(local), -1))
                    return F.log_softmax(logits[:, -1, :], dim=-1)
            else:
                codes = enc.recurrence_encode(sequence)
                prompt = torch.tensor([[cfg.rec_tag] + [cfg.rec_id(int(k)) for k in codes] + [cfg.plain_tag]],
                                      dtype=torch.long, device=device)
                if use_cache:
                    step_for_request = CachedCausalStep(model, prompt)
                else:
                    def step_for_request(local):
                        full = torch.cat((prompt.expand(len(local), -1), local), dim=1)
                        return F.log_softmax(model(full)[:, -1, :], dim=-1)
            active_index = index
        return step_for_request(generated)

    tokens, score, beams = length_matched_beam_search(
        step, num_steps=len(cipher), beam_width=beam_width,
        allowed_ids=torch.arange(cfg.vocab_size, dtype=torch.long, device=device),
        cipher_symbols=cipher, injective=injective, device=device, sequence_lengths=lengths,
    )
    return {"status": "ok", "tokens": tokens, "score": score, "beams": beams,
            "mapping": dict(zip(cipher, tokens)), "injective": injective,
            "sequence_lengths": lengths, "context_reset": "per_request",
            "variant": "shared_mapping_per_request_context_beam_probe"}
