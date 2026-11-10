"""Experimental neural-LM key search inspired by Kambhatla et al. (2018).

https://aclanthology.org/D18-1102/ describes search over partial substitution
keys and LM-filled global rest cost. This implementation is an engineering
variant, not an author-code reproduction or a registered paper experiment.
"""
from __future__ import annotations

from collections import Counter
import heapq
import math
from collections.abc import Callable, Sequence

import numpy as np


def _ids(values, name):
    result = list(values)
    if not result or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                         or not 0 <= x < 2**63 for x in result):
        raise ValueError(f"{name} must contain non-negative integer IDs")
    return [int(x) for x in result]


def lm_key_beam_search(ciphertext: Sequence[int], plaintext_alphabet: Sequence[int],
                       next_log_probs: Callable, *, beam_width: int, seed: int,
                       homophonic: bool = False, symbol_order: Sequence[int] | None = None,
                       sequence_lengths: Sequence[int] | None = None) -> dict:
    """Search keys using a frozen public LM; the callback sees generated IDs only.

    It returns normalized log probabilities indexed by plaintext token ID for
    the next token given the generated prefix. For each partial key, unsolved
    positions are sampled independently left-to-right from the alphabet, while
    solved positions are forced. The score is the full sampled sequence's LM
    log probability, without the optional paper frequency heuristic. This
    specific completion policy and ordering are not author-code verified.
    Optional sequence lengths share the key but reset LM context per request.
    """
    cipher = _ids(ciphertext, "ciphertext")
    alphabet = _ids(plaintext_alphabet, "plaintext_alphabet")
    lengths = [len(cipher)] if sequence_lengths is None else list(sequence_lengths)
    if (not lengths or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                           or n <= 0 for n in lengths) or sum(lengths) != len(cipher)):
        raise ValueError("sequence_lengths must be positive integers covering ciphertext exactly")
    lengths = [int(n) for n in lengths]
    starts = set(np.cumsum(lengths[:-1]).tolist())
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("plaintext_alphabet must contain distinct IDs")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(homophonic, bool) or not callable(next_log_probs):
        raise ValueError("homophonic must be boolean and next_log_probs must be callable")
    counts = Counter(cipher)
    if not homophonic and len(counts) > len(alphabet):
        raise ValueError("injective decoding needs at least as many plaintext as ciphertext symbols")
    if symbol_order is None:
        first = {x: cipher.index(x) for x in counts}
        order = sorted(counts, key=lambda x: (-counts[x], first[x]))
    else:
        order = _ids(symbol_order, "symbol_order")
        if len(order) != len(counts) or set(order) != set(counts):
            raise ValueError("symbol_order must contain each observed ciphertext symbol exactly once")
    rng = np.random.default_rng(seed)

    def score(mapping):
        prefix = []
        completed = []
        total = 0.0
        for position, symbol in enumerate(cipher):
            if position in starts:
                prefix = []
            logp = np.asarray(next_log_probs(tuple(prefix)), dtype=np.float64)
            if logp.ndim != 1 or len(logp) <= max(alphabet) or np.isnan(logp).any() or np.isposinf(logp).any():
                raise ValueError("LM must return a valid log-probability vector covering the alphabet")
            peak = float(np.max(logp))
            if not math.isfinite(peak) or abs(peak + math.log(float(np.exp(logp - peak).sum()))) > 1e-4:
                raise ValueError("LM log probabilities must be normalized")
            if symbol in mapping:
                token = mapping[symbol]
            else:
                choices = logp[alphabet]
                best = float(np.max(choices))
                if not math.isfinite(best):
                    return None
                weights = np.exp(choices - best)
                token = alphabet[int(rng.choice(len(alphabet), p=weights / weights.sum()))]
            value = float(logp[token])
            if not math.isfinite(value):
                return None
            total += value
            prefix.append(token)
            completed.append(token)
        return total, completed

    beam = [({}, 0.0, [])]
    steps = []
    for symbol in order:
        heap = []
        attempted = scored = 0
        for mapping, _, _ in beam:
            used = set(mapping.values())
            for token in alphabet:
                if not homophonic and token in used:
                    continue
                extended = {**mapping, symbol: token}
                result = score(extended)
                attempted += 1
                if result is None:
                    continue
                value, completed = result
                entry = (value, -attempted, extended, completed)
                if len(heap) < beam_width:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
                scored += 1
        if not heap:
            raise ValueError("no finite-scoring key extensions remain")
        beam = [(mapping, value, completed) for value, _, mapping, completed
                in sorted(heap, key=lambda x: (-x[0], -x[1]))]
        steps.append({"cipher_symbol": symbol, "attempted": attempted,
                      "finite_scored": scored, "retained": len(beam)})
    candidates = [{"mapping": mapping, "tokens": completed, "log_probability": value}
                  for mapping, value, completed in beam]
    return {"variant": "lm_key_search_global_sample_probe", "seed": seed,
            "beam_width": beam_width, "homophonic": homophonic, "symbol_order": order,
            "sequence_lengths": lengths, "context_reset": "per_sequence",
            "completion": "seeded left-to-right alphabet sampling", "frequency_heuristic": False,
            "steps": steps, "candidates": candidates, "tokens": candidates[0]["tokens"]}


class TorchCausalNextLogProbs:
    """Adapt a caller-supplied, frozen causal LM with explicit beginning token."""

    def __init__(self, model, *, bos_token_id: int, device: str = "cpu"):
        import torch
        from ._neural import validate_sda_device
        _ids([bos_token_id], "bos_token_id")
        validate_sda_device(device)
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.bos = int(bos_token_id)

    def __call__(self, prefix):
        import torch
        ids = torch.tensor([[self.bos, *prefix]], dtype=torch.long, device=self.device)
        capacity = getattr(self.model.config, "max_position_embeddings", None)
        if capacity is not None and ids.shape[1] > int(capacity):
            raise ValueError("LM prefix exceeds positional capacity; no truncation was performed")
        with torch.inference_mode():
            logits = self.model(input_ids=ids).logits
            if logits.ndim != 3 or logits.shape[:2] != ids.shape:
                raise ValueError("causal LM returned invalid logits")
            return torch.log_softmax(logits[0, -1].float(), dim=-1).cpu().numpy()
