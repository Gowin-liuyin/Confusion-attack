"""A tiny deterministic causal model for tests and smoke runs.

``CausalToyModel`` is a real, self-consistent causal map (not a mock): the
hidden state at position ``i`` is a fixed linear-recurrent function of the token
embeddings at positions ``<= i``. Distinct tokens produce distinct states given
a fixed prefix, which is exactly what lets the autoregressive VMA recover the
correct token by matching hidden states. It is deterministic given a seed so
tests can assert exact behaviour.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import ModelHandle


class CausalToyModel(ModelHandle):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 8,
        num_layers: int = 2,
        seed: int = 0,
        recurrence: float = 0.5,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.recurrence = float(recurrence)
        rng = np.random.default_rng(seed)
        # Well-separated embedding rows so the correct token is the unique argmin.
        self._embed = rng.standard_normal((self.vocab_size, self.hidden_size))
        # One transition matrix per layer; contraction keeps states bounded.
        self._transitions = [
            0.3 * rng.standard_normal((self.hidden_size, self.hidden_size))
            for _ in range(self.num_layers)
        ]

    def embedding_matrix(self) -> np.ndarray:
        return self._embed.copy()

    def causal_hidden_states(
        self, token_ids: Sequence[int], layer: Optional[int] = None
    ) -> np.ndarray:
        ids = np.asarray(list(token_ids), dtype=int)
        if ids.ndim != 1:
            raise ValueError("token_ids must be 1-D")
        depth = self.num_layers if layer is None else int(layer)
        if not (1 <= depth <= self.num_layers):
            raise ValueError(f"layer must be in [1, {self.num_layers}], got {depth}")
        states = self._embed[ids]  # (N, d)
        for li in range(depth):
            trans = self._transitions[li]
            out = np.empty_like(states)
            carry = np.zeros(self.hidden_size)
            for i in range(states.shape[0]):
                h = states[i] + self.recurrence * (carry @ trans)
                out[i] = h
                carry = h
            states = out
        return states
