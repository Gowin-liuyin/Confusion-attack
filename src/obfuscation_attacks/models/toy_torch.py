"""A tiny differentiable torch model shared by the neural attacks.

``TorchToyModel`` provides exactly the hooks the optimization-based ISA and the
learned IMA/SDA attacks need from a victim/clean model:

* ``embedding_matrix()``        - the input token-embedding matrix (for TBS SVD
  and for cosine token recovery);
* ``forward_from_embeddings``   - a differentiable map from input embeddings to
  the layer-``l`` hidden state ``psi_l`` (ISA optimizes its input against this);
* ``hidden_states``             - detached hidden states for building targets.

It is a genuine, deterministic multi-layer map (Linear + tanh), not a mock, so
gradient-based inversion really has to recover the input to match the state.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class TorchToyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 16,
        num_layers: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        gen = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.randn(self.vocab_size, self.hidden_size, generator=gen)
            )
        self.blocks = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)]
        )
        with torch.no_grad():
            for blk in self.blocks:
                blk.weight.copy_(
                    0.5 * torch.randn(self.hidden_size, self.hidden_size, generator=gen)
                )
                blk.bias.zero_()
        self.act = nn.Tanh()

    def embedding_matrix(self) -> torch.Tensor:
        return self.embed.weight.detach().clone()

    def _depth(self, layer: Optional[int]) -> int:
        depth = self.num_layers if layer is None else int(layer)
        if not (1 <= depth <= self.num_layers):
            raise ValueError(f"layer must be in [1, {self.num_layers}], got {depth}")
        return depth

    def forward_from_embeddings(
        self, inputs_embeds: torch.Tensor, layer: Optional[int] = None
    ) -> torch.Tensor:
        """Differentiable map from input embeddings to layer-``l`` hidden states."""
        depth = self._depth(layer)
        h = inputs_embeds
        for i in range(depth):
            h = self.act(self.blocks[i](h))
        return h

    @torch.no_grad()
    def hidden_states(
        self, token_ids: Sequence[int], layer: Optional[int] = None
    ) -> torch.Tensor:
        ids = torch.as_tensor(list(token_ids), dtype=torch.long)
        return self.forward_from_embeddings(self.embed(ids), layer).detach()

    def freeze(self) -> "TorchToyModel":
        for p in self.parameters():
            p.requires_grad_(False)
        return self
