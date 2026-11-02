"""Abstract model handle.

Attacks that need to *replay* a model (VMA autoregressive) or that need its
input embedding matrix (NN reference, ISA/TBS SVD, IMA) depend only on this
small interface, so a tiny toy model can stand in for a real LLM in tests while
a full HuggingFace model is used for large-scale runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class ModelHandle(ABC):
    vocab_size: int
    hidden_size: int
    num_layers: int

    @abstractmethod
    def embedding_matrix(self) -> np.ndarray:
        """Return the (vocab_size, hidden_size) input token-embedding matrix."""

    @abstractmethod
    def causal_hidden_states(
        self, token_ids: Sequence[int], layer: Optional[int] = None
    ) -> np.ndarray:
        """Return per-position hidden states at ``layer`` for ``token_ids``.

        The computation must be *causal*: position ``i`` depends only on tokens
        at positions ``<= i``. This is the property the autoregressive VMA relies
        on to recover one token at a time.
        """
