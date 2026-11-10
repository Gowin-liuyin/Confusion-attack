"""Model handles used by attacks that replay a model or need its embeddings."""

from .base import ModelHandle
from .toy import CausalToyModel

try:  # optional torch backend
    from .toy_torch import TorchToyModel
except Exception:  # pragma: no cover - torch not installed
    TorchToyModel = None  # type: ignore[assignment]

__all__ = ["ModelHandle", "CausalToyModel", "TorchToyModel"]
