"""Shared exact nearest-frequency ranking with deterministic token-ID ties."""
from __future__ import annotations

import numpy as np


def nearest_frequency_order(frequency: float, reference_frequencies, token_ids) -> np.ndarray:
    values = np.asarray(reference_frequencies, dtype=np.float64)
    ids = np.asarray(token_ids, dtype=np.int64)
    if values.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("reference frequencies and token IDs must be aligned finite vectors")
    if not np.isfinite(frequency):
        raise ValueError("observed frequency must be finite")
    return np.lexsort((ids, np.abs(values - frequency)))
