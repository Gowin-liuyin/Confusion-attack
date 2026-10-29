"""Deterministic seeding across random / numpy / torch.

Every experiment and test that uses randomness must call :func:`seed_everything`
so that results are reproducible and recorded in the resolved config.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def torch_available() -> bool:
    """Return True if the optional torch backend is importable."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> int:
    """Seed python, numpy and (if present) torch RNGs.

    Parameters
    ----------
    seed:
        The integer seed to apply.
    deterministic_torch:
        If True and torch is present, force deterministic cuDNN/cuBLAS behaviour.

    Returns
    -------
    int
        The seed, so callers can record exactly what was applied.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch_available():
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed
