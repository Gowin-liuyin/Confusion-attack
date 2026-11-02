"""A generic configurable scheme used to wire synthetic scenarios.

``GenericScheme`` is a thin :class:`ObfuscationScheme` whose exposed capability
set, replay function and obfuscation function are supplied at construction time.
The per-family *data* (what the obfuscation produced) lives on the
:class:`AttackSample`; this scheme just declares what an attacker is allowed to
see and, through :func:`build_context`, enforces it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..data.contracts import Capability
from .base import ObfuscationScheme


class GenericScheme(ObfuscationScheme):
    def __init__(
        self,
        name: str,
        exposed_capabilities: frozenset[Capability],
        *,
        replay_fn: Optional[Callable[[Sequence[int]], np.ndarray]] = None,
        obfuscation_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.name = name
        self._caps = frozenset(exposed_capabilities)
        self._replay_fn = replay_fn
        self._obfuscation_fn = obfuscation_fn

    def exposed_capabilities(self) -> frozenset[Capability]:
        return self._caps

    def replay_fn(self):
        return self._replay_fn

    def obfuscation_fn(self):
        return self._obfuscation_fn
