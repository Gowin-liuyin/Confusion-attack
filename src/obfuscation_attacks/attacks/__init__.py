"""Attack implementations, grouped by family.

Each family lives in its own sub-package and keeps the *original*, *AloePri* and
*OSNIP* variants in separate modules. Attacks register themselves in the global
:data:`REGISTRY` via :func:`register` on import.

Family modules are imported lazily through :func:`load_all` (not at package
import time) so that importing one family never forces importing the optional
torch-backed families. The experiment runner calls :func:`load_all`; tests that
need the full registry call it from ``conftest``.
"""

from __future__ import annotations

import importlib

from .base import Attack, register, get_attack, available_attacks, REGISTRY

_FAMILY_MODULES = [
    "obfuscation_attacks.attacks.nn_knn",
    "obfuscation_attacks.attacks.vma",
    "obfuscation_attacks.attacks.ia",
    "obfuscation_attacks.attacks.isa",
    "obfuscation_attacks.attacks.ima",
    "obfuscation_attacks.attacks.tfma",
    "obfuscation_attacks.attacks.sda",
]

_loaded = False


def load_all() -> list[str]:
    """Import every family module for registration side effects.

    Returns the sorted list of registered attack names.
    """
    global _loaded
    for mod in _FAMILY_MODULES:
        importlib.import_module(mod)
    _loaded = True
    return available_attacks()


__all__ = ["Attack", "register", "get_attack", "available_attacks", "REGISTRY", "load_all"]
