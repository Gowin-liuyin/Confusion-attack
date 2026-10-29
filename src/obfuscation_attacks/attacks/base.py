"""Attack base class and registry.

An :class:`Attack` declares the family, the variant (which paper lineage it
faithfully implements) and the capabilities it requires. The common base only
manages dispatch and capability checks; it never alters the internal maths of a
family (Goal section 5.1 rule 4 / section 3 rule 8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..data.contracts import (
    AttackContext,
    AttackResult,
    Capability,
    CapabilityError,
)

Variant = str  # "original" | "aloepri" | "osnip"


class Attack(ABC):
    """Base class for all attacks.

    Subclasses set :attr:`name`, :attr:`family`, :attr:`variant` and
    :attr:`required_capabilities`, and implement :meth:`_run`.
    """

    name: str = ""
    family: str = ""
    variant: Variant = ""
    required_capabilities: frozenset[Capability] = frozenset()

    def __init__(self, **params) -> None:
        self.params = params

    # The runner calls this. It performs the capability gate, then dispatches to
    # the family-specific ``_run``. Capability shortfalls become an honest
    # ``not_applicable`` result rather than a crash or a fake 0% success rate.
    def run(self, ctx: AttackContext) -> AttackResult:
        missing = [c for c in self.required_capabilities if not ctx.has(c)]
        if missing:
            names = ", ".join(c.value for c in missing)
            return AttackResult.not_applicable(
                f"{self.name} requires capabilities not exposed by the scheme: {names}"
            )
        try:
            return self._run(ctx)
        except CapabilityError as exc:
            return AttackResult.not_applicable(str(exc))

    @abstractmethod
    def _run(self, ctx: AttackContext) -> AttackResult:
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "variant": self.variant,
            "required_capabilities": sorted(c.value for c in self.required_capabilities),
            "params": self.params,
        }


REGISTRY: dict[str, type[Attack]] = {}


def register(cls: type[Attack]) -> type[Attack]:
    """Class decorator registering an attack under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate attack name: {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


def get_attack(name: str, **params) -> Attack:
    if name not in REGISTRY:
        raise KeyError(f"unknown attack {name!r}; available: {available_attacks()}")
    return REGISTRY[name](**params)


def available_attacks() -> list[str]:
    return sorted(REGISTRY)
