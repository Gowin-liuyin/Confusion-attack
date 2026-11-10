"""Cross-request linkage family (plan class C12, P0-9).

Variants are *not* separated by paper lineage here (the plan records linkage as a
project extension), but the SH-P / SH-B distinction is: the same attack code
reports ``attacker_knowledge="SH-P"`` when the scheme exposes request metadata
and ``"SH-B"`` otherwise. The main table takes SH-P; SH-B is the sensitivity
appendix (plan D13 / G15).
"""

from .core import (  # noqa: F401
    ApproximateLinkage,
    ExactLinkage,
    DEFAULT_THRESHOLDS,
    cosine_scores,
    enumerate_pairs,
    nearest_partner,
    score_linkage,
    symbol_equality,
)

__all__ = [
    "ApproximateLinkage",
    "ExactLinkage",
    "DEFAULT_THRESHOLDS",
    "cosine_scores",
    "enumerate_pairs",
    "nearest_partner",
    "score_linkage",
    "symbol_equality",
]
