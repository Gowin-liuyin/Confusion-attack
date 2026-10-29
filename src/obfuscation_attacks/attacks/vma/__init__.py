"""Vocabulary-Matching Attack - three independent paths.

* ``autoregressive``   - B1, original *Hidden No More* (Thomas et al. 2025):
  recover one token at a time by L1-matching replayed hidden states, with the
  proposal-ordered epsilon early-accept (Algorithm 1) and the permutation
  variants - plain (Alg 1), hidden-dimension sorted-L1 (Alg 5),
  sequence-dimension with removal (Alg 4), and factorized-2D (Alg 6).
* ``weight_rowsort``   - B2, AloePri App. D.1 / Table 9: recover permutations
  Z1, Z2 from ``Y = Z1 X Z2`` via RowSort + nearest-neighbor (L1) matching
  (noise-tolerant), with per-layer majority voting and unique-first duplicate
  resolution (unresolved groups reported as a partial result).
* ``osnip_projection`` - B3, OSNIP App. C.2: the *autoregressive* L1-replay
  vocabulary-matching attack (reuses the B1 helper; no epsilon, no proposal).

The three paths are deliberately *not* reduced to one another; B3 shares only
the causal L1-replay helper with B1 and stays its own class/variant.
"""

from . import autoregressive, weight_rowsort, osnip_projection

__all__ = ["autoregressive", "weight_rowsort", "osnip_projection"]
