"""Vocabulary-Matching Attack - three independent paths.

* ``autoregressive``   - B1, original *Hidden No More* (Thomas et al. 2025):
  recover one token at a time by L1-matching replayed hidden states, with the
  epsilon early-accept rule and sorted-L1 for hidden-dim permutation.
* ``weight_rowsort``   - B2, AloePri App. D.1: recover permutations Z1, Z2 from
  ``Y = Z1 X Z2`` via row-sorting + fingerprint matching, with multi-layer
  voting and duplicate-row disambiguation.
* ``osnip_projection`` - B3, OSNIP App. C.2: project perturbed hidden states to
  a vocabulary candidate pool and match (L1 / sorted-L1), Total/Clean ASR.

The three paths are deliberately *not* reduced to one another.
"""

from . import autoregressive, weight_rowsort, osnip_projection

__all__ = ["autoregressive", "weight_rowsort", "osnip_projection"]
