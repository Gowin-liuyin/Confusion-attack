"""Invariant Attack family.

* ``ednn`` - original EDNN (Lin et al. 2024): coordinate-difference invariant of
  the glide-reflection transform, matched with a circular-left-shift difference
  representation.
* ``gate`` - AloePri Gate-IA: the gate-weighted average invariant
  ``Avg(e W_gate) = Avg(e~ W~_gate)`` (requires alpha_e = 0).
* ``attn`` - AloePri Attn-IA: the Mahalanobis/whitening quadratic-form invariant
  of the query (and optionally key) projection, invariant under the invertible
  RoPE-block transforms. Handles single or stacked per-head/per-layer query/key
  weights, selectable RoPE ``pairing`` (``adjacent`` or ``half_split``), and
  concatenates every valid block into one per-token invariant vector; singular
  Gram blocks are skipped (no pseudo-inverse substitution). Exact only at
  ``alpha_e = 0``; AloePri defends with noise plus head/block permutation.
"""

from . import ednn, gate, attn

__all__ = ["ednn", "gate", "attn"]
