"""Invariant Attack family.

* ``ednn`` - original EDNN (Lin et al. 2024): coordinate-difference invariant of
  the glide-reflection transform, matched with a circular-left-shift difference
  representation.
* ``gate`` - AloePri Gate-IA: the gate-weighted average invariant
  ``Avg(e W_gate) = Avg(e~ W~_gate)`` (requires alpha_e = 0).
* ``attn`` - AloePri Attn-IA: the Mahalanobis/whitening quadratic-form invariant
  of the query projection, invariant under the invertible block transforms;
  singular Gram blocks are skipped (no pseudo-inverse substitution).
"""

from . import ednn, gate, attn

__all__ = ["ednn", "gate", "attn"]
