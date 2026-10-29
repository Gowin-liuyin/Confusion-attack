"""ISA - Internal-State Attack family (optimization-based inversion).

Inversion objective (ISA paper Eq. 1, Dong et al. 2025):

    xhat = argmin_{x'} d( h_l^{fA}(x'), h_l^{fV}(x) )

the attacker optimizes a continuous input representation so that its model's
layer-``l`` state matches the observed victim state, then recovers discrete
tokens by cosine similarity with the input-embedding matrix.

Modules:

* ``er``            - Embedding Recovery (original): optimize the dummy
  embeddings directly with ``L_inv = L_im + lambda * L_dm`` (Eq. 3), where
  ``L_dm`` is the random-Gaussian-network distribution-matching penalty.
* ``tbs``           - Token Basis Selection (original): optimize a low-dim
  coefficient over the SVD basis of ``w_in`` with ``what = arctan(z @ B_r)``.
* ``aloepri_state`` - AloePri App. D.1 State1/State2 adaptation: generic
  state-matching optimization (``L_im`` only, random init); does not claim to
  reproduce ER or TBS in full.

Importing this package registers all three attacks in the global registry.
"""

from . import er, tbs, aloepri_state

__all__ = ["er", "tbs", "aloepri_state"]
