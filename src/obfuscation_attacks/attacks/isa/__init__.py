"""ISA - Internal-State Attack family (optimization-based inversion).

Inversion objective (ISA paper Eq. 1, Dong et al. 2025):

    xhat = argmin_{x'} d( h_l^{fA}(x'), h_l^{fV}(x) )

the attacker optimizes a continuous input representation so that its model's
layer-``l`` state matches the observed victim state, then recovers discrete
tokens by cosine similarity with the input-embedding matrix.

Modules:

* ``er``            - Embedding Recovery (original): optimize the dummy
  embeddings directly with ``L_inv = L_im + lambda * L_dm`` (Eq. 3), where
  ``L_dm`` is the random-Gaussian-network distribution-matching penalty
  (**mean**-matching only, phi and the real batch resampled every step).
* ``tbs``           - Token Basis Selection (original): optimize the projection
  ``z`` over the **full** ``d_in`` orthogonal basis ``B`` of ``w_in`` with
  ``what = alpha * arctan(z @ B)`` (``alpha = 5/pi``); ``B = V`` (unbiased, the
  paper default) or ``B = V^T``. No low-rank truncation.
* ``aloepri_state`` - AloePri App. D.1 State1/State2 adaptation: generic
  state-matching optimization (``L_im`` only, random init); targets HiddenState
  and honestly reports AttnScore as not implemented; does not claim to reproduce
  ER or TBS in full.

All three optimize with AdamW (ISA Sec. 5.1) over a frozen model.

Importing this package registers all three attacks in the global registry.
"""

from . import er, tbs, aloepri_state

__all__ = ["er", "tbs", "aloepri_state"]
