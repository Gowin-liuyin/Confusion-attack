"""Token-Frequency Matching Attack (AloePri App. D.1 / section 7.6).

* ``tfma.independent`` - the method AloePri actually uses: match each ciphertext
  token to the plaintext token with the closest prior frequency (per-symbol
  nearest frequency), reporting Top-10/Top-100.
* ``tfma.global_assignment`` - a clearly separated EXTENSION that solves the
  one-to-one assignment ``argmin_pi sum_c (f_C(c) - f_R(pi(c)))^2`` with the
  Hungarian algorithm. It is NOT the paper's method and is named/reported apart.
"""

from . import frequency_matching

__all__ = ["frequency_matching"]
