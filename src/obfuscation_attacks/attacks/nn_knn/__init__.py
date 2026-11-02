"""Nearest-neighbour embedding inversion (NN / KNN).

* ``euclidean_knn`` - Song & Raghunathan (2020) style L2 retrieval; the OSNIP
  App. C.1 KNN attack ranks vocabulary tokens by Euclidean distance.
* ``cosine_knn`` - cosine-distance variant used as the direction-stable metric.

Both return the full vocabulary ranking (every id, ascending distance) plus the
first/second candidate margin. The ranking is never truncated to ``top_k`` here;
the scorer slices the Top-k it needs, so ASR@k is not capped by the attack.
"""

from . import euclidean_knn, cosine_knn

__all__ = ["euclidean_knn", "cosine_knn"]
