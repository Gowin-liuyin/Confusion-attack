"""Nearest-neighbour embedding inversion (NN / KNN).

* ``euclidean_knn`` - Song & Raghunathan (2020) style L2 retrieval; the OSNIP
  App. C.1 KNN attack ranks vocabulary tokens by Euclidean distance.
* ``cosine_knn`` - cosine-distance variant used as the direction-stable metric.

Both return the full ranking (top-k) and the first/second candidate margin.
"""

from . import euclidean_knn, cosine_knn

__all__ = ["euclidean_knn", "cosine_knn"]
