"""Cluster bootstrap intervals for micro and macro estimates (protocol section 8.1).

The point estimate and the interval are the *same* estimand:

* ``micro_cluster_bootstrap``: ratio of summed hits to summed counts over independent
  units (document / request / session). Each bootstrap replicate resamples *clusters*
  (with replacement) and recomputes the ratio over all rows of the drawn clusters; it
  never averages per-row ratios. This is the protocol's micro average.
* ``macro_bootstrap``: plain mean of per-unit values, with the same cluster resampling.

A cluster is the independent sampling unit (e.g. a session when documents are nested
in sessions). Rows are never resampled independently of their cluster.

Intervals are bias-corrected and accelerated (BCa) percentile intervals computed over
the cluster bootstrap replicates; the acceleration is the cluster-level jackknife. BCa
keeps the interval tied to the ratio-of-sums estimate even when the replicate
distribution is skewed or degenerate.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Iterable

import numpy as np


def _as_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _bca_interval(point: float, replicates: np.ndarray, jackknife: np.ndarray,
                  alpha: float) -> tuple[float, float]:
    """BCa percentile interval over cluster replicates, centred on ``point``.

    ``jackknife`` holds leave-one-cluster-out estimates of the same statistic.
    Degenerate resampling distributions fall back to the percentile interval.
    """
    valid = np.asarray(replicates, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return point, point

    n = valid.size
    prop_less = float(np.mean(valid < point))
    if 0.0 < prop_less < 1.0:
        z0 = NormalDist().inv_cdf(prop_less)
    else:
        z0 = 0.0

    jack = np.asarray(jackknife, dtype=np.float64)
    jack = jack[np.isfinite(jack)]
    accel = 0.0
    if jack.size > 1:
        diffs = jack.mean() - jack
        denom = 6.0 * float(np.sum(diffs ** 2)) ** 1.5
        if denom > 0.0:
            accel = float(np.sum(diffs ** 3)) / denom

    normal = NormalDist()

    def _adjust(z: float) -> float:
        denom = 1.0 - accel * (z0 + z)
        if denom == 0.0:
            return alpha / 2.0 if z < 0 else 1.0 - alpha / 2.0
        return normal.cdf(z0 + (z0 + z) / denom)

    a_lo = _adjust(normal.inv_cdf(alpha / 2.0))
    a_hi = _adjust(normal.inv_cdf(1.0 - alpha / 2.0))
    eps = 1.0 / (n + 1.0)
    a_lo = min(max(a_lo, eps), 1.0 - eps)
    a_hi = min(max(a_hi, eps), 1.0 - eps)
    if a_lo > a_hi:
        a_lo, a_hi = a_hi, a_lo
    lo = float(np.quantile(valid, a_lo))
    hi = float(np.quantile(valid, a_hi))
    return max(0.0, min(lo, point)), max(hi, point) if hi < point else hi


def _cluster_index(clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map cluster labels to contiguous row indices, preserving sorted label order."""
    return np.unique(clusters, return_inverse=True)


def _jackknife_ratio(cluster_hits: np.ndarray, cluster_counts: np.ndarray) -> np.ndarray:
    total_hits = float(cluster_hits.sum())
    total_counts = float(cluster_counts.sum())
    leave_counts = total_counts - cluster_counts
    with np.errstate(divide="ignore", invalid="ignore"):
        jack = np.where(leave_counts > 0, (total_hits - cluster_hits) / leave_counts, np.nan)
    return jack


def micro_cluster_bootstrap(hits: Iterable, counts: Iterable, clusters: Iterable, *,
                            n_boot: int = 2000, seed: int = 0,
                            alpha: float = 0.05) -> dict:
    """Micro (ratio-of-sums) point estimate with a cluster bootstrap interval.

    Returns ``{micro, lo, hi, n_clusters, n_units, denominator, n_boot}``; empty input
    yields ``micro=lo=hi=None`` (never zero).
    """
    hits = _as_1d(hits, "hits")
    counts = _as_1d(counts, "counts")
    clusters = _as_1d(clusters, "clusters")
    if not (len(hits) == len(counts) == len(clusters)):
        raise ValueError("hits, counts and clusters must have the same length")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    if np.any(hits < 0):
        raise ValueError("hits must be non-negative")
    if np.any(hits > counts):
        raise ValueError("hits must not exceed counts on any row")

    n_units = int(len(hits))
    denominator = int(counts.sum())
    n_clusters = int(len(np.unique(clusters))) if n_units else 0
    if n_units == 0 or denominator == 0:
        # No estimable interval: report n_boot=0 / no bounds so a consumer can
        # tell the difference between "resampled 2000 times" and "no data".
        return {"micro": None, "lo": None, "hi": None,
                "n_clusters": n_clusters, "n_units": n_units,
                "denominator": denominator, "n_boot": 0}

    micro = float(hits.sum() / denominator)
    uniq, inverse = _cluster_index(clusters)
    n_clusters = int(len(uniq))
    if n_clusters == 1:
        # Only one independent cluster: no resampling variance is identifiable.
        # Returning a content-free 2000-replicate interval would let an audit
        # mistake a degenerate point for evidence; report no interval instead.
        return {"micro": micro, "lo": None, "hi": None,
                "n_clusters": n_clusters, "n_units": n_units,
                "denominator": denominator, "n_boot": 0}

    # Per-cluster sums; a replicate draws clusters and sums both numerators and
    # denominators over all rows of the drawn clusters.
    cluster_hits = np.bincount(inverse, weights=hits.astype(np.float64),
                               minlength=n_clusters)
    cluster_counts = np.bincount(inverse, weights=counts.astype(np.float64),
                                 minlength=n_clusters)

    rng = np.random.default_rng(seed)
    replicates = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        drawn_counts = cluster_counts[drawn].sum()
        if drawn_counts == 0:
            replicates[b] = np.nan
        else:
            replicates[b] = cluster_hits[drawn].sum() / drawn_counts
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        lo = hi = None
    else:
        lo, hi = _bca_interval(
            micro, finite, _jackknife_ratio(cluster_hits, cluster_counts), alpha
        )
    return {"micro": micro, "lo": lo, "hi": hi,
            "n_clusters": n_clusters, "n_units": n_units,
            "denominator": denominator,
            "n_boot": 0 if finite.size < 2 else int(finite.size)}


def macro_bootstrap(values: Iterable, clusters: Iterable, *, n_boot: int = 2000,
                    seed: int = 0, alpha: float = 0.05) -> dict:
    """Macro (mean-of-units) point estimate with a cluster bootstrap interval.

    Returns ``{macro, lo, hi, n_clusters, n_units, n_boot}``; empty input yields
    ``macro=lo=hi=None``. With fewer than two independent clusters no variance is
    identifiable: the point is reported with ``n_boot=0`` and ``lo/hi=None``
    (never a content-free zero-width interval labelled as 2000 replicates).
    """
    values = _as_1d(values, "values").astype(np.float64)
    clusters = _as_1d(clusters, "clusters")
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have the same length")
    n_units = int(len(values))
    if n_units == 0:
        # No estimable interval: n_boot=0 / no bounds distinguishes "no data"
        # from a real 2000-replicate resampling.
        return {"macro": None, "lo": None, "hi": None, "n_clusters": 0,
                "n_units": 0, "n_boot": 0}

    macro = float(values.mean())
    uniq, inverse = _cluster_index(clusters)
    n_clusters = int(len(uniq))
    if n_clusters == 1:
        # One independent cluster identifies no resampling variance; report the
        # point without a content-free interval.
        return {"macro": macro, "lo": None, "hi": None, "n_clusters": 1,
                "n_units": n_units, "n_boot": 0}

    cluster_sum = np.bincount(inverse, weights=values, minlength=n_clusters)
    cluster_n = np.bincount(inverse, minlength=n_clusters).astype(np.float64)

    rng = np.random.default_rng(seed)
    replicates = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        n = cluster_n[drawn].sum()
        replicates[b] = cluster_sum[drawn].sum() / n if n > 0 else np.nan
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        lo = hi = None
    else:
        total_sum = float(cluster_sum.sum())
        total_n = float(cluster_n.sum())
        leave_n = total_n - cluster_n
        with np.errstate(divide="ignore", invalid="ignore"):
            jack = np.where(leave_n > 0, (total_sum - cluster_sum) / leave_n, np.nan)
        lo, hi = _bca_interval(macro, finite, jack, alpha)
    return {"macro": macro, "lo": lo, "hi": hi, "n_clusters": n_clusters,
            "n_units": n_units,
            "n_boot": 0 if finite.size < 2 else int(finite.size)}
