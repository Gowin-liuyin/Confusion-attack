"""Ratio (micro) bootstrap intervals for sum-of-hits / sum-of-counts points.

Protocol §8.1 reports occurrence-level rates as micro averages
``sum(hits) / sum(counts)`` over all sampled positions. A macro interval built
from per-unit rates answers a *different* estimand: with one hit in a 1-position
text and zero in a 9-position text the micro point is 0.1 while the macro mean is
0.5, and attaching a macro interval to the micro point reports an interval around
a number the run never estimated.

This module resamples the independent unit (text / request / session / prompt
key) and **recomputes the ratio of sums in every replicate**; it never averages
per-unit rates. A macro mean is still available, but only as an explicitly
labelled supplementary number.

It builds on :func:`obfuscation_attacks.metrics.intervals.micro_cluster_bootstrap`,
whose replicates already recompute the ratio over all rows of the drawn clusters.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .intervals import micro_cluster_bootstrap


def _macro_supplementary(hits: np.ndarray, counts: np.ndarray) -> Optional[float]:
    valid = counts > 0
    if not bool(valid.any()):
        return None
    return float(np.mean(hits[valid] / counts[valid]))


def ratio_ci(
    hits: Iterable,
    counts: Iterable,
    *,
    clusters: Optional[Iterable] = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    unit: str = "positions",
) -> dict:
    """Micro ratio point with a *ratio* bootstrap interval.

    ``hits``/``counts`` are per independent unit (a text, a request, a prompt-key
    observation). ``clusters`` groups nested units (for example the five fresh-key
    observations of one prompt) so the interval resamples prompts, not rows.
    Every replicate recomputes ``sum(hits) / sum(counts)`` over the drawn
    clusters, so the interval is centred on the same micro estimand as the
    reported point.

    Returns ``{micro, lo, hi, n_boot, n_clusters, n_units, denominator,
    estimand, estimand_detail, macro_supplementary, ...}``. ``estimand`` is the
    plain ``"micro"`` token the protocol auditor expects; ``estimand_detail``
    spells out the formula. ``macro_supplementary`` is the mean of per-unit rates
    and is *not* the interval's estimand.
    """
    hit_arr = np.asarray(list(hits) if not isinstance(hits, np.ndarray) else hits)
    count_arr = np.asarray(list(counts) if not isinstance(counts, np.ndarray) else counts)
    if clusters is None:
        cluster_arr = np.arange(len(hit_arr))
    else:
        cluster_arr = np.asarray(
            list(clusters) if not isinstance(clusters, np.ndarray) else clusters
        )
    base = micro_cluster_bootstrap(
        hit_arr, count_arr, cluster_arr, n_boot=n_boot, seed=seed, alpha=alpha
    )
    return {
        "micro": base["micro"],
        "lo": base["lo"],
        "hi": base["hi"],
        "n_boot": int(base["n_boot"]),
        "n_clusters": int(base["n_clusters"]),
        "n_units": int(base["n_units"]),
        "denominator": int(base["denominator"]),
        "estimand": "micro",
        "estimand_detail": f"micro (sum hits / sum {unit})",
        "interval": "cluster bootstrap of sum(hits)/sum(counts); never a mean of unit rates",
        "macro_supplementary": _macro_supplementary(hit_arr, count_arr),
        "macro_note": (
            "mean of per-unit rates (a different estimand); supplementary only, "
            "never the interval's point"
        ),
    }


def paired_gain_ci(hits, baseline_hits, counts, *, clusters, n_boot=2000, seed=0) -> dict:
    """Paired micro-rate difference with a percentile cluster bootstrap."""
    a = np.asarray(hits, dtype=np.int64)
    b = np.asarray(baseline_hits, dtype=np.int64)
    n = np.asarray(counts, dtype=np.int64)
    labels = np.asarray(clusters)
    if (a.ndim != 1 or not (a.shape == b.shape == n.shape == labels.shape)
            or np.any(n < 0) or np.any(a < 0) or np.any(b < 0)
            or np.any(a > n) or np.any(b > n) or n_boot < 2000):
        raise ValueError("invalid paired counts or bootstrap budget")
    _, inverse = np.unique(labels, return_inverse=True)
    groups = int(inverse.max() + 1) if len(inverse) else 0
    total = int(n.sum())
    point = float((a.sum() - b.sum()) / total) if total else None
    if groups < 2 or not total:
        return {"value": point, "lo": None, "hi": None, "n_boot": 0,
                "n_clusters": groups, "denominator": total}
    delta = np.bincount(inverse, weights=a - b, minlength=groups)
    sizes = np.bincount(inverse, weights=n, minlength=groups)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        draw = rng.integers(0, groups, size=groups)
        denom = sizes[draw].sum()
        if denom:
            samples.append(float(delta[draw].sum() / denom))
    return {"value": point, "lo": float(np.quantile(samples, .025)) if samples else None,
            "hi": float(np.quantile(samples, .975)) if samples else None, "n_boot": len(samples),
            "n_clusters": groups, "denominator": total,
            "method": "paired cluster bootstrap, ratio of summed differences"}
