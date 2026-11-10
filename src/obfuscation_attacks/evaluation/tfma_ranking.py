"""TFMA nearest-frequency candidate ranking over an explicit reference support."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from ..attacks.tfma.matcher import nearest_frequency_order

Context = Callable[[int, int], float]


def rank_candidates(
    symbol_counts: Mapping[int, int],
    ref_counts: Mapping[int, int],
    k: int,
    *,
    context: Context | None = None,
    vocab_size: int | None = None,
) -> dict[int, list[int]]:
    """Return up to ``k`` distinct valid reference ids per symbol, best first.

    The paper's TFMA-2 score is ``argmin_t |f_C(c) - f_R(t)|`` over normalised
    frequencies. Equal distances are resolved by ascending token ID. Supplying
    ``vocab_size`` includes zero-frequency rows; otherwise only provided IDs are
    candidates. A context scorer defines a separate ordering.
    """
    if not symbol_counts or not ref_counts:
        return {}
    if any(int(count) < 0 for count in ref_counts.values()) or any(int(count) < 0 for count in symbol_counts.values()):
        raise ValueError("frequency counts cannot be negative")
    ref_tokens = list(range(vocab_size)) if vocab_size is not None else sorted(int(token) for token in ref_counts)
    if vocab_size is not None and (vocab_size < 1 or any(int(token) < 0 or int(token) >= vocab_size for token in ref_counts)):
        raise ValueError("reference token outside vocabulary")
    limit = max(0, min(int(k), len(ref_tokens)))
    if limit == 0:
        return {int(symbol): [] for symbol in symbol_counts}
    total_ref = float(sum(int(count) for count in ref_counts.values()))
    total_symbol = float(sum(int(count) for count in symbol_counts.values()))
    if total_ref <= 0 or total_symbol <= 0:
        raise ValueError("reference and observed counts must have positive mass")
    frequencies = np.array([int(ref_counts.get(token, 0)) / total_ref for token in ref_tokens])

    symbols = sorted(symbol_counts.items(), key=lambda item: (-item[1], item[0]))
    ranking: dict[int, list[int]] = {}
    for symbol, count in symbols:
        f_c = int(count) / total_symbol
        if context is None:
            order = nearest_frequency_order(f_c, frequencies, ref_tokens)
            ordered = [ref_tokens[index] for index in order]
        else:
            ordered = sorted(
                ref_tokens,
                key=lambda token: (
                    float(context(int(symbol), token)),
                    abs(f_c - int(ref_counts.get(token, 0)) / total_ref),
                    token,
                ),
            )
        ranking[int(symbol)] = [int(token) for token in ordered[:limit]]
    return ranking


def max_candidates(ranking: Mapping[int, list[int]]) -> int:
    """Minimum candidate-list length; 0 when empty (never a padded count)."""
    if not ranking:
        return 0
    return int(min(len(candidates) for candidates in ranking.values()))
