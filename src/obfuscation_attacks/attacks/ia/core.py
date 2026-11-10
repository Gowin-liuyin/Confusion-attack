"""Shared invariant-matching helper for the AloePri IAs.

Given a plaintext invariant matrix (one row per plaintext token) and an
obfuscated invariant matrix (one row per obfuscated token), recover the token
mapping by nearest-neighbour matching of the invariant vectors. Both Gate-IA
and Attn-IA reduce to this once their respective invariant is computed.
"""

from __future__ import annotations

import numpy as np


def match_by_invariant(
    plain_vecs: np.ndarray, obf_vecs: np.ndarray, top_k: int = 10
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Return (recovered_mapping obf->plain, mapping_candidates obf->ranked plain)."""
    P = np.asarray(plain_vecs, dtype=float)
    O = np.asarray(obf_vecs, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    if O.ndim == 1:
        O = O[:, None]
    if P.shape[1] != O.shape[1]:
        raise ValueError(
            f"invariant widths differ: plain {P.shape[1]} vs obfuscated {O.shape[1]}; "
            "the two sides expose different block structures and cannot be matched"
        )
    n_obf = O.shape[0]
    n_plain = P.shape[0]
    if n_plain == 0 or n_obf == 0:
        return {}, {}
    k = min(top_k, n_plain)
    mapping: dict[int, int] = {}
    candidates: dict[int, list[int]] = {}
    for i in range(n_obf):
        d = np.sum((P - O[i]) ** 2, axis=1)  # squared L2 over invariant vector
        order = np.argsort(d, kind="stable")
        candidates[i] = order[:k].tolist()
        mapping[i] = int(order[0])
    return mapping, candidates
