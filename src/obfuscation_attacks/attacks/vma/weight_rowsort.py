"""B2 - AloePri weight-structure VMA (AloePri App. D.1, Table 9).

Known relationship (FORMULA_LEDGER: VMA-B2-1/2):

    Y = Z1 X Z2                       (Z1 row permutation, Z2 column permutation)
    RowSort(Y) = RowSort(X Z2)        (left permutation cancels under row-sort)

Algorithm: eliminate Z1 by row-sorting, recover Z1 by matching per-row sorted
fingerprints between X and Y, then recover Z2 by matching the columns of
``Z1^{-1} Y`` against X. For AloePri the token-level permutation Pi (= Z1) is
*shared* across decoder layers, so fingerprints are concatenated across layers
(the paper's voting mechanism) to break ties. Duplicate rows are never matched
arbitrarily: remaining ambiguity is resolved by a verified search over the
ambiguous groups and the full relation ``Y = Z1 X Z2`` is checked exactly.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np

from ...data.contracts import AttackContext, AttackResult, Capability
from ..base import Attack, register

_AMBIGUITY_CAP = 5040  # 7!; refuse to brute force larger ambiguous groups


def _as_layers(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"weight must be 2-D or 3-D (L,n,m); got shape {arr.shape}")


def _row_fingerprint(layers: np.ndarray, row: int) -> tuple:
    # Concatenate the sorted rows across layers -> multi-layer voting fingerprint.
    return tuple(np.round(np.concatenate([np.sort(layers[l, row]) for l in range(layers.shape[0])]), 9))


def _col_fingerprint(mat: np.ndarray, col: int) -> tuple:
    return tuple(np.round(mat[:, col], 9))


def _group_by_fingerprint(fps: list[tuple]) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for idx, fp in enumerate(fps):
        groups.setdefault(fp, []).append(idx)
    return groups


def recover_col_permutation(X: np.ndarray, M: np.ndarray) -> Optional[np.ndarray]:
    """Return c with M[:,k] == X[:, c[k]] for all k, or None if impossible."""
    m = X.shape[1]
    x_groups = _group_by_fingerprint([_col_fingerprint(X, j) for j in range(m)])
    used: dict[tuple, int] = {}
    c = np.empty(m, dtype=int)
    for k in range(m):
        fp = _col_fingerprint(M, k)
        if fp not in x_groups:
            return None
        pool = x_groups[fp]
        pos = used.get(fp, 0)
        if pos >= len(pool):
            return None
        c[k] = pool[pos]
        used[fp] = pos + 1
    return c


def _reconstruct(X_layers: np.ndarray, p1: np.ndarray, c_layers: list[np.ndarray]) -> np.ndarray:
    L, n, m = X_layers.shape
    out = np.empty_like(X_layers)
    for l in range(L):
        out[l] = X_layers[l][np.ix_(p1, c_layers[l])]
    return out


def recover_permutations(X: np.ndarray, Y: np.ndarray) -> dict:
    """Recover Z1 (row perm) and per-layer Z2 (col perm) from Y = Z1 X Z2.

    Accepts 2-D (single layer) or 3-D (L, n, m) stacks sharing Z1.
    """
    X_layers = _as_layers(X)
    Y_layers = _as_layers(Y)
    if X_layers.shape != Y_layers.shape:
        return {"status": "failed", "reason": f"shape mismatch X{X_layers.shape} Y{Y_layers.shape}"}
    L, n, m = X_layers.shape

    x_fps = [_row_fingerprint(X_layers, r) for r in range(n)]
    y_fps = [_row_fingerprint(Y_layers, i) for i in range(n)]
    x_groups = _group_by_fingerprint(x_fps)

    # Candidate input rows for each output row (by matching sorted fingerprints).
    candidates: list[list[int]] = []
    for i in range(n):
        if y_fps[i] not in x_groups:
            return {"status": "failed", "reason": f"output row {i} has no matching fingerprint"}
        candidates.append(list(x_groups[y_fps[i]]))

    # Partition output rows by their fingerprint group; each group is an
    # independent assignment problem of size len(group)!. A valid row
    # permutation requires each group to have equal multiplicity in X and Y.
    out_groups = _group_by_fingerprint(y_fps)
    for fp, out_rows in out_groups.items():
        if len(x_groups.get(fp, [])) != len(out_rows):
            return {
                "status": "failed",
                "reason": f"fingerprint multiplicity mismatch for group {fp[:2]}...",
            }
    ambiguity = 1
    for fp, out_rows in out_groups.items():
        ambiguity *= int(math.factorial(len(out_rows)))
    collisions = any(len(v) > 1 for v in out_groups.values())

    def _assignments():
        # Yield every bijection consistent with the fingerprint groups.
        per_group = []
        order = []
        for fp, out_rows in out_groups.items():
            in_rows = x_groups[fp]
            order.extend(out_rows)
            per_group.append((out_rows, list(itertools.permutations(in_rows))))
        for combo in itertools.product(*[perms for _, perms in per_group]):
            p1 = np.empty(n, dtype=int)
            for (out_rows, _), assigned in zip(per_group, combo):
                for o, src in zip(out_rows, assigned):
                    p1[o] = src
            yield p1

    if ambiguity > _AMBIGUITY_CAP:
        return {
            "status": "ambiguous",
            "reason": f"duplicate-row ambiguity {ambiguity} exceeds cap {_AMBIGUITY_CAP}",
            "candidates": candidates,
        }

    inv = np.empty(n, dtype=int)
    for p1 in _assignments():
        inv[p1] = np.arange(n)
        c_layers = []
        ok = True
        for l in range(L):
            M = Y_layers[l][inv]  # unsort rows: M == X_layer Z2_layer
            c = recover_col_permutation(X_layers[l], M)
            if c is None:
                ok = False
                break
            c_layers.append(c)
        if not ok:
            continue
        recon = _reconstruct(X_layers, p1, c_layers)
        if np.allclose(recon, Y_layers, atol=1e-6):
            return {
                "status": "ok",
                "Z1": p1,
                "Z2_layers": c_layers,
                "collisions": collisions,
                "ambiguity": ambiguity,
            }
    return {"status": "failed", "reason": "no consistent (Z1, Z2) reproduced Y"}


@register
class WeightRowSortVMA(Attack):
    name = "vma.weight_rowsort"
    family = "vma"
    variant = "aloepri"
    required_capabilities = frozenset({Capability.WHITE_BOX_WEIGHTS})

    def _run(self, ctx: AttackContext) -> AttackResult:
        names = ctx.weight_names()
        if "X" not in names or "Y" not in names:
            return AttackResult.not_applicable(
                f"weight VMA needs known X and observed Y; have {names}"
            )
        X = ctx.weight("X")
        Y = ctx.weight("Y")
        res = recover_permutations(X, Y)
        status = res.get("status")
        if status == "ok":
            p1 = res["Z1"]
            mapping = {int(i): int(p1[i]) for i in range(len(p1))}
            return AttackResult(
                recovered_permutations={
                    "Z1": p1,
                    **{f"Z2_layer{l}": c for l, c in enumerate(res["Z2_layers"])},
                },
                recovered_mapping=mapping,
                diagnostics={
                    "verified": True,
                    "collisions": bool(res["collisions"]),
                    "ambiguity": int(res["ambiguity"]),
                },
            )
        if status == "ambiguous":
            return AttackResult(
                status=AttackResult.not_applicable(res["reason"]).status,
                status_reason=res["reason"],
                diagnostics={"candidates": res.get("candidates")},
            )
        return AttackResult.failed(res.get("reason", "unknown failure"))
