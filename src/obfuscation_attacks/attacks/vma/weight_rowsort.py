"""B2 - AloePri weight-structure VMA (AloePri App. D.1, Table 9).

Known relationship (FORMULA_LEDGER: VMA-B2-1/2), paper lines 1580-1599:

    Y = Z1 X Z2                       (Z1 row permutation, Z2 column permutation)

``RowSort(.)`` sorts the entries *within* each row. The exact identity is

    RowSort(Y) = Z1 . RowSort(X)

because the column permutation Z2 vanishes under a per-row sort and the row
permutation Z1 commutes with it (sorting each row then permuting the rows equals
permuting the rows then sorting each row). The paper writes this loosely as
``RowSort(Y) = RowSort(X Z2)``; that holds only as a *set* of rows, i.e. up to
the row ordering Z1. Matching the sorted rows of Y against those of X therefore
recovers Z1 ("first eliminate one permutation via column-wise sorting, then
recover the other via neighbor matching", paper line 1594).

AloePri weights are *noisy*: ``W*_embed = W_e + alpha_e . E_embed`` with the
recommended ``alpha_e = 1.0`` (paper lines 473-475, 872), so ``Y = Z1 X Z2``
holds only approximately. Exact sorted-row fingerprints do not match; we recover
Z1 by **RowSort then nearest-neighbor (L1) matching** of the row-sorted rows,
which tolerates the additive noise (the distance is UNSPECIFIED in AloePri; we
record L1 after Thomas et al. as an EXPERIMENTAL_COMPLETION). Z2 is then
recovered by nearest-neighbor matching of columns on the assigned rows.

Multi-layer voting (paper lines 1597-1599): for AloePri the token-level
permutation Pi (= Z1) is *shared* across decoder layers, whose Table-9 matrices
may differ in shape. We recover a neighbor-match per layer and take a per-row
**majority vote** across layers (deterministic tie-break by aggregated sorted-L1
distance) - not one concatenated fingerprint.

Duplicate rows: rows that are identical after sorting across *all* layers form a
fingerprint group that RowSort cannot order. We assign the unique rows first and
recover Z2 from them, then resolve each duplicate group individually using the
full (Z2-corrected) relation ``Y[i] ~ X[j] Z2``. Groups that remain ambiguous
are reported as a *partial* result (``diagnostics['partial'] = True`` with
candidate sets); such rows are never reported as a verified match, and a wrong
Z1 is never claimed verified.

The min-cost (Hungarian) assignment used for Z2 columns, for within-group
resolution and for completing the bijection is a deterministic
EXPERIMENTAL_COMPLETION on top of the paper's neighbor matching; it does not
change the RowSort + L1 maths.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...data.contracts import AttackContext, AttackResult, AttackStatus, Capability
from ..base import Attack, register


# ---------------------------------------------------------------------------
# small array helpers
# ---------------------------------------------------------------------------
def _as_layer_list(arr) -> list[np.ndarray]:
    """Normalise X / Y into a list of 2-D layers (allowing different shapes)."""
    if isinstance(arr, (list, tuple)):
        return [np.asarray(a, dtype=float) for a in arr]
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        return [a]
    if a.ndim == 3:
        return [a[l] for l in range(a.shape[0])]
    raise ValueError(f"weight must be 2-D, 3-D (L,n,m) or a list of 2-D layers; got {a.shape}")


def _row_sort(mat: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(mat, dtype=float), axis=1)


def _pairwise_l1(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """L1 distance between every row of A and every row of B -> (nA, nB)."""
    return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)


def _input_row_groups(Xs: list[np.ndarray], atol: float) -> tuple[list[list[int]], list[int]]:
    """Group input rows that are identical (after sorting) across *all* layers."""
    L = len(Xs)
    n = Xs[0].shape[0]
    group_of = [-1] * n
    groups: list[list[int]] = []
    for j in range(n):
        placed = False
        for gid, g in enumerate(groups):
            rep = g[0]
            if all(np.allclose(Xs[l][j], Xs[l][rep], atol=atol, rtol=0.0) for l in range(L)):
                g.append(j)
                group_of[j] = gid
                placed = True
                break
        if not placed:
            groups.append([j])
            group_of[j] = len(groups) - 1
    return groups, group_of


def recover_col_permutation(X: np.ndarray, M: np.ndarray) -> Optional[np.ndarray]:
    """Return c with ``M[:,k] ~ X[:, c[k]]`` via nearest-neighbor column matching.

    Uses a min-cost (Hungarian) assignment on inter-column L1 distances so the
    result is a genuine permutation even under noise; exact in the noise-free
    case. ``None`` only on a shape mismatch.
    """
    X = np.asarray(X, dtype=float)
    M = np.asarray(M, dtype=float)
    if X.shape != M.shape:
        return None
    m = X.shape[1]
    if m == 0:
        return np.empty(0, dtype=int)
    # D[k, c] = L1(M[:,k], X[:,c])
    D = np.abs(M.T[:, None, :] - X.T[None, :, :]).sum(axis=2)
    rows, cols = linear_sum_assignment(D)
    cperm = np.empty(m, dtype=int)
    cperm[rows] = cols
    return cperm


def _recover_col_perms(
    Xl: list[np.ndarray], Yl: list[np.ndarray], p1: np.ndarray, assigned: list[int]
) -> list[np.ndarray]:
    """Per-layer Z2 recovered from the confidently assigned (unique) rows."""
    z2: list[np.ndarray] = []
    for l in range(len(Xl)):
        m = Xl[l].shape[1]
        if not assigned:
            z2.append(np.arange(m))
            continue
        A = Xl[l][[int(p1[i]) for i in assigned]]  # (k, m)
        B = Yl[l][assigned]                         # (k, m)
        c = recover_col_permutation(A, B)
        z2.append(c if c is not None else np.arange(m))
    return z2


def _resolve_duplicate_group(
    Xl: list[np.ndarray],
    Yl: list[np.ndarray],
    z2_layers: list[np.ndarray],
    members: list[int],
    outs: list[int],
    atol: float,
) -> Optional[dict[int, int]]:
    """Resolve a duplicate fingerprint group with the full Z2-corrected relation.

    Returns ``{output_row: input_row}`` if a unique, strictly-separated bijection
    exists, otherwise ``None`` (genuinely ambiguous -> caller emits candidates).
    """
    if len(outs) != len(members):
        return None
    k = len(members)
    if k == 0:
        return {}
    R = np.zeros((k, k))
    for a, i in enumerate(outs):
        for b, j in enumerate(members):
            s = 0.0
            for l in range(len(Xl)):
                pred = Xl[l][j][z2_layers[l]]  # X row j with columns permuted by Z2
                s += float(np.sum(np.abs(Yl[l][i] - pred)))
            R[a, b] = s
    rows, cols = linear_sum_assignment(R)
    # Accept only if every matched input is the strict, unique nearest for its
    # output row; identical (truly duplicate) rows tie and are rejected.
    for a, b in zip(rows, cols):
        order = np.argsort(R[a], kind="stable")
        if int(order[0]) != int(b):
            return None
        if k > 1 and not (R[a, order[0]] < R[a, order[1]] - atol):
            return None
    return {outs[a]: members[b] for a, b in zip(rows, cols)}


def _complete_bijection(p1: np.ndarray, S: np.ndarray, unresolved: set[int]) -> None:
    """Fill unassigned output rows with a best-effort min-cost bijection."""
    n = len(p1)
    free_out = [i for i in range(n) if p1[i] < 0]
    if not free_out:
        return
    used = {int(p1[i]) for i in range(n) if p1[i] >= 0}
    free_in = [j for j in range(n) if j not in used]
    sub = S[np.ix_(free_out, free_in)]
    rows, cols = linear_sum_assignment(sub)
    for a, b in zip(rows, cols):
        p1[free_out[a]] = free_in[b]
        unresolved.add(free_out[a])


# ---------------------------------------------------------------------------
# core recovery
# ---------------------------------------------------------------------------
def recover_permutations(X, Y, *, atol: float = 1e-6) -> dict:
    """Recover Z1 (row perm) and per-layer Z2 (col perm) from ``Y = Z1 X Z2``.

    Accepts 2-D (single layer), 3-D ``(L, n, m)`` stacks, or a list of per-layer
    2-D matrices sharing Z1 but possibly differing in column count (Table 9).
    Tolerant to additive noise via RowSort + nearest-neighbor matching.

    Returns a dict with ``status`` in {``ok``, ``partial``, ``failed``}, the
    best-effort ``Z1`` / ``Z2_layers``, and diagnostics (``verified``,
    ``partial``, ``collisions``, ``candidates``, ``reconstruction_residual``,
    ``min_margin``).
    """
    Xl = _as_layer_list(X)
    Yl = _as_layer_list(Y)
    if len(Xl) != len(Yl):
        return {"status": "failed", "reason": f"layer count mismatch {len(Xl)} vs {len(Yl)}"}
    L = len(Xl)
    if L == 0:
        return {"status": "failed", "reason": "no layers provided"}
    n = Xl[0].shape[0]
    for l in range(L):
        if Xl[l].shape[0] != n or Yl[l].shape[0] != n:
            return {"status": "failed", "reason": "row count differs across layers"}
        if Xl[l].shape != Yl[l].shape:
            return {"status": "failed", "reason": f"layer {l} shape mismatch X{Xl[l].shape} Y{Yl[l].shape}"}
    if n == 0:
        return {"status": "failed", "reason": "empty matrices"}

    Xs = [_row_sort(Xl[l]) for l in range(L)]
    Ys = [_row_sort(Yl[l]) for l in range(L)]

    # Per-layer sorted-row L1 distance matrices, and their aggregate.
    D = [_pairwise_l1(Ys[l], Xs[l]) for l in range(L)]  # each (n_out, n_in)
    S = np.sum(D, axis=0)

    groups, group_of = _input_row_groups(Xs, atol)
    collisions = any(len(g) > 1 for g in groups)

    # --- M1: per-layer neighbor match, then per-row majority vote -----------
    per_layer_nn = [np.argmin(D[l], axis=1) for l in range(L)]  # each (n,)
    voted_group = np.empty(n, dtype=int)
    for i in range(n):
        tally = Counter(int(group_of[int(per_layer_nn[l][i])]) for l in range(L))
        best = max(tally.values())
        tied = [g for g, c in tally.items() if c == best]
        if len(tied) == 1:
            voted_group[i] = tied[0]
        else:
            voted_group[i] = min(tied, key=lambda g: (min(S[i, j] for j in groups[g]), g))

    outs_by_group: dict[int, list[int]] = {}
    for i in range(n):
        outs_by_group.setdefault(int(voted_group[i]), []).append(i)

    # --- M2: assign unique rows first ---------------------------------------
    p1 = np.full(n, -1, dtype=int)
    candidates: dict[int, list[int]] = {}
    unresolved: set[int] = set()

    for gid, members in enumerate(groups):
        if len(members) != 1:
            continue
        j = members[0]
        outs = outs_by_group.get(gid, [])
        if len(outs) == 1:
            p1[outs[0]] = j
        elif len(outs) > 1:  # noise-induced collision onto one unique row
            keep = min(outs, key=lambda i: (S[i, j], i))
            p1[keep] = j
            for i in outs:
                if i != keep:
                    unresolved.add(i)

    # Z2 from the confidently assigned unique rows.
    assigned = [i for i in range(n) if p1[i] >= 0]
    z2_layers = _recover_col_perms(Xl, Yl, p1, assigned)

    # Resolve each duplicate group individually.
    for gid, members in enumerate(groups):
        if len(members) <= 1:
            continue
        outs = outs_by_group.get(gid, [])
        resolved = _resolve_duplicate_group(Xl, Yl, z2_layers, members, outs, atol)
        if resolved is not None:
            for i, j in resolved.items():
                p1[i] = j
        else:
            for i in outs:
                candidates[i] = list(members)
                unresolved.add(i)

    # Complete the bijection (best-effort) and record candidate sets.
    _complete_bijection(p1, S, unresolved)
    for i in sorted(unresolved):
        candidates.setdefault(i, list(groups[int(voted_group[i])]))

    # --- verification: reconstruction residual + separation margin ----------
    residual = 0.0
    for l in range(L):
        residual = max(residual, float(np.max(np.abs(Ys[l] - Xs[l][p1]))))
    min_margin = float("inf")
    for i in range(n):
        gi = group_of[int(p1[i])]
        comp = [S[i, j] for j in range(n) if group_of[j] != gi]
        if comp:
            min_margin = min(min_margin, min(comp) - S[i, int(p1[i])])
    if min_margin == float("inf"):
        min_margin = 0.0

    partial = len(unresolved) > 0
    bijection = sorted(int(v) for v in p1) == list(range(n))
    verified = bool((not partial) and bijection and (min_margin > 0.0))
    status = "ok" if verified else "partial"

    return {
        "status": status,
        "Z1": p1,
        "Z2_layers": z2_layers,
        "verified": verified,
        "partial": bool(partial),
        "collisions": bool(collisions),
        "candidates": candidates,
        "reconstruction_residual": residual,
        "min_margin": float(min_margin),
    }


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
        if status == "failed":
            return AttackResult.failed(res.get("reason", "unknown failure"))

        p1 = res["Z1"]
        n = len(p1)
        mapping = {int(i): int(p1[i]) for i in range(n)}
        permutations = {
            "Z1": p1,
            **{f"Z2_layer{l}": c for l, c in enumerate(res["Z2_layers"])},
        }
        diagnostics = {
            "verified": bool(res["verified"]),
            "partial": bool(res["partial"]),
            "collisions": bool(res["collisions"]),
            "distance": "rowsort+l1_nn",
            "voting": "per-layer majority vote",
            "reconstruction_residual": float(res["reconstruction_residual"]),
            "min_margin": float(res["min_margin"]),
        }

        if res["partial"]:
            cand = res["candidates"]
            # ranked candidates per output row (resolved rows -> singleton).
            mapping_candidates = {
                int(i): (list(cand[i]) if i in cand else [int(p1[i])]) for i in range(n)
            }
            diagnostics["candidates"] = {int(i): list(v) for i, v in cand.items()}
            return AttackResult(
                status=AttackStatus.OK,
                status_reason=f"partial: {len(cand)} row(s) unresolved (duplicate fingerprints)",
                recovered_permutations=permutations,
                recovered_mapping=mapping,
                mapping_candidates=mapping_candidates,
                diagnostics=diagnostics,
            )

        return AttackResult(
            status=AttackStatus.OK,
            recovered_permutations=permutations,
            recovered_mapping=mapping,
            diagnostics=diagnostics,
        )
