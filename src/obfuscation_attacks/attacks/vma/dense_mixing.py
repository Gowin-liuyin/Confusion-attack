"""Dense-mixing inversion: the weight attack this artifact actually permits.

Measured fact (``reports/ab_probe/axis_probe.json``): on the packaged artifact the
hidden-axis obfuscation is a **dense** linear re-basing, not the monomial permutation the
plan's Table 9 attack assumes --

    W'_q = W_q M          (least-squares residual 0.001 on q_proj, i.e. only alpha_h=0.01 noise)
    E'   = E M            (same M on the embedding rows' hidden axis)

so a monomial RowSort is the wrong invariant: rows of ``W'_q`` are dense mixtures of all
hidden coordinates, which is why it lands at 0.5-1.0% instead of recovering anything.

The plan's §6.9(3) gives the attacker both matrices ("部署侧 AloePri artifact 的 server/
权重 + 公开模型权重"), which is exactly the white-box assumption AloePri's Table 9 works
under. With both, the mixing is solvable in closed form: least-squares ``M`` from one
attention projection, then ``M^-1`` un-mixes the embedding table and the token permutation
falls out of a nearest-neighbour lookup. That is a *stronger* attack than RowSort and it is
reported as its own variant (``dense_mixing_inversion``) so it can never be quoted as
"the paper's VMA-W".

Both variants are kept: the paper-literal RowSort documents the premise, this one measures
the actual artifact.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: Matrices whose *input* axis is the hidden axis (W' = W M), i.e. they pin down M.
HIDDEN_INPUT_GROUPS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
)


def solve_mixing_matrix(clean: np.ndarray, deployment: np.ndarray) -> dict:
    """Solve ``deployment ~ clean @ M`` by least squares; return M and its residual.

    ``clean``/``deployment`` are ``[out, in]`` weight matrices. The residual is the
    evidence that decides whether this variant applies at all: ~alpha_h means the
    artifact is dense-linear on this axis, ~1.0 means it is not.
    """
    c = np.asarray(clean, dtype=np.float64)
    d = np.asarray(deployment, dtype=np.float64)
    if c.shape != d.shape or c.ndim != 2:
        raise ValueError(f"expect matching 2-D matrices, got {c.shape} and {d.shape}")
    M, *_ = np.linalg.lstsq(c, d, rcond=None)
    residual = float(np.linalg.norm(d - c @ M) / (np.linalg.norm(d) + 1e-12))
    return {"M": M, "relative_residual": residual, "shape": list(M.shape)}


def recover_mixing(dep, *, layers: Optional[Sequence[int]] = None,
                   groups: Sequence[str] = HIDDEN_INPUT_GROUPS,
                   getter=None, max_cond: float = 1e4) -> dict:
    """Recover one shared hidden-axis mixing matrix from several attention projections.

    Each projection proposes an M; the median across proposals is used so a single noisy
    matrix cannot move the result. Agreement between proposals is reported, because a
    per-layer differing M would mean one shared un-mixing is invalid. Since a square
    in-sample least-squares fit is exact for any deployment matrix, the function also
    reports ``heldout_residual_mean``: the error of each proposal on the *other*
    matrices, which is small only when one dense M really generated all of them.
    """
    # ``getter(name, side)`` keeps peak memory at one matrix per call; the default keeps the
    # old behaviour for callers that already hold the state dicts.
    if getter is None:
        w_clean = dep.weights("clean")
        w_dep = dep.weights("deployment")

        def getter(name: str, side: str):  # type: ignore[misc]
            return (w_clean if side == "clean" else w_dep)[name]

    layer_ids = list(range(dep.n_layers)) if layers is None else list(layers)
    proposals: list[np.ndarray] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    per_matrix: dict[str, dict] = {}
    skipped_groups: list[dict] = []
    for layer in layer_ids:
        for group in groups:
            name = f"model.layers.{layer}.{group}.weight"
            try:
                clean_matrix = getter(name, "clean")
                dep_matrix = getter(name, "deployment")
                solved = solve_mixing_matrix(clean_matrix, dep_matrix)
            except KeyError as exc:
                skipped_groups.append({
                    "layer": int(layer), "group": group, "code": "weight_missing",
                    "reason": f"weight tensor unavailable ({exc})",
                })
                continue
            except ValueError as exc:
                # solve_mixing_matrix rejects shape mismatches here; a wrong
                # shape is a side/shape problem, not a missing tensor.
                skipped_groups.append({
                    "layer": int(layer), "group": group,
                    "code": ("side_mismatch" if "matching" in str(exc)
                             else "shape_invalid"),
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            except (OSError, RuntimeError, TypeError, MemoryError) as exc:
                skipped_groups.append({
                    "layer": int(layer), "group": group, "code": "tensor_unreadable",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            proposals.append(solved["M"])
            pairs.append((np.asarray(clean_matrix, dtype=np.float64),
                          np.asarray(dep_matrix, dtype=np.float64)))
            per_matrix[f"layer{layer}.{group}"] = {
                "relative_residual": solved["relative_residual"]}
    if not proposals:
        raise RuntimeError("no hidden-input weight matrices available")
    # Leave-one-out cross-prediction: an in-sample square fit is exact for *any*
    # deployment matrix, so its residual cannot establish that one shared dense M
    # generated all the matrices. Applying M fitted on one matrix to a different
    # matrix is the honest check: it is small only when the transform really is
    # dense and shared.
    cross_residuals: list[float] = []
    if len(pairs) >= 2:
        for i in range(len(pairs)):
            for j, (clean_j, dep_j) in enumerate(pairs):
                if i == j or clean_j.shape[1] != proposals[i].shape[0]:
                    continue
                predicted = clean_j @ proposals[i]
                cross_residuals.append(
                    float(
                        np.linalg.norm(dep_j - predicted)
                        / (np.linalg.norm(dep_j) + 1e-12)
                    )
                )
    heldout = float(np.mean(cross_residuals)) if cross_residuals else None
    stack = np.stack(proposals)
    # The median is only meaningful if the proposals agree and the result is well posed. A
    # square projection (q_proj is [2048, 2048]) makes the least-squares fit exact, so an
    # ill-conditioned W_q yields an ill-conditioned M (cond 2.6e7 measured). The k/v
    # projections have fewer rows than hidden columns ([512, 2048] on the paper model), so
    # they are under-determined on this axis and cannot regularise the square fit; guard the
    # conditioning of the median instead of trusting agreement between rank-deficient fits.
    M = np.median(stack, axis=0)
    cond = float(np.linalg.cond(M))
    if cond > max_cond:
        best = min(proposals, key=lambda m: float(np.linalg.cond(m)))
        M, cond = best, float(np.linalg.cond(best))
    spread = float(np.mean(np.std(stack, axis=0)))
    # residual of the agreed matrix on the first matrix it was fitted from
    return {
        "M": M,
        "n_proposals": len(proposals),
        "per_matrix": per_matrix,
        "conditional_number": cond,
        "proposal_spread_mean": spread,
        "heldout_residual_mean": heldout,
        "residual_mean": float(np.mean([v["relative_residual"] for v in per_matrix.values()])),
        "residual_max": float(np.max([v["relative_residual"] for v in per_matrix.values()])),
        "skipped_groups": skipped_groups,
    }


def unmix_table(deployment_table: np.ndarray, M: np.ndarray, *,
                chunk_rows: int = 8192) -> np.ndarray:
    """Pure attack step: ``E ~= E' M^-1``, row-chunked.

    Takes arrays only -- no deployment object, therefore no permutation, key or client secret
    is reachable from here. That is deliberate: the isolation the plan demands (§1.7-9) is
    enforced by the signature rather than by a comment (see
    ``tests/unit/test_attack_isolation.py``).

    Chunking matters on this machine: a single ``[128256, 2048] @ [2048, 2048]`` matmul
    allocates a 1 GiB output on top of the two loaded models and the two embedding tables, and
    the resulting swap pressure turns a ~30 s GEMM into >10 min.
    """
    table = np.asarray(deployment_table, dtype=np.float32)
    M_inv = np.linalg.inv(np.asarray(M, dtype=np.float64)).astype(np.float32)
    unmixed = np.empty_like(table)
    for start in range(0, table.shape[0], chunk_rows):
        stop = min(start + chunk_rows, table.shape[0])
        unmixed[start:stop] = table[start:stop] @ M_inv
    return unmixed


def unmix_rows(deployment_table: np.ndarray, M: np.ndarray,
               rows: np.ndarray) -> np.ndarray:
    """Un-mix only the requested deployment rows: ``E[rows] ~= E'[rows] @ M^-1``.

    Same maths as :func:`unmix_table`, but the cost stays proportional to the rows the attack
    actually needs. That matters here: the full 128,256 x 2048 table times a 2048x2048 inverse
    is ~0.54 TFLOP and, on this CPU, took >8 min of wall time per run while the answer only ever
    needs the candidate rows (stage-1 outputs or an evaluation sample of ~1e3 rows).
    """
    table = np.asarray(deployment_table, dtype=np.float32)
    ids = np.asarray(rows, dtype=np.int64)
    M_inv = np.linalg.inv(np.asarray(M, dtype=np.float64)).astype(np.float32)
    return table[ids] @ M_inv


def map_rows(queries: np.ndarray, unmixed_table: np.ndarray, *,
             chunk: int = 256) -> np.ndarray:
    """Pure attack step: nearest un-mixed row for each query row (cosine).

    ``queries`` are reference rows the attacker chose from the *public* clean table;
    ``unmixed_table`` is indexed by deployment row. Returns the recovered deployment ids.
    No permutation, no plaintext ids, no key.
    """
    q = np.asarray(queries, dtype=np.float32)
    table = np.asarray(unmixed_table, dtype=np.float32)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    tn = table / (np.linalg.norm(table, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(q), dtype=np.int64)
    for start in range(0, len(q), chunk):
        stop = min(start + chunk, len(q))
        out[start:stop] = (qn[start:stop] @ tn.T).argmax(axis=1)
    return out


def forward_match(deployment_rows: np.ndarray, transformed_clean: np.ndarray) -> np.ndarray:
    """Nearest clean id for each deployment row, in the *forward* parameterisation.

    ``E' = E M`` means matching ``E'`` against ``E M`` is algebraically the same as un-mixing
    both sides with ``M^-1``, but it never forms the inverse: that matters because the mixing
    matrix can be ill-conditioned (cond 2.6e7 for a single square projection), and an explicit
    inverse then amplifies noise enough to sink the nearest-neighbour lookup. Cost is one GEMM
    over the clean table (128k x 2048 @ 2048 x 2048, ~2.8 s here).
    """
    ids, _, _ = forward_match_stats(deployment_rows, transformed_clean)
    return ids


def forward_match_stats(
    deployment_rows: np.ndarray, transformed_clean: np.ndarray, *,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward match plus its public confidence: top-1 cosine and top1-top2 margin.

    The confidence is what distinguishes "this artifact is a dense re-basing that
    this M aligns with" from "the nearest row is merely the least bad of 128k
    unrelated rows". It uses only the public tables and the recovered M.
    """
    rows = np.asarray(deployment_rows, dtype=np.float32)
    table = np.asarray(transformed_clean, dtype=np.float32)
    rn = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)
    tn = table / (np.linalg.norm(table, axis=1, keepdims=True) + 1e-12)
    ids = np.empty(len(rows), dtype=np.int64)
    best = np.empty(len(rows), dtype=np.float32)
    margin = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), max(1, chunk)):
        stop = min(start + max(1, chunk), len(rows))
        sims = rn[start:stop] @ tn.T
        ids[start:stop] = sims.argmax(axis=1)
        top = np.take_along_axis(sims, ids[start:stop, None], axis=1)[:, 0]
        second = np.partition(sims, -2, axis=1)[:, -2]
        best[start:stop] = top
        margin[start:stop] = top - second
    return ids, best, margin


def map_deployment_ids(unmixed_table: np.ndarray, clean_table: np.ndarray,
                       deployment_ids: np.ndarray, *, chunk: int = 256) -> dict:
    """Deployment row id -> clean id, by querying the **clean table** (not itself).

    The first version nearest-neighboured the un-mixed table *against itself*, which returns
    the same row and degenerates into the identity map -- a bug that silently poisons a
    composed chain. The correct lookup is: the un-mixed row holds the clean embedding of its
    token, so match it against the clean table's rows.

    Returns the mapping and the fraction of ids that fell inside the clean table.
    """
    ids = np.asarray(deployment_ids, dtype=np.int64)
    U = np.asarray(unmixed_table, dtype=np.float32)
    E = np.asarray(clean_table, dtype=np.float32)
    en = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(ids), dtype=np.int64)
    for start in range(0, len(ids), chunk):
        stop = min(start + chunk, len(ids))
        q = U[ids[start:stop]]
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        out[start:stop] = (qn @ en.T).argmax(axis=1)
    return {"clean_ids": out, "n_ids": int(len(ids))}


def unmix_diagnostics(unmixed: np.ndarray, clean_table: np.ndarray,
                      perm: np.ndarray) -> dict:
    """Known-secret diagnostics for the scorer only.

    This is the single place in the dense-mixing path that reads the secret permutation, which
    is why it is a separate function taking it as an explicit argument: any caller that imports
    it is visibly opting into ground-truth use.
    """
    E = np.asarray(clean_table, dtype=np.float32)
    U = np.asarray(unmixed, dtype=np.float32)
    p = np.asarray(perm, dtype=np.int64)
    paired = U[p]                      # clean rows in clean order
    e_sq = float(np.sum(E.astype(np.float64) ** 2)) or 1.0
    return {
        "relative_residual": float((np.sum((paired - E) ** 2, dtype=np.float64) / e_sq) ** 0.5),
        "relative_residual_raw": float(
            (np.sum((U - E) ** 2, dtype=np.float64) / e_sq) ** 0.5),
        "row_cosine_mean": float(np.mean(
            (paired * E).sum(axis=1)
            / (np.linalg.norm(paired, axis=1) * np.linalg.norm(E, axis=1) + 1e-12))),
        "note": "known-secret diagnostic (pairs rows with the secret permutation)",
    }
