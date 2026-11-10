"""Weight attacks on the public deployment surface (F01; AloePri App. D.1).

Gate-IA and Attn-IA rank the evaluated clean tokens against **all** deployment
rows. The secret permutation is never used to build the candidate set; it appears
only in the scorer, which checks whether the true deployment row
``perm[clean_type]`` is among the attack's candidates.

The invariant formulas are the ones in ``attacks/ia/gate.py`` and
``attacks/ia/attn.py``; this module only supplies public data and exact chunked
matching.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from ..attacks.ia.attn import rope_block_columns
from ..data.contracts import CapabilityError
from ..schemes.public_view import PublicDeployment

#: Tensor-name templates consumed by each weight attack.  ``{layer}`` is
#: substituted with the model layer index; a bare suffix (no ``{layer}``) is
#: expanded as ``model.layers.{layer}.<suffix>``.
GATE_WEIGHT = "model.layers.{layer}.mlp.gate_proj.weight"
ATTN_WEIGHTS = (
    "model.layers.{layer}.self_attn.q_proj.weight",
    "model.layers.{layer}.self_attn.k_proj.weight",
)

_WEIGHT_SUFFIX_ROLES = (
    ("gate", ".mlp.gate_proj.weight"),
    ("q", ".self_attn.q_proj.weight"),
    ("k", ".self_attn.k_proj.weight"),
)

#: Machine-checkable codes for ``missing_weight_layers`` (protocol_audit
#: ``WEIGHT_LAYER_MISSING_CODES``).  The order is also the priority used when a
#: layer fails several checks: the most fundamental gap names the record.
_MISSING_CODE_PRIORITY = (
    "weight_missing", "tensor_unreadable", "shape_invalid", "side_mismatch",
    "head_dim_mismatch", "zero_columns", "singular_gram", "condition_exceeded",
)


def _unreadable_code(exc: BaseException) -> str:
    """Classify a weight-source failure: absent vs present-but-unreadable."""
    if isinstance(exc, (KeyError, CapabilityError, FileNotFoundError)):
        return "weight_missing"
    return "tensor_unreadable"


def _weight_name(template: str, layer: int) -> str:
    """Expand one required-weight template for ``layer``."""
    if "{layer}" in template:
        return template.format(layer=int(layer))
    suffix = template.lstrip(".")
    if suffix.startswith("model.layers."):
        return suffix
    return f"model.layers.{int(layer)}.{suffix}"


def _weight_role(name: str) -> str:
    """Classify a tensor name so the matching shape convention is checked."""
    for role, suffix in _WEIGHT_SUFFIX_ROLES:
        if name.endswith(suffix):
            return role
    return "tensor"


def _weight_shape_problem(public: PublicDeployment, name: str,
                          shape: Sequence[int]) -> str | None:
    """Return why ``shape`` cannot feed the attack's transpose convention."""
    if len(shape) != 2:
        return f"expected a 2-D weight tensor; got shape {tuple(int(v) for v in shape)}"
    out_features, in_features = int(shape[0]), int(shape[1])
    if out_features <= 0 or in_features <= 0:
        return f"empty weight tensor shape {tuple(int(v) for v in shape)}"
    hidden = int(
        getattr(public, "hidden_size", 0) or getattr(public, "hidden", 0) or 0
    )
    n_heads = max(1, int(getattr(public, "n_heads", 1) or 1))
    n_kv_heads = max(1, int(getattr(public, "n_kv_heads", n_heads) or n_heads))
    # build_public_view never sets ``head_dim``, so fall back to the frozen
    # model's config relation head_dim = hidden_size / num_attention_heads
    # (2048/32 = 64 for llama-3.2-1b); an explicit value still wins.
    head_dim = getattr(public, "head_dim", None)
    if not head_dim and hidden:
        config_heads = max(1, int(getattr(public, "n_heads", 1) or 1))
        if hidden % config_heads == 0:
            head_dim = hidden // config_heads
    role = _weight_role(name)
    if role == "gate":
        # HF stores (intermediate, hidden) and the attack transposes it to
        # (hidden, intermediate); an already-transposed tensor is also usable.
        if hidden and hidden not in (in_features, out_features):
            return (f"gate weight shape {tuple(int(v) for v in shape)} has no "
                    f"hidden axis of size {hidden}")
        return None
    if role in {"q", "k"}:
        # q/k are consumed as ``table @ weight.T``, i.e. raw HF layout
        # (heads * head_dim, hidden); a transposed tensor would be misread.
        if hidden and in_features != hidden:
            return (f"{role}_proj weight shape {tuple(int(v) for v in shape)} must be "
                    f"(heads*head_dim, hidden={hidden}); the attack transposes it")
        heads = n_heads if role == "q" else n_kv_heads
        if out_features % heads:
            return (f"{role}_proj out_features {out_features} is not divisible by "
                    f"{heads} heads")
        if head_dim:
            expected = heads * int(head_dim)
            if out_features != expected:
                return (f"{role}_proj out_features {out_features} != "
                        f"heads*head_dim {expected}")
        return None
    return None


def weight_layers(public: PublicDeployment, *,
                  required: Sequence[str]) -> tuple[list[int], list[dict]]:
    """Enumerate every model layer that can supply the ``required`` weights.

    Weight attacks need the real transformer stack, so the layers are
    ``0 .. n_layers-1`` from the public view - **not** ``public.layers``, which
    is the hidden-state sampling tuple used by state attacks and covers only a
    handful of layers.  When ``n_layers`` is unknown (0), the sampled layers are
    the only enumerable fallback and are used verbatim; if neither is present a
    :class:`CapabilityError` is raised.

    ``required`` holds tensor-name templates (``{layer}`` placeholder, or a bare
    suffix such as ``"mlp.gate_proj.weight"``).  Each template is probed on the
    clean and deployment sides and checked against the shape convention the
    attack actually consumes.  A layer is returned as usable only when every
    required tensor is present and usable on both sides.  Unusable layers are
    recorded in the returned ``missing`` list as ``{"layer": i, "reason": ...}``
    - a present-but-unusable layer is an execution gap, never a silent skip.

    Returns ``(usable_layers, missing)``; both are ordered by layer index.
    """
    templates = tuple(str(item) for item in required)
    if not templates:
        raise ValueError("required must name at least one weight tensor template")

    n_layers = int(getattr(public, "n_layers", 0) or 0)
    if n_layers > 0:
        candidates = list(range(n_layers))
    else:
        candidates = [int(layer) for layer in (getattr(public, "layers", ()) or ())]
        if not candidates:
            raise CapabilityError(
                "public view reports n_layers=0 and no sampled layers; cannot "
                "enumerate the weight layers"
            )

    n_heads = max(1, int(getattr(public, "n_heads", 1) or 1))
    n_kv_heads = max(1, int(getattr(public, "n_kv_heads", n_heads) or n_heads))
    usable: list[int] = []
    missing: list[dict] = []
    for layer in candidates:
        reasons: list[tuple[str, str]] = []
        shapes: dict[tuple[str, str], tuple[int, ...]] = {}
        for template in templates:
            name = _weight_name(template, layer)
            role = _weight_role(name)
            per_side: dict[str, tuple[int, ...]] = {}
            for side in ("clean", "deployment"):
                try:
                    tensor = np.asarray(public.weight_tensor(name, side))
                except (KeyError, CapabilityError, OSError, ValueError) as exc:
                    reasons.append((
                        _unreadable_code(exc),
                        f"{side} {name!r} unavailable "
                        f"({type(exc).__name__}: {exc})",
                    ))
                    continue
                problem = _weight_shape_problem(public, name, tensor.shape)
                if problem is not None:
                    reasons.append(
                        ("shape_invalid", f"{side} {name!r} unusable: {problem}")
                    )
                    continue
                per_side[side] = tuple(int(v) for v in tensor.shape)
                shapes[(role, side)] = per_side[side]
            if len(per_side) == 2 and per_side["clean"] != per_side["deployment"]:
                reasons.append((
                    "side_mismatch",
                    f"{name!r} shape mismatch: clean {per_side['clean']} vs "
                    f"deployment {per_side['deployment']}",
                ))
        for side in ("clean", "deployment"):
            q_shape = shapes.get(("q", side))
            k_shape = shapes.get(("k", side))
            if q_shape is not None and k_shape is not None:
                q_dim = int(q_shape[0]) // n_heads
                k_dim = int(k_shape[0]) // n_kv_heads
                if q_dim != k_dim:
                    reasons.append((
                        "head_dim_mismatch",
                        f"{side} q head_dim {q_dim} != k head_dim {k_dim}",
                    ))
        if reasons:
            codes = {code for code, _ in reasons}
            code = next(item for item in _MISSING_CODE_PRIORITY if item in codes)
            missing.append({
                "layer": int(layer),
                "code": code,
                "reason": "; ".join(text for _, text in reasons),
            })
        else:
            usable.append(int(layer))
    return usable, missing


def _head_view(x: np.ndarray, n_heads: int) -> np.ndarray:
    tokens, width = x.shape
    if n_heads <= 1:
        return x[None]
    if width % n_heads:
        raise ValueError(f"width {width} not divisible by n_heads {n_heads}")
    return x.reshape(tokens, n_heads, width // n_heads).transpose(1, 0, 2)


def _merge_topk(
    best_dist: np.ndarray,
    best_rows: np.ndarray,
    block_dist: np.ndarray,
    block_rows: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a row block into the running Top-k, deterministically.

    Entries are selected and ordered by ``(squared distance, deployment row)``,
    exactly like a stable argsort of the full table, so the returned ranking does
    not depend on how the deployment rows were chunked.  Exact distance ties go
    to the lower row index.
    """
    dist = np.concatenate([best_dist, block_dist], axis=1)
    rows = np.concatenate([best_rows, block_rows], axis=1)
    n_rows, width = dist.shape
    k = max(1, min(int(top_k), width))
    if k == width:
        keep = np.ones((n_rows, width), dtype=bool)
    else:
        kth = np.partition(dist, k - 1, axis=1)[:, k - 1]
        keep = dist < kth[:, None]
        tied = dist == kth[:, None]
        need = k - keep.sum(axis=1)
        for index in np.flatnonzero(need > 0):
            tie_positions = np.flatnonzero(tied[index])
            take = int(min(need[index], tie_positions.size))
            if take < tie_positions.size:
                tie_order = np.argsort(rows[index, tie_positions], kind="stable")
                tie_positions = tie_positions[tie_order[:take]]
            keep[index, tie_positions] = True
    # Sorting ``inf`` onto every non-selected entry leaves the selected k first;
    # a stable argsort then keeps distance order, and any exact tie is fixed by
    # row index below (ties are rare, so the repair stays off the hot path).
    masked_dist = np.where(keep, dist, np.inf)
    order = np.argsort(masked_dist, axis=1, kind="stable")[:, :k]
    kept_dist = np.take_along_axis(dist, order, axis=1)
    kept_rows = np.take_along_axis(rows, order, axis=1)
    if k > 1:
        tied_rows = np.flatnonzero(np.any(kept_dist[:, 1:] == kept_dist[:, :-1], axis=1))
        for index in tied_rows:
            tie_order = np.lexsort((kept_rows[index], kept_dist[index]))
            kept_dist[index] = kept_dist[index][tie_order]
            kept_rows[index] = kept_rows[index][tie_order]
    return kept_dist, kept_rows


def _topk_against_full_table(
    clean_inv: np.ndarray,
    row_invariants: Iterator[tuple[int, np.ndarray]],
    top_k: int,
    n_rows: int,
    *, return_distances: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    clean_inv = np.asarray(clean_inv)
    if clean_inv.ndim != 2 or not len(clean_inv) or int(top_k) < 1 or int(n_rows) < 1:
        raise ValueError("full-table matching requires non-empty queries, rows and positive top_k")
    if not np.isfinite(clean_inv).all():
        raise ValueError("query invariants must be finite")
    best_dist = np.full((clean_inv.shape[0], 0), np.inf)
    best_rows = np.full((clean_inv.shape[0], 0), -1, dtype=np.int64)
    expected_start = 0
    for start, block in row_invariants:
        block = np.asarray(block)
        if (int(start) != expected_start or block.ndim != 2 or block.shape[1] != clean_inv.shape[1]
                or len(block) == 0 or expected_start + len(block) > int(n_rows)):
            raise ValueError("deployment invariant chunks must cover the full table exactly once in order")
        if not np.isfinite(block).all():
            raise ValueError("deployment invariants must be finite")
        expected_start += len(block)
        d = cdist(clean_inv, block, metric="sqeuclidean")
        rows = np.broadcast_to(np.arange(start, start + block.shape[0]), d.shape)
        best_dist, best_rows = _merge_topk(best_dist, best_rows, d, rows, top_k)
    if expected_start != int(n_rows):
        raise ValueError("deployment invariant chunks do not cover the full table")
    order = np.argsort(best_dist, axis=1, kind="stable")
    ranked = np.take_along_axis(best_rows, order, axis=1)
    if ranked.shape[0] == 0:  # pragma: no cover - defensive
        raise RuntimeError(f"no deployment rows available (vocab {n_rows})")
    distances = np.take_along_axis(best_dist, order, axis=1)
    return (ranked, distances) if return_distances else ranked


def _chunks(n_rows: int, chunk: int | None, features: int, queries: int = 1) -> Iterator[slice]:
    """Row chunks bounded by the distance-tensor budget ``queries x chunk x features``."""
    if chunk is None:
        budget = int(2 ** 25)  # ~128 MiB float32
        chunk = max(1, min(n_rows, budget // max(1, int(features) * int(queries))))
    if int(chunk) < 1:
        raise ValueError("chunk must be positive")
    for start in range(0, n_rows, int(chunk)):
        yield slice(start, min(start + int(chunk), n_rows))


def _gate_weight(public: PublicDeployment, layer: int, side: str) -> np.ndarray:
    """One gate matrix as ``(hidden, intermediate)``, matching the invariant."""
    name = f"model.layers.{int(layer)}.mlp.gate_proj.weight"
    weight = np.asarray(public.weight_tensor(name, side), dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"gate weight {name!r} must be 2-D; got {weight.shape}")
    hidden = int(getattr(public, "hidden_size", 0) or 0)
    if hidden and weight.shape[1] == hidden:
        return weight.T  # HF Linear layout (intermediate, hidden)
    if hidden and weight.shape[0] == hidden:
        return weight  # already (hidden, intermediate)
    return weight.T  # hidden size unknown: keep the historical transpose


def _gate_mean_weights(public: PublicDeployment, layers: Sequence[int],
                       side: str) -> np.ndarray:
    """Reduce each layer before loading the next weight tensor."""
    return np.stack([
        _gate_weight(public, layer, side).mean(axis=1, dtype=np.float64)
        for layer in layers
    ], axis=1)


def gate_public_candidates(
    public: PublicDeployment,
    clean_types: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
    top_k: int = 10,
    chunk: int | None = None,
    reverse: bool = False,
) -> dict:
    """Rank the full table by Gate-IA (deployment-to-clean when reverse=True).

    By default every model layer that supplies a usable gate weight is used
    (``0 .. n_layers-1``); ``layers=`` overrides that list verbatim.  The layers
    that actually participated and those excluded because a required tensor was
    missing or unusable are both recorded in the result.
    """
    types = np.asarray(clean_types, dtype=np.int64)
    e_clean = public.table("plaintext")
    e_dep = public.table(public.arm)
    if layers is None:
        used_layers, missing = weight_layers(public, required=(GATE_WEIGHT,))
    else:
        used_layers, missing = [int(layer) for layer in layers], []
    if not used_layers:
        raise CapabilityError(
            f"Gate-IA found no usable weight layer among "
            f"{int(getattr(public, 'n_layers', 0) or 0)} model layers; "
            f"gaps: {missing}"
        )
    query_table, candidate_table = (e_dep, e_clean) if reverse else (e_clean, e_dep)
    query_side, candidate_side = ("deployment", "clean") if reverse else ("clean", "deployment")
    if np.any(types < 0) or np.any(types >= len(query_table)):
        raise ValueError("query IDs are outside the public query table")
    clean_inv = query_table[types] @ _gate_mean_weights(public, used_layers, query_side)
    candidate_gate = _gate_mean_weights(public, used_layers, candidate_side)
    n_rows = len(candidate_table)
    features = int(clean_inv.shape[1])
    row_iter = (
        (sl.start, candidate_table[sl] @ candidate_gate)
        for sl in _chunks(n_rows, chunk, features + 2 * len(types))
    )
    ranked, distances = _topk_against_full_table(clean_inv, row_iter, top_k, n_rows,
                                                return_distances=True)
    return {
        "candidates": {
            int(token): [int(row) for row in ranked[i]]
            for i, token in enumerate(types)
        },
        "candidate_distances": {
            int(token): [float(value) for value in distances[i]]
            for i, token in enumerate(types)
        },
        "n_candidates": n_rows,
        "invariant_dims": features,
        "layers": [int(layer) for layer in used_layers],
        "missing_weight_layers": missing,
        "attack_id": "gate_ia",
    }


def _attn_gram(public: PublicDeployment, layers: Sequence[int], side: str,
               pairing: str, block_size: int,
               chunk: int | None) -> tuple[dict, list, list]:
    """Accumulate the full-vocabulary whitening Gram per layer/group/head/block.

    Returns ``(inverse_grams, sorted_keys, skipped)``.  Singular or
    ill-conditioned Gram blocks are skipped (never pseudo-inverted) and each one
    is recorded in ``skipped`` with its layer/group/head/block, condition and
    side, so a layer that contributes no invariant column is visible to the
    caller instead of silently disappearing.
    """
    table = np.asarray(
        public.table("plaintext") if side == "clean" else public.table(public.arm),
        dtype=np.float32,
    )
    n_heads = max(1, int(getattr(public, "n_heads", 1)))
    n_kv_heads = max(1, int(getattr(public, "n_kv_heads", n_heads)))
    grams: dict[tuple, np.ndarray] = {}
    skipped: list[dict] = []
    for layer in layers:
        for group in ("q_proj", "k_proj"):
            name = f"model.layers.{int(layer)}.self_attn.{group}.weight"
            weight = np.asarray(public.weight_tensor(name, side), dtype=np.float32).T
            features = int(weight.shape[1])
            accum: dict[tuple, np.ndarray] = {}
            for sl in _chunks(table.shape[0], chunk, features):
                projected = table[sl] @ weight
                heads = _head_view(
                    projected, n_kv_heads if group == "k_proj" else n_heads
                )
                for head_index in range(heads.shape[0]):
                    head = heads[head_index]
                    for columns in rope_block_columns(
                        head.shape[1], block_size, pairing
                    ):
                        key = (int(layer), group, int(head_index), tuple(columns))
                        block = head[:, list(columns)].astype(np.float64)
                        gram = block.T @ block
                        accum[key] = gram if key not in accum else accum[key] + gram
            for key, gram in accum.items():
                condition = float(np.linalg.cond(gram))
                if np.isfinite(condition) and condition <= 1e12:
                    grams[key] = np.linalg.inv(gram)
                    continue
                skipped.append({
                    "layer": int(key[0]),
                    "group": str(key[1]),
                    "head": int(key[2]),
                    "block": [int(c) for c in key[3]],
                    "condition": condition,
                    "code": (
                        "singular_gram"
                        if not np.isfinite(condition)
                        else "condition_exceeded"
                    ),
                    "reason": (
                        "non-finite Gram condition (singular block)"
                        if not np.isfinite(condition)
                        else "Gram condition above the 1e+12 limit"
                    ),
                    "side": side,
                })
    return grams, sorted(grams), skipped


def _attn_invariants_rows(
    public: PublicDeployment,
    rows: np.ndarray,
    side: str,
    grams: dict,
    keys: Sequence[tuple],
) -> np.ndarray:
    """Per-token whitening invariants for selected rows, from the cached Grams."""
    rows = np.asarray(rows, dtype=np.int64)
    table = np.asarray(
        public.table("plaintext") if side == "clean" else public.table(public.arm),
        dtype=np.float32,
    )
    n_heads = max(1, int(getattr(public, "n_heads", 1)))
    n_kv_heads = max(1, int(getattr(public, "n_kv_heads", n_heads)))
    invariant = np.empty((len(rows), len(keys)), dtype=np.float64)
    previous_group = None
    embeddings = table[rows]
    for column, key in enumerate(keys):
        layer, group, head_index, columns = key
        heads_for_group = n_kv_heads if group == "k_proj" else n_heads
        cache_key = (int(layer), group)
        if cache_key != previous_group:
            name = f"model.layers.{int(layer)}.self_attn.{group}.weight"
            weight = np.asarray(public.weight_tensor(name, side), dtype=np.float32).T
            heads = _head_view(embeddings @ weight, heads_for_group)
            previous_group = cache_key
        head = heads[int(head_index)]
        block = head[:, list(columns)]
        gram_inv = grams[key]
        invariant[:, column] = np.einsum("nb,bc,nc->n", block, gram_inv, block)
    return invariant


def attn_public_candidates(
    public: PublicDeployment,
    clean_types: Sequence[int],
    layers: Sequence[int] | None = None,
    *,
    top_k: int = 10,
    pairing: str = "half_split",
    block_size: int = 2,
    chunk: int | None = None,
    reverse: bool = False,
) -> dict:
    """Rank the full table by Attn-IA (deployment-to-clean when reverse=True).

    By default every model layer that supplies usable query *and* key weights is
    used (``0 .. n_layers-1``); ``layers=`` overrides that list verbatim.  A layer
    whose Gram blocks are all singular / ill-conditioned contributes no invariant
    column and is removed from the reported ``layers`` and recorded in
    ``missing_weight_layers``; the individual skipped blocks are listed in
    ``skipped_gram_blocks``.  Layers that still contribute are kept.

    Two passes keep memory bounded: pass 1 accumulates the full-vocabulary Gram
    (the invariant is defined over all tokens), pass 2 computes invariants for one
    row chunk at a time and maintains the running Top-k. No ``V x D`` invariant
    matrix or ``n_types x V x D`` distance tensor is materialised.
    """
    types = np.asarray(clean_types, dtype=np.int64)
    if layers is None:
        used_layers, missing = weight_layers(public, required=ATTN_WEIGHTS)
    else:
        used_layers, missing = [int(layer) for layer in layers], []
    if not used_layers:
        raise CapabilityError(
            f"Attn-IA found no usable weight layer among "
            f"{int(getattr(public, 'n_layers', 0) or 0)} model layers; "
            f"gaps: {missing}"
        )
    grams_clean, keys_clean, skipped_clean = _attn_gram(
        public, used_layers, "clean", pairing, block_size, chunk
    )
    grams_dep, keys_dep, skipped_dep = _attn_gram(
        public, used_layers, "deployment", pairing, block_size, chunk
    )
    skipped_gram_blocks = skipped_clean + skipped_dep
    keys = sorted(set(keys_clean) & set(keys_dep))
    if not keys:
        raise CapabilityError(
            "Attn-IA found no query/key Gram block invertible on both the clean "
            f"and deployment sides; skipped blocks: {skipped_gram_blocks}"
        )
    # A layer only participates through the invariant columns common to both
    # sides; a layer whose blocks were all skipped (e.g. all-zero weights) must
    # be reported as a gap, not counted as covered.
    columns_by_layer: dict[int, int] = {}
    for key in keys:
        columns_by_layer[int(key[0])] = columns_by_layer.get(int(key[0]), 0) + 1
    participating = [
        int(layer) for layer in used_layers if columns_by_layer.get(int(layer), 0)
    ]
    empty_layers = [
        {
            "layer": int(layer),
            "code": "zero_columns",
            "reason": (
                "contributed 0 of the invariant columns: every query/key Gram "
                "block was singular or non-finite, above the 1e+12 condition "
                "limit, or unmatched between the clean and deployment sides"
            ),
        }
        for layer in used_layers if not columns_by_layer.get(int(layer), 0)
    ]
    if not participating:  # pragma: no cover - covered by the ``not keys`` guard
        raise CapabilityError(
            "Attn-IA has no layer contributing an invariant column; "
            f"skipped blocks: {skipped_gram_blocks}"
        )
    used_layers = participating
    missing = sorted(
        list(missing) + empty_layers, key=lambda record: record["layer"]
    )
    query_side, candidate_side = ("deployment", "clean") if reverse else ("clean", "deployment")
    query_grams, candidate_grams = (grams_dep, grams_clean) if reverse else (grams_clean, grams_dep)
    query_arm, candidate_arm = (public.arm, "plaintext") if reverse else ("plaintext", public.arm)
    if np.any(types < 0) or np.any(types >= len(public.table(query_arm))):
        raise ValueError("query IDs are outside the public query table")
    clean_inv = _attn_invariants_rows(
        public, types, query_side, query_grams, keys
    )
    n_rows = len(public.table(candidate_arm))

    def row_iter():
        for sl in _chunks(n_rows, chunk, len(keys) + 2 * len(types)):
            yield sl.start, _attn_invariants_rows(
                public, np.arange(sl.start, sl.stop), candidate_side, candidate_grams,
                keys,
            )

    n_chunks = 0
    def counted_iter():
        nonlocal n_chunks
        for start, values in row_iter():
            n_chunks += 1
            yield start, values

    ranked = _topk_against_full_table(clean_inv, counted_iter(), top_k, n_rows)
    return {
        "candidates": {
            int(token): [int(row) for row in ranked[i]]
            for i, token in enumerate(types)
        },
        "n_candidates": n_rows,
        "invariant_dims": int(len(keys)),
        "layers": [int(layer) for layer in used_layers],
        "missing_weight_layers": missing,
        "skipped_gram_blocks": skipped_gram_blocks,
        "pairing": pairing,
        "block_size": block_size,
        "n_chunks": int(n_chunks),
        "attack_id": "attn_ia",
    }


def recover_mapping_gate_ia(
    public: PublicDeployment,
    *,
    n_types: int = 4000,
    top_k: int = 3,
    seed: int = 0,
    layers: Sequence[int] | None = None,
    chunk: int | None = None,
) -> dict:
    """Sample clean types and rank deployment rows; scoring stays outside."""
    vocab = int(public.vocab_size)
    rng = np.random.default_rng(seed)
    types = np.sort(rng.choice(vocab, size=min(int(n_types), vocab), replace=False))
    out = gate_public_candidates(public, types, layers=layers, top_k=top_k, chunk=chunk)
    reverse: dict[int, list[dict]] = {}
    for token, candidates in out["candidates"].items():
        for rank, (row, distance) in enumerate(zip(candidates, out["candidate_distances"][token]), start=1):
            reverse.setdefault(int(row), []).append({"token_id": int(token), "distance": float(distance), "rank": rank})
    for candidates in reverse.values():
        candidates.sort(key=lambda item: (item["distance"], item["rank"], item["token_id"]))
    mapping = {row: candidates[0]["token_id"] for row, candidates in reverse.items()}
    return {
        "types": [int(t) for t in types],
        "mapping": mapping,
        "mapping_candidates": reverse,
        "collision_rows": sum(len(candidates) > 1 for candidates in reverse.values()),
        "selection_rule": "minimum invariant distance; ties by forward rank then clean ID; uncalibrated",
        "candidates": out["candidates"],
        "n_types": int(len(types)),
        "n_candidates": out["n_candidates"],
        "invariant_dims": out["invariant_dims"],
        "layers": out["layers"],
        "missing_weight_layers": out["missing_weight_layers"],
    }


def attention_invariants(public: PublicDeployment, clean_types: Sequence[int],
                         layers: Sequence[int] | None = None, *,
                         pairing: str = "half_split",
                         top_k: int = 10) -> dict:
    """Backwards-compatible alias for the public Attn-IA entry."""
    return attn_public_candidates(
        public, clean_types, layers, top_k=top_k, pairing=pairing
    )
