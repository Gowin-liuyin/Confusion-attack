"""Surface-generic privacy runner and the eight-arm x attack status matrix.

The adapter layer (:mod:`obfuscation_attacks.schemes.adapters`) gives every
defense one executable contract. This module turns that contract into privacy
tests:

* :func:`run_vector_knn` runs the exact full-vocabulary KNN on any adapter that
  exposes a candidate table and a query surface (uploaded vectors, server
  embeddings, or hidden states). The same scoring code therefore applies to
  plaintext, AloePri, A+B, OSNIP-only and DP-Forward without inventing a wire.
* :func:`run_classification` runs the classification arm on its real server
  classifier and reports accuracy + confusion.
* :func:`run_text_surface` records the real perturbed-text upload and server
  continuation (the InferDPT path) instead of faking a vector.
* :func:`adapter_attack_matrix` reports, for every arm x paper attack, whether
  the adapter is ready, blocked (with the exact shortage) or not applicable to
  that surface/task, and whether a paper-v1 executor entry is wired.

None of these functions fabricate a surface: an arm that uploads text never
gets a vector query, and a classification-only arm never gets a generation
metric.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..data.contracts import Capability
from ..schemes.ab_deployment import knn_topk
from ..schemes.adapters import (
    ADAPTER_CLASSES,
    SchemeAdapter,
    UploadSurface,
    adapter_status_table,
)
from ..schemes.arms import ARM_DECLARATIONS, PAPER_ARMS
from .paper_profile import PAPER_ATTACKS
from .real_battery import asr_from_ranks, micro_bootstrap, truth_rank

#: attacks a surface can carry, before arm-specific capability filters
SURFACE_ATTACKS: dict[UploadSurface, tuple[str, ...]] = {
    UploadSurface.IDS: (
        "knn_l2", "knn_cosine", "knn_l2_key_robustness", "ednn",
        "tfma", "sda",
        "linkage_exact", "linkage_cosine", "linkage_knn_aligned",
        "linkage_recovered_text", "aggregation",
    ),
    UploadSurface.VECTORS: (
        "knn_l2", "knn_cosine", "knn_l2_key_robustness", "ednn",
        "ima_ridge", "ima_neural", "aggregation",
        "tfma_after_inversion", "sda",
        "linkage_exact", "linkage_cosine", "linkage_knn_aligned",
        "linkage_recovered_text",
    ),
    UploadSurface.TEXT: (
        "tfma", "sda",
        "text_bayes", "text_lm_rerank",
        "linkage_recovered_text",
    ),
}

#: attacks that also need a server observation capability
CAPABILITY_ATTACKS: dict[str, tuple[str, ...]] = {
    "HIDDEN_STATES": (
        "vma_h_recovered_prefix", "vma_h_observed_prefix", "isa_er", "isa_tbs",
        "gate_ia", "attn_ia",
    ),
    "WHITE_BOX_WEIGHTS": ("vma_w", "least_squares"),
    "MODEL_REPLAY": ("linkage_exact", "linkage_cosine"),
}

GENERATION_ATTACKS = frozenset({"output_pii", "utility_generation", "text_bayes",
                                "text_lm_rerank"})
CLASSIFICATION_ATTACKS = frozenset({"utility_classification"})

#: attack ids with a paper-v1 executor entry in this round
WIRED_ATTACKS = frozenset({"knn_l2", "knn_cosine", "knn_l2_key_robustness"})
WIRED_CELLS = {
    ("plaintext", "knn_l2"), ("plaintext", "knn_cosine"),
    ("aloepri", "knn_l2"), ("aloepri", "knn_cosine"),
    ("ab", "knn_l2"), ("ab", "knn_cosine"),
    ("ab", "knn_l2_key_robustness"),
}


def _wired(arm: str, attack: str) -> bool:
    return (arm, attack) in WIRED_CELLS


#: capabilities each arm exposes by declaration, used when no runtime deployment
#: is supplied so the matrix still reports the right blocked/not-applicable cells
DECLARED_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "plaintext": frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.EMBEDDINGS,
        Capability.HIDDEN_STATES, Capability.MODEL_REPLAY,
        Capability.WHITE_BOX_WEIGHTS}),
    "aloepri": frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.EMBEDDINGS,
        Capability.HIDDEN_STATES, Capability.WHITE_BOX_WEIGHTS,
        Capability.DETERMINISTIC_PERMUTATION}),
    "osnip": frozenset({
        Capability.EMBEDDINGS, Capability.HIDDEN_STATES, Capability.MODEL_REPLAY}),
    "ab": frozenset({
        Capability.EMBEDDINGS, Capability.HIDDEN_STATES,
        Capability.WHITE_BOX_WEIGHTS, Capability.MODEL_REPLAY}),
    "sentinellms": frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.WHITE_BOX_WEIGHTS}),
    "obfuslm_plus": frozenset({
        Capability.TOKEN_CIPHERTEXT, Capability.WHITE_BOX_WEIGHTS}),
    "inferdpt": frozenset({Capability.TOKEN_CIPHERTEXT}),
    "dp_forward": frozenset({
        Capability.EMBEDDINGS, Capability.WHITE_BOX_WEIGHTS}),
}


def attacks_for(adapter: SchemeAdapter) -> dict[str, dict]:
    """Which paper attacks can run on this adapter's real surface, and why."""
    result: dict[str, dict] = {}
    surfaces = SURFACE_ATTACKS[adapter.upload_surface]
    for attack in surfaces:
        result[attack] = {"surface": adapter.upload_surface.value,
                          "requires": "upload surface"}
    capabilities = {cap.name for cap in adapter.server_capabilities}
    for capability, attacks in CAPABILITY_ATTACKS.items():
        if capability not in capabilities:
            continue
        for attack in attacks:
            result.setdefault(attack, {
                "surface": adapter.upload_surface.value,
                "requires": capability,
            })
    tasks = set(adapter.tasks or adapter.declaration.tasks)
    implements_generation = type(adapter).generate is not SchemeAdapter.generate
    if "generation" in tasks and implements_generation:
        for attack in GENERATION_ATTACKS:
            if adapter.upload_surface == UploadSurface.TEXT or attack == "output_pii":
                result.setdefault(attack, {
                    "surface": adapter.upload_surface.value,
                    "requires": "generation task",
                })
        result.setdefault("utility_generation", {
            "surface": adapter.upload_surface.value, "requires": "generation task"})
    if "classification" in tasks and type(adapter).classify is not SchemeAdapter.classify:
        # decoder-only arms score classification through the utility harness, not
        # through a classification head; this adapter layer only carries real heads
        result.setdefault("utility_classification", {
            "surface": adapter.upload_surface.value, "requires": "classification head"})
    result.setdefault("efficiency", {
        "surface": adapter.upload_surface.value, "requires": "same frozen artifact"})
    # protocol §0.1: weight-alignment attacks need an obfuscated weight permutation
    if adapter.name not in {"aloepri", "ab"}:
        for attack in ("gate_ia", "attn_ia", "vma_w", "least_squares"):
            result.pop(attack, None)
    # direct symbol TFMA needs observed symbols; vector uploads compose it after inversion
    if adapter.upload_surface == UploadSurface.VECTORS:
        result.pop("tfma", None)
    else:
        result.pop("tfma_after_inversion", None)
    return result


def _clusters(uploads: Sequence, keys_per_text: int) -> list[int]:
    return [index // max(int(keys_per_text), 1) for index in range(len(uploads))]


def run_vector_knn(
    adapter: SchemeAdapter,
    texts: Sequence[str],
    *,
    metric: str = "l2",
    top_k: int = 100,
    keys_per_text: int = 1,
    layer: int = 1,
    use_hidden: bool = False,
    manifest: Mapping | None = None,
) -> dict:
    """Exact full-vocabulary KNN on whatever representation the adapter exposes."""
    adapter.ensure_ready(manifest)
    uploads = adapter.client_upload(texts, keys_per_text=keys_per_text)
    try:
        queries = adapter.attack_queries(uploads, forward=use_hidden, layer=layer)
        table = adapter.candidate_table()
        truth = adapter.scoring_truth(uploads)
    except NotImplementedError as exc:
        return {
            "status": "not_applicable",
            "arm": adapter.name,
            "reason": str(exc),
            "surface": adapter.upload_surface.value,
        }
    candidates = knn_topk(np.asarray(queries, dtype=np.float32),
                          np.asarray(table, dtype=np.float32),
                          k=int(top_k), metric=metric)
    ranks = truth_rank(candidates, np.asarray(truth).ravel())
    stage1 = asr_from_ranks(ranks, (1, 5, 10, int(top_k)))
    lengths = [int(upload.seq_len) for upload in uploads]
    clusters = _clusters(uploads, keys_per_text)
    hits = [int((ranks[start:start + n] <= 1).sum())
            for start, n in _blocks(lengths)]
    per_unit = []
    start = 0
    for index, n in enumerate(lengths):
        block = ranks[start:start + n]
        per_unit.append({
            "unit": index,
            "text": int(clusters[index]),
            "key": int(index % max(int(keys_per_text), 1)),
            "hits": int((block <= 1).sum()),
            "count": int(n),
            "rank": block.tolist(),
            "predicted": candidates[start:start + n, 0].tolist(),
            "truth": np.asarray(truth).ravel()[start:start + n].tolist(),
        })
        start += n
    return {
        "status": "ok",
        "arm": adapter.name,
        "surface": adapter.upload_surface.value,
        "query_source": "hidden_states" if use_hidden else adapter.upload_surface.value,
        "metric": metric,
        "top_k": int(top_k),
        "n_prompts": len(texts),
        "n_keys": int(keys_per_text),
        "n_tokens": int(np.asarray(truth).size),
        "metrics": {"stage1_permuted_id": stage1},
        "per_unit_predictions": per_unit,
        "ci": {"top1_micro": micro_bootstrap(hits, lengths, clusters=clusters)},
        "candidates": candidates,
        "ranks": ranks,
    }


def _blocks(lengths: Sequence[int]):
    start = 0
    for n in lengths:
        yield start, int(n)
        start += int(n)


def run_classification(
    adapter: SchemeAdapter,
    examples: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping | None = None,
) -> dict:
    """Server classification accuracy on the adapter's real classifier path."""
    adapter.ensure_ready(manifest)
    texts = [str(example["text"]) for example in examples]
    labels = np.asarray([int(example["label"]) for example in examples], dtype=np.int64)
    try:
        probabilities = np.asarray(adapter.classify(texts))
    except NotImplementedError as exc:
        return {"status": "not_applicable", "arm": adapter.name, "reason": str(exc)}
    predictions = probabilities.argmax(axis=1)
    n_classes = int(max(labels.max(initial=0), predictions.max(initial=0)) + 1)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(labels, predictions):
        confusion[int(truth), int(pred)] += 1
    per_class = {}
    for label in range(n_classes):
        mask = labels == label
        per_class[label] = float((predictions[mask] == label).mean()) if mask.any() else None
    return {
        "status": "ok",
        "arm": adapter.name,
        "n": int(len(labels)),
        "accuracy": float((predictions == labels).mean()) if len(labels) else None,
        "per_class_accuracy": per_class,
        "confusion": confusion.tolist(),
    }


def run_text_surface(
    adapter: SchemeAdapter,
    texts: Sequence[str],
    *,
    max_new_tokens: int = 32,
    manifest: Mapping | None = None,
) -> dict:
    """Record the real perturbed-text upload and server continuation."""
    adapter.ensure_ready(manifest)
    uploads = adapter.client_upload(texts)
    try:
        observations = adapter.server_observe(uploads, forward=True)
    except NotImplementedError as exc:
        return {"status": "not_applicable", "arm": adapter.name, "reason": str(exc)}
    traces = []
    for upload, observation in zip(uploads, observations):
        traces.append({
            "request_id": int(upload.request_id),
            "uploaded_text": str(upload.payload),
            "server_completion": observation.generated_text,
        })
    return {
        "status": "ok",
        "arm": adapter.name,
        "surface": "text",
        "n": len(traces),
        "traces": traces,
    }


def adapter_attack_matrix(
    manifest: Mapping | None = None,
    deployments: Mapping[str, Any] | None = None,
    backends: Mapping[str, Any] | None = None,
) -> dict:
    """Every arm x paper-v1 attack: ready / blocked / not_applicable + wiring."""
    status = adapter_status_table(manifest, deployments, backends)
    matrix: dict[str, dict] = {}
    for arm in PAPER_ARMS:
        row = status[arm]
        try:
            if arm in {"plaintext", "aloepri", "ab"}:
                deployment = (deployments or {}).get(arm)
                adapter = ADAPTER_CLASSES[arm](deployment) if deployment is not None else None
            elif arm == "osnip":
                adapter = ADAPTER_CLASSES[arm]((deployments or {}).get("osnip"))
            else:
                adapter = ADAPTER_CLASSES[arm]((backends or {}).get(arm))
            if adapter is not None:
                available = attacks_for(adapter)
            else:
                declaration = ARM_DECLARATIONS[arm]
                surface = UploadSurface(declaration.upload)
                available = {
                    attack: {"surface": surface.value, "requires": "upload surface"}
                    for attack in SURFACE_ATTACKS[surface]
                }
                if arm == "inferdpt":
                    # only InferDPT's adapter contract implements generate() here
                    available.setdefault("utility_generation", {
                        "surface": surface.value, "requires": "generation task"})
                    for attack in ("text_bayes", "text_lm_rerank", "output_pii"):
                        available.setdefault(attack, {
                            "surface": surface.value, "requires": "generation task"})
                if arm in {"sentinellms", "dp_forward", "obfuslm_plus"}:
                    available.setdefault("utility_classification", {
                        "surface": surface.value, "requires": "classification head"})
                available.setdefault("efficiency", {
                    "surface": surface.value, "requires": "same frozen artifact"})
                for capability in DECLARED_CAPABILITIES.get(arm, frozenset()):
                    for attack in CAPABILITY_ATTACKS.get(capability.name, ()):
                        available.setdefault(attack, {
                            "surface": surface.value, "requires": capability.name})
                if arm not in {"aloepri", "ab"}:
                    for attack in ("gate_ia", "attn_ia", "vma_w", "least_squares"):
                        available.pop(attack, None)
                if arm != "inferdpt":
                    available.pop("tfma", None)
                    available.pop("tfma_after_inversion", None)
        except Exception:
            adapter = None
            available = {}
        cells = {}
        for attack in PAPER_ATTACKS:
            if attack not in available:
                cells[attack] = {"status": "not_applicable",
                                 "reason": "surface/task does not expose this attack"}
            elif row["ready"]:
                cells[attack] = {
                    "status": "adapter_ready",
                    "surface": available[attack]["surface"],
                    "requires": available[attack]["requires"],
                    "executor_wired": _wired(arm, attack),
                }
            else:
                cells[attack] = {
                    "status": "blocked",
                    "surface": available[attack]["surface"],
                    "requires": available[attack]["requires"],
                    "executor_wired": _wired(arm, attack),
                    "shortages": list(row["shortages"]),
                }
        matrix[arm] = cells
    return {
        "arms": {arm: {k: status[arm][k] for k in
                       ("ready", "shortages", "status", "tasks", "upload_surface")}
                 for arm in PAPER_ARMS},
        "matrix": matrix,
        "wired_attacks": sorted(WIRED_ATTACKS),
    }
