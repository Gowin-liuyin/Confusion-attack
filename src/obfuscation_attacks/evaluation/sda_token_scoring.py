"""Scoring-side token recovery; private reference IDs never enter the attacker."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from ..metrics.intervals import macro_bootstrap, micro_cluster_bootstrap
from ..metrics.ratio_ci import paired_gain_ci


def _ids(values, *, reference: bool) -> list[int]:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and
            (not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_))):
        raise ValueError("token IDs must be one-dimensional integers")
    if np.any(array < 0):
        raise ValueError("token IDs must be non-negative")
    if reference and not array.size:
        raise ValueError("reference token sequences must not be empty")
    return [int(value) for value in array]


def _dependency_components(request_ids, clusters):
    if not isinstance(clusters, Mapping) or set(clusters) != set(request_ids):
        raise ValueError("cluster request IDs must exactly match reference request IDs")
    if any(isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)) for key in clusters):
        raise ValueError("cluster request IDs must be integers")
    parent = list(range(len(request_ids)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    seen = {}
    for index, request_id in enumerate(request_ids):
        row = clusters[request_id]
        if not isinstance(row, Mapping):
            raise ValueError("each cluster record must identify document, session and key group")
        for field in ("document", "session", "key_group"):
            label = row.get(field)
            if not isinstance(label, str) or not label:
                raise ValueError(f"cluster {field} must be a non-empty string")
            key = (field, label)
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    return [find(index) for index in range(len(request_ids))]


def score_sda_token_rows(references: Mapping[int, object], predictions: Mapping[int, object], *,
                         clusters: Mapping[int, Mapping[str, str]] | None = None,
                         seed: int = 0) -> dict:
    """Score by request ID; absent predictions are misses in the full denominator.

    Token accuracy is positionwise, so use a separate alignment metric for
    variable-length or cross-tokenizer reconstructions. No truth is returned.
    If supplied, clusters must identify each reference's document, session and
    actual repeated-key group; overlapping labels form one resampling unit.
    """
    if not isinstance(references, Mapping) or not isinstance(predictions, Mapping):
        raise ValueError("references and predictions must be keyed by request ID")
    for key in list(references) + list(predictions):
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)):
            raise ValueError("request IDs must be integers")
    if set(predictions) - set(references):
        raise ValueError("prediction request ID is absent from references")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    rows = []
    for request_id, original in references.items():
        truth = _ids(original, reference=True)
        predicted = _ids(predictions.get(request_id, []), reference=False)
        hits = sum(a == b for a, b in zip(truth, predicted))
        rows.append({"request_id": int(request_id), "token_hits": hits,
                     "reference_tokens": len(truth), "prediction_tokens": len(predicted),
                     "exact_sequence": truth == predicted,
                     "prediction_missing": request_id not in predictions})
    hits = sum(row["token_hits"] for row in rows)
    total = sum(row["reference_tokens"] for row in rows)
    exact = sum(row["exact_sequence"] for row in rows)
    ci = {"status": "not_evaluated", "reason": "independent document/session and key clusters not supplied"}
    exact_ci = dict(ci)
    macro = {"status": "not_evaluated", "reason": "document groups not supplied"}
    if clusters is not None:
        components = _dependency_components([row["request_id"] for row in rows], clusters)
        interval = micro_cluster_bootstrap([row["token_hits"] for row in rows],
                                           [row["reference_tokens"] for row in rows], components,
                                           n_boot=2000, seed=seed)
        ci = {**interval, "status": "descriptive" if interval["n_boot"] else "not_evaluated",
              "estimand": "micro_token_accuracy", "seed": seed,
              "resampling_unit": "connected component of shared document, session or repeated-key group",
              "independence": "caller-supplied groups do not establish independence between components"}
        if not interval["n_boot"]:
            ci["reason"] = "at least two nonempty dependency components are required"
        exact_interval = micro_cluster_bootstrap([int(row["exact_sequence"]) for row in rows],
                                                 [1] * len(rows), components, n_boot=2000, seed=seed)
        exact_ci = {**exact_interval, "status": "descriptive" if exact_interval["n_boot"] else "not_evaluated",
                    "estimand": "exact_sequence_recovery", "seed": seed,
                    "resampling_unit": ci["resampling_unit"], "independence": ci["independence"]}
        if not exact_interval["n_boot"]:
            exact_ci["reason"] = "at least two nonempty dependency components are required"
        documents = {}
        for row, component in zip(rows, components):
            doc = clusters[row["request_id"]]["document"]
            sums = documents.setdefault(doc, [0, 0, component])
            sums[0] += row["token_hits"]
            sums[1] += row["reference_tokens"]
        document_values = [hits / count for hits, count, _ in documents.values()]
        macro_interval = macro_bootstrap(document_values, [entry[2] for entry in documents.values()],
                                          n_boot=2000, seed=seed)
        macro = {**macro_interval, "status": "descriptive" if macro_interval["n_boot"] else "not_evaluated",
                 "estimand": "mean_of_document_micro_token_accuracies", "seed": seed,
                 "resampling_unit": ci["resampling_unit"], "independence": ci["independence"]}
        if not macro_interval["n_boot"]:
            macro["reason"] = "at least two nonempty dependency components are required"
    return {
        "status": "scored" if rows else "not_evaluated",
        "n_requests": len(rows), "missing_predictions": sum(row["prediction_missing"] for row in rows),
        "token_accuracy": {"value": hits / total if total else None,
                           "numerator": hits, "denominator": total,
                           "alignment": "positionwise"},
        "exact_sequence": {"value": exact / len(rows) if rows else None,
                           "numerator": exact, "denominator": len(rows), "confidence_interval": exact_ci},
        "confidence_interval": ci,
        "macro_token_accuracy": macro,
        "rows": rows,
    }


def compare_sda_token_predictions(references: Mapping[int, object], predictions: Mapping[int, object],
                                  baseline_predictions: Mapping[int, object], *,
                                  clusters: Mapping[int, Mapping[str, str]], seed: int = 0) -> dict:
    """Compare two fixed predictions on the same references and dependency components."""
    attack = score_sda_token_rows(references, predictions, seed=seed)
    baseline = score_sda_token_rows(references, baseline_predictions, seed=seed)
    attack_rows = attack["rows"]
    baseline_by_id = {row["request_id"]: row for row in baseline["rows"]}
    baseline_rows = [baseline_by_id[row["request_id"]] for row in attack_rows]
    components = _dependency_components([row["request_id"] for row in attack_rows], clusters)
    interval = paired_gain_ci([row["token_hits"] for row in attack_rows],
                              [row["token_hits"] for row in baseline_rows],
                              [row["reference_tokens"] for row in attack_rows],
                              clusters=components, n_boot=2000, seed=seed)
    return {**interval, "status": "descriptive" if interval["n_boot"] else "not_evaluated",
            "estimand": "micro_token_accuracy_difference_attack_minus_baseline",
            "resampling_unit": "connected component of shared document, session or repeated-key group",
            "independence": "caller-supplied groups do not establish independence between components",
            "seed": seed, "attack_missing_predictions": attack["missing_predictions"],
            "baseline_missing_predictions": baseline["missing_predictions"],
            "rows": [{"request_id": row["request_id"], "attack_hits": row["token_hits"],
                      "baseline_hits": other["token_hits"], "reference_tokens": row["reference_tokens"]}
                     for row, other in zip(attack_rows, baseline_rows)]}
