"""Read-only protocol audit for one result payload (protocol §4, §6, §8).

The auditor is separate from the run: it reads a result, applies the paper-v1
floors for the exact attack variant, and returns a verdict. It never edits the
measurement and never upgrades a cell because a directory name or a schema flag
says so. A run's own ``main_table_eligible`` is not an input.

``not_applicable`` is preserved verbatim. A failed positive control is
``control_failed``. An iterative/learned attack without saturation evidence is
``not_converged``. Only a payload that meets every floor, carries the required
per-unit evidence and passes the control is ``audited_paper_scale``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from .paper_profile import PAPER_ARMS, PAPER_PROFILE_NAME, applicable

AUDITOR = "protocol_auditor"
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")

#: every attack id with an explicit paper-v1 floor (unknown ids are fail-closed)
FLOOR_ATTACKS = frozenset(
    {
        "knn_l2", "knn_cosine", "ednn", "knn_l2_key_robustness",
        "vma_h_recovered_prefix", "vma_h_observed_prefix",
        "isa_er", "isa_tbs", "vma_w", "gate_ia", "attn_ia", "least_squares",
        "ima_ridge", "ima_neural",
        "linkage_exact", "linkage_cosine", "linkage_knn_aligned", "linkage_recovered_text",
        "tfma", "tfma_after_inversion", "sda", "sda2021", "sda2023",
        "text_bayes", "text_lm_rerank", "output_pii",
        "utility_generation", "utility_classification", "efficiency",
    }
)

#: attacks whose §4.2 saturation check is required before a paper-scale verdict
SATURATION_REQUIRED = frozenset(
    {"isa_er", "isa_tbs", "ima_ridge", "ima_neural", "sda", "sda2021", "sda2023", "tfma",
     "tfma_after_inversion"}
)

#: attacks that must declare a full-vocabulary candidate set
FULL_VOCAB_REQUIRED = frozenset(
    {
        "knn_l2", "knn_cosine", "ednn", "knn_l2_key_robustness",
        "vma_h_recovered_prefix", "vma_h_observed_prefix",
        "vma_w", "gate_ia", "attn_ia",
    }
)

TEXT_ATTACKS = frozenset({"text_bayes", "text_lm_rerank"})
LINKAGE_ATTACKS = frozenset(
    {"linkage_exact", "linkage_cosine", "linkage_knn_aligned", "linkage_recovered_text"}
)

#: attacks whose point estimate is over sampled texts/sessions and must carry a
#: >=2,000-replicate cluster bootstrap interval (§8.1)
CI_REQUIRED = frozenset(
    {
        "knn_l2", "knn_cosine", "ednn", "knn_l2_key_robustness",
        "vma_h_recovered_prefix", "vma_h_observed_prefix",
        "isa_er", "isa_tbs", "ima_ridge", "ima_neural",
        "tfma", "tfma_after_inversion", "sda", "sda2021", "sda2023",
        "text_bayes", "text_lm_rerank", "output_pii",
        "linkage_exact", "linkage_cosine", "linkage_knn_aligned",
        "linkage_recovered_text",
    }
)

#: arms whose defense artifact/encoder is trained; §8.1 requires seeds 17/29/43
TRAINED_DEFENSE_ARMS = frozenset({
    "ab", "aloepri", "osnip", "sentinellms", "obfuslm_plus", "inferdpt", "dp_forward",
})
REQUIRED_TRAINING_SEEDS = frozenset({17, 29, 43})
CONTROL_FLOORS = {
    "knn_l2": 0.99, "knn_cosine": 0.99, "ednn": 0.99,
    "knn_l2_key_robustness": 0.99,
    "vma_h_recovered_prefix": 0.99, "vma_h_observed_prefix": 0.99,
    "vma_w": 0.99, "least_squares": 0.99,
    "gate_ia": 0.90, "attn_ia": 0.90,
    "isa_er": 0.90, "isa_tbs": 0.90, "tfma": 0.90,
    "tfma_after_inversion": 0.90,
    "linkage_exact": 0.95, "linkage_cosine": 0.95,
    "linkage_knn_aligned": 0.95, "linkage_recovered_text": 0.95,
    "ima_ridge": 0.50, "ima_neural": 0.50,
    "text_bayes": 0.50, "text_lm_rerank": 0.50,
    "output_pii": 0.50, "utility_generation": 0.50,
    "utility_classification": 0.50, "efficiency": 0.50,
}

_EXPECTED_LAYERS = {
    "llama-3.2-1b-base": {1, 5, 10, 15},
    "llama-3.2-3b-instruct": {1, 10, 19, 27},
    "qwen2.5-7b-instruct": {1, 10, 19, 27},
}

#: total transformer depth per frozen model. Hidden-state attacks sample the four
#: ``_EXPECTED_LAYERS`` above (§8); weight attacks must cover every applicable
#: weight layer 0..depth-1, or record why a layer is unusable.
_MODEL_LAYER_COUNTS = {
    "llama-3.2-1b-base": 16,
    "llama-3.2-3b-instruct": 28,
    "qwen2.5-7b-instruct": 28,
}

_WEIGHT_LAYER_ATTACKS = frozenset({"gate_ia", "attn_ia", "vma_w", "least_squares"})

#: A layer may be excluded only for one of these machine-checkable reasons.
#: Free-form prose (e.g. "not requested") is not evidence that a layer is unusable.
WEIGHT_LAYER_MISSING_CODES = frozenset({
    "weight_missing", "tensor_unreadable", "shape_invalid", "side_mismatch",
    "head_dim_mismatch", "zero_columns", "singular_gram", "condition_exceeded",
})


_ARM_ALIASES = {
    "aloepri_weights": "aloepri",
    "ab_composed": "ab",
    "ab_deployment": "ab",
}


def _payload_arm(payload: Mapping[str, Any]) -> str:
    arm = str(payload.get("arm") or "").strip()
    if not arm:
        scheme = str(payload.get("scheme_name") or "")
        if ":" in scheme:
            arm = scheme.split(":")[-1].strip()
        if not arm:
            target = str(payload.get("target") or "")
            arm = target.split("/")[-1].strip() if "/" in target else ""
    if arm in _ARM_ALIASES:
        return _ARM_ALIASES[arm]
    for name in PAPER_ARMS:
        if arm == name or arm.startswith(name + "_") or arm.startswith(name + "-"):
            return name
    return arm


def _weight_layer_gaps(payload: Mapping[str, Any], diag: Mapping[str, Any],
                       used: set[int], depth: int) -> list[str]:
    """Weight attacks cover every layer 0..depth-1, unless a layer is recorded as unusable.

    A layer may be excluded only through ``missing_weight_layers`` with a non-empty
    reason; using a layer and calling it missing at once is contradictory. Every
    layer outside ``used`` must therefore appear in that record, so a truncated
    four-layer run can no longer pass as "all applicable weight layers".
    """
    gaps: list[str] = []
    all_layers = set(range(int(depth)))
    outside = sorted(used - all_layers)
    if outside:
        gaps.append(f"layers: {outside} outside model layer range 0..{depth - 1}")
    records = payload.get("missing_weight_layers")
    if records is None:
        records = diag.get("missing_weight_layers")
    if records is None:
        records = []
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)):
        return gaps + ["missing_weight_layers: not a list of {layer, reason}"]
    missing: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            gaps.append("missing_weight_layers: entry is not a mapping with layer/reason")
            continue
        try:
            layer = int(record.get("layer"))
        except (TypeError, ValueError, OverflowError):
            gaps.append("missing_weight_layers: non-integer layer id")
            continue
        reason = str(record.get("reason") or "").strip()
        code = record.get("code")
        if code not in WEIGHT_LAYER_MISSING_CODES:
            gaps.append(
                f"missing_weight_layers: layer {layer} has no recognised machine-checkable "
                f"code (got {code!r})"
            )
        if layer not in all_layers:
            gaps.append(
                f"missing_weight_layers: layer {layer} outside model layer range 0..{depth - 1}"
            )
        if not reason:
            gaps.append(f"missing_weight_layers: layer {layer} has no recorded reason")
        if layer in missing:
            gaps.append(f"missing_weight_layers: layer {layer} recorded twice")
        if layer in used:
            gaps.append(f"missing_weight_layers: layer {layer} is used and also recorded missing")
        missing.add(layer)
    uncovered = sorted(all_layers - used - missing)
    if uncovered:
        gaps.append(
            f"layers: {uncovered} neither used nor recorded as missing; weight attacks "
            "must cover every applicable layer or record why it is unusable"
        )
    return gaps


def _layer_gaps(payload: Mapping[str, Any], diag: Mapping[str, Any], *,
                attack: str = "") -> list[str]:
    layers = payload.get("layers") or diag.get("layers") or []
    model = str(payload.get("model") or payload.get("target") or "").lower()
    expected = None
    key = None
    if "qwen2.5-7b" in model:
        key = "qwen2.5-7b-instruct"
    elif "3b" in model:
        key = "llama-3.2-3b-instruct"
    elif "1b" in model:
        key = "llama-3.2-1b-base"
    if key is not None:
        expected = _EXPECTED_LAYERS[key]
    if not layers:
        return ["layers: missing"]
    if expected is None:
        return [
            "layers: cannot determine the frozen layer set from the model/target record"
        ]
    try:
        got = {int(layer) for layer in layers}
    except (TypeError, ValueError, OverflowError):
        return ["layers: non-integer layer id"]
    if attack in _WEIGHT_LAYER_ATTACKS:
        return _weight_layer_gaps(payload, diag, got, _MODEL_LAYER_COUNTS[key])
    if got != expected:
        return [f"layers: {sorted(got)} != frozen {sorted(expected)}"]
    return []


def _least_squares_group_gaps(payload: Mapping[str, Any], diag: Mapping[str, Any]) -> list[str]:
    """Require an accounting of every projection used by the dense solver."""
    groups = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    matrices = diag.get("mixing_group_residuals")
    skipped = diag.get("mixing_skipped_groups")
    if not isinstance(matrices, Mapping) or not isinstance(skipped, list):
        return ["mixing groups: per-matrix results and skipped-group records are required"]
    gaps: list[str] = []
    skipped_keys: set[str] = set()
    for record in skipped:
        if not isinstance(record, Mapping):
            gaps.append("mixing skipped groups: invalid record")
            continue
        try:
            key = f"layer{int(record.get('layer'))}.{record.get('group')}"
        except (TypeError, ValueError, OverflowError):
            gaps.append("mixing skipped groups: invalid layer")
            continue
        if key in skipped_keys:
            gaps.append(f"mixing group {key}: recorded as skipped twice")
        skipped_keys.add(key)
        if record.get("code") not in WEIGHT_LAYER_MISSING_CODES or not str(record.get("reason") or "").strip():
            gaps.append(f"mixing group {key}: skipped without a recognized code and reason")
        if key in matrices:
            gaps.append(f"mixing group {key}: both solved and skipped")
    try:
        layers = {int(layer) for layer in (payload.get("layers") or diag.get("layers") or [])}
    except (TypeError, ValueError, OverflowError):
        return gaps
    for layer in sorted(layers):
        for group in groups:
            key = f"layer{layer}.{group}"
            if key in skipped_keys:
                continue
            result = matrices.get(key)
            try:
                residual = float(result["relative_residual"])
                if not math.isfinite(residual) or residual < 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError, OverflowError):
                gaps.append(f"mixing group {key}: missing valid per-matrix result or skipped record")
    return gaps


TOPK_SHAPE_ATTACKS = frozenset({
    "knn_l2", "knn_cosine", "ednn", "knn_l2_key_robustness",
    "vma_h_recovered_prefix", "vma_h_observed_prefix",
    "isa_er", "isa_tbs", "tfma", "tfma_after_inversion",
    "gate_ia", "attn_ia", "vma_w", "least_squares",
})

TOPK_CONTAINERS = {
    "knn_l2": {"stage1_permuted_id"},
    "knn_cosine": {"stage1_permuted_id"},
    "knn_l2_key_robustness": {"stage1_permuted_id"},
    "ednn": {"stage1_permuted_id", "mapping_asr", "occurrence_level"},
    "vma_h_recovered_prefix": {"occurrence_level"},
    "vma_h_observed_prefix": {"occurrence_level"},
    "isa_er": {"budget_curve"},
    "isa_tbs": {"budget_curve"},
    "tfma": {"occurrence_level"},
    "tfma_after_inversion": {"occurrence_level"},
    "gate_ia": {"mapping_asr"},
    "attn_ia": {"mapping_asr"},
    "vma_w": {"mapping_asr", "occurrence_level", "mapping_recovery_rate"},
    "least_squares": {"mapping_asr", "occurrence_level", "mapping_recovery_rate"},
}
PREMISE_CODES = frozenset({
    "no_generation_path", "no_decoder_hidden_state", "no_weight_permutation",
    "no_request_stream", "no_pii_annotation", "no_dense_hidden_rebasing",
})

#: premise code -> attacks for which that absence is a real, related reason.
#: A "real_premise_absent" claim outside these sets is rejected: an applicable
#: attack cannot relabel its own failure with an unrelated controlled code.
PREMISE_ATTACKS = {
    "no_generation_path": {
        "text_bayes", "text_lm_rerank", "output_pii",
    },
    "no_decoder_hidden_state": {
        "vma_h_recovered_prefix", "vma_h_observed_prefix", "isa_er", "isa_tbs",
        "ednn", "gate_ia", "attn_ia", "beamclean",
    },
    "no_weight_permutation": {
        "vma_w", "gate_ia", "attn_ia", "least_squares",
        "ima_ridge", "ima_neural",
    },
    "no_request_stream": {
        "knn_l2", "knn_cosine", "knn_l2_key_robustness", "ednn",
        "vma_h_recovered_prefix", "vma_h_observed_prefix", "isa_er", "isa_tbs",
        "tfma", "tfma_after_inversion", "sda", "sda2021", "sda2023",
        "linkage_exact", "linkage_cosine", "linkage_knn_aligned",
        "linkage_recovered_text",
    },
    "no_pii_annotation": {"output_pii"},
    "no_dense_hidden_rebasing": {"vma_w", "least_squares"},
}


def _all_mapping_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            keys.add(str(key))
            keys |= _all_mapping_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            keys |= _all_mapping_keys(value)
    return keys


def _find_key(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        for value in obj.values():
            found = _find_key(value, name)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_key(value, name)
            if found is not None:
                return found
    return None


def _collect_key_values(obj: Any, name: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == name:
                values.append(value)
            values.extend(_collect_key_values(value, name))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            values.extend(_collect_key_values(value, name))
    return values


def _metric_shape_gaps(attack: str, metrics: Mapping[str, Any]) -> list[str]:
    if not isinstance(metrics, Mapping):
        return ["metrics: missing mapping"]
    fractions = _fractions(metrics)
    paths = [path for path, _ in fractions]
    gaps: list[str] = []
    if attack in TOPK_SHAPE_ATTACKS:
        containers = TOPK_CONTAINERS.get(attack, set())
        for need in ("top1", "top5", "top10"):
            if not any(
                path.endswith(need)
                and "random_baseline" not in path
                and any(segment in containers for segment in path.split("."))
                for path in paths
            ):
                gaps.append(
                    f"metrics: missing a numerator/denominator {need} point estimate"
                )
    if attack in LINKAGE_ATTACKS:
        if _find_key(metrics, "test") is None:
            gaps.append("metrics: missing linkage test-split block")
        if not any(
            _find_key(metrics, key) is not None
            for key in ("exact_test", "approximate", "auc")
        ):
            gaps.append("metrics: missing linkage exact/approximate/AUC result")
    if attack in {"sda", "sda2021", "sda2023"}:
        block = _find_key(metrics, "token_accuracy")
        if not isinstance(block, Mapping):
            gaps.append("metrics: missing a numeric token_accuracy point")
        else:
            value = block.get("value")
            try:
                value_f = float(value)
            except (TypeError, ValueError, OverflowError):
                gaps.append("metrics: token_accuracy.value is not numeric")
                value_f = None
            if value_f is not None and not math.isfinite(value_f):
                gaps.append("metrics: token_accuracy.value is not finite")
                value_f = None
            if "numerator" not in block or "denominator" not in block:
                gaps.append(
                    "metrics: token_accuracy must carry numerator and denominator"
                )
            elif value_f is not None:
                try:
                    expected = int(block["numerator"]) / int(block["denominator"])
                    if abs(value_f - expected) > 1e-9:
                        gaps.append(
                            "metrics: token_accuracy.value != numerator/denominator"
                        )
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    gaps.append("metrics: token_accuracy fraction is invalid")
    return gaps


def _rank_reconciliation_gaps(
    attack: str, metrics: Mapping[str, Any], diag: Mapping[str, Any],
) -> list[str]:
    if attack not in TOPK_SHAPE_ATTACKS:
        return []
    rows = diag.get("per_unit_predictions")
    if not isinstance(rows, list) or not rows:
        return []
    gaps: list[str] = []

    def _totals(rows_block, label: str):
        block_gaps: list[str] = []
        block_totals = {1: 0, 5: 0, 10: 0, 100: 0}
        block_positions = 0
        for index, row in enumerate(rows_block):
            ranks = row.get("rank") if isinstance(row, Mapping) else None
            count = row.get("count") if isinstance(row, Mapping) else None
            if (
                not isinstance(ranks, list)
                or not isinstance(count, int)
                or len(ranks) != count
            ):
                block_gaps.append(
                    f"evidence: {label}[{index}] rank list required for top-k"
                )
                continue
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in ranks
            ):
                block_gaps.append(
                    f"evidence: {label}[{index}] rank values are not positive integers"
                )
                continue
            block_positions += count
            for k in block_totals:
                block_totals[k] += sum(1 for value in ranks if value <= k)
        return block_totals, block_positions, block_gaps

    totals, total_positions, gaps = _totals(rows, "per_unit_predictions")
    # ISA reports every sampled layer; each layer's curve must reconcile with
    # that layer's own per-unit rows, not with the headline layer's table.
    layer_totals: dict[str, tuple[dict[int, int], int]] = {}
    by_layer = diag.get("per_unit_predictions_by_layer")
    if isinstance(by_layer, Mapping):
        for name, layer_rows in by_layer.items():
            if not isinstance(layer_rows, list) or not layer_rows:
                gaps.append(
                    f"evidence: per_unit_predictions_by_layer[{name}] is empty"
                )
                continue
            layer_totals[str(name)] = _totals(
                layer_rows, f"per_unit_predictions_by_layer[{name}]"
            )[:2]
            gaps.extend(
                _totals(layer_rows, f"per_unit_predictions_by_layer[{name}]")[2]
            )
    if gaps:
        return gaps
    import re

    entries: list[tuple[int, str, int, int]] = []
    containers = TOPK_CONTAINERS.get(attack, set())
    for path, block in _fractions(metrics):
        match = re.search(r"top(\d+)$", path)
        if not match:
            continue
        if not any(segment in containers for segment in path.split(".")):
            continue
        k = int(match.group(1))
        if k not in totals:
            continue
        try:
            numerator = int(block["numerator"])
            denominator = int(block["denominator"])
        except (TypeError, ValueError, KeyError, OverflowError):
            gaps.append(f"{path}: numerator/denominator are not integers")
            continue
        entries.append((k, path, numerator, denominator))
    if attack in {"isa_er", "isa_tbs"}:
        # Only the final budget point is the reported estimate. Group by layer
        # and top-k and keep the largest checkpoint step; paths without a step
        # (fixture/legacy layouts) fall back to the largest numerator.
        best: dict[tuple[str, int], tuple[int, tuple[int, str, int, int]]] = {}
        for entry in entries:
            k = entry[0]
            match = re.search(r"budget_curve\.([^.]+)\.steps(\d+)\.", entry[1])
            if match:
                key = (match.group(1), k)
                rank = int(match.group(2))
            else:
                key = ("_", k)
                rank = entry[2]
            current = best.get(key)
            if current is None or rank > current[0]:
                best[key] = (rank, entry)
        entries = [value for _, value in best.values()]
    for k, path, numerator, denominator in entries:
        layer_match = re.search(r"budget_curve\.([^.]+)\.steps(\d+)\.", path)
        if layer_match and layer_match.group(1) in layer_totals:
            use_totals, use_positions = layer_totals[layer_match.group(1)]
        else:
            use_totals, use_positions = totals, total_positions
        if denominator != use_positions:
            gaps.append(
                f"{path}.denominator {denominator} != per-unit "
                f"count {use_positions}"
            )
            continue
        if numerator != use_totals[k]:
            gaps.append(
                f"{path}.numerator {numerator} != ranks<= {k} "
                f"recomputed {use_totals[k]}"
            )
    return gaps


def _ci_gaps(payload: Mapping[str, Any], attack: str = "",
             metrics: Mapping[str, Any] | None = None) -> list[str]:
    ci = payload.get("ci")
    if not isinstance(ci, Mapping) or not ci:
        return ["ci: cluster-bootstrap interval block missing (>=2000 replicates)"]
    gaps: list[str] = []
    found = False
    bounded = False
    points: list[float] = []
    named_points: list[tuple[str, float]] = []
    for name, block in ci.items():
        if not isinstance(block, Mapping):
            continue
        interesting = any(
            key in block for key in (
                "n_boot", "lo", "hi", "micro", "value", "mean", "top1",
                "rate", "estimate", "point",
            )
        )
        if not interesting:
            continue
        found = True
        has_bounds = "lo" in block and "hi" in block
        if not has_bounds:
            gaps.append(f"ci.{name}: missing lo/hi bounds")
        point = None
        for key in ("micro", "macro", "value", "mean", "top1", "rate", "estimate", "point"):
            if key in block:
                point = block[key]
                break
        if has_bounds:
            bounded = True
            try:
                lo_raw = block["lo"]
                hi_raw = block["hi"]
                if (
                    isinstance(lo_raw, bool) or isinstance(hi_raw, bool)
                    or not isinstance(lo_raw, (int, float))
                    or not isinstance(hi_raw, (int, float))
                ):
                    raise TypeError("non-numeric bounds")
                lo = float(lo_raw)
                hi = float(hi_raw)
                if not (math.isfinite(lo) and math.isfinite(hi)):
                    raise ValueError("non-finite bounds")
                if not (0.0 <= lo <= hi <= 1.0):
                    gaps.append(f"ci.{name}: bounds {lo}..{hi} are not a probability interval")
                if point is None:
                    gaps.append(f"ci.{name}: interval has no point estimate field")
                elif isinstance(point, bool) or not isinstance(point, (int, float)):
                    gaps.append(f"ci.{name}: point estimate is not numeric")
                else:
                    point_value = float(point)
                    if not math.isfinite(point_value):
                        gaps.append(f"ci.{name}: point estimate is not finite")
                        continue
                    points.append(point_value)
                    named_points.append((str(name), point_value))
                    if not (lo <= point_value <= hi):
                        gaps.append(
                            f"ci.{name}: interval [{lo}, {hi}] does not contain its "
                            f"point estimate {point_value}"
                        )
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"ci.{name}: lo/hi/point are not numeric")
        if "n_boot" not in block:
            gaps.append(f"ci.{name}.n_boot: missing (>=2000 required)")
        else:
            try:
                boot = int(block["n_boot"])
                if boot < 2000 or boot > 1_000_000:
                    gaps.append(
                        f"ci.{name}.n_boot: {block['n_boot']} outside [2000, 1000000]"
                    )
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"ci.{name}.n_boot: {block['n_boot']!r} is not numeric")
    if not found:
        gaps.append("ci: no interval declares n_boot (>=2000 replicates)")
    if not bounded:
        gaps.append("ci: no interval carries both lo and hi bounds")
    if attack in TOPK_SHAPE_ATTACKS and isinstance(metrics, Mapping):
        primary = None
        containers = TOPK_CONTAINERS.get(attack, set())
        isa_final: dict[str, tuple[int, float]] = {}
        isa_fallback: tuple[int, float] | None = None
        for path, block in _fractions(metrics):
            if (
                path.endswith("top1")
                and "random_baseline" not in path
                and any(segment in containers for segment in path.split("."))
            ):
                try:
                    value = float(
                        block.get("value")
                        if block.get("value") is not None
                        else int(block["numerator"]) / int(block["denominator"])
                    )
                except (TypeError, ValueError, KeyError, OverflowError, ZeroDivisionError):
                    continue
                if attack in {"isa_er", "isa_tbs"}:
                    match = re.search(r"budget_curve\.([^.]+)\.steps(\d+)\.", path)
                    if match:
                        layer, step = match.group(1), int(match.group(2))
                        if layer not in isa_final or step > isa_final[layer][0]:
                            isa_final[layer] = (step, value)
                        continue
                    try:
                        numerator = int(block["numerator"])
                    except (TypeError, ValueError, KeyError, OverflowError):
                        continue
                    if isa_fallback is None or numerator > isa_fallback[0]:
                        isa_fallback = (numerator, value)
                    continue
                primary = value if primary is None else max(primary, value)
        if attack in {"isa_er", "isa_tbs"}:
            if isa_final:
                declared_primary = payload.get("primary_layer")
                if declared_primary is None:
                    declared_layers = payload.get("layers")
                    if isinstance(declared_layers, (list, tuple)) and declared_layers:
                        try:
                            declared_primary = f"layer{int(declared_layers[0])}"
                        except (TypeError, ValueError, OverflowError):
                            declared_primary = None
                if (
                    isinstance(declared_primary, str)
                    and declared_primary in isa_final
                ):
                    primary = isa_final[declared_primary][1]
                else:
                    primary = max(value for _, value in isa_final.values())
            elif isa_fallback is not None:
                primary = isa_fallback[1]
        if primary is not None:
            micro_points = [
                point for name, point in named_points
                if "micro" in name.lower()
            ]
            if not micro_points:
                micro_points = [
                    point for name, point in named_points
                    if "top1" in name.lower() and "macro" not in name.lower()
                ]
            if not micro_points:
                gaps.append("ci: no micro/top1 interval for the primary estimate")
            elif not any(abs(point - primary) <= 1e-9 for point in micro_points):
                gaps.append(
                    "ci: no micro/top1 interval reports the primary top1 point estimate "
                    f"{primary:.6f}"
                )
    if attack in CI_REQUIRED and attack not in LINKAGE_ATTACKS and isinstance(ci, Mapping):
        micro_est = [
            block for block in ci.values()
            if isinstance(block, Mapping) and block.get("estimand") == "micro"
        ]
        if not micro_est:
            gaps.append("ci: no micro-estimand interval for the primary rate")
    if attack in TOPK_SHAPE_ATTACKS and isinstance(ci, Mapping):
        micro_blocks = [
            (name, block) for name, block in ci.items()
            if isinstance(block, Mapping) and "micro" in str(name).lower()
        ]
        if not micro_blocks:
            gaps.append("ci: no micro-named cluster bootstrap block")
    if attack in CI_REQUIRED and isinstance(ci, Mapping):
        for name, block in ci.items():
            if not isinstance(block, Mapping):
                continue
            is_micro = (
                "micro" in str(name).lower()
                or str(block.get("estimand") or "").strip().lower() == "micro"
            )
            if not is_micro:
                continue
            if block.get("estimand") != "micro":
                gaps.append(f"ci.{name}: estimand {block.get('estimand')!r} != 'micro'")
            if "lo" not in block or "hi" not in block:
                gaps.append(f"ci.{name}: missing lo/hi bounds")
            try:
                boot = int(block.get("n_boot"))
                if boot < 2000 or boot > 1_000_000:
                    gaps.append(f"ci.{name}: n_boot {boot} outside [2000, 1000000]")
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"ci.{name}: n_boot missing or not numeric")
            try:
                clusters = int(_ci_cluster_count(block))
                if clusters < 2 or clusters > 1_000_000:
                    gaps.append(f"ci.{name}: n_clusters {clusters} outside [2, 1000000]")
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"ci.{name}: n_clusters missing or not numeric")
        for name, block in ci.items():
            if not isinstance(block, Mapping):
                continue
            is_macro = (
                "macro" in str(name).lower()
                or str(block.get("estimand") or "").strip().lower() == "macro"
            )
            if not is_macro:
                continue
            if block.get("estimand") != "macro":
                gaps.append(f"ci.{name}: estimand {block.get('estimand')!r} != 'macro'")
            if "lo" not in block or "hi" not in block:
                gaps.append(f"ci.{name}: missing lo/hi bounds")
            try:
                clusters = int(_ci_cluster_count(block))
                if clusters < 2 or clusters > 1_000_000:
                    gaps.append(
                        f"ci.{name}: n_clusters {clusters} outside [2, 1000000]"
                    )
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"ci.{name}: n_clusters missing or not numeric")
            try:
                boot = int(block.get("n_boot"))
                if boot < 2000:
                    gaps.append(f"ci.{name}: n_boot {boot} < 2000")
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"ci.{name}: n_boot missing or not numeric")
    if attack in LINKAGE_ATTACKS and isinstance(metrics, Mapping):
        exact = metrics.get("exact_test") if isinstance(metrics, Mapping) else None
        exact_points = [
            point for name, point in named_points if "exact_test" in name.lower()
        ]
        if isinstance(exact, Mapping) and exact.get("value") is not None:
            try:
                primary = float(exact["value"])
            except (TypeError, ValueError, OverflowError):
                primary = None
            if primary is not None:
                if not exact_points:
                    gaps.append("ci: no exact_test interval for the linkage point")
                elif not any(abs(point - primary) <= 1e-9 for point in exact_points):
                    gaps.append("ci: exact_test interval contradicts the linkage point")
        exact_ci = ci.get("exact_test") if isinstance(ci, Mapping) else None
        if not isinstance(exact_ci, Mapping):
            gaps.append("ci: linkage exact_test interval missing")
        else:
            if exact_ci.get("estimand") != "session bootstrap":
                gaps.append("ci: linkage interval is not a session bootstrap")
            try:
                clusters = int(exact_ci.get("n_clusters"))
                if clusters < 2 or clusters > 1_000_000:
                    gaps.append(f"ci: linkage n_clusters {clusters} outside [2, 1000000]")
            except (TypeError, ValueError, OverflowError):
                gaps.append("ci: linkage n_clusters missing or not numeric")
    if attack in {"sda", "sda2021", "sda2023"} and isinstance(metrics, Mapping):
        token = _find_key(metrics, "token_accuracy")
        if isinstance(token, Mapping) and token.get("value") is not None:
            try:
                primary = float(token["value"])
            except (TypeError, ValueError, OverflowError):
                primary = None
            token_points = [
                point for name, point in named_points
                if "token_accuracy" in name.lower()
            ]
            if primary is not None:
                if not token_points:
                    gaps.append("ci: no token_accuracy interval for the SDA point")
                elif not any(abs(point - primary) <= 1e-9 for point in token_points):
                    gaps.append("ci: token_accuracy interval contradicts the SDA point")
    return gaps


def _unit_count(attack: str, payload: Mapping[str, Any],
                diag: Mapping[str, Any]):
    if attack in TEXT_ATTACKS:
        return _number(payload, diag, "n_test")
    if attack in {"utility_generation", "utility_classification"}:
        return _number(payload, diag, "n_docs") or _number(payload, diag, "n_test")
    if attack == "efficiency":
        return (
            _number(payload, diag, "scored_requests")
            or _number(payload, diag, "n_requests")
        )
    if attack in {"ima_ridge", "ima_neural"}:
        return _number(payload, diag, "n_test_docs")
    if attack == "output_pii":
        return _number(payload, diag, "n_prompts")
    if attack in {"gate_ia", "attn_ia", "vma_w", "least_squares"}:
        return _number(payload, diag, "n_types")
    if attack in {"sda", "sda2021", "sda2023"}:
        return (
            _number(payload, diag, "n_requests")
            or _number(payload, diag, "n_prompts")
        )
    return None


def _number(payload: Mapping[str, Any], diag: Mapping[str, Any], name: str):
    for source in (payload, diag):
        value = source.get(name)
        if value is not None:
            return value
    return None


def _is_true(value) -> bool:
    """Accept the truthy encodings a JSON writer may emit, reject the rest."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "1", "yes"}


def _diagnostic_flag(value) -> bool:
    """Presence of a diagnostic marker counts as diagnostic unless explicitly false."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (dict, list, tuple)):
        return True
    text = str(value).strip().lower()
    return text not in {"false", "no", "0", "none", "null"}


_HYPHEN_LIKE = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-", "﹣": "-",
    "－": "-",
})


def _normalise_grouping(value) -> str:
    """Upper-case grouping label with dashes/spaces/underscores unified."""
    text = str(value).strip().translate(_HYPHEN_LIKE).upper()
    return text.replace("_", "-").replace(" ", "")


def _normalise_key(key) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def _collect_key_values_containing(obj: Any, needle: str,
                                   *, exclude=()) -> list[Any]:
    """Collect values whose (normalised) key contains ``needle`` at any depth.

    ``exclude`` names container keys (e.g. "diagnostics") that match the needle
    but are not markers themselves; their nested entries are still scanned.
    """
    excluded = {_normalise_key(name) for name in exclude}
    values: list[Any] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            normalised = _normalise_key(key)
            if needle in normalised and normalised not in excluded:
                values.append(value)
            values.extend(_collect_key_values_containing(value, needle, exclude=exclude))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            values.extend(_collect_key_values_containing(value, needle, exclude=exclude))
    return values


def _collect_key_values_normalised(obj: Any, name: str) -> list[Any]:
    """Like :func:`_collect_key_values`, but key spelling/case/space tolerant."""
    target = _normalise_key(name)
    values: list[Any] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if _normalise_key(key) == target:
                values.append(value)
            values.extend(_collect_key_values_normalised(value, name))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            values.extend(_collect_key_values_normalised(value, name))
    return values


def _ci_cluster_count(block: Mapping[str, Any]):
    """Clusters for an interval block; legacy payloads record only n_units."""
    value = block.get("n_clusters")
    if value is None:
        value = block.get("n_units")
    return value


def _control_top1(payload: Mapping[str, Any],
                  diag: Mapping[str, Any]):
    record = diag.get("positive_control")
    if not isinstance(record, Mapping):
        record = payload.get("positive_control")
    if isinstance(record, Mapping):
        return record.get("top1")
    return None


def _floor_gap(gaps: list[str], name: str, got, need) -> None:
    if got is None:
        gaps.append(f"{name}: missing < {need}")
    else:
        try:
            value = float(got)
            if not math.isfinite(value):
                gaps.append(f"{name}: {got!r} is not finite (need >= {need})")
            elif value < float(need):
                gaps.append(f"{name}: {got} < {need}")
        except (TypeError, ValueError, OverflowError):
            gaps.append(f"{name}: {got!r} < {need} (not numeric)")


def _evidence_gaps(payload: Mapping[str, Any], diag: Mapping[str, Any],
                   attack: str = "") -> list[str]:
    gaps: list[str] = []
    predictions = diag.get("per_unit_predictions")
    if not isinstance(predictions, list) or not predictions:
        gaps.append("evidence: per_unit_predictions missing")
    else:
        total = 0
        for index, row in enumerate(predictions):
            if not isinstance(row, Mapping):
                gaps.append(f"evidence: per_unit_predictions[{index}] is not a mapping")
                continue
            hits = row.get("hits")
            count = row.get("count")
            if not isinstance(hits, int) or not isinstance(count, int) or hits < 0 or count < 0:
                gaps.append(f"evidence: per_unit_predictions[{index}] has invalid hits/count")
                continue
            if hits > count:
                gaps.append(f"evidence: per_unit_predictions[{index}] hits {hits} > count {count}")
            if count <= 0:
                gaps.append(
                    f"evidence: per_unit_predictions[{index}] has no counted positions"
                )
            ranks = row.get("rank")
            if ranks is not None:
                if not isinstance(ranks, list) or len(ranks) != count:
                    gaps.append(
                        f"evidence: per_unit_predictions[{index}] rank length != count {count}"
                    )
                else:
                    if any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 1
                        for value in ranks
                    ):
                        gaps.append(
                            f"evidence: per_unit_predictions[{index}] rank has invalid values"
                        )
                    else:
                        recomputed = int(sum(1 for value in ranks if value <= 1))
                        if recomputed != hits:
                            gaps.append(
                                f"evidence: per_unit_predictions[{index}] hits {hits} "
                                f"!= ranks<=1 {recomputed}"
                            )
                        predicted = row.get("predicted")
                        truth = row.get("truth")
                        if isinstance(predicted, list) and isinstance(truth, list):
                            if len(predicted) != count or len(truth) != count:
                                gaps.append(
                                    f"evidence: per_unit_predictions[{index}] predicted/truth "
                                    f"length != count {count}"
                                )
                            else:
                                for position, (pred, true, rank) in enumerate(
                                    zip(predicted, truth, ranks)
                                ):
                                    if (rank == 1) != (pred == true):
                                        gaps.append(
                                            f"evidence: per_unit_predictions[{index}] position "
                                            f"{position} rank {rank} contradicts predicted==truth"
                                        )
                                        break
            total += count
        if attack in LINKAGE_ATTACKS:
            metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
            test = metrics.get("test") if isinstance(metrics, Mapping) else None
            expected_total = test.get("positives") if isinstance(test, Mapping) else None
            expected_rows = expected_total
        else:
            units = _unit_count(attack, payload, diag)
            if units is not None:
                expected_total = (_number(payload, diag, "n_tokens")
                                  if attack in {"sda", "sda2021", "sda2023", "ima_ridge", "ima_neural"}
                                  else units)
                if expected_total is None:
                    gaps.append("evidence: n_tokens missing for token counts")
                expected_rows = units
            else:
                n_tokens = _number(payload, diag, "n_tokens")
                n_types = _number(payload, diag, "n_types")
                expected_total = n_tokens if n_tokens else n_types
                n_prompts = _number(payload, diag, "n_prompts")
                try:
                    keys = int(payload.get("n_keys") or 1)
                except (TypeError, ValueError, OverflowError):
                    keys = 1
                    gaps.append("evidence: declared n_keys is not an integer")
                try:
                    expected_rows = (
                        int(n_prompts) * max(1, keys) if n_prompts else None
                    )
                except (TypeError, ValueError, OverflowError):
                    expected_rows = None
                    gaps.append("evidence: declared n_prompts is not an integer")
                if expected_rows is None and n_types:
                    try:
                        expected_rows = int(n_types)
                    except (TypeError, ValueError, OverflowError):
                        gaps.append("evidence: declared n_types is not an integer")
        try:
            expected_total_int = int(expected_total) if expected_total is not None else None
        except (TypeError, ValueError, OverflowError):
            expected_total_int = None
            gaps.append("evidence: declared size is not an integer")
        if expected_total_int is not None and total != expected_total_int:
            gaps.append(
                f"evidence: per-unit counts sum {total} != declared size {expected_total}"
            )
        try:
            expected_rows_int = int(expected_rows) if expected_rows is not None else None
        except (TypeError, ValueError, OverflowError):
            expected_rows_int = None
            gaps.append("evidence: expected per-unit row count is not an integer")
        if expected_rows_int is not None and len(predictions) != expected_rows_int:
            gaps.append(
                f"evidence: {len(predictions)} per-unit rows != expected "
                f"{expected_rows_int} (n_prompts x n_keys)"
            )
    data_sha = diag.get("data_sha256", payload.get("data_sha256"))
    if not isinstance(data_sha, str) or not _HEX64.match(data_sha):
        gaps.append("evidence: data_sha256 is not a 64-hex content hash")
    source_sha = diag.get("source_sha256", payload.get("source_sha256"))
    if not isinstance(source_sha, str) or not _HEX64.match(source_sha):
        gaps.append("evidence: source_sha256 is not a 64-hex content hash")
    control = diag.get("positive_control", payload.get("positive_control"))
    if not isinstance(control, Mapping) or not control:
        gaps.append("evidence: positive control record missing")
    return gaps


def _fractions(obj: Any, path: str = "metrics") -> list[tuple[str, Mapping]]:
    found: list[tuple[str, Mapping]] = []
    if isinstance(obj, Mapping):
        if "numerator" in obj and "denominator" in obj:
            found.append((path, obj))
        for key, value in obj.items():
            found.extend(_fractions(value, f"{path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_fractions(value, f"{path}[{index}]"))
    return found


def _fraction_gaps(fractions, *, skip_paths=()) -> list[str]:
    gaps: list[str] = []
    for path, block in fractions:
        if path in skip_paths:
            continue
        numerator = block.get("numerator")
        denominator = block.get("denominator")
        if (
            isinstance(numerator, bool)
            or isinstance(denominator, bool)
            or not isinstance(numerator, int)
            or not isinstance(denominator, int)
        ):
            gaps.append(f"{path}: numerator/denominator are not integers")
            continue
        if denominator <= 0 or numerator < 0 or numerator > denominator:
            gaps.append(
                f"{path}: numerator {numerator} / denominator {denominator} "
                "is not a valid fraction"
            )
            continue
        value = block.get("value")
        if value is None:
            gaps.append(f"{path}: point estimate value is missing")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            gaps.append(f"{path}: value is not numeric")
            continue
        try:
            value_f = float(value)
        except (OverflowError, ValueError):
            gaps.append(f"{path}: value is not representable")
            continue
        if not math.isfinite(value_f):
            gaps.append(f"{path}: value is not finite")
            continue
        try:
            expected = numerator / denominator
        except (OverflowError, ZeroDivisionError):
            gaps.append(f"{path}: fraction is not representable")
            continue
        if abs(value_f - expected) > 1e-9:
            gaps.append(f"{path}: value {value} != numerator/denominator {expected}")
    return gaps


def _primary_denominator_gaps(payload: Mapping[str, Any],
                              diag: Mapping[str, Any],
                              attack: str = "") -> list[str]:
    """Reconcile every metric fraction with the declared size and per-unit evidence."""
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    fractions = _fractions(metrics)
    if not fractions:
        # §8.1: every reported point carries numerator and denominator. This is
        # not attack-specific; a payload with no fraction has no auditable point.
        return ["metrics: no numerator/denominator point estimate"]
    if attack in LINKAGE_ATTACKS:
        gaps: list[str] = []
        test = metrics.get("test") if isinstance(metrics, Mapping) else None
        exact = metrics.get("exact_test") if isinstance(metrics, Mapping) else None
        if not isinstance(test, Mapping) or not isinstance(exact, Mapping):
            return ["linkage: test split / exact_test block missing"]
        try:
            allowed_raw = test.get("positives")
            denominator_raw = exact.get("denominator")
            numerator_raw = exact.get("numerator")
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (allowed_raw, denominator_raw, numerator_raw)
            ):
                raise ValueError("non-integer linkage counts")
            allowed = int(allowed_raw)
            denominator = int(denominator_raw)
            numerator = int(numerator_raw)
            if denominator != allowed:
                gaps.append(
                    f"exact_test.denominator: {denominator} != test.positives {allowed}"
                )
            if not 0 <= numerator <= denominator:
                gaps.append("exact_test.numerator: outside [0, denominator]")
            value = exact.get("value")
            if value is None:
                gaps.append("exact_test.value: missing point estimate")
            else:
                try:
                    expected_value = numerator / denominator if denominator else 0.0
                    value_f = float(value)
                    if not math.isfinite(value_f):
                        gaps.append("exact_test.value: not finite")
                    elif abs(value_f - expected_value) > 1e-9:
                        gaps.append(
                            f"exact_test.value {value} != numerator/denominator "
                            f"{expected_value}"
                        )
                except (TypeError, ValueError, OverflowError):
                    gaps.append("exact_test.value: not numeric")
        except (TypeError, ValueError, OverflowError):
            gaps.append("linkage: exact_test numerator/denominator are not integers")
            return gaps
        rows = diag.get("per_unit_predictions")
        if isinstance(rows, list) and rows:
            try:
                total_hits = sum(int(row["hits"]) for row in rows)
                if total_hits != numerator:
                    gaps.append(
                        f"evidence: per-unit hits sum {total_hits} != "
                        f"exact_test.numerator {numerator}"
                    )
            except (KeyError, TypeError, ValueError, OverflowError):
                gaps.append("evidence: linkage per-unit hits are invalid")
        gaps.extend(_fraction_gaps(fractions))
        auc = metrics.get("auc") if isinstance(metrics, Mapping) else None
        try:
            auc_value = float(auc)
            if not math.isfinite(auc_value) or not (0.0 <= auc_value <= 1.0):
                gaps.append(f"metrics.auc: {auc!r} is not a probability")
        except (TypeError, ValueError, OverflowError):
            gaps.append("metrics.auc: missing or not numeric")
        return gaps
    n_tokens = payload.get("n_tokens")
    n_prompts = payload.get("n_prompts")
    n_types = payload.get("n_types")
    if attack in {"sda", "sda2021", "sda2023", "ima_ridge", "ima_neural"}:
        gaps = _fraction_gaps(fractions)
        rows = diag.get("per_unit_predictions")
        declared_tokens = _number(payload, diag, "n_tokens")
        if not isinstance(rows, list) or not rows:
            return gaps + ["evidence: token-level per_unit_predictions missing"]
        try:
            count = sum(int(row["count"]) for row in rows)
            hits = sum(int(row["hits"]) for row in rows)
            if declared_tokens is None or count != int(declared_tokens):
                gaps.append(f"evidence: token count {count} != n_tokens {declared_tokens}")
        except (KeyError, TypeError, ValueError, OverflowError):
            return gaps + ["evidence: invalid token counts"]
        token_points = [(path, block) for path, block in fractions
                        if re.search(r"(?:token_accuracy|token_recovery.*top1|top1)$", path)]
        if not token_points:
            gaps.append("metrics: token-level primary fraction missing")
        for path, block in token_points:
            if block.get("denominator") != count or block.get("numerator") != hits:
                gaps.append(f"{path}: token numerator/denominator differs from per-unit evidence")
        return gaps
    allowed = _unit_count(attack, payload, diag) or n_tokens or n_prompts or n_types
    gaps: list[str] = _fraction_gaps(fractions)
    try:
        allowed_int = int(allowed) if allowed is not None else None
    except (TypeError, ValueError, OverflowError):
        allowed_int = None
        gaps.append(f"declared size {allowed!r} is not an integer")
    if allowed_int is not None and allowed_int <= 0:
        gaps.append(f"declared size {allowed_int} is not positive")
    if allowed_int is not None:
        for path, block in fractions:
            denominator = block.get("denominator")
            if isinstance(denominator, int) and denominator > allowed_int:
                gaps.append(
                    f"{path}.denominator {denominator} exceeds declared size {allowed_int}"
                )
    if allowed_int is None:
        gaps.append("declared size: no n_tokens/n_prompts/n_types/unit count")
        return gaps
    allowed = allowed_int
    for path, block in fractions:
        if not re.search(r"top\d+$", path):
            continue
        denominator = block.get("denominator")
        if isinstance(denominator, int) and denominator != allowed:
            gaps.append(
                f"{path}.denominator: {denominator} != declared unit count {allowed}"
            )
    def _int_denominator(item):
        try:
            return int(item[1]["denominator"])
        except (TypeError, ValueError, KeyError, OverflowError):
            return -1

    primary = max(
        ((path, block) for path, block in fractions
         if block.get("denominator") is not None),
        key=_int_denominator,
        default=None,
    )
    if primary is not None and _int_denominator(primary) != int(allowed):
        gaps.append(
            f"{primary[0]}.denominator: {primary[1]['denominator']} != declared size {allowed}"
        )
    predictions = diag.get("per_unit_predictions")
    if isinstance(predictions, list) and predictions:
        total_hits = 0
        valid = True
        for row in predictions:
            if not isinstance(row, Mapping) or not isinstance(row.get("hits"), int):
                valid = False
                break
            total_hits += int(row["hits"])
        if valid:
            numerators: set[int] = set()
            for _path, block in fractions:
                try:
                    if (
                        block.get("denominator") is not None
                        and int(block["denominator"]) == int(allowed)
                        and isinstance(block.get("numerator"), (int, float))
                    ):
                        numerators.add(int(block["numerator"]))
                except (TypeError, ValueError, KeyError, OverflowError, ZeroDivisionError):
                    continue
            if numerators and total_hits not in numerators:
                gaps.append(
                    f"evidence: per-unit hits sum {total_hits} matches no primary numerator "
                    f"{sorted(numerators)}"
                )
    return gaps


def _declared_size_gaps(payload: Mapping[str, Any],
                        diag: Mapping[str, Any]) -> list[str]:
    gaps: list[str] = []
    for field in (
        "n_prompts", "n_tokens", "n_types", "n_keys", "top_k",
        "keys_per_prompt", "n_observations", "n_seeds", "n_epochs",
        "n_train_tokens", "n_train_docs", "n_test_docs", "executed_steps",
        "restarts", "reference_tokens", "top100_distinct",
        "n_streams", "n_train_sequences", "n_val_sequences", "n_test",
        "n_docs", "n_requests", "scored_requests", "rounds",
        "output_length", "max_new_tokens",
    ):
        value = _number(payload, diag, field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            gaps.append(f"{field}: {value!r} is not an integer")
    return gaps


def _paired_saturation_gap(saturation: Mapping[str, Any], points: list, attack: str) -> str | None:
    """Recompute the validation-set gain and its paired cluster bootstrap interval."""
    import numpy as np

    final = points[-2:]
    try:
        budgets = [float(point["budget"]) for point in final]
        if not all(math.isfinite(value) and value > 0 for value in budgets):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        return "saturation-record: final budgets must be finite positive numbers"
    documented_isa = attack in {"isa_er", "isa_tbs"} and tuple(budgets) in {
        (1000, 10000), (10000, 25000), (25000, 50000),
    }
    if budgets[1] <= budgets[0] or (budgets[1] < 2 * budgets[0] and not documented_isa):
        return "saturation-record: final budgets must increase by at least twofold or use the ISA protocol checkpoints"
    for field in ("validation_data_sha256", "model_sha256"):
        values = [point.get(field) if isinstance(point, Mapping) else None for point in final]
        if (not all(isinstance(value, str) and _HEX64.fullmatch(value) for value in values)
                or values[0] != values[1]):
            return f"saturation-record: {field} must identify the same frozen inputs at both budgets"
    if any(point.get("split") != "validation" for point in final):
        return "saturation-record: both points must use the frozen validation split"
    rows = saturation.get("paired_units")
    if not isinstance(rows, list) or len(rows) < 2:
        return "saturation-record: paired validation-unit evidence missing"
    clusters: dict[str, list[int]] = {}
    units = set()
    try:
        for row in rows:
            unit, cluster = row["unit"], row["cluster"]
            if not isinstance(unit, str) or not unit or unit in units or not isinstance(cluster, str) or not cluster:
                raise ValueError
            units.add(unit)
            count, before, after = row["count"], row["hits_before"], row["hits_after"]
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (count, before, after)):
                raise ValueError
            if count <= 0 or not (0 <= before <= count and 0 <= after <= count):
                raise ValueError
            totals = clusters.setdefault(cluster, [0, 0, 0])
            totals[0] += count
            totals[1] += before
            totals[2] += after
    except (KeyError, TypeError, ValueError):
        return "saturation-record: invalid or duplicate paired validation units"
    if len(clusters) < 2:
        return "saturation-record: at least two independent validation clusters required"
    totals = np.asarray(list(clusters.values()), dtype=np.float64)
    count, before, after = totals.sum(axis=0)
    try:
        if abs(before / count - float(final[0]["top1"])) > 1e-9 or abs(after / count - float(final[1]["top1"])) > 1e-9:
            return "saturation-record: points contradict paired validation units"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "saturation-record: malformed validation points"
    rng = np.random.default_rng(0)
    gains = np.empty(2000, dtype=np.float64)
    for index in range(len(gains)):
        sampled = totals[rng.integers(0, len(totals), size=len(totals))].sum(axis=0)
        gains[index] = 100 * (sampled[2] - sampled[1]) / sampled[0]
    upper = float(np.quantile(gains, 0.975))
    if upper > 2.0:
        return f"saturation: paired validation gain CI upper bound {upper:.4g}pp > 2pp"
    return None


def _saturation_gap(diag: Mapping[str, Any], metrics: Mapping[str, Any] | None = None,
                    attack: str = "", payload: Mapping[str, Any] | None = None) -> str | None:
    saturation = diag.get("saturation")
    if not isinstance(saturation, Mapping):
        return "saturation: missing (last two budget points required)"
    gain = saturation.get("gain_pp")
    points = saturation.get("points")
    if points is None:
        return "saturation: points missing (the last two measured budget points are required)"
    point_gain = None
    if points is not None:
        if not isinstance(points, list) or len(points) < 2:
            return "saturation: points must contain the last two budget points"
        if attack in {"tfma", "tfma_after_inversion"}:
            allowed_budgets = {4_450_000, 10_000_000, 20_000_000}
            budgets: list[int] = []
            for point in points:
                if not isinstance(point, Mapping):
                    return "saturation: points are malformed"
                raw_budget = point.get("budget")
                if isinstance(raw_budget, bool) or not isinstance(raw_budget, (int, float)):
                    return "saturation-record: TFMA point budget is not numeric"
                try:
                    if float(raw_budget) != int(raw_budget):
                        return "saturation-record: TFMA point budget is not an integer"
                    budget = int(raw_budget)
                except (TypeError, ValueError, OverflowError):
                    return "saturation-record: TFMA point budget is not representable"
                if budget not in allowed_budgets:
                    return (
                        f"saturation-record: TFMA budget {budget} is not one of the "
                        f"protocol reference-token budgets {sorted(allowed_budgets)}"
                    )
                try:
                    value = float(point.get("top1"))
                except (TypeError, ValueError, OverflowError):
                    return "saturation: points are malformed"
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    return "saturation: TFMA point top1 is not a probability"
                budgets.append(budget)
            if len(set(budgets)) != len(budgets):
                return "saturation-record: TFMA budget duplicated"
            if budgets != sorted(budgets):
                return "saturation-record: TFMA budgets are not ascending"
            ladder = sorted(allowed_budgets)
            if budgets != ladder[: len(budgets)]:
                return (
                    f"saturation-record: TFMA budgets {budgets} must be a contiguous "
                    f"prefix of the protocol ladder {ladder}"
                )
            reference = None
            for source in (payload, diag):
                if isinstance(source, Mapping) and source.get("reference_tokens") is not None:
                    try:
                        reference = int(source["reference_tokens"])
                    except (TypeError, ValueError, OverflowError):
                        reference = None
                    break
            if reference is None:
                return (
                    "saturation-record: reference_tokens missing; TFMA budgets cannot "
                    "be bound to the measured corpus"
                )
            if budgets[-1] != reference:
                return (
                    f"saturation-record: last TFMA budget {budgets[-1]} != "
                    f"reference_tokens {reference}"
                )
            # Each declared point must be a measured recovery value: earlier
            # budgets come from the recorded saturation curve, the final budget
            # must equal the run's own occurrence-level top1.
            curve_map = None
            if isinstance(diag, Mapping) and isinstance(
                diag.get("saturation_curve"), Mapping
            ):
                curve_map = diag["saturation_curve"]
            metrics_top1 = None
            if isinstance(metrics, Mapping):
                occurrence = metrics.get("occurrence_level")
                if isinstance(occurrence, Mapping):
                    top1_block = occurrence.get("top1")
                    if isinstance(top1_block, Mapping):
                        value = top1_block.get("value")
                    else:
                        value = top1_block
                    try:
                        metrics_top1 = float(value)
                    except (TypeError, ValueError, OverflowError):
                        metrics_top1 = None
            for point in points:
                budget = int(point["budget"])
                point_top1 = float(point["top1"])
                if budget == reference:
                    if metrics_top1 is None:
                        return (
                            "saturation-record: final TFMA point cannot be bound to "
                            "the run's occurrence-level top1"
                        )
                    if abs(point_top1 - metrics_top1) > 1e-9:
                        return (
                            f"saturation-record: final TFMA point {point_top1} != "
                            f"measured top1 {metrics_top1}"
                        )
                    continue
                if not isinstance(curve_map, Mapping):
                    return (
                        f"saturation-record: no measured saturation_curve for "
                        f"budget {budget}"
                    )
                measured_value = curve_map.get(budget)
                if measured_value is None:
                    measured_value = curve_map.get(str(budget))
                try:
                    measured_f = float(measured_value)
                except (TypeError, ValueError, OverflowError):
                    return (
                        f"saturation-record: no measured saturation_curve value "
                        f"for budget {budget}"
                    )
                if abs(point_top1 - measured_f) > 1e-9:
                    return (
                        f"saturation-record: point at budget {budget} contradicts "
                        "the measured saturation_curve"
                    )
        if attack in {"isa_er", "isa_tbs"}:
            curve = (
                metrics.get("budget_curve") if isinstance(metrics, Mapping) else None
            )
            if not isinstance(curve, Mapping):
                return "saturation-record: budget_curve missing; points cannot be bound"
            measured: dict[int, set[float]] = {}
            primary = None
            if isinstance(payload, Mapping):
                primary = payload.get("primary_layer")
                if primary is None:
                    layers = payload.get("layers")
                    if isinstance(layers, (list, tuple)) and layers:
                        try:
                            primary = f"layer{int(layers[0])}"
                        except (TypeError, ValueError, OverflowError):
                            primary = None
            if isinstance(primary, str) and primary in curve:
                layer_blocks = [curve[primary]]
            else:
                layer_blocks = list(curve.values())
            for layer_block in layer_blocks:
                if not isinstance(layer_block, Mapping):
                    continue
                for key, block in layer_block.items():
                    match = re.fullmatch(r"steps(\d+)", str(key))
                    if not match or not isinstance(block, Mapping):
                        continue
                    top1 = block.get("top1")
                    value = top1.get("value") if isinstance(top1, Mapping) else top1
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if math.isfinite(value_f):
                        measured.setdefault(int(match.group(1)), set()).add(value_f)
            for point in points:
                if not isinstance(point, Mapping):
                    return "saturation: points are malformed"
                try:
                    budget = int(point.get("budget"))
                except (TypeError, ValueError, OverflowError):
                    return "saturation: point budget is missing or not an integer"
                if budget not in measured:
                    return (
                        f"saturation-record: declared point budget {budget} is not a "
                        "measured budget_curve checkpoint"
                    )
                try:
                    point_top1 = float(point.get("top1"))
                except (TypeError, ValueError, OverflowError):
                    return "saturation: points are malformed"
                if not math.isfinite(point_top1):
                    return "saturation: points contain a non-finite top1"
                if not any(abs(point_top1 - value) <= 1e-9 for value in measured[budget]):
                    return (
                        f"saturation-record: point at budget {budget} contradicts the "
                        "measured budget_curve"
                    )
            final_two = sorted(measured)[-2:]
            if sorted({int(p["budget"]) for p in points}) != final_two:
                return (
                    "saturation-record: ISA points must be the final two measured "
                    f"checkpoints {final_two}, got "
                    f"{sorted({int(p['budget']) for p in points})}"
                )
        try:
            first = float(points[-2]["top1"])
            last = float(points[-1]["top1"])
            if not (math.isfinite(first) and math.isfinite(last)):
                return "saturation: points contain a non-finite top1"
            point_gain = (last - first) * 100.0
        except (KeyError, TypeError, ValueError, OverflowError):
            return "saturation: points are malformed"
        if gain is None:
            gain = point_gain
        else:
            try:
                if abs(float(gain) - point_gain) > 1e-9:
                    return "saturation: gain_pp does not match the points"
            except (TypeError, ValueError, OverflowError):
                return f"saturation: gain_pp {gain!r} is not numeric"
    if gain is None:
        return "saturation: gain_pp missing"
    try:
        gain_value = float(gain)
        if not math.isfinite(gain_value):
            return f"saturation: gain_pp {gain!r} is not finite"
        if gain_value > 2.0:
            return f"saturation: last-interval gain {gain}pp > 2pp"
    except (TypeError, ValueError, OverflowError):
        return f"saturation: gain_pp {gain!r} is not numeric"
    return _paired_saturation_gap(saturation, points, attack)


def _stream_floor_gaps(gaps: list[str], payload: Mapping[str, Any],
                       diag: Mapping[str, Any]) -> None:
    """§8.2 shared stream floor: 3 streams x 5,000 requests, >=100k tokens each."""
    _floor_gap(gaps, "n_streams", _number(payload, diag, "n_streams"), 3)
    requests = (
        _number(payload, diag, "n_prompts")
        or _number(payload, diag, "n_requests")
    )
    _floor_gap(gaps, "stream requests", requests, 15000)
    _floor_gap(gaps, "stream tokens", _number(payload, diag, "n_tokens"), 300000)
    _floor_gap(gaps, "stream types", _number(payload, diag, "n_types"), 1000)


def _key_floor_gaps(gaps: list[str], payload: Mapping[str, Any],
                    diag: Mapping[str, Any]) -> None:
    """Enforce the 200-prompts x 5-fresh-keys floor on the observation count.

    ``n_keys`` alone cannot establish it: an earlier wiring recorded one fresh key
    per prompt (200 keys) while looking like ">= 5 keys". The payload must state
    ``keys_per_prompt`` and the total number of observed wires.
    """
    n_prompts = _number(payload, diag, "n_prompts")
    _floor_gap(gaps, "keys_per_prompt", _number(payload, diag, "keys_per_prompt"), 5)
    try:
        need = int(float(n_prompts)) * 5
    except (TypeError, ValueError, OverflowError):
        need = 1000
    n_obs = _number(payload, diag, "n_observations")
    if n_obs is None:
        gaps.append("n_observations: missing (prompts x 5 fresh keys)")
    else:
        try:
            value = float(n_obs)
            if not math.isfinite(value):
                gaps.append(f"n_observations: {n_obs!r} is not finite")
            elif value < need:
                gaps.append(
                    f"n_observations: {n_obs} < {need} (prompts x 5 fresh keys)"
                )
        except (TypeError, ValueError, OverflowError):
            gaps.append(f"n_observations: {n_obs!r} is not numeric")


def _strata_gaps(payload: Mapping[str, Any],
                 diag: Mapping[str, Any]) -> list[str]:
    """ISA length strata: 50 prompts in each token-length band (§8.1)."""
    floor = {"32-64": 50, "65-128": 50, "129-256": 50, "257-512": 50}
    block = _number(payload, diag, "length_strata")
    if not isinstance(block, Mapping):
        return ["length_strata: missing (50 prompts per token-length band)"]
    observed = block.get("observed") if isinstance(block.get("observed"), Mapping) else block
    gaps: list[str] = []
    for name, need in floor.items():
        got = observed.get(name)
        try:
            if int(got) < need:
                gaps.append(f"length_strata.{name}: {got} < {need}")
        except (TypeError, ValueError, OverflowError):
            gaps.append(f"length_strata.{name}: {got!r} is not numeric")
    return gaps


def _isa_validation_gaps(payload: Mapping[str, Any],
                         diag: Mapping[str, Any]) -> list[str]:
    """ISA hyper-parameters must be selected on the independent 40-prompt set."""
    block = _number(payload, diag, "selected_on_validation")
    if not isinstance(block, Mapping):
        return ["selected_on_validation: missing (40 prompts, 10 per length band)"]
    gaps: list[str] = []
    try:
        if int(block.get("n_prompts")) < 40:
            gaps.append(
                f"selected_on_validation.n_prompts: {block.get('n_prompts')} < 40"
            )
    except (TypeError, ValueError, OverflowError):
        gaps.append("selected_on_validation.n_prompts: missing or not numeric")
    strata = block.get("strata")
    if not isinstance(strata, Mapping):
        gaps.append("selected_on_validation.strata: missing (10 per band)")
    else:
        for name in ("32-64", "65-128", "129-256", "257-512"):
            try:
                if int(strata.get(name)) < 10:
                    gaps.append(f"selected_on_validation.strata.{name}: {strata.get(name)} < 10")
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"selected_on_validation.strata.{name}: missing or not numeric")
    lr = block.get("lr")
    try:
        lr_ok = math.isfinite(float(lr)) and float(lr) in {1e-4, 5e-4, 1e-3, 1e-2}
    except (TypeError, ValueError, OverflowError):
        lr_ok = False
    if not lr_ok:
        gaps.append(f"selected_on_validation.lr: {lr!r} not in the fixed grid")
    lam = block.get("lambda_dm")
    try:
        lam_ok = math.isfinite(float(lam)) and float(lam) in {0.0, 1e-3}
    except (TypeError, ValueError, OverflowError):
        lam_ok = False
    if not lam_ok:
        gaps.append(f"selected_on_validation.lambda_dm: {lam!r} not in the fixed grid")
    digest = block.get("data_sha256")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        gaps.append("selected_on_validation.data_sha256: validation content hash missing")
    if block.get("split") != "validation" or block.get("metric") != "token_top1":
        gaps.append("selected_on_validation: expected validation split and token_top1 metric")
    trials = block.get("trials")
    trial_scores: dict[tuple[float, float, str], float] = {}
    try:
        if not isinstance(trials, list):
            raise ValueError
        for trial in trials:
            rate, penalty = float(trial["lr"]), float(trial["lambda_dm"])
            initial = trial["initialization"]
            score = float(trial["token_top1"])
            key = (rate, penalty, initial)
            if (rate not in {1e-4, 5e-4, 1e-3, 1e-2} or penalty not in {0.0, 1e-3}
                    or not isinstance(initial, str) or not initial or key in trial_scores
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError
            trial_scores[key] = score
        if {(a, b) for a, b, _ in trial_scores} != {
            (a, b) for a in (1e-4, 5e-4, 1e-3, 1e-2) for b in (0., 1e-3)
        }:
            raise ValueError
        selected_key = (float(lr), float(lam), block.get("initialization"))
        if selected_key not in trial_scores or trial_scores[selected_key] != max(trial_scores.values()):
            gaps.append("selected_on_validation: selected configuration is not a best validation trial")
    except (KeyError, TypeError, ValueError, OverflowError):
        gaps.append("selected_on_validation.trials: incomplete or invalid validation grid")
    metrics = payload.get("metrics")
    executed_lr = metrics.get("lr") if isinstance(metrics, Mapping) else None
    executed_lam = metrics.get("lambda_dm") if isinstance(metrics, Mapping) else None
    if executed_lr is None:
        executed_lr = _number(payload, diag, "lr")
    if executed_lam is None:
        executed_lam = _number(payload, diag, "lambda_dm")
    try:
        if executed_lr is not None and float(executed_lr) != float(lr):
            gaps.append(
                f"selected_on_validation.lr {lr!r} != executed lr {executed_lr!r}"
            )
    except (TypeError, ValueError, OverflowError):
        gaps.append("executed lr: not numeric")
    try:
        if executed_lam is not None and float(executed_lam) != float(lam):
            gaps.append(
                "selected_on_validation.lambda_dm "
                f"{lam!r} != executed {executed_lam!r}"
            )
    except (TypeError, ValueError, OverflowError):
        gaps.append("executed lambda_dm: not numeric")
    layers = payload.get("layers") or diag.get("layers") or []
    layer_ok = isinstance(layers, (list, tuple)) and bool(layers)
    lr_by_layer = metrics.get("lr_by_layer") if isinstance(metrics, Mapping) else None
    try:
        if isinstance(lr_by_layer, Mapping) and layer_ok:
            for layer in layers:
                layer_i = int(layer)
                value = lr_by_layer.get(str(layer_i))
                if value is None:
                    value = lr_by_layer.get(layer_i)
                if value is None or float(value) != float(lr):
                    gaps.append(
                        f"lr_by_layer.{layer_i}: {value!r} != selected validation lr {lr!r}"
                    )
    except (TypeError, ValueError, OverflowError):
        gaps.append("lr_by_layer: not numeric")
    return gaps


def _ridge_selection_gaps(payload: Mapping[str, Any], diag: Mapping[str, Any]) -> list[str]:
    """Audit the validation-selected ridge regularization without requiring epochs."""
    record = _number(payload, diag, "ridge_validation")
    if not isinstance(record, Mapping):
        return ["ridge_validation: missing lambda selection evidence"]
    gaps: list[str] = []
    _floor_gap(gaps, "ridge_validation.n_docs", record.get("n_docs"), 1024)
    digest = record.get("data_sha256")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        gaps.append("ridge_validation.data_sha256: missing validation data content hash")
    if record.get("split") != "validation" or record.get("metric") != "token_top1":
        gaps.append("ridge_validation: expected validation split and token_top1 metric")
    trials = record.get("trials")
    scores: dict[float, float] = {}
    try:
        if not isinstance(trials, list):
            raise ValueError
        for row in trials:
            lam, score = float(row["lambda"]), float(row["token_top1"])
            if not math.isfinite(lam) or not math.isfinite(score) or not 0 <= score <= 1 or lam in scores:
                raise ValueError
            scores[lam] = score
        if set(scores) != {0.01, 0.1, 1.0, 10.0}:
            raise ValueError
    except (TypeError, KeyError, ValueError, OverflowError):
        gaps.append("ridge_validation.trials: expected one finite score for each protocol lambda")
        return gaps
    try:
        selected = float(record["selected_lambda"])
        executed = float(_number(payload, diag, "ridge_lambda"))
        if selected not in scores or scores[selected] != max(scores.values()):
            gaps.append("ridge_validation.selected_lambda: not a best validation score")
        if selected != executed:
            gaps.append("ridge_lambda: executed value differs from validation selection")
    except (TypeError, KeyError, ValueError, OverflowError):
        gaps.append("ridge_validation: selected or executed lambda missing")
    return gaps


def _attack_floors(
    attack: str,
    payload: Mapping[str, Any],
    diag: Mapping[str, Any],
) -> list[str]:
    gaps: list[str] = []
    if attack in {"knn_l2", "knn_cosine", "ednn"}:
        _floor_gap(gaps, "n_prompts", _number(payload, diag, "n_prompts"), 1000)
        _floor_gap(gaps, "top_k", _number(payload, diag, "top_k"), 100)
        _floor_gap(gaps, "positive_control.top1", _control_top1(payload, diag), 0.99)
    elif attack == "knn_l2_key_robustness":
        _floor_gap(gaps, "n_prompts", _number(payload, diag, "n_prompts"), 200)
        _key_floor_gaps(gaps, payload, diag)
        _floor_gap(gaps, "top_k", _number(payload, diag, "top_k"), 100)
    elif attack in {"vma_h_recovered_prefix", "vma_h_observed_prefix"}:
        _floor_gap(gaps, "n_prompts", _number(payload, diag, "n_prompts"), 200)
        _key_floor_gaps(gaps, payload, diag)
        layers = payload.get("layers") or diag.get("layers") or []
        try:
            n_layers = len(layers)
        except TypeError:
            gaps.append("layers: not a list of layer ids")
            n_layers = 0
        if n_layers < 4:
            gaps.append(f"layers: {n_layers} < 4")
        gaps.extend(_layer_gaps(payload, diag, attack=attack))
        expected_prefix = (
            "recovered" if attack == "vma_h_recovered_prefix" else "observed"
        )
        prefix = payload.get("prefix") or diag.get("prefix")
        if prefix != expected_prefix:
            gaps.append(f"prefix: {prefix!r} != {expected_prefix!r}")
    elif attack in {"isa_er", "isa_tbs"}:
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        if metrics.get("trajectory_mode") != "continuous_checkpoints":
            gaps.append("trajectory_mode: continuous checkpoints are not evidenced")
        _floor_gap(gaps, "n_prompts", _number(payload, diag, "n_prompts"), 200)
        _key_floor_gaps(gaps, payload, diag)
        gaps.extend(_strata_gaps(payload, diag))
        gaps.extend(_isa_validation_gaps(payload, diag))
        steps = payload.get("executed_steps")
        if steps is None:
            steps = diag.get("executed_steps")
        if steps is None:
            saturation_record = diag.get("saturation")
            steps = (
                saturation_record.get("max_budget")
                if isinstance(saturation_record, Mapping) else None
            )
        _floor_gap(gaps, "steps", steps, 50000)
        _floor_gap(gaps, "restarts", _number(payload, diag, "restarts"), 3)
        actual_steps = _number(payload, diag, "optimizer_steps_total")
        if actual_steps is None:
            gaps.append("optimizer_steps_total: missing actual compute record")
        else:
            try:
                if int(actual_steps) <= 0:
                    gaps.append("optimizer_steps_total: 0 actual optimizer steps")
                elif steps is not None and int(actual_steps) < int(steps):
                    gaps.append(
                        "optimizer_steps_total: less than one full checkpoint"
                    )
            except (TypeError, ValueError, OverflowError):
                gaps.append("optimizer_steps_total: not numeric")
        failed = _number(payload, diag, "failed_observations")
        observations = _number(payload, diag, "n_observations")
        if failed is not None and observations is not None:
            try:
                if int(observations) > 0 and int(failed) >= int(observations):
                    gaps.append("failed_observations: every observation failed")
            except (TypeError, ValueError, OverflowError):
                gaps.append("failed_observations: not numeric")
        # Continuous trajectories execute only the largest checkpoint per restart.
        # The lower bound assumes
        # one successful layer per surviving wire; the upper bound assumes all.
        try:
            metrics = payload.get("metrics")
            curve = metrics.get("budget_curve") if isinstance(metrics, Mapping) else None
            layer_blocks = list(curve.values()) if isinstance(curve, Mapping) else []
            checkpoint_set: set[int] = set()
            for layer_block in layer_blocks:
                if not isinstance(layer_block, Mapping):
                    continue
                for key in layer_block:
                    match = re.fullmatch(r"steps(\d+)", str(key))
                    if match:
                        checkpoint_set.add(int(match.group(1)))
            layers_list = payload.get("layers") or diag.get("layers") or []
            n_layers_used = len(layers_list) if isinstance(layers_list, (list, tuple)) else 0
            restarts_n = int(_number(payload, diag, "restarts"))
            obs_n = int(_number(payload, diag, "n_observations"))
            failed_n = int(failed) if failed is not None else 0
            if checkpoint_set and n_layers_used and obs_n > 0 and actual_steps is not None:
                per_schedule = max(checkpoint_set) * restarts_n
                lower = per_schedule * max(1, obs_n - failed_n)
                upper = per_schedule * n_layers_used * obs_n
                if not (lower <= int(actual_steps) <= upper):
                    gaps.append(
                        "optimizer_steps_total: "
                        f"{actual_steps} outside the checkpoint identity "
                        f"[{lower}, {upper}] (max {sorted(checkpoint_set)} x "
                        f"{restarts_n} restarts x layers/wires)"
                    )
        except (TypeError, ValueError, OverflowError):
            gaps.append("optimizer_steps_total: identity inputs not numeric")
        layers = payload.get("layers") or diag.get("layers") or []
        try:
            n_layers = len(layers)
        except TypeError:
            gaps.append("layers: not a list of layer ids")
            n_layers = 0
        if n_layers < 4:
            gaps.append(f"layers: {n_layers} < 4")
        gaps.extend(_layer_gaps(payload, diag, attack=attack))
        metrics = payload.get("metrics")
        curve = metrics.get("budget_curve") if isinstance(metrics, Mapping) else None
        required_steps = {1000, 10000, 25000, 50000}
        if not isinstance(curve, Mapping) or not curve:
            gaps.append("budget_curve: missing the 1k/10k/25k/50k checkpoint ladder")
        else:
            for layer_name, layer_block in curve.items():
                steps: set[int] = set()
                if isinstance(layer_block, Mapping):
                    for key in layer_block:
                        match = re.fullmatch(r"steps(\d+)", str(key))
                        if match:
                            steps.add(int(match.group(1)))
                missing = sorted(required_steps - steps)
                if missing:
                    gaps.append(
                        f"budget_curve.{layer_name}: missing checkpoints {missing} "
                        "(protocol §8 requires 1k/10k/25k/50k)"
                    )
    elif attack in {"vma_w", "gate_ia", "attn_ia"}:
        _floor_gap(gaps, "n_types", _number(payload, diag, "n_types"), 1000)
        layers = payload.get("layers") or diag.get("layers") or []
        if not layers:
            gaps.append("layers: missing (all applicable weight layers)")
        elif attack in {"gate_ia", "attn_ia", "vma_w"}:
            gaps.extend(_layer_gaps(payload, diag, attack=attack))
    elif attack == "least_squares":
        _floor_gap(gaps, "n_types", _number(payload, diag, "n_types"), 1000)
        gaps.extend(_layer_gaps(payload, diag, attack=attack))
        gaps.extend(_least_squares_group_gaps(payload, diag))
    elif attack in {"ima_ridge", "ima_neural"}:
        _floor_gap(gaps, "n_train_tokens", _number(payload, diag, "n_train_tokens"), 1_000_000)
        _floor_gap(gaps, "n_train_docs", _number(payload, diag, "n_train_docs"), 8192)
        _floor_gap(gaps, "n_test_docs", _number(payload, diag, "n_test_docs"), 1000)
        _floor_gap(gaps, "n_seeds", _number(payload, diag, "n_seeds"), 3)
        if attack == "ima_neural":
            _floor_gap(gaps, "n_epochs", _number(payload, diag, "n_epochs"), 10)
        else:
            gaps.extend(_ridge_selection_gaps(payload, diag))
        seeds = payload.get("training_seeds")
        if seeds is None:
            seeds = diag.get("training_seeds")
        try:
            got_seeds = {int(seed) for seed in seeds or []}
        except (TypeError, ValueError, OverflowError):
            got_seeds = set()
        if got_seeds != set(REQUIRED_TRAINING_SEEDS):
            gaps.append(
                "training_seeds: "
                f"{sorted(got_seeds)} != required {sorted(REQUIRED_TRAINING_SEEDS)}"
            )
    elif attack in LINKAGE_ATTACKS:
        metrics = payload.get("metrics") or {}
        test = metrics.get("test") if isinstance(metrics, Mapping) else None
        if not isinstance(test, Mapping):
            test = diag.get("test")
        if not isinstance(test, Mapping):
            gaps.append("test split: missing per-split positive/negative counts")
        else:
            _floor_gap(gaps, "test.positives", test.get("positives"), 10000)
            _floor_gap(gaps, "test.negatives", test.get("negatives"), 100000)
            exact = (
                metrics.get("exact_test") if isinstance(metrics, Mapping) else None
            ) or diag.get("exact_test")
            if isinstance(exact, Mapping):
                try:
                    if int(exact.get("denominator")) != int(test.get("positives")):
                        gaps.append(
                            "exact_test.denominator != test.positives "
                            "(scored test set differs from the recorded budget)"
                        )
                except (TypeError, ValueError, OverflowError):
                    gaps.append("exact_test.denominator: missing or not numeric")
        _floor_gap(gaps, "n_streams", _number(payload, diag, "n_streams"), 3)
    elif attack in {"tfma", "tfma_after_inversion"}:
        _floor_gap(gaps, "reference_tokens", _number(payload, diag, "reference_tokens"), 4_450_000)
        _floor_gap(gaps, "reference_types", _number(payload, diag, "reference_types"), 50000)
        for field in ("n_prompts", "n_tokens", "n_streams", "n_requests"):
            declared_value = payload.get(field)
            measured_value = diag.get(field)
            if declared_value is None or measured_value is None:
                continue
            try:
                if int(declared_value) != int(measured_value):
                    gaps.append(
                        f"tfma {field}: declared {declared_value!r} != "
                        f"measured {measured_value!r}"
                    )
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"tfma {field}: non-integer record")
        declared_types = payload.get("reference_types")
        measured_types = diag.get("reference_types")
        if measured_types is None:
            gaps.append(
                "tfma reference_types: measured value missing from diagnostics"
            )
        if declared_types is not None and measured_types is not None:
            try:
                if int(declared_types) != int(measured_types):
                    gaps.append(
                        "tfma reference_types: declared "
                        f"{declared_types!r} != measured {measured_types!r}"
                    )
            except (TypeError, ValueError, OverflowError):
                gaps.append("tfma reference_types: non-integer record")
        _floor_gap(gaps, "top100_distinct", _number(payload, diag, "top100_distinct"), 100)
        reference_tokens = _number(payload, diag, "reference_tokens")
        observed_tokens = _number(payload, diag, "n_tokens")
        n_streams = _number(payload, diag, "n_streams")
        try:
            if reference_tokens is not None and observed_tokens is not None:
                if n_streams is None or int(n_streams) < 1:
                    gaps.append(
                        "tfma ratio: n_streams missing; the per-stream observed "
                        "share cannot be computed"
                    )
                else:
                    per_stream = int(observed_tokens) / int(n_streams)
                    if per_stream * 100 > int(reference_tokens) + 1e-9:
                        gaps.append(
                            f"tfma ratio: observed per stream {per_stream} x100 > "
                            f"reference {reference_tokens} (protocol requires "
                            "observed <= reference/100 per stream)"
                        )
        except (TypeError, ValueError, OverflowError):
            gaps.append("tfma ratio: non-integer token/stream counts")
        _floor_gap(gaps, "positive_control.top1", _control_top1(payload, diag), 0.90)
        _stream_floor_gaps(gaps, payload, diag)
        per_stream = diag.get("per_stream") if isinstance(diag, Mapping) else None
        if not isinstance(per_stream, list) or not per_stream:
            gaps.append(
                "tfma streams: per-stream evidence missing (3 streams x 5000 "
                "requests and >=100k tokens each)"
            )
        else:
            if n_streams is not None:
                try:
                    if int(n_streams) != 3:
                        gaps.append(
                            f"tfma streams: n_streams {n_streams} != the protocol's "
                            "fixed 3 streams"
                        )
                    if len(per_stream) != int(n_streams):
                        gaps.append(
                            f"tfma streams: {len(per_stream)} per-stream records != "
                            f"n_streams {n_streams}"
                        )
                except (TypeError, ValueError, OverflowError):
                    pass
            for index, record in enumerate(per_stream):
                if not isinstance(record, Mapping):
                    gaps.append(f"tfma stream {index}: record is not a mapping")
                    continue
                requests = record.get("n_requests")
                tokens = record.get("n_positions")
                if tokens is None:
                    tokens = record.get("n_tokens")
                for label, value in (
                    ("requests", requests), ("tokens", tokens),
                    ("types", record.get("n_types")),
                ):
                    if value is None:
                        continue
                    try:
                        numeric = float(value)
                        whole = math.isfinite(numeric) and numeric == int(numeric)
                    except (TypeError, ValueError, OverflowError):
                        whole = False
                    if not whole:
                        gaps.append(
                            f"tfma stream {index}: {label} is not a whole number"
                        )
                try:
                    if int(requests) != 5000:
                        gaps.append(
                            f"tfma stream {index}: {requests} requests != the "
                            "protocol's fixed 5000"
                        )
                except (TypeError, ValueError, OverflowError):
                    gaps.append(f"tfma stream {index}: requests missing or not numeric")
                try:
                    if int(tokens) < 100_000:
                        gaps.append(
                            f"tfma stream {index}: {tokens} tokens < 100000"
                        )
                except (TypeError, ValueError, OverflowError):
                    gaps.append(f"tfma stream {index}: tokens missing or not numeric")
                stream_types = record.get("n_types")
                try:
                    if int(stream_types) < 1000:
                        gaps.append(
                            f"tfma stream {index}: {stream_types} types < 1000"
                        )
                except (TypeError, ValueError, OverflowError):
                    gaps.append(f"tfma stream {index}: types missing or not numeric")
            try:
                request_sum = sum(
                    int(record["n_requests"]) for record in per_stream
                    if isinstance(record, Mapping)
                )
                token_sum = sum(
                    int(
                        record.get("n_positions")
                        if record.get("n_positions") is not None
                        else record.get("n_tokens")
                    )
                    for record in per_stream if isinstance(record, Mapping)
                )
                total_prompts = _number(payload, diag, "n_prompts")
                if total_prompts is None:
                    total_prompts = _number(payload, diag, "n_requests")
                if total_prompts is not None and request_sum != int(total_prompts):
                    gaps.append(
                        f"tfma streams: per-stream requests sum {request_sum} != "
                        f"n_prompts {total_prompts}"
                    )
                declared_requests = _number(payload, diag, "n_requests")
                if (
                    declared_requests is not None
                    and request_sum != int(declared_requests)
                ):
                    gaps.append(
                        f"tfma streams: per-stream requests sum {request_sum} != "
                        f"n_requests {declared_requests}"
                    )
                observed_total = _number(payload, diag, "n_tokens")
                if observed_total is not None and token_sum != int(observed_total):
                    gaps.append(
                        f"tfma streams: per-stream tokens sum {token_sum} != "
                        f"n_tokens {observed_total}"
                    )
                reference = _number(payload, diag, "reference_tokens")
                if reference is not None:
                    for index, record in enumerate(per_stream):
                        if not isinstance(record, Mapping):
                            continue
                        tokens = record.get("n_positions")
                        if tokens is None:
                            tokens = record.get("n_tokens")
                        if int(tokens) * 100 > int(reference) + 1e-9:
                            gaps.append(
                                f"tfma stream {index}: observed {tokens} x100 > "
                                f"reference {reference} (per-stream ratio)"
                            )
            except (TypeError, ValueError, KeyError, OverflowError):
                gaps.append("tfma streams: per-stream sums are not numeric")
        canonical = {"SH-P", "SH-B", "SH-ALIGN"}
        variant_records = [
            payload.get("attack_variant"),
            payload.get("rotation_grouping"),
            diag.get("rotation_grouping"),
        ]
        # Nested records count too: a grouping label hidden one level down must
        # not disagree with the top-level cell identity.
        for source in (payload, diag):
            variant_records.extend(_collect_key_values(source, "rotation_grouping"))
            for value in _collect_key_values(source, "attack_variant"):
                normalised = _normalise_grouping(value)
                if any(
                    label in normalised
                    for label in ("SH-P", "SH-B", "SH-ALIGN")
                ):
                    variant_records.append(normalised)
                    for label in ("SH-P", "SH-B", "SH-ALIGN"):
                        if label in normalised:
                            variant_records.append(label)
        # The human-readable grouping description must not describe a different
        # cell than the declared label (e.g. an SH-ALIGN description under SH-P).
        variants: set[str] = set()
        for text in (
            _collect_key_values(payload, "grouping")
            + _collect_key_values(diag, "grouping")
        ):
            normalised_text = _normalise_grouping(text)
            lowered = normalised_text.lower()
            for label in canonical:
                if label in normalised_text:
                    variants.add(label)
            if "raw-symbol" in lowered or "rawsymbol" in lowered:
                variants.add("SH-B")
            if (
                "stage-1knn" in lowered
                or "deploymentid" in lowered
                or "knndeployment" in lowered
            ):
                variants.add("SH-ALIGN")
            if "group-id" in lowered or "groupid" in lowered:
                variants.add("SH-P")
        # ``diagnostics.attack_variant`` normally names the algorithmic variant
        # (e.g. "embedding_level"); only a canonical grouping value there is a
        # second grouping record that must agree.
        diag_variant = _normalise_grouping(diag.get("attack_variant") or "")
        if diag_variant in canonical:
            variant_records.append(diag_variant)
        variants = set(variants)
        for value in variant_records:
            if value in (None, ""):
                continue
            normalised = _normalise_grouping(value)
            variants.add(normalised)
        unknown = sorted(variants - canonical)
        if unknown:
            gaps.append(
                f"tfma cell: non-canonical grouping label(s) {unknown}; expected "
                f"one of {sorted(canonical)}"
            )
        knowledge = payload.get("attacker_knowledge") or diag.get("attacker_knowledge")
        rotation = payload.get("rotation_aware", diag.get("rotation_aware"))
        payload_knowledge = payload.get("attacker_knowledge")
        diag_knowledge = diag.get("attacker_knowledge")
        if payload_knowledge is not None and diag_knowledge is not None:
            if (
                str(payload_knowledge).strip().upper()
                != str(diag_knowledge).strip().upper()
            ):
                gaps.append(
                    "tfma cell: conflicting attacker_knowledge records "
                    f"{payload_knowledge!r} vs {diag_knowledge!r}"
                )
        payload_rotation = payload.get("rotation_aware")
        diag_rotation = diag.get("rotation_aware")
        if payload_rotation is not None and diag_rotation is not None:
            if _is_true(payload_rotation) != _is_true(diag_rotation):
                gaps.append(
                    "tfma cell: conflicting rotation_aware records "
                    f"{payload_rotation!r} vs {diag_rotation!r}"
                )
        if not knowledge:
            gaps.append(
                "tfma cell: attacker_knowledge missing (the SH-P cell must be declared)"
            )
        elif str(knowledge).strip().upper() != "SH-P":
            gaps.append(
                f"tfma cell: attacker_knowledge {knowledge!r} is not the protocol SH-P cell"
            )
        if rotation is None:
            gaps.append("tfma cell: rotation_aware missing (the SH-P cell is rotation-aware)")
        elif not _is_true(rotation):
            gaps.append(
                "tfma cell: rotation_aware is not true; this is not the protocol SH-P cell"
            )
        if not variants:
            gaps.append(
                "tfma cell: attack_variant/rotation_grouping missing; the grouping "
                "(SH-P/SH-B/SH-ALIGN) must be declared"
            )
        if len(variants) > 1:
            gaps.append(
                f"tfma cell: conflicting grouping records {sorted(variants)}"
            )
        if any(
            _diagnostic_flag(value)
            for source in (payload, diag)
            for value in _collect_key_values_containing(
                source, "diagnostic", exclude=("diagnostics", "evidence")
            )
        ):
            gaps.append(
                "tfma: diagnostic-only path cannot be certified paper-scale"
            )
        payload_reference = payload.get("reference_tokens")
        diag_reference = diag.get("reference_tokens")
        if diag_reference is None:
            gaps.append(
                "tfma reference_tokens: measured diagnostics value missing"
            )
        if payload_reference is not None and diag_reference is not None:
            try:
                payload_value = float(payload_reference)
                diag_value = float(diag_reference)
                if (
                    payload_value != int(payload_value)
                    or diag_value != int(diag_value)
                    or payload_value != diag_value
                ):
                    gaps.append(
                        "tfma reference_tokens: declared "
                        f"{payload_reference!r} != recorded {diag_reference!r}"
                    )
            except (TypeError, ValueError, OverflowError):
                gaps.append("tfma reference_tokens: non-numeric record")
        if any("SH-ALIGN" in value for value in variants):
            gaps.append(
                "tfma: SH-ALIGN is not the protocol SH-P cell and cannot be "
                "certified as it"
            )
        if any("SH-B" in value for value in variants):
            gaps.append(
                "tfma: the SH-B raw-symbol path is diagnostic-only and cannot be "
                "certified as the paper-scale SH-P cell"
            )
            premise = diag.get("premise")
            holds = (
                premise.get("premise_holds")
                if isinstance(premise, Mapping) else None
            )
            measured = diag.get("symbol_invariance")
            measured_holds = (
                measured.get("premise_holds")
                if isinstance(measured, Mapping) else None
            )
            if not _is_true(holds):
                gaps.append(
                    "tfma SH-B: measured symbol-invariance premise_holds is not true; "
                    "raw-symbol pooling is not a valid paper-scale row"
                )
            if holds is not None and measured_holds is not None and _is_true(holds) != _is_true(measured_holds):
                gaps.append(
                    "tfma SH-B: declared premise contradicts the measured "
                    "symbol_invariance record"
                )
            premise_measured = (
                premise.get("measured")
                if isinstance(premise, Mapping) else None
            )
            if not _is_true(premise_measured):
                gaps.append(
                    "tfma SH-B: symbol-invariance premise is not marked measured"
                )
            if _is_true(diag.get("diagnostic_only")):
                gaps.append(
                    "tfma SH-B: diagnostic-only path cannot be certified paper-scale"
                )
        if diag.get("candidate_lists_unique") is not True:
            gaps.append(
                "tfma candidates: the run must verify its candidate lists are distinct "
                "(candidate_lists_unique is not true)"
            )
        for source in (payload, diag):
            for value in _collect_key_values(source, "attacker_knowledge"):
                if _normalise_grouping(value) != "SH-P":
                    gaps.append(
                        f"tfma cell: nested attacker_knowledge {value!r} is not SH-P"
                    )
            for value in _collect_key_values(source, "rotation_aware"):
                if not _is_true(value):
                    gaps.append(
                        f"tfma cell: nested rotation_aware {value!r} is not true"
                    )
        declared_top100 = payload.get("top100_distinct")
        measured_top100 = diag.get("top100_distinct")
        if measured_top100 is None:
            gaps.append(
                "tfma top100_distinct: measured value missing from diagnostics"
            )
        if (
            declared_top100 is not None and measured_top100 is not None
        ):
            try:
                if int(declared_top100) != int(measured_top100):
                    gaps.append(
                        "tfma top100_distinct: declared "
                        f"{declared_top100!r} != measured {measured_top100!r}"
                    )
            except (TypeError, ValueError, OverflowError):
                gaps.append("tfma top100_distinct: non-integer declaration")
    elif attack in {"sda", "sda2021", "sda2023"}:
        _floor_gap(gaps, "n_train_sequences", _number(payload, diag, "n_train_sequences"), 60000)
        _floor_gap(gaps, "n_val_sequences", _number(payload, diag, "n_val_sequences"), 2000)
        _floor_gap(gaps, "n_epochs", _number(payload, diag, "n_epochs"), 6)
        metrics = payload.get("metrics")
        for name in ("sampling_evidence", "validation_sampling"):
            evidence = metrics.get(name) if isinstance(metrics, Mapping) else None
            width = evidence.get("window_length") if isinstance(evidence, Mapping) else None
            if isinstance(width, bool) or not isinstance(width, int) or width != 128:
                gaps.append(f"SDA {name}: fixed 128-token window evidence missing or inconsistent (got {width!r})")
        _stream_floor_gaps(gaps, payload, diag)
    elif attack in TEXT_ATTACKS:
        _floor_gap(gaps, "n_test", _number(payload, diag, "n_test"), 1000)
    elif attack == "output_pii":
        _floor_gap(gaps, "n_prompts", _number(payload, diag, "n_prompts"), 1000)
        _floor_gap(gaps, "max_new_tokens", _number(payload, diag, "max_new_tokens"), 256)
        if not _number(payload, diag, "generation_record"):
            gaps.append("generation: real generation record missing")
    elif attack in {"utility_generation", "utility_classification"}:
        docs = (
            _number(payload, diag, "n_docs")
            or _number(payload, diag, "n_test")
            or _number(payload, diag, "n_prompts")
        )
        _floor_gap(gaps, "utility docs", docs, 1000)
    elif attack == "efficiency":
        requests = _number(payload, diag, "scored_requests")
        if requests is None:
            requests = _number(payload, diag, "n_requests")
        _floor_gap(gaps, "scored_requests", requests, 200)
        _floor_gap(gaps, "rounds", _number(payload, diag, "rounds"), 3)
        _floor_gap(gaps, "output_length", _number(payload, diag, "output_length"), 128)
        lengths = _number(payload, diag, "input_lengths")
        try:
            got_lengths = {int(value) for value in lengths or []}
        except (TypeError, ValueError, OverflowError):
            got_lengths = set()
        if not {128, 512, 1024} <= got_lengths:
            gaps.append(f"input_lengths: {sorted(got_lengths)} missing 128/512/1024")
    if attack in FULL_VOCAB_REQUIRED:
        mode = payload.get("candidate_mode")
        if mode != "full":
            gaps.append(f"candidate_mode: {mode!r} != 'full'")
    return gaps


def audit_payload(
    payload: Mapping[str, Any],
    *,
    profile,
    protocol_sha256: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict:
    """Return the audit verdict for one result payload. Read-only."""
    raw_metrics = payload.get("metrics")
    if raw_metrics is None:
        raw_metrics = {}
    if not isinstance(raw_metrics, Mapping):
        return {
            "verdict": "below_budget",
            "status": "below_budget",
            "reason": "metrics is not a mapping",
            "gaps": ["metrics: not a mapping"],
            "control": {"passed": None},
            "saturation": None,
            "auditor": AUDITOR,
            "protocol_sha256": protocol_sha256,
        }
    metrics = raw_metrics
    raw_diag = payload.get("diagnostics")
    if raw_diag is not None and not isinstance(raw_diag, Mapping):
        return {
            "verdict": "below_budget",
            "status": "below_budget",
            "reason": "diagnostics is not a mapping",
            "gaps": ["diagnostics: not a mapping"],
            "control": {"passed": None},
            "saturation": None,
            "auditor": AUDITOR,
            "protocol_sha256": protocol_sha256,
        }
    diag = dict(raw_diag or {})
    if evidence:
        diag.update(evidence)
    attack = str(payload.get("attack_id") or payload.get("attack") or "")

    # Binding checks run before any self-declared status. A payload that is not
    # bound to the audited paper-v1 protocol cannot become not_applicable either.
    binding_gaps: list[str] = []
    profile_hash = getattr(profile, "protocol_sha256", None)
    if getattr(profile, "name", None) != PAPER_PROFILE_NAME:
        binding_gaps.append(
            f"profile: {getattr(profile, 'name', None)!r} != '{PAPER_PROFILE_NAME}'"
        )
    if not profile_hash:
        binding_gaps.append("protocol: profile carries no protocol hash")
    elif profile_hash != protocol_sha256:
        binding_gaps.append("protocol: profile hash differs from the audited protocol file")
    if payload.get("profile") != PAPER_PROFILE_NAME:
        binding_gaps.append(
            f"profile: payload {payload.get('profile')!r} != '{PAPER_PROFILE_NAME}'"
        )
    if payload.get("protocol_sha256") != protocol_sha256:
        binding_gaps.append(
            "protocol: payload protocol_sha256 differs from the audited protocol file"
        )
    if binding_gaps:
        return {
            "verdict": "below_budget",
            "status": "below_budget",
            "reason": "; ".join(binding_gaps[:4]),
            "gaps": binding_gaps,
            "control": {"passed": diag.get("positive_control_passed")},
            "saturation": None,
            "auditor": AUDITOR,
            "protocol_sha256": protocol_sha256,
        }

    reason = diag.get("not_applicable_reason") or metrics.get("reason")
    if metrics.get("status") == "not_applicable" or reason:
        premise = diag.get("not_applicable_premise")
        na_arm = _payload_arm(payload)
        if attack in FLOOR_ATTACKS and (not na_arm or na_arm not in PAPER_ARMS):
            return {
                "verdict": "below_budget",
                "status": "below_budget",
                "reason": "not_applicable cannot be checked without a valid paper arm",
                "gaps": ["not_applicable: arm unresolved or not a section-8 arm"],
                "control": {"passed": diag.get("positive_control_passed")},
                "saturation": None,
                "auditor": AUDITOR,
                "protocol_sha256": protocol_sha256,
            }
        if premise not in {"task_not_supported", "attack_not_valid_for_arm",
                           "real_premise_absent"}:
            return {
                "verdict": "below_budget",
                "status": "below_budget",
                "reason": "not_applicable requires a known real-premise code",
                "gaps": ["not_applicable: missing a known premise code"],
                "control": {"passed": diag.get("positive_control_passed")},
                "saturation": None,
                "auditor": AUDITOR,
                "protocol_sha256": protocol_sha256,
            }
        if premise == "real_premise_absent":
            evidence = str(diag.get("not_applicable_evidence") or "").strip()
            scenario = str(
                payload.get("adapter_scenario") or payload.get("scenario") or ""
            )
            if not evidence:
                return {
                    "verdict": "below_budget",
                    "status": "below_budget",
                    "reason": "not_applicable real_premise_absent needs evidence",
                    "gaps": ["not_applicable: real_premise_absent without evidence"],
                    "control": {"passed": diag.get("positive_control_passed")},
                    "saturation": None,
                    "auditor": AUDITOR,
                    "protocol_sha256": protocol_sha256,
                }
            premise_code = diag.get("not_applicable_premise_code")
            if premise_code not in PREMISE_CODES:
                return {
                    "verdict": "below_budget",
                    "status": "below_budget",
                    "reason": "real_premise_absent requires a controlled premise code",
                    "gaps": [
                        "not_applicable: missing a controlled not_applicable_premise_code"
                    ],
                    "control": {"passed": diag.get("positive_control_passed")},
                    "saturation": None,
                    "auditor": AUDITOR,
                    "protocol_sha256": protocol_sha256,
                }
            reason_values = (
                _collect_key_values(payload, "reason")
                + _collect_key_values(payload, "not_applicable_reason")
            )
            reason_text = " ".join(str(value) for value in reason_values)
            lowered = f"{evidence} {reason_text}".lower()
            resource_words = (
                "missing", "unavailable", "not available", "not installed",
                "no encoder", "encoder unavailable", "encoder artifact",
                "artifact unavailable", "checkpoint unavailable",
                "requires gpu", "gpu hours", "no artifact", "no gpu",
                "could not obtain", "resource blocked", "resource_blocked",
                "no weights", "weights are not provided", "not provided",
                "not present", "is absent", "are absent",
                "out of memory", "oom", "compute budget",
                "budget exhausted", "budget insufficient",
                "insufficient budget", "not enough budget",
                "insufficient gpu", "gpu memory",
                "resources exhausted", "resource exhausted",
            )
            if any(word in lowered for word in resource_words):
                return {
                    "verdict": "below_budget",
                    "status": "below_budget",
                    "reason": "a missing resource is resource_blocked/not_run, not not_applicable",
                    "gaps": ["not_applicable: missing resource laundered as a premise absence"],
                    "control": {"passed": diag.get("positive_control_passed")},
                    "saturation": None,
                    "auditor": AUDITOR,
                    "protocol_sha256": protocol_sha256,
                }
            arm = _payload_arm(payload)
            related = PREMISE_ATTACKS.get(str(premise_code), set())
            if arm and applicable(arm, attack) and attack not in related:
                return {
                    "verdict": "below_budget",
                    "status": "below_budget",
                    "reason": (
                        f"the {premise_code} premise is unrelated to an applicable "
                        f"{attack}/{arm} cell"
                    ),
                    "gaps": [
                        f"not_applicable: {premise_code} is not a premise of {attack}"
                    ],
                    "control": {"passed": diag.get("positive_control_passed")},
                    "saturation": None,
                    "auditor": AUDITOR,
                    "protocol_sha256": protocol_sha256,
                }
            if scenario == "S-public" and attack in {
                "gate_ia", "attn_ia", "vma_w", "least_squares"
            }:
                return {
                    "verdict": "below_budget",
                    "status": "below_budget",
                    "reason": "S-public provides public clean reference weights for this attack",
                    "gaps": ["not_applicable: contradicts the S-public premise"],
                    "control": {"passed": diag.get("positive_control_passed")},
                    "saturation": None,
                    "auditor": AUDITOR,
                    "protocol_sha256": protocol_sha256,
                }
        if premise in {"attack_not_valid_for_arm", "task_not_supported"}:
            arm = _payload_arm(payload)
            if not arm or applicable(arm, attack):
                return {
                    "verdict": "below_budget",
                    "status": "below_budget",
                    "reason": f"{premise} is false for this cell",
                    "gaps": [
                        f"not_applicable: {attack}/{arm} is applicable under section 8"
                    ],
                    "control": {"passed": diag.get("positive_control_passed")},
                    "saturation": None,
                    "auditor": AUDITOR,
                    "protocol_sha256": protocol_sha256,
                }
        return {
            "verdict": "not_applicable",
            "status": "not_applicable",
            "reason": str(reason or "not applicable"),
            "gaps": [],
            "control": {"passed": diag.get("positive_control_passed")},
            "saturation": None,
            "auditor": AUDITOR,
            "protocol_sha256": protocol_sha256,
        }

    control = {"passed": diag.get("positive_control_passed"), "detail": diag.get("positive_control")}
    if control["passed"] is False:
        return {
            "verdict": "control_failed",
            "status": "control_failed",
            "reason": "positive control not passed",
            "gaps": ["control: positive_control_passed is false"],
            "control": control,
            "saturation": diag.get("saturation"),
            "auditor": AUDITOR,
            "protocol_sha256": protocol_sha256,
        }

    gaps: list[str] = []
    arm = _payload_arm(payload)
    if attack in FLOOR_ATTACKS:
        if not arm:
            gaps.append("arm: cannot determine the paper arm for applicability")
        elif arm not in PAPER_ARMS:
            gaps.append(f"arm: {arm!r} is not a section-8 arm")
    if arm in TRAINED_DEFENSE_ARMS:
        seeds = payload.get("defense_training_seeds")
        if seeds is None:
            seeds = diag.get("defense_training_seeds")
        try:
            got_seeds = {int(seed) for seed in seeds or []}
        except (TypeError, ValueError, OverflowError):
            got_seeds = set()
        if got_seeds != set(REQUIRED_TRAINING_SEEDS):
            gaps.append(
                "defense_training_seeds: "
                f"{sorted(got_seeds)} != required {sorted(REQUIRED_TRAINING_SEEDS)}"
            )
    if arm in PAPER_ARMS and not applicable(arm, attack):
        gaps.append(
            f"applicability: {attack}/{arm} is not applicable under section 8"
        )
    # The spec cell may carry a symbolic scenario (e.g. "S-B"); the adapter
    # scenario is what the public view actually granted and is authoritative
    # for the clean-access rules.
    scenario = str(
        payload.get("adapter_scenario") or payload.get("scenario") or ""
    )

    def _truthy_flag(value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return value is True or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 1
        )

    def _flag_true_any(obj: Any, flag: str) -> bool:
        if isinstance(obj, Mapping):
            if flag in obj and _truthy_flag(obj.get(flag)):
                return True
            return any(_flag_true_any(value, flag) for value in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(_flag_true_any(value, flag) for value in obj)
        return False

    for flag in (
        "attack_uses_key", "attack_uses_client_secret", "attack_uses_encoder",
        "uses_known_pairs", "uses_client_key", "uses_client_secret",
        "attack_uses_root_key",
    ):
        allowed = {"S-pair", "S-encoder"} if flag == "attack_uses_encoder" else ({"S-pair"} if flag == "uses_known_pairs" else set())
        if scenario in allowed:
            continue
        if any(_flag_true_any(source, flag) for source in (payload, diag, metrics)):
            gaps.append(
                f"permission: {flag}=true is not allowed in scenario {scenario!r}"
            )
    if attack in {"gate_ia", "attn_ia", "vma_w", "least_squares"}:
        clean_values = (
            _collect_key_values(payload, "clean_model_public")
            + _collect_key_values(metrics, "clean_model_public")
            + _collect_key_values(diag, "clean_model_public")
        )
        if not clean_values or any(value is not True for value in clean_values):
            gaps.append(
                f"scenario: {attack} requires every clean_model_public record to be true; "
                f"got {clean_values!r}"
            )
        if scenario not in {"S-public", "S-encoder", "S-pair"}:
            gaps.append(
                f"scenario: {attack} requires a recorded clean-weight "
                f"scenario; got {scenario!r}"
            )
    if attack not in FLOOR_ATTACKS:
        gaps.append(f"attack: no paper-v1 floor for {attack!r}")
    if attack in {"sda", "sda2021", "sda2023"}:
        gaps.append("control: SDA paired independent-document comparison against frequency and LM priors is not yet verified")
    if (attack == "least_squares"
            and diag.get("attack_variant") == "dense_mixing_composed_engineering_variant"):
        gaps.append("control: deployment-level composed positive control at the same candidate, data, layer and budget settings is not verified; synthetic control alone is insufficient")
    if control["passed"] is not True:
        gaps.append(f"control: positive_control_passed is {control['passed']!r}, not True")
    control_record = diag.get("positive_control") or payload.get("positive_control") or {}
    if control["passed"] is True:
        required = (
            control_record.get("required_top1")
            if isinstance(control_record, Mapping) else None
        )
        declared = (
            control_record.get("top1")
            if isinstance(control_record, Mapping) else None
        )
        floor = CONTROL_FLOORS.get(attack)
        declared_f = None
        required_f = None
        try:
            if declared is not None:
                declared_f = float(declared)
                if not math.isfinite(declared_f):
                    gaps.append(f"control: top1 {declared!r} is not finite")
                    declared_f = None
        except (TypeError, ValueError, OverflowError):
            gaps.append(f"control: top1 {declared!r} is not numeric")
        try:
            if required is not None:
                required_f = float(required)
                if not math.isfinite(required_f):
                    gaps.append(f"control: required_top1 {required!r} is not finite")
                    required_f = None
        except (TypeError, ValueError, OverflowError):
            gaps.append(f"control: required_top1 {required!r} is not numeric")
        if floor is not None:
            if declared_f is None or declared_f < floor:
                gaps.append(
                    f"control: passed=true but top1 {declared!r} < "
                    f"pre-registered floor {floor}"
                )
            if required is not None and required_f is not None and required_f < floor:
                gaps.append(
                    f"control: declared anchor {required!r} < "
                    f"pre-registered floor {floor}"
                )
        elif required is None:
            gaps.append("control: passed=true but no required_top1 declared")
        elif declared_f is None or required_f is None or declared_f < required_f:
            gaps.append(
                f"control: passed=true but top1 {declared!r} < required {required!r}"
            )

    by_layer = (
        control_record.get("positive_control_by_layer")
        if isinstance(control_record, Mapping) else None
    )
    if isinstance(by_layer, Mapping) and by_layer:
        layer_values: list[float] = []
        layer_floor = CONTROL_FLOORS.get(attack)
        for name, record in by_layer.items():
            if not isinstance(record, Mapping):
                gaps.append(f"control: layer {name} record is not a mapping")
                continue
            try:
                value = float(record.get("top1"))
            except (TypeError, ValueError, OverflowError):
                gaps.append(f"control: layer {name} top1 is not numeric")
                continue
            if not math.isfinite(value):
                gaps.append(f"control: layer {name} top1 is not finite")
                continue
            layer_values.append(value)
            if record.get("passed") is not True:
                gaps.append(f"control: layer {name} did not pass its own control")
            if layer_floor is not None and value < layer_floor:
                gaps.append(
                    f"control: layer {name} top1 {value} < pre-registered floor "
                    f"{layer_floor}"
                )
        if layer_values and declared_f is not None:
            minimum = min(layer_values)
            if abs(declared_f - minimum) > 1e-9:
                gaps.append(
                    f"control: declared top1 {declared_f} != min over layers {minimum} "
                    "(a passing shallow layer cannot mask a failing deep layer)"
                )

    gaps.extend(_attack_floors(attack, payload, diag))
    gaps.extend(_declared_size_gaps(payload, diag))
    gaps.extend(_metric_shape_gaps(attack, metrics))
    gaps.extend(_evidence_gaps(payload, diag, attack))
    gaps.extend(_primary_denominator_gaps(payload, diag, attack))
    gaps.extend(_rank_reconciliation_gaps(attack, metrics, diag))
    if attack in CI_REQUIRED:
        gaps.extend(_ci_gaps(payload, attack, metrics))

    saturation = None
    if attack in SATURATION_REQUIRED:
        saturation = diag.get("saturation")
        sat_gap = _saturation_gap(diag, metrics=metrics, attack=attack, payload=payload)
        if sat_gap:
            gaps.append(sat_gap)

    non_saturation = [
        gap for gap in gaps
        if not gap.startswith(("saturation:", "saturation-record:"))
    ]
    if non_saturation:
        verdict = "below_budget"
    elif any(
        gap.startswith(("saturation:", "saturation-record:")) for gap in gaps
    ):
        verdict = "not_converged"
    else:
        verdict = "audited_paper_scale"

    return {
        "verdict": verdict,
        "status": verdict,
        "reason": "; ".join(gaps[:8]) if gaps else "all paper-v1 floors, control and evidence met",
        "gaps": gaps,
        "control": control,
        "saturation": saturation,
        "auditor": AUDITOR,
        "protocol_sha256": protocol_sha256,
    }
