"""Acceptance tests for the protocol auditor (F10, protocol §4/§6/§8).

A complete fixture must be accepted as ``audited_paper_scale``; every tampered
field must be rejected with a specific gap. The auditor never edits the
measurement.
"""

from __future__ import annotations

import copy

import pytest

from obfuscation_attacks.evaluation.paper_profile import protocol_sha256, require_profile
from obfuscation_attacks.evaluation.protocol_audit import (
    _evidence_gaps, _layer_gaps, audit_payload,
)

pytestmark = pytest.mark.unit

PROFILE = require_profile("paper-v1")
DIGEST = protocol_sha256()


def _evidence(control: float = 1.0) -> dict:
    return {
        "positive_control_passed": True,
        "clean_model_public": True,
        "positive_control": {"name": "known_construction", "top1": control,
                             "required_top1": 0.99},
        "per_unit_predictions": [
            {"unit": index, "hits": 20, "count": 20, "rank": [1] * 20}
            for index in range(1000)
        ],
        "data_sha256": "d" * 64,
        "source_sha256": "e" * 64,
    }


def knn_payload(**overrides) -> dict:
    payload = {
        "profile": "paper-v1",
        "protocol_sha256": DIGEST,
        "attack": "knn_l2",
        "arm": "ab",
        "scenario": "S-public",
        "defense_training_seeds": [17, 29, 43],
        "metrics": {
            "stage1_permuted_id": {
                "top1": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                "top5": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                "top10": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                "top100": {"value": 1.0, "numerator": 20000, "denominator": 20000},
            }
        },
        "n_prompts": 1000,
        "n_tokens": 20000,
        "n_keys": 1,
        "top_k": 100,
        "candidate_mode": "full",
        "ci": {
            "top1_micro": {
                "n_boot": 2000, "value": 1.0, "lo": 1.0, "hi": 1.0,
                "denominator": 20000, "estimand": "micro",
                "n_clusters": 1000, "n_units": 1000,
            },
            "top1_per_text_macro": {
                "n_boot": 2000, "mean": 1.0, "lo": 1.0, "hi": 1.0,
                "estimand": "macro", "n_clusters": 1000, "n_units": 1000,
            },
        },
        "diagnostics": _evidence(),
    }
    payload.update(overrides)
    return payload


def _topk_metrics(container: str = "stage1_permuted_id") -> dict:
    block = {
        f"top{k}": {"value": 1.0, "numerator": 20000, "denominator": 20000}
        for k in (1, 5, 10, 100)
    }
    return {container: block}


def vma_payload(**overrides) -> dict:
    metrics = overrides.pop("metrics", _topk_metrics("occurrence_level"))
    payload = knn_payload(
        attack="vma_h_recovered_prefix",
        target="llama-3.2-1b-base/ab",
        n_prompts=200,
        n_keys=5,
        keys_per_prompt=5,
        n_observations=1000,
        layers=[1, 5, 10, 15],
        prefix="recovered",
        metrics=metrics,
    )
    payload.update(overrides)
    return payload


def _isa_curve() -> dict:
    steps = (1000, 10000, 25000, 50000)
    layer = {
        f"steps{step}": {
            f"top{k}": {"value": 1.0, "numerator": 20000, "denominator": 20000}
            for k in (1, 5, 10, 100)
        }
        for step in steps
    }
    return {"trajectory_mode": "continuous_checkpoints", "budget_curve": {f"layer{n}": copy.deepcopy(layer) for n in (1, 5, 10, 15)}}


def _paired_saturation(first, last, before, after):
    shared = {"split": "validation", "model_sha256": "a" * 64,
              "validation_data_sha256": "b" * 64}
    return {
        "points": [{**shared, "budget": first, "top1": before / 100},
                   {**shared, "budget": last, "top1": after / 100}],
        "gain_pp": float(after - before),
        "paired_units": [
            {"unit": str(i), "cluster": str(i), "count": 100,
             "hits_before": before, "hits_after": after} for i in range(2)
        ],
    }


def isa_payload(**overrides) -> dict:
    strata = {name: 50 for name in ("32-64", "65-128", "129-256", "257-512")}
    metrics = overrides.pop("metrics", _isa_curve())
    payload = knn_payload(
        attack="isa_er",
        target="llama-3.2-1b-base/ab",
        n_prompts=200,
        n_keys=5,
        keys_per_prompt=5,
        n_observations=1000,
        length_strata={"required": dict(strata), "observed": dict(strata)},
        selected_on_validation={
            "n_prompts": 40,
            "strata": {name: 10 for name in strata},
            "lr": 1e-3,
            "lambda_dm": 0.0,
            "initialization": "paper",
            "split": "validation", "metric": "token_top1", "data_sha256": "b" * 64,
            "trials": [
                {"lr": rate, "lambda_dm": penalty, "initialization": "paper",
                 "token_top1": .8 if rate == 1e-3 and penalty == 0 else .5}
                for rate in (1e-4, 5e-4, 1e-3, 1e-2) for penalty in (0., 1e-3)
            ],
        },
        layers=[1, 5, 10, 15],
        restarts=3,
        executed_steps=50000,
        optimizer_steps_total=600_000_000,
        failed_observations=0,
        metrics=metrics,
        diagnostics={
            **_evidence(),
            "saturation": _paired_saturation(25000, 50000, 100, 100),
        },
    )
    payload.update(overrides)
    return payload


def test_complete_fixture_is_accepted_without_editing_it():
    payload = knn_payload()
    before = copy.deepcopy(payload)
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "audited_paper_scale"
    assert audit["auditor"] == "protocol_auditor"
    assert audit["protocol_sha256"] == DIGEST
    assert payload == before


def test_one_key_per_prompt_cannot_satisfy_the_fresh_key_floor():
    for base in (vma_payload, isa_payload):
        audit = audit_payload(
            base(keys_per_prompt=1, n_observations=200),
            profile=PROFILE, protocol_sha256=DIGEST,
        )
        assert audit["verdict"] == "below_budget"
        assert any(gap.startswith("keys_per_prompt") for gap in audit["gaps"])
        assert any(gap.startswith("n_observations") for gap in audit["gaps"])


def test_isa_length_strata_must_be_filled():
    payload = isa_payload()
    payload["length_strata"]["observed"]["32-64"] = 49
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("length_strata") for gap in audit["gaps"])
    missing = isa_payload()
    missing.pop("length_strata")
    audit = audit_payload(missing, profile=PROFILE, protocol_sha256=DIGEST)
    assert any(gap.startswith("length_strata") for gap in audit["gaps"])


def test_isa_hyperparameters_must_be_selected_on_validation():
    payload = isa_payload()
    payload.pop("selected_on_validation")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("selected_on_validation") for gap in audit["gaps"])


def test_payload_without_any_fraction_is_below_budget():
    payload = knn_payload(attack="utility_generation", metrics={"score": 0.5})
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert "metrics: no numerator/denominator point estimate" in audit["gaps"]


def test_placeholder_metric_does_not_certify():
    payload = knn_payload()
    payload["metrics"] = {
        "placeholder": {"numerator": 0, "denominator": 20000},
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 0, "count": 20, "rank": [50] * 20}
        for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("top1" in gap or "top5" in gap or "top10" in gap
               for gap in audit["gaps"])


def test_topk_numerators_must_match_the_stored_ranks():
    payload = knn_payload()
    stage = payload["metrics"]["stage1_permuted_id"]
    stage["top1"]["numerator"] = 0
    stage["top5"]["numerator"] = 0
    stage["top10"]["numerator"] = 20000
    stage["top100"]["numerator"] = 0
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 0, "count": 20, "rank": [50] * 20}
        for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("ranks<=" in gap for gap in audit["gaps"])


def test_ci_interval_must_have_numeric_bounds_and_contain_its_point():
    payload = knn_payload()
    payload["ci"] = {"x": {"n_boot": 2000, "lo": None, "hi": None}}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("ci.x") for gap in audit["gaps"])


def test_permission_flags_cannot_be_numeric_or_nested_to_evade():
    for mutate in (
        lambda payload: payload.__setitem__("attack_uses_key", 1),
        lambda payload: payload["metrics"].__setitem__(
            "threat_model", {"attack_uses_key": True}
        ),
    ):
        payload = knn_payload()
        mutate(payload)
        audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
        assert audit["verdict"] == "below_budget"
        assert any(gap.startswith("permission:") for gap in audit["gaps"])


def test_pair_permission_never_grants_target_key():
    payload = knn_payload()
    payload["adapter_scenario"] = "S-pair"
    payload["uses_known_pairs"] = True
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any("uses_known_pairs" in gap for gap in audit["gaps"])
    payload["attack_uses_key"] = True
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("permission: attack_uses_key" in gap for gap in audit["gaps"])


def test_document_and_token_denominators_are_distinct():
    from obfuscation_attacks.evaluation.protocol_audit import _primary_denominator_gaps

    payload = {"n_requests": 2, "n_tokens": 6, "metrics": {
        "token_accuracy": {"numerator": 4, "denominator": 6, "value": 4 / 6},
        "entity_recovery": {"numerator": 1, "denominator": 2, "value": .5},
    }}
    diag = {"per_unit_predictions": [{"hits": 2, "count": 2}, {"hits": 2, "count": 4}]}
    assert _primary_denominator_gaps(payload, diag, "sda") == []
    payload["metrics"]["token_accuracy"] = {"numerator": 2, "denominator": 2, "value": 1.0}
    assert _primary_denominator_gaps(payload, diag, "sda")


def test_conflicting_clean_model_records_fail_closed():
    payload = knn_payload(
        attack="vma_w", n_types=1000, candidate_mode="full",
    )
    payload["layers"] = [1, 5, 10, 15]
    payload["diagnostics"]["clean_model_public"] = False
    payload["metrics"]["threat_model"] = {"clean_model_public": True}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("scenario:") for gap in audit["gaps"])


def test_real_premise_absence_requires_a_controlled_code():
    payload = knn_payload(
        metrics={"status": "not_applicable", "reason": "no generation path"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    payload["diagnostics"]["not_applicable_evidence"] = "this wire has no generation path"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("premise_code" in gap for gap in audit["gaps"])


def test_float_topk_keys_are_not_point_estimates():
    payload = knn_payload()
    payload["metrics"]["stage1_permuted_id"]["top1"] = 1.0
    payload["metrics"]["stage1_permuted_id"]["top5"] = 1.0
    payload["metrics"]["stage1_permuted_id"]["top10"] = 1.0
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("numerator/denominator top" in gap for gap in audit["gaps"])


def test_ci_block_needs_a_point_and_the_primary_estimate():
    payload = knn_payload()
    payload["ci"] = {"unrelated": {"n_boot": 2000, "lo": 0.0, "hi": 1.0}}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("no point estimate" in gap for gap in audit["gaps"])
    payload["ci"] = {"unrelated": {"n_boot": 2000, "lo": 0.0, "hi": 0.5}}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("micro/top1" in gap or "primary top1" in gap for gap in audit["gaps"])


def test_weight_clean_model_records_are_checked_at_every_level():
    for mutate in (
        lambda payload: payload.__setitem__("clean_model_public", False),
        lambda payload: payload.__setitem__(
            "threat_model", {"clean_model_public": False}
        ),
    ):
        payload = knn_payload(
            attack="vma_w", n_types=1000, candidate_mode="full",
        )
        payload["layers"] = [1, 5, 10, 15]
        mutate(payload)
        audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
        assert audit["verdict"] == "below_budget"
        assert any(gap.startswith("scenario:") for gap in audit["gaps"])


def test_not_applicable_reason_field_is_scanned_for_resources():
    payload = knn_payload(
        metrics={"status": "not_applicable", "reason": "resource blocked: encoder"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    payload["diagnostics"]["not_applicable_premise_code"] = "no_generation_path"
    payload["diagnostics"]["not_applicable_evidence"] = "this wire has no decoder output"
    payload["diagnostics"]["not_applicable_reason"] = "resource blocked: encoder"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("laundered" in gap for gap in audit["gaps"])


def test_permission_flags_are_found_at_any_nesting_depth():
    payload = knn_payload()
    payload["threat_model"] = {"permissions": {"attack_uses_key": True}}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("permission:") for gap in audit["gaps"])


def test_malformed_ranks_fail_closed_instead_of_crashing():
    payload = knn_payload()
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 0, "count": 20, "rank": ["1"] * 20}
        for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("rank values" in gap for gap in audit["gaps"])


def test_fraction_value_must_equal_numerator_over_denominator():
    payload = knn_payload()
    payload["metrics"]["stage1_permuted_id"]["top1"] = {
        "value": 1999.998, "numerator": 999999, "denominator": 500,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("value" in gap or "numerator" in gap for gap in audit["gaps"])


def test_topk_denominator_must_match_the_per_unit_count():
    payload = knn_payload()
    stage = payload["metrics"]["stage1_permuted_id"]
    stage["top1"] = {"value": 1.0, "numerator": 1, "denominator": 1}
    stage["top5"] = {"value": 0.0, "numerator": 0, "denominator": 20000}
    stage["top10"] = {"value": 0.0, "numerator": 0, "denominator": 20000}
    stage["top100"] = {"value": 1.0, "numerator": 20000, "denominator": 20000}
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 0, "count": 20, "rank": [50] * 20}
        for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("denominator" in gap and "per-unit" in gap for gap in audit["gaps"])


def test_linkage_exact_value_must_match_its_fraction():
    payload = knn_payload(attack="linkage_exact", n_streams=3)
    payload["metrics"]["test"] = {"positives": 10000, "negatives": 100000}
    payload["metrics"]["exact_test"] = {
        "value": 1.0, "numerator": 0, "denominator": 10000,
    }
    payload["ci"] = {
        "exact_test": {
            "value": 1.0, "n_boot": 2000, "lo": 1.0, "hi": 1.0,
            "numerator": 0, "denominator": 10000,
        }
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("exact_test.value" in gap for gap in audit["gaps"])


def test_sda_token_accuracy_must_be_numeric():
    payload = knn_payload(attack="sda")
    payload["metrics"]["token_accuracy"] = {"value": "garbage"}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("token_accuracy" in gap for gap in audit["gaps"])


def test_malformed_denominator_fails_closed_instead_of_crashing():
    payload = knn_payload()
    payload["metrics"]["stage1_permuted_id"]["top1"] = {
        "value": 0.0, "numerator": 0, "denominator": "abc",
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"


def test_ci_block_without_n_boot_is_not_skipped():
    payload = knn_payload()
    del payload["ci"]["top1_per_text_macro"]["n_boot"]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("n_boot" in gap for gap in audit["gaps"])


def test_resource_words_in_reason_are_rejected():
    payload = knn_payload(
        metrics={"status": "not_applicable", "reason": "out of memory on the GPU"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    payload["diagnostics"]["not_applicable_premise_code"] = "no_generation_path"
    payload["diagnostics"]["not_applicable_evidence"] = "no decoder output path"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("laundered" in gap for gap in audit["gaps"])


def test_non_mapping_metrics_or_diagnostics_fail_closed():
    payload = knn_payload()
    payload["metrics"] = ["x"]
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    payload = knn_payload()
    payload["diagnostics"] = ["x"]
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"


def test_one_unit_cannot_stand_in_for_a_thousand_prompts():
    payload = knn_payload()
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": 0, "hits": 20000, "count": 20000, "rank": [1] * 20000}
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("per-unit rows" in gap for gap in audit["gaps"])


def test_isa_budget_curve_reconciles_the_final_checkpoint():
    payload = isa_payload()
    payload["metrics"] = {
        "trajectory_mode": "continuous_checkpoints",
        "budget_curve": {
            "layer1": {
                "steps1000": {
                    "top1": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top5": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top10": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                },
                "steps10000": {
                    "top1": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top5": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top10": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                },
                "steps25000": {
                    "top1": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                    "top5": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                    "top10": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                },
                "steps50000": {
                    "top1": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                    "top5": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                    "top10": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                },
            }
        }
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "audited_paper_scale", audit["gaps"]


def test_sda_token_accuracy_needs_its_fraction():
    payload = knn_payload(attack="sda")
    payload["metrics"]["token_accuracy"] = {"value": 0.999}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("token_accuracy" in gap for gap in audit["gaps"])


def test_utility_and_efficiency_floors_are_enforced():
    utility = knn_payload(
        attack="utility_generation",
        metrics={"score": {"value": 0.9, "numerator": 9, "denominator": 10}},
        n_prompts=10, n_tokens=10,
    )
    utility["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1} for index in range(10)
    ]
    assert audit_payload(utility, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    efficiency = knn_payload(
        attack="efficiency",
        metrics={"rate": {"value": 0.9, "numerator": 9, "denominator": 10}},
        n_prompts=10, n_tokens=10,
    )
    efficiency["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1} for index in range(10)
    ]
    assert audit_payload(efficiency, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"


def test_one_aggregated_row_cannot_stand_in_for_weight_types():
    payload = knn_payload(
        attack="vma_w", n_prompts=0, n_tokens=0, n_types=1000,
        candidate_mode="full",
    )
    payload["layers"] = [1, 5, 10, 15]
    payload["metrics"] = {
        "mapping_asr": {
            f"top{k}": {"value": 0.001, "numerator": 1, "denominator": 1000}
            for k in (1, 5, 10, 100)
        }
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": 0, "hits": 1, "count": 1000, "rank": [1]}
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("per-unit rows" in gap for gap in audit["gaps"])


def test_weight_attacks_need_a_topk_mapping_metric():
    payload = knn_payload(
        attack="gate_ia", n_prompts=0, n_tokens=0, n_types=1000,
    )
    payload["layers"] = list(range(16))
    payload["metrics"] = {
        "unrelated_score": {"value": 1.0, "numerator": 1000, "denominator": 1000},
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("top1" in gap for gap in audit["gaps"])


def test_non_integer_fraction_counts_and_infinity_fail_closed():
    payload = knn_payload()
    payload["metrics"]["stage1_permuted_id"]["top1"] = {
        "value": 1.0, "numerator": 20000.9, "denominator": 20000.9,
    }
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    payload = knn_payload()
    payload["metrics"]["stage1_permuted_id"]["top1"] = {
        "value": 1.0, "numerator": 0, "denominator": float("inf"),
    }
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"


def test_all_na_reason_fields_are_scanned_for_resources():
    payload = knn_payload(
        metrics={"status": "not_applicable", "reason": "out of memory on GPU"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    payload["diagnostics"]["not_applicable_premise_code"] = "no_decoder_hidden_state"
    payload["diagnostics"]["not_applicable_evidence"] = "this wire has no hidden state"
    payload["diagnostics"]["not_applicable_reason"] = "no decoder state exposed"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("laundered" in gap for gap in audit["gaps"])


def test_nan_values_and_floors_fail_closed():
    payload = knn_payload()
    payload["metrics"]["stage1_permuted_id"]["top1"]["value"] = float("nan")
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    efficiency = knn_payload(
        attack="efficiency",
        metrics={"rate": {"value": 0.9, "numerator": 9, "denominator": 10}},
        n_prompts=10, n_tokens=10,
    )
    efficiency["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1} for index in range(10)
    ]
    efficiency["scored_requests"] = float("nan")
    efficiency["rounds"] = float("nan")
    efficiency["output_length"] = float("nan")
    efficiency["input_lengths"] = [128, 512, 1024]
    audit = audit_payload(efficiency, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("not finite" in gap for gap in audit["gaps"])


def test_isa_final_checkpoint_is_the_one_reconciled():
    payload = isa_payload()
    payload["metrics"] = {
        "budget_curve": {
            "layer1": {
                "steps1000": {
                    "top1": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                    "top5": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                    "top10": {"value": 1.0, "numerator": 20000, "denominator": 20000},
                },
                "steps50000": {
                    "top1": {"value": 0.95, "numerator": 19000, "denominator": 20000},
                    "top5": {"value": 0.95, "numerator": 19000, "denominator": 20000},
                    "top10": {"value": 0.95, "numerator": 19000, "denominator": 20000},
                },
            }
        }
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("steps50000" in gap and "ranks<=" in gap for gap in audit["gaps"])


def test_all_reason_fields_are_scanned_for_resources():
    payload = knn_payload(
        metrics={"status": "not_applicable", "reason": "no generation path"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    payload["diagnostics"]["not_applicable_premise_code"] = "no_generation_path"
    payload["diagnostics"]["not_applicable_evidence"] = "no decoder output"
    payload["diagnostics"]["reason"] = "insufficient GPU memory"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("laundered" in gap for gap in audit["gaps"])


def test_client_key_flag_and_string_true_are_rejected():
    for mutate in (
        lambda payload: payload.__setitem__("uses_client_key", True),
        lambda payload: payload.__setitem__("attack_uses_key", "true"),
    ):
        payload = knn_payload()
        mutate(payload)
        audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
        assert audit["verdict"] == "below_budget"
        assert any(gap.startswith("permission:") for gap in audit["gaps"])


def test_zero_count_rows_cannot_be_padding():
    payload = knn_payload()
    rows = [
        {"unit": index, "hits": 0, "count": 0, "rank": []}
        for index in range(999)
    ]
    rows.append({"unit": 999, "hits": 20000, "count": 20000, "rank": [1] * 20000})
    payload["diagnostics"]["per_unit_predictions"] = rows
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("no counted positions" in gap for gap in audit["gaps"])


def test_nan_declared_size_fails_closed_without_crashing():
    payload = knn_payload(
        attack="vma_w", n_prompts=0, n_tokens=0, n_types=float("nan"),
        candidate_mode="full",
    )
    payload["layers"] = [1, 5, 10, 15]
    payload["metrics"] = {
        "mapping_asr": {
            f"top{k}": {"value": 1.0, "numerator": 1000, "denominator": 1000}
            for k in (1, 5, 10, 100)
        }
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"


def test_nan_saturation_never_counts_as_converged():
    payload = isa_payload()
    payload["diagnostics"]["saturation"]["gain_pp"] = float("nan")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] != "audited_paper_scale"
    assert any("not finite" in gap for gap in audit["gaps"])


def test_non_integer_declared_sizes_fail_closed():
    payload = knn_payload(n_prompts=1000.9)
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("not an integer" in gap for gap in audit["gaps"])


def test_infinite_n_boot_and_control_fields_do_not_crash():
    payload = knn_payload()
    payload["ci"]["top1_micro"]["n_boot"] = float("inf")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    payload = knn_payload()
    payload["diagnostics"]["positive_control"] = ["not", "a", "mapping"]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"


def test_nonfinite_control_and_saturation_values_fail_closed():
    payload = vma_payload()
    payload["diagnostics"]["positive_control"]["top1"] = float("nan")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("not finite" in gap for gap in audit["gaps"])
    payload = isa_payload()
    payload["diagnostics"]["saturation"] = {
        "gain_pp": 1.0,
        "points": [
            {"budget": 50000, "top1": 0.0},
            {"budget": 100000, "top1": 1.0},
        ],
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] != "audited_paper_scale"
    assert any("does not match" in gap or "saturation" in gap for gap in audit["gaps"])


def test_huge_integers_and_infinite_layers_fail_closed():
    payload = knn_payload(n_prompts=10 ** 400)
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    payload = vma_payload(layers=[1, 5, 10, float("inf")])
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"


def test_zero_denominator_and_micro_macro_disagreement_fail_closed():
    payload = knn_payload()
    payload["metrics"]["stage1_permuted_id"]["top1"] = {
        "numerator": 5, "denominator": 0,
    }
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    payload = knn_payload()
    payload["ci"]["top1_micro"]["value"] = 0.0
    payload["ci"]["top1_micro"]["lo"] = 0.0
    payload["ci"]["top1_micro"]["hi"] = 0.0
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("micro/top1" in gap for gap in audit["gaps"])


def test_isa_zero_steps_cannot_fall_back_to_a_saturation_budget():
    payload = isa_payload(executed_steps=0)
    payload["diagnostics"]["saturation"]["max_budget"] = 50000
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("steps" in gap for gap in audit["gaps"])


def test_efficiency_zero_scored_requests_fail():
    payload = knn_payload(
        attack="efficiency", n_prompts=10, n_tokens=10,
        metrics={"rate": {"value": 0.9, "numerator": 9, "denominator": 10}},
    )
    payload["scored_requests"] = 0
    payload["n_requests"] = 250
    payload["rounds"] = 3
    payload["output_length"] = 128
    payload["input_lengths"] = [128, 512, 1024]
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1} for index in range(10)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("scored_requests" in gap for gap in audit["gaps"])


def test_huge_integers_in_auditor_specific_fields_fail_closed():
    payload = vma_payload(n_observations=10 ** 400)
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    payload = isa_payload()
    payload["selected_on_validation"]["lr"] = 10 ** 400
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"
    payload = knn_payload(attack="sda")
    payload["metrics"]["token_accuracy"] = {
        "value": 10 ** 400, "numerator": 9, "denominator": 10,
    }
    assert audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "below_budget"


def test_non_integer_stream_and_sequence_sizes_fail_closed():
    payload = knn_payload(attack="sda")
    payload["n_train_sequences"] = 60000.5
    payload["n_val_sequences"] = "2000"
    payload["n_epochs"] = 6
    payload["metrics"]["token_accuracy"] = {
        "value": 1.0, "numerator": 1, "denominator": 1,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("not an integer" in gap for gap in audit["gaps"])


def test_sda_budget_requires_evidence_of_actual_128_token_windows():
    from obfuscation_attacks.evaluation.protocol_audit import _attack_floors
    payload = {"metrics": {"sampling_evidence": {"window_length": 32},
                           "validation_sampling": {"window_length": 128}}}
    gaps = _attack_floors("sda", payload, {})
    assert any("sampling_evidence: fixed 128-token" in gap for gap in gaps)
    assert not any("validation_sampling: fixed 128-token" in gap for gap in gaps)
    payload["metrics"]["sampling_evidence"]["window_length"] = 128
    payload["metrics"]["validation_sampling"]["window_length"] = True
    gaps = _attack_floors("sda", payload, {})
    assert not any("SDA sampling_evidence:" in gap for gap in gaps)
    assert any("validation_sampling: fixed 128-token" in gap for gap in gaps)


def test_huge_bootstrap_count_is_not_certified():
    payload = knn_payload()
    payload["ci"]["top1_micro"]["n_boot"] = 10 ** 400
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("outside" in gap for gap in audit["gaps"])


def test_isa_ci_must_match_the_final_checkpoint():
    payload = isa_payload()
    payload["metrics"] = {
        "trajectory_mode": "continuous_checkpoints",
        "budget_curve": {
            "layer1": {
                "steps1000": {
                    "top1": {"value": 0.9, "numerator": 18000, "denominator": 20000},
                    "top5": {"value": 0.9, "numerator": 18000, "denominator": 20000},
                    "top10": {"value": 0.9, "numerator": 18000, "denominator": 20000},
                },
                "steps10000": {
                    "top1": {"value": 0.9, "numerator": 18000, "denominator": 20000},
                    "top5": {"value": 0.9, "numerator": 18000, "denominator": 20000},
                    "top10": {"value": 0.9, "numerator": 18000, "denominator": 20000},
                },
                "steps25000": {
                    "top1": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top5": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top10": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                },
                "steps50000": {
                    "top1": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top5": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                    "top10": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                },
            }
        }
    }
    payload["diagnostics"]["saturation"] = _paired_saturation(25000, 50000, 50, 50)
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 20 if index < 500 else 0, "count": 20,
         "rank": [1] * 20 if index < 500 else [50] * 20}
        for index in range(1000)
    ]
    payload["ci"]["top1_micro"].update({"value": 0.9, "lo": 0.85, "hi": 0.95})
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("micro/top1" in gap for gap in audit["gaps"])
    payload["ci"]["top1_micro"].update({"value": 0.5, "lo": 0.45, "hi": 0.55})
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "audited_paper_scale", audit["gaps"]


def test_utility_declaration_cannot_outrun_the_scored_docs():
    payload = knn_payload(
        attack="utility_generation",
        n_prompts=1000, n_docs=5, n_tokens=5,
        metrics={
            "accuracy": {
                "top1": {"value": 1.0, "numerator": 5, "denominator": 5},
            }
        },
    )
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1} for index in range(5)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"


def test_weight_types_cannot_be_bypassed_by_smaller_alias_counts():
    payload = knn_payload(
        attack="vma_w", n_types=1000, n_prompts=5, n_tokens=5, n_keys=1,
        candidate_mode="full",
    )
    payload["layers"] = [1, 5, 10, 15]
    payload["metrics"] = {
        "mapping_asr": {
            f"top{k}": {"value": 1.0, "numerator": 5, "denominator": 5}
            for k in (1, 5, 10, 100)
        }
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(5)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"


def test_isa_decoy_outside_budget_curve_is_not_reconciled():
    payload = isa_payload()
    payload["metrics"] = {
        "budget_curve": {
            "layer1": {
                "top1": {"value": 0.5, "numerator": 10000, "denominator": 20000},
                "top5": {"value": 0.95, "numerator": 19000, "denominator": 20000},
                "top10": {"value": 0.95, "numerator": 19000, "denominator": 20000},
            }
        },
        "extra": {
            "top1": {"value": 0.9, "numerator": 18000, "denominator": 20000},
        },
    }
    rows = []
    for index in range(1000):
        if index < 900:
            rows.append({"unit": index, "hits": 20, "count": 20, "rank": [1] * 20})
        elif index < 950:
            rows.append({"unit": index, "hits": 20, "count": 20, "rank": [5] * 20})
        else:
            rows.append({"unit": index, "hits": 0, "count": 20, "rank": [50] * 20})
    payload["diagnostics"]["per_unit_predictions"] = rows
    payload["ci"]["top1_micro"].update({"value": 0.5, "lo": 0.45, "hi": 0.55})
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("budget_curve" in gap and "numerator" in gap for gap in audit["gaps"])


def test_secondary_headline_fraction_must_match_the_unit_count():
    payload = knn_payload(
        attack="text_bayes", n_test=1000, n_prompts=1000, n_tokens=20000,
        metrics={
            "accuracy": {
                "top1": {"value": 1.0, "numerator": 5, "denominator": 5},
            },
            "decoy": {"value": 1.0, "numerator": 1000, "denominator": 1000},
        },
    )
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("declared unit count" in gap for gap in audit["gaps"])


def test_trained_defense_requires_the_three_fixed_training_seeds():
    payload = knn_payload()
    payload["defense_training_seeds"] = [42]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("defense_training_seeds") for gap in audit["gaps"])


def test_measured_payload_must_be_applicable_to_its_arm():
    payload = knn_payload(
        attack="tfma", arm="osnip", n_tokens=20000,
        reference_tokens=4_450_000, n_types=50000, top100_distinct=100,
    )
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("applicability") for gap in audit["gaps"])


def test_measured_payload_without_a_resolvable_arm_is_rejected():
    payload = knn_payload()
    payload.pop("arm")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("arm:") for gap in audit["gaps"])


def test_weight_attack_reads_clean_model_public_from_diagnostics():
    payload = knn_payload(
        attack="vma_w", n_types=1000, candidate_mode="full",
    )
    payload["diagnostics"]["clean_model_public"] = False
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("scenario:") for gap in audit["gaps"])


def test_not_applicable_cannot_hide_a_missing_resource():
    payload = knn_payload(
        metrics={"status": "not_applicable", "reason": "encoder is missing"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    payload["diagnostics"]["not_applicable_premise_code"] = "no_generation_path"
    payload["diagnostics"]["not_applicable_evidence"] = "encoder missing from the artifact"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("laundered" in gap or "resource" in gap for gap in audit["gaps"])


def test_weight_attack_on_a_non_public_clean_scenario_is_rejected():
    payload = knn_payload(
        attack="vma_w", arm="aloepri", scenario="S-deploy",
        n_types=1000, candidate_mode="full",
    )
    payload["metrics"]["threat_model"] = {"clean_model_public": False}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("scenario:") for gap in audit["gaps"])


def test_layer_list_must_match_the_frozen_choice():
    payload = vma_payload(
        target="llama-3.2-1b-base/ab", layers=[0, 1, 2, 3],
    )
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("layers:") for gap in audit["gaps"])


def _weight_layers_payload(attack: str = "gate_ia", layers=None,
                           missing=None) -> dict:
    payload = knn_payload(
        attack=attack, n_types=1000, n_prompts=0, n_tokens=0,
        target="llama-3.2-1b-base/aloepri_weights",
    )
    payload["layers"] = list(range(16)) if layers is None else layers
    payload["missing_weight_layers"] = [] if missing is None else missing
    return payload


def test_weight_attacks_must_cover_every_model_layer():
    payload = _weight_layers_payload(layers=[1, 5, 10, 15])
    gaps = _layer_gaps(payload, payload["diagnostics"], attack="gate_ia")
    assert any("neither used nor recorded as missing" in gap for gap in gaps)


def test_weight_attack_all_layers_pass_the_layer_rule():
    payload = _weight_layers_payload()
    for attack in ("gate_ia", "attn_ia"):
        gaps = _layer_gaps(payload, payload["diagnostics"], attack=attack)
        assert gaps == []


def test_weight_attack_can_record_an_unusable_layer_with_a_reason():
    used = [layer for layer in range(16) if layer != 7]
    payload = _weight_layers_payload(
        layers=used,
        missing=[{"layer": 7, "code": "weight_missing",
                  "reason": "gate_proj missing from artifact"}],
    )
    assert _layer_gaps(payload, payload["diagnostics"], attack="gate_ia") == []


def test_weight_attack_missing_layer_needs_a_machine_checkable_code():
    used = [layer for layer in range(16) if layer != 7]
    payload = _weight_layers_payload(
        layers=used,
        missing=[{"layer": 7, "reason": "not requested"}],
    )
    gaps = _layer_gaps(payload, payload["diagnostics"], attack="gate_ia")
    assert any("machine-checkable code" in gap for gap in gaps)


def test_weight_attack_missing_layer_needs_a_reason():
    used = [layer for layer in range(16) if layer != 7]
    payload = _weight_layers_payload(
        layers=used,
        missing=[{"layer": 7, "code": "weight_missing", "reason": ""}],
    )
    gaps = _layer_gaps(payload, payload["diagnostics"], attack="gate_ia")
    assert any("no recorded reason" in gap for gap in gaps)


def test_weight_attack_cannot_both_use_and_skip_a_layer():
    payload = _weight_layers_payload(
        missing=[{"layer": 3, "code": "weight_missing", "reason": "claimed unusable"}],
    )
    gaps = _layer_gaps(payload, payload["diagnostics"], attack="gate_ia")
    assert any("used and also recorded missing" in gap for gap in gaps)


def test_weight_attack_missing_layer_must_be_inside_the_model_depth():
    payload = _weight_layers_payload(
        missing=[{"layer": 99, "code": "weight_missing", "reason": "junk"}],
    )
    gaps = _layer_gaps(payload, payload["diagnostics"], attack="gate_ia")
    assert any("outside model layer range" in gap for gap in gaps)


def test_vma_w_is_held_to_the_same_full_layer_rule():
    payload = knn_payload(
        attack="vma_w", target="llama-3.2-1b/aloepri_weights",
        n_types=1000, candidate_mode="full",
    )
    payload["layers"] = [0]
    payload["missing_weight_layers"] = []
    gaps = _layer_gaps(payload, payload["diagnostics"], attack="vma_w")
    assert any("neither used nor recorded as missing" in gap for gap in gaps)


def test_legacy_macro_ci_with_n_units_only_is_accepted():
    payload = knn_payload()
    payload["ci"]["top1_per_text_macro"].pop("n_clusters")
    payload["ci"]["top1_per_text_macro"]["n_units"] = 1000
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any("n_clusters" in gap for gap in audit["gaps"])


def test_micro_estimand_block_is_checked_even_without_a_micro_name():
    payload = knn_payload()
    payload["ci"] = {
        "top1": {
            "micro": 1.0, "lo": 0.9, "hi": 1.0,
            "n_boot": 2000, "estimand": "micro",
        }
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("n_clusters missing" in gap for gap in audit["gaps"])


def test_tfma_after_inversion_reaches_the_tfma_floors():
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
        top100_distinct=10,
        attacker_knowledge="SH-P", rotation_aware=True,
        attack_variant="SH-P",
    )
    payload["diagnostics"]["top100_distinct"] = 10
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any("no paper-v1 floor" in gap for gap in audit["gaps"])
    assert any("reference_tokens" in gap for gap in audit["gaps"])
    assert any("top100_distinct" in gap for gap in audit["gaps"])


def test_isa_budget_ladder_must_have_all_four_checkpoints():
    payload = isa_payload()
    payload["metrics"]["budget_curve"]["layer1"].pop("steps25000")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("missing checkpoints" in gap and "25000" in gap for gap in audit["gaps"])


def test_isa_saturation_points_must_be_measured_checkpoints():
    payload = isa_payload()
    payload["diagnostics"]["saturation"] = {
        "points": [
            {"budget": 50000, "top1": 1.0},
            {"budget": 100000, "top1": 1.0},
        ],
        "gain_pp": 0.0,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "not_converged"
    assert any("not a measured budget_curve checkpoint" in gap for gap in audit["gaps"])


def test_isa_saturation_point_value_must_match_the_curve():
    payload = isa_payload()
    payload["diagnostics"]["saturation"]["points"][-1]["top1"] = 0.40
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "not_converged"
    assert any("contradicts the measured budget_curve" in gap for gap in audit["gaps"])


def test_isa_layer_control_cannot_take_the_best_layer():
    payload = isa_payload()
    payload["diagnostics"]["positive_control"] = {
        "top1": 1.0, "required_top1": 0.90, "passed": True,
        "positive_control_by_layer": {
            "layer1": {"top1": 1.0, "passed": True},
            "layer5": {"top1": 0.2, "passed": False},
        },
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("layer5" in gap or "min over layers" in gap for gap in audit["gaps"])


def test_tfma_shb_requires_a_true_symbol_invariance_premise():
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
        top100_distinct=10,
    )
    payload["attack_variant"] = "SH-B"
    payload["attacker_knowledge"] = "SH-B"
    payload["rotation_aware"] = False
    payload["diagnostics"]["premise"] = {"premise_holds": False}
    payload["diagnostics"]["diagnostic_only"] = True
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("tfma SH-B" in gap for gap in audit["gaps"])


def test_tfma_shb_declared_premise_must_match_the_measured_record():
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
        top100_distinct=10,
    )
    payload["attack_variant"] = "SH-B"
    payload["attacker_knowledge"] = "SH-B"
    payload["rotation_aware"] = False
    payload["diagnostics"]["premise"] = {"premise_holds": True}
    payload["diagnostics"]["symbol_invariance"] = {"premise_holds": False}
    payload["diagnostics"]["diagnostic_only"] = True
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("contradicts the measured" in gap for gap in audit["gaps"])


def test_tfma_sh_align_cannot_be_certified_as_the_shp_cell():
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
        top100_distinct=10,
        attacker_knowledge="SH", rotation_aware=False,
        attack_variant="SH-ALIGN",
    )
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("not the protocol SH-P cell" in gap for gap in audit["gaps"])


def test_tfma_cell_identity_fields_are_required():
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
        top100_distinct=10,
    )
    payload["diagnostics"]["top100_distinct"] = 10
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("attacker_knowledge missing" in gap for gap in audit["gaps"])
    assert any("rotation_aware missing" in gap for gap in audit["gaps"])
    assert any("grouping" in gap and "missing" in gap for gap in audit["gaps"])


def test_tfma_rotation_false_encodings_are_rejected():
    for value in (False, 0, "false", "0", "no"):
        payload = knn_payload(
            attack="tfma_after_inversion", target="llama-3.2-1b/ab",
            n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
            top100_distinct=10,
            attacker_knowledge="SH-P", rotation_aware=value,
            attack_variant="SH-P",
        )
        payload["diagnostics"]["top100_distinct"] = 10
        audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
        assert any("not true" in gap for gap in audit["gaps"]), value


def test_tfma_conflicting_grouping_records_are_rejected():
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
        top100_distinct=10,
        attacker_knowledge="SH-P", rotation_aware=True,
        attack_variant="SH-P",
    )
    payload["diagnostics"]["top100_distinct"] = 10
    payload["diagnostics"]["rotation_grouping"] = "SH-B"
    payload["diagnostics"]["premise"] = {"premise_holds": False}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("conflicting grouping" in gap for gap in audit["gaps"])


def test_tfma_declared_top100_must_match_the_measured_value():
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=10,
        top100_distinct=100,
        attacker_knowledge="SH-P", rotation_aware=True,
        attack_variant="SH-P",
    )
    payload["diagnostics"]["top100_distinct"] = 3
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("declared 100" in gap and "measured 3" in gap for gap in audit["gaps"])


def test_saturation_requires_measured_points_not_a_bare_gain():
    payload = isa_payload()
    payload["diagnostics"]["saturation"] = {"gain_pp": 0.0}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "not_converged"
    assert any("points missing" in gap for gap in audit["gaps"])


def _tfma_identity_payload(**overrides) -> dict:
    payload = knn_payload(
        attack="tfma_after_inversion", target="llama-3.2-1b/ab",
        n_prompts=0, n_tokens=0, n_types=10, reference_tokens=4_450_000,
        top100_distinct=10,
        attacker_knowledge="SH-P", rotation_aware=True,
        attack_variant="SH-P",
        **overrides,
    )
    payload["diagnostics"]["top100_distinct"] = 10
    payload["diagnostics"]["reference_types"] = 50_000
    return payload


def test_tfma_saturation_must_use_the_protocol_budget_ladder():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["saturation"] = {
        "points": [
            {"budget": 1, "top1": 0.30},
            {"budget": 2, "top1": 0.31},
        ],
        "gain_pp": 1.0,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("protocol reference-token budgets" in gap for gap in audit["gaps"])


def test_tfma_saturation_last_budget_must_equal_reference_tokens():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["saturation"] = {
        "points": [
            {"budget": 4_450_000, "top1": 0.50},
            {"budget": 10_000_000, "top1": 0.50},
        ],
        "gain_pp": 0.0,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("!= reference_tokens" in gap for gap in audit["gaps"])


def test_tfma_shb_premise_and_diagnostic_flags_use_truthy_encodings():
    payload = _tfma_identity_payload()
    payload["attack_variant"] = "SH-B"
    payload["attacker_knowledge"] = "SH-B"
    payload["rotation_aware"] = False
    payload["diagnostics"]["premise"] = {"premise_holds": True}
    payload["diagnostics"]["symbol_invariance"] = {"premise_holds": "false"}
    payload["diagnostics"]["diagnostic_only"] = "true"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("contradicts the measured" in gap for gap in audit["gaps"])
    assert any("diagnostic-only" in gap for gap in audit["gaps"])


def test_tfma_shb_is_diagnostic_even_when_its_premise_holds():
    payload = _tfma_identity_payload()
    payload["rotation_grouping"] = "SH-B"
    payload["diagnostics"]["premise"] = {"premise_holds": True}
    payload["diagnostics"]["symbol_invariance"] = {"premise_holds": True}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("diagnostic-only" in gap for gap in audit["gaps"])


def test_tfma_diagnostic_only_flag_is_rejected_for_any_variant():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["diagnostic_only"] = "true"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("diagnostic-only path cannot be certified" in gap for gap in audit["gaps"])


def test_tfma_payload_level_diagnostic_only_is_also_rejected():
    payload = _tfma_identity_payload()
    payload["diagnostic_only"] = "true"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("diagnostic-only path cannot be certified" in gap for gap in audit["gaps"])


def test_tfma_diagnostic_only_via_evidence_and_container_is_rejected():
    payload = _tfma_identity_payload()
    audit = audit_payload(
        payload, profile=PROFILE, protocol_sha256=DIGEST,
        evidence={"diagnostic_only": True},
    )
    assert any("diagnostic-only path cannot be certified" in gap for gap in audit["gaps"])
    payload2 = _tfma_identity_payload()
    payload2["diagnostics"]["diagnostic_only"] = {"enabled": True}
    audit2 = audit_payload(payload2, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("diagnostic-only path cannot be certified" in gap for gap in audit2["gaps"])


def test_tfma_saturation_budget_must_be_an_integer():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["saturation"] = {
        "points": [
            {"budget": 4_450_000, "top1": 0.5},
            {"budget": 20_000_000.9, "top1": 0.5},
        ],
        "gain_pp": 0.0,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("not an integer" in gap for gap in audit["gaps"])


def test_tfma_diagnostics_attack_variant_shb_is_a_grouping_record():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["attack_variant"] = "SH-B"
    payload["diagnostics"]["premise"] = {"premise_holds": True}
    payload["diagnostics"]["symbol_invariance"] = {"premise_holds": True}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("diagnostic-only" in gap or "SH-B" in gap for gap in audit["gaps"])


def test_tfma_grouping_description_must_match_the_declared_cell():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["grouping"] = "public stage-1 KNN deployment ids"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("SH-ALIGN" in gap for gap in audit["gaps"])


def test_tfma_nested_knowledge_and_rotation_records_are_checked():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["meta"] = {
        "attacker_knowledge": "SH-B", "rotation_aware": False,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("nested attacker_knowledge" in gap for gap in audit["gaps"])
    assert any("nested rotation_aware" in gap for gap in audit["gaps"])


def test_tfma_saturation_points_must_match_the_measured_curve():
    payload = _tfma_identity_payload()
    payload["n_streams"] = 3
    payload["n_tokens"] = 300_000
    payload["diagnostics"]["per_stream"] = [
        {"n_requests": 5000, "n_positions": 100_000} for _ in range(3)
    ]
    payload["reference_tokens"] = 10_000_000
    payload["diagnostics"]["reference_tokens"] = 10_000_000
    payload["metrics"]["occurrence_level"] = {
        "top1": {"value": 0.9, "numerator": 9, "denominator": 10}
    }
    payload["diagnostics"]["saturation"] = {
        "points": [
            {"budget": 4_450_000, "top1": 0.8},
            {"budget": 10_000_000, "top1": 0.9},
        ],
        "gain_pp": 10.0,
    }
    payload["diagnostics"]["saturation_curve"] = {
        4_450_000: 0.8, 10_000_000: 0.9,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any(
        "contradicts the measured saturation_curve" in gap
        or "final TFMA point" in gap
        or "no measured saturation_curve" in gap
        for gap in audit["gaps"]
    )
    payload["diagnostics"]["saturation_curve"][4_450_000] = 0.7
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any(
        "contradicts the measured saturation_curve" in gap for gap in audit["gaps"]
    )


def test_tfma_stream_count_and_requests_are_fixed():
    payload = _tfma_identity_payload()
    payload["n_streams"] = 4
    payload["diagnostics"]["per_stream"] = [
        {"n_requests": 5000, "n_positions": 100_000} for _ in range(4)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("fixed 3 streams" in gap for gap in audit["gaps"])
    payload["n_streams"] = 3
    payload["diagnostics"]["per_stream"] = [
        {"n_requests": 4999, "n_positions": 100_000} for _ in range(3)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("fixed 5000" in gap for gap in audit["gaps"])


def test_tfma_requires_per_stream_evidence():
    payload = _tfma_identity_payload()
    payload["n_streams"] = 3
    payload["n_prompts"] = 15_000
    payload["n_tokens"] = 300_000
    payload["reference_tokens"] = 10_000_000
    payload["diagnostics"]["reference_tokens"] = 10_000_000
    payload["diagnostics"]["per_stream"] = [
        {"n_requests": 5000, "n_positions": 100_000, "n_types": 1000}
        for _ in range(3)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any("tfma stream " in gap for gap in audit["gaps"])
    payload["diagnostics"]["per_stream"][1]["n_positions"] = 1
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("tokens < 100000" in gap for gap in audit["gaps"])


def test_tfma_observed_tokens_must_be_at_most_reference_over_100():
    payload = _tfma_identity_payload()
    payload["n_streams"] = 3
    payload["n_tokens"] = 150_000
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("observed per stream" in gap for gap in audit["gaps"])


def test_isa_lr_must_follow_validation_selection():
    payload = isa_payload()
    payload["metrics"]["lr_by_layer"] = {
        1: 1e-3, 5: 1e-3, 10: 1e-3, 15: 1e-3,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any("lr_by_layer" in gap for gap in audit["gaps"])
    payload["metrics"]["lr_by_layer"][10] = 1e-2
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("selected validation lr" in gap for gap in audit["gaps"])


def test_isa_can_select_small_rate_from_validation_grid():
    from obfuscation_attacks.evaluation.protocol_audit import _isa_validation_gaps
    from obfuscation_attacks.evaluation.real_battery import _isa_lr_by_layer

    payload = isa_payload()
    selection = payload["selected_on_validation"]
    selection["lr"] = 1e-4
    for trial in selection["trials"]:
        trial["token_top1"] = .8 if trial["lr"] == 1e-4 and trial["lambda_dm"] == 0 else .5
    payload["metrics"]["lr_by_layer"] = {layer: 1e-4 for layer in payload["layers"]}
    assert _isa_validation_gaps(payload, payload["diagnostics"]) == []
    assert set(_isa_lr_by_layer(1e-4, payload["layers"]).values()) == {1e-4}
    selection["trials"] = selection["trials"][:-1]
    assert any("trials" in gap for gap in _isa_validation_gaps(payload, payload["diagnostics"]))


def test_isa_executed_hyperparameters_must_match_the_validation_selection():
    payload = isa_payload()
    payload["layers"] = [1, 2, 3, 4]
    payload["metrics"]["lr"] = 0.1
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("executed lr" in gap for gap in audit["gaps"])
    payload["metrics"]["lr"] = 1e-3
    payload["metrics"]["lambda_dm"] = 0.0
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any("executed lr" in gap for gap in audit["gaps"])


def test_isa_zero_actual_steps_and_all_failed_are_rejected():
    payload = isa_payload()
    payload["optimizer_steps_total"] = 0
    payload["failed_observations"] = 1000
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("0 actual optimizer steps" in gap for gap in audit["gaps"])
    assert any("every observation failed" in gap for gap in audit["gaps"])


def test_tfma_reference_types_floor_is_enforced():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["reference_types"] = 10
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("reference_types" in gap for gap in audit["gaps"])


def test_tfma_diagnostics_container_is_not_a_diagnostic_marker():
    payload = _tfma_identity_payload()
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any(
        "diagnostic-only path cannot be certified" in gap
        for gap in audit["gaps"]
    )


def test_tfma_diagnostic_key_aliases_are_detected():
    for key in ("Diagnostic_Only", "diagnostic_only ", "diagnostic only",
                "diagnosticOnly", "diagnostic-only", "is_diagnostic_mode"):
        payload = _tfma_identity_payload()
        payload["diagnostics"][key] = True
        audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
        assert any(
            "diagnostic-only path cannot be certified" in gap
            for gap in audit["gaps"]
        ), key


def test_tfma_bare_and_unicode_grouping_descriptions_are_checked():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["grouping"] = "SH-ALIGN"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("SH-ALIGN" in gap for gap in audit["gaps"])
    payload = _tfma_identity_payload()
    payload["diagnostics"]["grouping"] = "stage‑1 KNN deployment ids"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("SH-ALIGN" in gap for gap in audit["gaps"])


def test_tfma_sh_align_grouping_cannot_certify_as_shp():
    payload = _tfma_identity_payload()
    payload["attack_variant"] = "SH-ALIGN"
    payload["diagnostics"]["rotation_grouping"] = "SH-ALIGN"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("SH-ALIGN is not the protocol SH-P cell" in gap for gap in audit["gaps"])


def test_tfma_reference_tokens_measured_record_is_required():
    payload = _tfma_identity_payload()
    payload["diagnostics"].pop("reference_tokens", None)
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("measured diagnostics value missing" in gap for gap in audit["gaps"])


def test_tfma_oversized_saturation_budget_fails_closed():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["reference_tokens"] = 4_450_000
    payload["diagnostics"]["saturation"] = {
        "points": [
            {"budget": 4_450_000, "top1": 0.5},
            {"budget": 10 ** 400, "top1": 0.5},
        ],
        "gain_pp": 0.0,
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("not representable" in gap for gap in audit["gaps"])


def test_tfma_numeric_diagnostic_only_marker_is_rejected():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["diagnostic_only"] = 2
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("diagnostic-only path cannot be certified" in gap for gap in audit["gaps"])


def test_tfma_empty_diagnostic_only_marker_is_still_a_marker():
    for value in ({}, [], " "):
        payload = _tfma_identity_payload()
        payload["diagnostics"]["diagnostic_only"] = value
        audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
        assert any(
            "diagnostic-only path cannot be certified" in gap
            for gap in audit["gaps"]
        ), value


def test_tfma_homoglyph_shb_label_is_detected():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["attack_variant"] = "SH‑B"
    payload["diagnostics"]["premise"] = {"premise_holds": True}
    payload["diagnostics"]["symbol_invariance"] = {"premise_holds": True}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any(
        "non-canonical grouping" in gap or "diagnostic-only" in gap
        for gap in audit["gaps"]
    )


def test_tfma_saturation_ladder_cannot_skip_a_fixed_tier():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["saturation"] = {
        "points": [
            {"budget": 4_450_000, "top1": 0.9},
            {"budget": 20_000_000, "top1": 0.9},
        ],
        "gain_pp": 0.0,
    }
    payload["reference_tokens"] = 20_000_000
    payload["diagnostics"]["reference_tokens"] = 20_000_000
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("contiguous prefix" in gap for gap in audit["gaps"])


def test_tfma_decorated_nested_shb_label_is_rejected():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["tfma"] = {"attack_variant": "SH-B (raw)"}
    payload["diagnostics"]["premise"] = {"premise_holds": True}
    payload["diagnostics"]["symbol_invariance"] = {"premise_holds": True}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any(
        "non-canonical grouping" in gap or "diagnostic-only" in gap
        for gap in audit["gaps"]
    )


def test_tfma_nested_rotation_grouping_is_read():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["tfma"] = {"rotation_grouping": "SH-B"}
    payload["diagnostics"]["premise"] = {"premise_holds": True}
    payload["diagnostics"]["symbol_invariance"] = {"premise_holds": True}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("diagnostic-only" in gap or "SH-B" in gap for gap in audit["gaps"])


def test_tfma_fractional_diagnostic_reference_tokens_are_rejected():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["reference_tokens"] = 4_450_000.9
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("reference_tokens: declared" in gap for gap in audit["gaps"])


def test_tfma_reference_tokens_records_must_agree():
    payload = _tfma_identity_payload()
    payload["reference_tokens"] = 10_000_000
    payload["diagnostics"]["reference_tokens"] = 4_450_000
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("reference_tokens: declared" in gap for gap in audit["gaps"])


def test_tfma_conflicting_identity_records_are_rejected():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["attacker_knowledge"] = "SH-B"
    payload["diagnostics"]["rotation_aware"] = False
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("conflicting attacker_knowledge" in gap for gap in audit["gaps"])
    assert any("conflicting rotation_aware" in gap for gap in audit["gaps"])


def test_tfma_diagnostics_algorithm_variant_is_not_a_grouping_record():
    payload = _tfma_identity_payload()
    payload["diagnostics"]["attack_variant"] = "embedding_level"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert not any("conflicting grouping" in gap for gap in audit["gaps"])


def test_tfma_measured_top100_record_is_required():
    payload = _tfma_identity_payload()
    payload["diagnostics"].pop("top100_distinct")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("measured value missing" in gap for gap in audit["gaps"])


def test_tfma_non_canonical_grouping_labels_are_rejected():
    payload = _tfma_identity_payload()
    payload["attack_variant"] = "SH_B"
    payload["diagnostics"]["rotation_grouping"] = "SH_B"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("tfma SH-B" in gap for gap in audit["gaps"])
    payload = _tfma_identity_payload()
    payload["attack_variant"] = "embedding_level"
    payload["diagnostics"]["rotation_grouping"] = "embedding_level"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("non-canonical grouping" in gap for gap in audit["gaps"])


def test_tfma_top_level_rotation_grouping_is_read():
    payload = _tfma_identity_payload()
    payload["rotation_grouping"] = "SH-B"
    payload["diagnostics"]["premise"] = {"premise_holds": False}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("tfma SH-B" in gap for gap in audit["gaps"])


def test_stream_attack_units_are_one_row_per_request():
    payload = knn_payload(
        attack="sda", target="llama-3.2-1b/ab",
        n_prompts=3, n_keys=3, n_requests=3, n_tokens=30, n_types=10,
        n_streams=3,
    )
    payload["metrics"] = {
        "token_accuracy": {"value": 0.5, "numerator": 15, "denominator": 30},
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 5, "count": 10} for index in range(3)
    ]
    gaps = _evidence_gaps(payload, payload["diagnostics"], "sda")
    assert not any("per-unit rows" in gap for gap in gaps)


def test_an_unrelated_premise_code_cannot_mark_an_applicable_cell_na():
    payload = knn_payload(
        attack_id="sda", arm="ab",
        metrics={"status": "not_applicable", "reason": "ciphertext-only pipeline"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    payload["diagnostics"]["not_applicable_premise_code"] = "no_generation_path"
    payload["diagnostics"]["not_applicable_evidence"] = (
        "SDA deciphers symbols and does not generate free text."
    )
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("not a premise of sda" in gap for gap in audit["gaps"])


def test_weight_layer_rule_is_not_applied_to_state_attacks():
    payload = vma_payload(target="llama-3.2-1b-base/ab", layers=[1, 5, 10, 15])
    assert _layer_gaps(payload, payload["diagnostics"], attack="vma_h_recovered_prefix") == []


def test_positive_control_cannot_lower_its_own_floor():
    payload = vma_payload()
    payload["diagnostics"]["positive_control"]["top1"] = 0.10
    payload["diagnostics"]["positive_control"]["required_top1"] = 0.10
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("control:") for gap in audit["gaps"])


def test_bootstrap_below_the_2000_floor_is_below_budget():
    payload = knn_payload()
    payload["ci"]["top1_micro"]["n_boot"] = 500
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("n_boot" in gap and "2000" in gap for gap in audit["gaps"])


def test_missing_bootstrap_block_is_below_budget():
    payload = knn_payload()
    payload.pop("ci")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any(gap.startswith("ci:") for gap in audit["gaps"])


def test_protocol_hash_drift_is_rejected():
    audit = audit_payload(knn_payload(), profile=PROFILE, protocol_sha256="0" * 64)
    assert audit["verdict"] == "below_budget"
    assert any("protocol" in gap for gap in audit["gaps"])


def test_below_budget_names_the_field():
    audit = audit_payload(knn_payload(n_prompts=200), profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("n_prompts" in gap for gap in audit["gaps"])


def test_failed_control_is_control_failed():
    payload = knn_payload()
    payload["diagnostics"]["positive_control_passed"] = False
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "control_failed"


def test_proposal_mode_is_below_budget():
    audit = audit_payload(
        knn_payload(candidate_mode="proposal"), profile=PROFILE, protocol_sha256=DIGEST
    )
    assert audit["verdict"] == "below_budget"
    assert any("candidate" in gap for gap in audit["gaps"])


def test_missing_evidence_is_below_budget():
    payload = knn_payload()
    payload["diagnostics"].pop("per_unit_predictions")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("evidence" in gap for gap in audit["gaps"])


def test_not_applicable_is_not_overwritten():
    payload = knn_payload(
        n_prompts=0,
        metrics={"status": "not_applicable", "reason": "text upload has no vector"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "task_not_supported"
    payload["arm"] = "inferdpt"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "not_applicable"
    assert "text upload" in audit["reason"]


def test_metrics_without_fractions_are_rejected():
    audit = audit_payload(
        knn_payload(metrics={"status": "ok"}), profile=PROFILE, protocol_sha256=DIGEST
    )
    assert audit["verdict"] == "below_budget"
    assert any("numerator/denominator" in gap for gap in audit["gaps"])


def test_self_reported_control_with_a_low_top1_is_rejected():
    payload = vma_payload()
    payload["diagnostics"]["positive_control"] = {
        "name": "known_construction", "top1": 0.1, "required_top1": 0.9,
    }
    payload["diagnostics"]["positive_control_passed"] = True
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("control" in gap and "0.1" in gap for gap in audit["gaps"])


def test_task_not_supported_is_rejected_for_an_applicable_cell():
    payload = knn_payload(
        arm="ab",
        metrics={"status": "not_applicable", "reason": "claimed unsupported task"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "task_not_supported"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("applicable" in gap for gap in audit["gaps"])


def test_unknown_attack_id_is_fail_closed():
    payload = knn_payload(attack="nn_knn.cosine")
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("no paper-v1 floor" in gap for gap in audit["gaps"])


def test_not_applicable_cannot_hide_bad_binding():
    payload = knn_payload(metrics={"status": "not_applicable", "reason": "missing artifact"})
    payload["diagnostics"]["not_applicable_premise"] = "task_not_supported"
    payload["protocol_sha256"] = "0" * 64
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"


def test_real_premise_absent_requires_evidence():
    payload = knn_payload(metrics={"status": "not_applicable", "reason": "no vector"})
    payload["diagnostics"]["not_applicable_premise"] = "real_premise_absent"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    payload["diagnostics"]["not_applicable_evidence"] = "text upload has no vector"
    payload["diagnostics"]["not_applicable_premise_code"] = "no_request_stream"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "not_applicable"


def test_attack_not_valid_for_arm_must_match_the_applicability_table():
    payload = knn_payload(
        arm="ab", metrics={"status": "not_applicable", "reason": "claimed"},
    )
    payload["diagnostics"]["not_applicable_premise"] = "attack_not_valid_for_arm"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"

    payload["arm"] = "inferdpt"
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "not_applicable"


def test_nested_metric_denominator_is_reconciled():
    payload = knn_payload()
    payload["metrics"] = {
        "status": "ok",
        "stage1_permuted_id": {
            "top1": {"value": 1.0, "numerator": 500, "denominator": 500}
        },
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("denominator" in gap for gap in audit["gaps"])


def test_per_unit_hits_must_match_a_primary_numerator():
    payload = knn_payload()
    payload["metrics"] = {
        "status": "ok",
        "stage1_permuted_id": {
            "top1": {"value": 1.0, "numerator": 20000, "denominator": 20000}
        },
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 0, "count": 20} for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("per-unit hits" in gap for gap in audit["gaps"])


def test_weight_attack_denominators_use_n_types():
    payload = knn_payload(
        attack="vma_w", n_tokens=0, n_prompts=0, n_types=1000,
        target="llama-3.2-1b-base/aloepri_weights",
    )
    payload["layers"] = list(range(16))
    payload["missing_weight_layers"] = []
    payload["metrics"] = {
        "status": "ok",
        "mapping_asr": {
            f"top{k}": {"value": 1.0, "numerator": 5000, "denominator": 5000}
            for k in (1, 5, 10, 100)
        },
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 0, "count": 0, "rank": []} for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("denominator" in gap for gap in audit["gaps"])

    payload["metrics"]["mapping_asr"] = {
        f"top{k}": {"value": 1.0, "numerator": 1000, "denominator": 1000}
        for k in (1, 5, 10, 100)
    }
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]} for index in range(1000)
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "audited_paper_scale", audit


def test_per_unit_evidence_must_reconcile_with_the_declared_size():
    payload = knn_payload()
    payload["diagnostics"]["per_unit_predictions"] = [
        {"unit": 0, "hits": 500, "count": 500}
    ]
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("per-unit counts" in gap for gap in audit["gaps"])


def test_primary_metric_denominator_must_match_the_declared_size():
    payload = knn_payload()
    payload["metrics"] = {
        "status": "ok",
        "top1": {"value": 1.0, "numerator": 500, "denominator": 500},
    }
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("denominator" in gap for gap in audit["gaps"])


def test_vma_recovered_and_observed_are_different_variants():
    good = audit_payload(vma_payload(), profile=PROFILE, protocol_sha256=DIGEST)
    assert good["verdict"] == "audited_paper_scale"
    observed = vma_payload(attack="vma_h_observed_prefix", prefix="observed")
    assert audit_payload(observed, profile=PROFILE, protocol_sha256=DIGEST)["verdict"] == (
        "audited_paper_scale"
    )
    wrong = vma_payload(prefix="observed")
    audit = audit_payload(wrong, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("prefix" in gap for gap in audit["gaps"])


def test_isa_reads_the_real_step_count_and_saturation():
    assert audit_payload(isa_payload(), profile=PROFILE, protocol_sha256=DIGEST)["verdict"] == (
        "audited_paper_scale"
    )
    low = audit_payload(isa_payload(executed_steps=20000), profile=PROFILE, protocol_sha256=DIGEST)
    assert low["verdict"] == "below_budget"
    assert any("steps" in gap for gap in low["gaps"])
    unsat = isa_payload()
    unsat["diagnostics"]["saturation"]["gain_pp"] = 5.0
    audit = audit_payload(unsat, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "not_converged"
    old = isa_payload()
    del old["metrics"]["trajectory_mode"]
    audit = audit_payload(old, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("trajectory_mode" in gap for gap in audit["gaps"])


def test_weight_attack_requires_a_real_type_sample_and_keeps_its_name():
    weight = knn_payload(attack="vma_w", n_types=0)
    audit = audit_payload(weight, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("n_types" in gap for gap in audit["gaps"])
    least = knn_payload(
        attack="least_squares", n_types=1000, n_tokens=0, n_prompts=0,
    )
    least["target"] = "llama-3.2-1b-base/ab"
    least["layers"] = list(range(16))
    least["diagnostics"]["mixing_group_residuals"] = {
        f"layer{layer}.self_attn.{projection}_proj": {"relative_residual": 0.1}
        for layer in range(16) for projection in ("q", "k", "v")
    }
    least["diagnostics"]["mixing_skipped_groups"] = []
    least["metrics"] = {
        "mapping_asr": {
            f"top{k}": {"value": 1.0, "numerator": 1000, "denominator": 1000}
            for k in (1, 5, 10, 100)
        }
    }
    least["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(1000)
    ]
    assert audit_payload(least, profile=PROFILE, protocol_sha256=DIGEST)["verdict"] == (
        "audited_paper_scale"
    )
    least["diagnostics"]["attack_variant"] = "dense_mixing_composed_engineering_variant"
    audit = audit_payload(least, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("deployment-level composed positive control" in gap for gap in audit["gaps"])
    del least["diagnostics"]["attack_variant"]
    result = least["diagnostics"]["mixing_group_residuals"].pop("layer3.self_attn.k_proj")
    audit = audit_payload(least, profile=PROFILE, protocol_sha256=DIGEST)
    assert any("layer3.self_attn.k_proj" in gap for gap in audit["gaps"])
    least["diagnostics"]["mixing_group_residuals"]["layer3.self_attn.k_proj"] = result
    least["layers"].remove(7)
    audit = audit_payload(least, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("neither used nor recorded" in gap for gap in audit["gaps"])


def test_ima_requires_tokens_docs_and_seeds():
    base = dict(
        attack="ima_ridge", n_tokens=1000, n_train_tokens=1_000_000, n_train_docs=8192,
        n_test_docs=1000, n_seeds=3,
        ridge_lambda=0.1,
        ridge_validation={"n_docs": 1024, "split": "validation", "metric": "token_top1",
                          "data_sha256": "b" * 64, "selected_lambda": 0.1,
                          "trials": [{"lambda": x, "token_top1": .8 if x == .1 else .5}
                                     for x in (.01, .1, 1., 10.)]},
        training_seeds=[17, 29, 43],
        metrics={
            "stage1_permuted_id": {
                f"top{k}": {"value": 1.0, "numerator": 1000, "denominator": 1000}
                for k in (1, 5, 10, 100)
            }
        },
        diagnostics={
            **_evidence(),
            "saturation": _paired_saturation(40, 80, 50, 51),
        },
    )
    ima = knn_payload(**base)
    ima["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(1000)
    ]
    assert audit_payload(ima, profile=PROFILE, protocol_sha256=DIGEST)[
        "verdict"
    ] == "audited_paper_scale"
    for field, value in (
        ("n_train_tokens", 100_000), ("n_train_docs", 1000),
        ("n_seeds", 1),
    ):
        audit = audit_payload(knn_payload(**{**base, field: value}), profile=PROFILE,
                              protocol_sha256=DIGEST)
        assert audit["verdict"] == "below_budget", field
    neural = knn_payload(**{**base, "attack": "ima_neural", "n_epochs": 3})
    assert any("n_epochs" in gap for gap in audit_payload(neural, profile=PROFILE, protocol_sha256=DIGEST)["gaps"])
    unsat = knn_payload(**{**base, "diagnostics": _evidence()})
    unsat["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(1000)
    ]
    assert audit_payload(unsat, profile=PROFILE, protocol_sha256=DIGEST)["verdict"] == (
        "not_converged"
    )


def test_linkage_and_tfma_floors():
    linkage = knn_payload(
        attack="linkage_exact", n_positives=5000, n_negatives=100_000, n_streams=3,
    )
    linkage["metrics"]["test"] = {"positives": 5000, "negatives": 100_000}
    linkage["metrics"]["exact_test"] = {
        "value": 0.5, "numerator": 2500, "denominator": 5000,
    }
    audit = audit_payload(linkage, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("test.positives" in gap for gap in audit["gaps"])
    tfma = knn_payload(attack="tfma", reference_tokens=4_450_000, n_types=50_000,
                       top100_distinct=7)
    audit = audit_payload(tfma, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("top100" in gap for gap in audit["gaps"])


def test_linkage_scored_test_denominator_must_match_the_budget():
    payload = knn_payload(attack="linkage_exact", n_streams=3)
    payload["metrics"]["test"] = {"positives": 10000, "negatives": 100000}
    payload["metrics"]["exact_test"] = {"numerator": 5000, "denominator": 9999}
    audit = audit_payload(payload, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("exact_test.denominator" in gap for gap in audit["gaps"])


def test_text_and_output_floors_require_real_generation():
    text = knn_payload(
        attack="text_bayes", n_test=1000,
        metrics={
            "accuracy": {
                f"top{k}": {"value": 1.0, "numerator": 1000, "denominator": 1000}
                for k in (1, 5, 10, 100)
            }
        },
    )
    text["diagnostics"]["per_unit_predictions"] = [
        {"unit": index, "hits": 1, "count": 1, "rank": [1]}
        for index in range(1000)
    ]
    assert audit_payload(text, profile=PROFILE, protocol_sha256=DIGEST)["verdict"] == (
        "audited_paper_scale"
    )
    output = knn_payload(attack="output_pii", n_prompts=1000, generation_record=None)
    audit = audit_payload(output, profile=PROFILE, protocol_sha256=DIGEST)
    assert audit["verdict"] == "below_budget"
    assert any("generation" in gap for gap in audit["gaps"])
