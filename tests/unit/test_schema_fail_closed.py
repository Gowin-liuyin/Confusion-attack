"""Acceptance tests for fail-closed result schemas (F08)."""

from __future__ import annotations

import copy

import pytest

from obfuscation_attacks.evaluation.attack_model import result_payload, validate_payload

pytestmark = pytest.mark.unit


def _valid_payload() -> dict:
    return result_payload(
        attack_name="nn_knn.euclidean",
        scheme_name="ab",
        metrics={"top1": {"value": 0.9, "numerator": 9, "denominator": 10}},
        positive_control={"top1": 1.0},
        diagnostics={"positive_control_passed": True},
        n_prompts=10,
        n_tokens=40,
    )


def test_valid_payload_passes_but_is_not_self_declared_eligible():
    payload = _valid_payload()
    assert validate_payload(payload) == []
    assert payload["main_table_eligible"] is False


def test_negative_counts_are_rejected():
    payload = _valid_payload()
    payload["n_prompts"] = -1
    assert any("n_prompts" in problem for problem in validate_payload(payload))


def test_failed_control_cannot_be_eligible():
    payload = _valid_payload()
    payload["diagnostics"]["positive_control_passed"] = False
    payload["positive_control"] = {"top1": 0.1}
    payload["main_table_eligible"] = True
    problems = validate_payload(payload)
    assert any("control" in problem for problem in problems)
    assert any("eligible" in problem for problem in problems)


def test_numerator_must_not_exceed_denominator():
    payload = _valid_payload()
    payload["metrics"]["top1"] = {"value": 2.0, "numerator": 2, "denominator": 1}
    problems = validate_payload(payload)
    assert any("numerator" in problem or "denominator" in problem for problem in problems)


def test_value_must_match_the_ratio():
    payload = _valid_payload()
    payload["metrics"]["top1"] = {"value": 0.1, "numerator": 9, "denominator": 10}
    problems = validate_payload(payload)
    assert any("value" in problem for problem in problems)


def test_self_declared_eligibility_without_audit_is_rejected():
    payload = _valid_payload()
    payload["main_table_eligible"] = True
    payload["diagnostics"]["positive_control_passed"] = True
    problems = validate_payload(payload)
    assert any("audit" in problem or "eligible" in problem for problem in problems)


def test_a_hand_written_audit_block_cannot_self_certify_eligibility():
    payload = _valid_payload()
    payload["main_table_eligible"] = True
    payload["audit"] = {
        "verdict": "audited_paper_scale",
        "auditor": "protocol_auditor",
        "protocol_sha256": "a" * 64,
    }
    problems = validate_payload(payload)
    assert any("self-certify" in problem or "eligible" in problem for problem in problems)


def test_missing_required_field_is_rejected():
    payload = _valid_payload()
    broken = copy.deepcopy(payload)
    broken.pop("data_sha256")
    assert any("data_sha256" in problem for problem in validate_payload(broken))


def test_unknown_enums_are_rejected():
    payload = _valid_payload()
    payload["candidate_mode"] = "magic"
    assert any("candidate_mode" in problem for problem in validate_payload(payload))
    payload = _valid_payload()
    payload["observation_surface"] = "S9"
    assert any("observation_surface" in problem for problem in validate_payload(payload))
