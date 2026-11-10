from copy import deepcopy

import pytest

from obfuscation_attacks.evaluation.protocol_audit import SATURATION_REQUIRED, _saturation_gap


def evidence():
    return {"saturation": {
        "points": [
            {"budget": 10, "top1": .5, "split": "validation", "model_sha256": "a" * 64, "validation_data_sha256": "b" * 64},
            {"budget": 20, "top1": .5, "split": "validation", "model_sha256": "a" * 64, "validation_data_sha256": "b" * 64},
        ],
        "gain_pp": 0.,
        "paired_units": [
            {"unit": "u1", "cluster": "s1", "count": 100, "hits_before": 50, "hits_after": 50},
            {"unit": "u2", "cluster": "s2", "count": 100, "hits_before": 50, "hits_after": 50},
        ],
    }}


@pytest.mark.parametrize("attack", ["sda", "sda2021", "sda2023"])
def test_sda_saturation_requires_paired_validation_evidence(attack):
    assert attack in SATURATION_REQUIRED
    diag = evidence()
    assert _saturation_gap(diag, attack=attack) is None
    diag["saturation"]["paired_units"] = []
    assert "paired" in _saturation_gap(diag, attack=attack)


def test_saturation_rejects_duplicate_budget_and_different_inputs():
    diag = evidence()
    diag["saturation"]["points"][1]["budget"] = 10
    assert "budgets" in _saturation_gap(diag, attack="sda")
    diag = evidence()
    diag["saturation"]["points"][1]["model_sha256"] = "c" * 64
    assert "model_sha256" in _saturation_gap(diag, attack="sda")


def test_saturation_rejects_interval_that_allows_large_gain():
    diag = evidence()
    diag["saturation"]["paired_units"][1]["hits_after"] = 54
    diag["saturation"]["points"][1]["top1"] = .52
    diag["saturation"]["gain_pp"] = 2.0
    assert "CI upper" in _saturation_gap(diag, attack="sda")


def test_saturation_rejects_forged_point_and_test_selection():
    diag = evidence()
    diag["saturation"]["paired_units"][0]["hits_after"] = 49
    assert "contradict" in _saturation_gap(diag, attack="sda")
    diag = evidence()
    diag["saturation"]["points"][0]["split"] = "test"
    assert "validation" in _saturation_gap(diag, attack="sda")
