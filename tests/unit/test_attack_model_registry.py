"""The attack-model registry must stay pinned to the plan's catalogue.

``evaluation/attack_model.py`` claims in its docstring that "a test fails when a
registered attack has no spec, or when a spec names an unknown attack". Until this
file existed that claim was untrue, so a new attack could be merged without
declaring which surface it observed or what the attacker knew -- exactly the
discipline the plan's §1.4 / App. A forbids. These tests make the claim real:

* registry <-> spec bijection (implemented attacks only);
* every spec carries a plan §3 sub-capability code (K0 / KW / KI / ...) and the
  code matches the catalogue, transcribed literally below;
* the plan's PB-required codes are present in the payload and validate;
* an appendix-only attack can never come out main-table eligible, and a
  ``proposal``-mode run cannot either (plan §1.7-4);
* all 14 plan classes are registered.
"""

from __future__ import annotations

import pytest

from obfuscation_attacks.attacks import available_attacks, load_all
from obfuscation_attacks.evaluation.attack_model import (
    ATTACK_MODELS,
    MAIN_TABLE_KNOWLEDGE,
    PLAN_SUBCAPABILITY,
    SUB_CAPABILITIES,
    result_payload,
    spec_for,
    validate_payload,
)

pytestmark = pytest.mark.unit

#: Plan §3 catalogue, "知识" column, transcribed by hand from
#: ``docs/privacy_test_plan_20261106.md`` lines 447-484. Attack name -> code.
PLAN_CATALOGUE = {
    # A1 KNN (K0), A4 adaptive KNN (KA)
    "nn_knn.cosine": "K0",
    "nn_knn.euclidean": "K0",
    "nn_knn.adaptive_wk": "KA",
    "nn_knn.adaptive_oracle": "KA",
    # A2 learned inversion (KI/KP), A5 transfer (KI)
    "ima.cross_instance": "KI",
    "ima.cross_checkpoint": "KI",
    # B1/B2 ISA (K0), B3 VMA-H (K0), B4 BeamClean (KI)
    "isa.er": "K0",
    "isa.tbs": "K0",
    "isa.aloepri_state": "K0",
    "vma.autoregressive": "K0",
    "vma.osnip_projection": "K0",
    "beamclean.gaussian": "KI",
    "beamclean.laplacian": "KI",
    # C1 VMA-W (KW), C2 Gate-IA (KW), C3 Attn-IA (KW)
    "vma.weight_rowsort": "KW",
    "ia.gate": "KW",
    "ia.attn": "KW",
    "ia.ednn": "K0",
    # D1/D2 same-key IMA (KP), D4 cross-key (KP+EXT-K), D5 mixed (KP), D6 adaptive (KA+KP)
    "ima.classify": "KP",
    "ima.seq2seq": "KP",
    "ima.qwen2_aloepri": "KP",
    "ima.cross_key": "KP+EXT-K",
    "ima.mixed_key": "KP",
    "ima.adaptive": "KA+KP",
    # E1/E2/E5 TFMA (K0), E3 SDA (K0), E6/E7 linkage (K0)
    "tfma.independent": "K0",
    "tfma.global_assignment": "K0",
    "tfma.per_key": "K0",
    "sda.supervised_substitution_diagnostic": "KP",
    "sda.sda2021": "K0",
    "sda.sda2023": "K0",
    "linkage.exact": "K0",
    "linkage.approximate": "K0",
    # F1 output TFMA (K0/KP), F2 PII (K0/KP), F3 LM recon (K0)
    "tfma.output": "K0",
    "pii.extract": "K0",
    "lm_recon.ngram": "K0",
    "lm_recon.local_lm": "K0",
    # A3 Vec2Text (K0)
    "vec2text.iterative": "K0",
}


def test_every_registered_attack_declares_a_model():
    load_all()
    registered = set(available_attacks())
    missing = sorted(registered - set(ATTACK_MODELS))
    assert not missing, f"registered attacks without an attack model: {missing}"


def test_every_implemented_spec_is_registered():
    load_all()
    registered = set(available_attacks())
    ghosts = sorted(
        name for name, spec in ATTACK_MODELS.items() if spec.implemented and name not in registered
    )
    assert not ghosts, f"specs claim implemented=True but no attack class exists: {ghosts}"


def test_subcapability_matches_the_plan_catalogue():
    """The K0/KW/KI code recorded for each attack must equal the plan's §3 table."""
    mismatched = {
        name: (PLAN_SUBCAPABILITY.get(name), code)
        for name, code in PLAN_CATALOGUE.items()
        if PLAN_SUBCAPABILITY.get(name) != code
    }
    assert not mismatched, f"sub-capability drift vs plan §3: {mismatched}"
    undocumented = sorted(set(ATTACK_MODELS) - set(PLAN_CATALOGUE))
    assert not undocumented, f"attacks with no plan §3 catalogue row: {undocumented}"


def test_every_spec_has_a_valid_subcapability_code():
    bad = {
        name: PLAN_SUBCAPABILITY.get(name)
        for name in ATTACK_MODELS
        if PLAN_SUBCAPABILITY.get(name) not in SUB_CAPABILITIES
    }
    assert not bad, f"invalid sub-capability codes: {bad}"


def test_payload_carries_the_subcapability_and_validates():
    payload = result_payload(
        attack_name="nn_knn.cosine",
        scheme_name="plaintext",
        metrics={"top1": 1.0},
        n_prompts=1,
        n_tokens=4,
    )
    assert payload["knowledge_subcapability"] == "K0"
    assert validate_payload(payload) == []


def test_appendix_only_attack_can_never_be_main_table_eligible():
    """KP/KA/EXT-K attacks are appendix material regardless of their flags (plan D8)."""
    for name in ("ima.classify", "ima.qwen2_aloepri", "ima.cross_key", "ima.adaptive"):
        payload = result_payload(
            attack_name=name,
            scheme_name="aloepri",
            metrics={"top1": 0.99},
            n_prompts=1,
            n_tokens=4,
        )
        assert payload["main_table_eligible"] is False, name
        assert payload["attacker_knowledge"] in {"KP", "KA", "EXT-K"}


def test_proposal_mode_is_never_main_table_eligible():
    payload = result_payload(
        attack_name="tfma.independent",
        scheme_name="ab",
        metrics={},
        overrides={"candidate_mode": "proposal"},
    )
    assert payload["main_table_eligible"] is False
    assert payload["candidate_mode"] == "proposal"


def test_all_fourteen_plan_classes_are_registered():
    classes = {spec.class_id for spec in ATTACK_MODELS.values()}
    expected = {f"C{i:02d}" for i in range(1, 15)}
    assert classes == expected, f"missing classes: {sorted(expected - classes)}"


def test_main_table_knowledge_set_matches_plan_d1():
    assert MAIN_TABLE_KNOWLEDGE == {"SH", "SH-P"}
    for name, spec in ATTACK_MODELS.items():
        if spec.main_table:
            assert spec.attacker_knowledge in MAIN_TABLE_KNOWLEDGE, name
        if spec.attacker_knowledge in {"KP", "KA", "EXT-K"}:
            assert spec.main_table is False, name


def test_spec_for_falls_back_to_family():
    spec = spec_for("ia.gate")
    assert spec.class_id == "C08"
    with pytest.raises(KeyError):
        spec_for("does.not.exist")
