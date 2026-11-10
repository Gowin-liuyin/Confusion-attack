"""Acceptance tests for the eight-arm declaration matrix (W5)."""

from __future__ import annotations

import pytest

from obfuscation_attacks.evaluation.paper_profile import (
    PAPER_ARMS,
    PreflightError,
    clear_faithful_entries,
    preflight_formal_start,
    require_profile,
)
from obfuscation_attacks.schemes.arms import (
    ARM_DECLARATIONS,
    arm_matrix,
    manifest_arms,
    text_upload_arms,
    vector_upload_arms,
)

pytestmark = pytest.mark.unit


def test_all_eight_arms_are_declared_with_real_capabilities():
    assert set(ARM_DECLARATIONS) == set(PAPER_ARMS)
    for arm, declaration in ARM_DECLARATIONS.items():
        cap = declaration.capability()
        assert cap["upload"] in {"text", "ids", "vectors", "none"}
        assert cap["tasks"]
        assert cap["server_visible"]
        assert cap["randomization"]
        assert cap["key_mechanism"]
        assert cap["status"] in {"implemented", "resource_blocked", "unverified"}
        assert declaration.name == arm


def test_text_arms_never_invent_a_vector_upload():
    assert "inferdpt" in text_upload_arms()
    assert "inferdpt" not in vector_upload_arms()
    assert ARM_DECLARATIONS["inferdpt"].upload == "text"
    assert ARM_DECLARATIONS["sentinellms"].upload in {"ids", "text"}


def test_classification_only_arms_do_not_claim_generation():
    for arm in ("sentinellms", "dp_forward"):
        assert "generation" not in ARM_DECLARATIONS[arm].tasks
        assert "classification" in ARM_DECLARATIONS[arm].tasks


def test_unverified_arms_name_the_exact_missing_dependency():
    for arm in ("sentinellms", "obfuslm_plus", "inferdpt", "dp_forward"):
        declaration = ARM_DECLARATIONS[arm]
        assert declaration.status in {"unverified", "resource_blocked"}
        assert declaration.missing
        assert declaration.commit is None or declaration.license is None


def test_aloepri_is_not_labelled_a_full_paper_reproduction():
    notes = ARM_DECLARATIONS["aloepri"].notes.lower()
    assert "engineering" in notes or "not a complete" in notes


def test_manifest_exposes_external_arm_gaps_to_preflight(monkeypatch):
    clear_faithful_entries()
    manifest = {
        "models": {
            "llama-3.2-1b-base": {
                "name": "Llama-3.2-1B", "variant": "base", "revision": "r",
                "config_sha256": "a" * 64, "tokenizer_sha256": "a" * 64,
                "weights_sha256": "a" * 64,
            },
            "llama-3.2-3b-instruct": {
                "name": "Llama-3.2-3B-Instruct", "variant": "instruct", "revision": "r",
                "config_sha256": "b" * 64, "tokenizer_sha256": "b" * 64,
                "weights_sha256": "b" * 64,
            },
            "qwen2.5-7b-instruct": {
                "name": "Qwen2.5-7B-Instruct", "variant": "instruct", "revision": "r",
                "config_sha256": "c" * 64, "tokenizer_sha256": "c" * 64,
                "weights_sha256": "c" * 64,
            },
        },
        "data": {
            "split_seed": 20261107, "train_docs": 8192, "val_docs": 1024,
            "test_docs": 1000, "train_tokens": 1_000_000, "knn_texts": 1000,
            "hashes_disjoint": True,
        },
        "arms": manifest_arms(),
    }
    with pytest.raises(PreflightError) as exc:
        preflight_formal_start(require_profile("paper-v1"), manifest)
    text = " ".join(exc.value.shortages)
    for arm in ("sentinellms", "obfuslm_plus", "inferdpt", "dp_forward"):
        assert arm in text


def test_matrix_contains_the_manifest_entry_for_every_arm():
    matrix = arm_matrix()
    assert set(matrix) == set(PAPER_ARMS)
    for entry in matrix.values():
        assert "adapter" in entry["manifest_entry"]
        assert "status" in entry["declaration"]


def test_sda_vector_cells_are_supported_but_formal_executor_unwired():
    from obfuscation_attacks.evaluation.generic_runner import adapter_attack_matrix
    matrix = adapter_attack_matrix()["matrix"]
    for arm in ("ab", "osnip", "dp_forward"):
        cell = matrix[arm]["sda"]
        assert cell["status"] != "not_applicable"
        assert cell["executor_wired"] is False
