"""Negative/mutation-derived acceptance tests for the W0 profile gate.

Added after the independent review found that several section-8 values and
bypass paths were not pinned by the original fixture suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from obfuscation_attacks.evaluation.paper_profile import (
    PAPER_ATTACKS,
    PAPER_ARMS,
    PreflightError,
    ProfileError,
    check_requested_budget,
    clear_faithful_entries,
    gate_battery_args,
    plan_jobs,
    preflight_formal_start,
    register_faithful_entry,
    require_profile,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_faithful_entries()
    yield
    clear_faithful_entries()


def _model(variant: str, digest: str) -> dict:
    return {
        "name": f"fixture-{variant}", "source": "fixture", "path": "fixture-model",
        "revision": "rev-fixture", "variant": variant, "config_sha256": digest,
        "tokenizer_sha256": digest, "weights_sha256": digest, "license": "fixture",
    }


def _manifest() -> dict:
    return {
        "models": {
            "llama-3.2-1b-base": _model("base", "a" * 64),
            "llama-3.2-3b-instruct": _model("instruct", "b" * 64),
            "qwen2.5-7b-instruct": _model("instruct", "c" * 64),
        },
        "data": {
            "split_seed": 20261107, "train_docs": 8192, "val_docs": 1024,
            "test_docs": 1000, "train_tokens": 1_000_000, "knn_texts": 1000,
            "knn_sha256": "e" * 64,
            "hashes_disjoint": True, "content_sha256": "d" * 64,
            "split_strategy": "char5_containment80_representative_v2", "split_sha256": "1" * 64,
        },
        "artifacts": {
            "ab": {
                "path": "fixture-artifact",
                "files_sha256": {"model_server/config.json": "f" * 64},
            },
        },
        "arms": {
            arm: {"adapter": "present", "license": "fixture", "commit": "abc",
                  "source_verified": True}
            for arm in PAPER_ARMS
        },
    }


def _write_carrier(root: Path, mutate=None) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "PAPER_EVALUATION_PROTOCOL.md").write_bytes(
        (ROOT / "docs" / "PAPER_EVALUATION_PROTOCOL.md").read_bytes()
    )
    doc = json.loads((ROOT / "configs" / "paper" / "paper-v1.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(doc)
    (root / "configs" / "paper").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "paper" / "paper-v1.json").write_text(json.dumps(doc), encoding="utf-8")


def test_weakened_carrier_is_refused(tmp_path):
    def weaken(doc):
        doc["floors"]["knn"]["main_texts"] = 200

    _write_carrier(tmp_path, weaken)
    with pytest.raises(ProfileError, match="floor"):
        require_profile("paper-v1", root=tmp_path)


def test_carrier_role_and_authority_cannot_be_relabelled(tmp_path):
    def relabel(doc):
        doc["role"] = "policy"

    _write_carrier(tmp_path, relabel)
    with pytest.raises(ProfileError):
        require_profile("paper-v1", root=tmp_path)


def test_every_section8_floor_is_pinned():
    floors = require_profile("paper-v1", root=ROOT).floors
    isa = floors["isa"]
    assert isa["keys"] == 5
    assert floors["ima"]["ridge_lambda"] == pytest.approx([0.01, 0.1, 1, 10])
    assert floors["ima"]["lr"] == pytest.approx([2e-5, 1e-4, 3e-4])
    assert floors["stream"]["n_streams"] == 3
    assert floors["stream"]["min_types"] == 1000
    assert floors["linkage"]["fpr"] == pytest.approx([0.01, 0.001])
    assert floors["tfma"]["min_types"] == 50000
    assert floors["sda"]["val_sequences"] == 2000
    assert floors["sda"]["length"] == 128
    assert floors["sda"]["epochs"] == [6, 12, 24]
    assert floors["text"]["val"] == 200
    assert floors["text"]["lm_model"]["id"] == "qwen2.5-7b-instruct"
    assert floors["text"]["decoding"] == "greedy"
    assert floors["output"]["decoding"] == "greedy"
    assert floors["output"]["entity_types"] == ["name", "email", "phone", "address", "id"]
    assert floors["utility"]["summary_max_input_tokens"] == 1024
    assert floors["utility"]["summary_max_new_tokens"] == 256
    assert floors["efficiency"]["input_lengths"] == [128, 512, 1024]
    assert floors["efficiency"]["output_length"] == 128
    assert floors["efficiency"]["rounds"] == 3
    assert floors["baseline_grids"]["obfuslm_epsilon"] == pytest.approx([0.1, 1, 3])
    assert floors["models"]["scale"]["variant"] == "instruct"
    assert floors["models"]["scale"]["id"] == "llama-3.2-3b-instruct"
    assert floors["models"]["cross_family"]["variant"] == "instruct"
    assert floors["models"]["cross_family"]["id"] == "qwen2.5-7b-instruct"


def test_non_primary_model_identity_is_enforced():
    manifest = _manifest()
    del manifest["models"]["llama-3.2-3b-instruct"]["revision"]
    del manifest["models"]["qwen2.5-7b-instruct"]["weights_sha256"]
    with pytest.raises(PreflightError) as exc:
        preflight_formal_start(require_profile("paper-v1", root=ROOT), manifest)
    text = " ".join(exc.value.shortages)
    assert "llama-3.2-3b-instruct" in text and "revision" in text
    assert "qwen2.5-7b-instruct" in text and "weights_sha256" in text


def test_no_manifest_never_configures_a_cell():
    register_faithful_entry("knn_l2", arms=set(PAPER_ARMS))
    jobs = plan_jobs(require_profile("paper-v1", root=ROOT), None)["jobs"]
    assert not any(job["status"] == "configured" for job in jobs)
    for job in jobs:
        if job["attack"] == "knn_l2" and job["applicable"]:
            assert job["status"] in {"resource_blocked", "unverified"}


def test_gate_success_path_reaches_run_paper():
    register_faithful_entry("knn_l2", arms=set(PAPER_ARMS))
    decision = gate_battery_args(
        "paper-v1", plan_only=False, requested={"attacks": ["knn_l2"]},
        manifest=_manifest(), root=ROOT,
    )
    assert decision["action"] == "run_paper"
    assert decision["code"] == 0


def test_split_seed_overlap_and_arm_provenance_are_enforced():
    manifest = _manifest()
    manifest["data"]["split_seed"] = 1
    manifest["data"]["hashes_disjoint"] = False
    del manifest["arms"]["sentinellms"]["license"]
    with pytest.raises(PreflightError) as exc:
        preflight_formal_start(require_profile("paper-v1", root=ROOT), manifest)
    text = " ".join(exc.value.shortages)
    assert "split_seed" in text
    assert "disjoint" in text
    assert "sentinellms" in text and "license" in text


def test_split_hash_lists_must_not_overlap():
    manifest = _manifest()
    manifest["data"]["split_hashes"] = {
        "train": ["a", "b"], "val": ["b"], "test": ["c"],
    }
    with pytest.raises(PreflightError, match="overlap"):
        preflight_formal_start(require_profile("paper-v1", root=ROOT), manifest)


def test_full_job_grid_is_emitted():
    jobs = plan_jobs(require_profile("paper-v1", root=ROOT), None)["jobs"]
    assert len(jobs) == len(PAPER_ARMS) * 3 * 4 * len(PAPER_ATTACKS)
    for arm in PAPER_ARMS:
        assert sum(1 for job in jobs if job["arm"] == arm) == 3 * 4 * len(PAPER_ATTACKS)
    for model in ("llama-3.2-1b-base", "llama-3.2-3b-instruct", "qwen2.5-7b-instruct"):
        assert sum(1 for job in jobs if job["model"] == model) == 4 * len(PAPER_ATTACKS) * len(PAPER_ARMS)


def test_non_primary_layer_jobs_do_not_invent_layers():
    from obfuscation_attacks.evaluation.paper_profile import LAYER_ATTACKS

    jobs = plan_jobs(require_profile("paper-v1", root=ROOT), None)["jobs"]
    for job in jobs:
        if job["model"] == "llama-3.2-1b-base":
            continue
        if job["attack"] in LAYER_ATTACKS:
            assert "layers" not in job["budget"]
            assert job["blockers"]


def test_non_numeric_budget_is_named_not_crashing():
    with pytest.raises(PreflightError) as exc:
        check_requested_budget(
            require_profile("paper-v1", root=ROOT), {"knn_texts": "many"}
        )
    assert "not numeric" in " ".join(exc.value.shortages)


def test_non_numeric_manifest_data_is_named_not_crashing():
    manifest = _manifest()
    manifest["data"]["knn_texts"] = "many"
    with pytest.raises(PreflightError) as exc:
        preflight_formal_start(require_profile("paper-v1", root=ROOT), manifest)
    assert "not numeric" in " ".join(exc.value.shortages)
    jobs = plan_jobs(require_profile("paper-v1", root=ROOT), manifest)["jobs"]
    knn = next(
        job for job in jobs
        if job["attack"] == "knn_l2" and job["arm"] == "ab"
        and job["model"] == "llama-3.2-1b-base" and job["scenario"] == "S-public"
    )
    assert knn["status"] != "configured"
    assert any("not numeric" in blocker for blocker in knn["blockers"])


def test_plan_requires_the_exact_model_variant():
    manifest = _manifest()
    manifest["models"]["llama-3.2-1b-base"]["variant"] = None
    jobs = plan_jobs(require_profile("paper-v1", root=ROOT), manifest)["jobs"]
    knn = next(
        job for job in jobs
        if job["attack"] == "knn_l2" and job["arm"] == "ab"
        and job["model"] == "llama-3.2-1b-base" and job["scenario"] == "S-public"
    )
    assert knn["status"] != "configured"
    assert any("variant" in blocker for blocker in knn["blockers"])


def test_plan_knn_texts_floor_blocks_a_configured_cell():
    manifest = _manifest()
    manifest["data"]["knn_texts"] = 200
    register_faithful_entry("knn_l2", arms={"ab"})
    jobs = plan_jobs(require_profile("paper-v1", root=ROOT), manifest)["jobs"]
    job = next(
        j for j in jobs
        if j["attack"] == "knn_l2" and j["arm"] == "ab" and j["model"] == "llama-3.2-1b-base"
        and j["scenario"] == "S-public"
    )
    assert job["status"] != "configured"
    assert any("knn_texts" in blocker for blocker in job["blockers"])


def test_runner_preflights_before_any_backend(monkeypatch, tmp_path):
    import obfuscation_attacks.evaluation.real_battery as battery
    from obfuscation_attacks.evaluation.runner import run_from_config

    def explode(*args, **kwargs):
        raise AssertionError("legacy battery ran before the paper-v1 preflight")

    monkeypatch.setattr(battery, "run_full_config", explode)
    out = run_from_config(
        {"name": "paper", "mode": "full", "profile": "paper-v1", "knn_texts": 200,
         "classes": ["C99"]},
        reports_root=str(tmp_path),
    )
    assert out["status"] in {"refused", "preflight_failed"}
    assert "token_recovery" not in json.dumps(out)
