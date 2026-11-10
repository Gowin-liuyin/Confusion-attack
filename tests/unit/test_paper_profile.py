"""Acceptance tests for the paper-v1 execution carrier.

These tests are the W0 contract. They encode docs/PAPER_EVALUATION_PROTOCOL.md
section 8. Do not weaken them to match a thinner implementation.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_SUMMARY = ROOT / "reports" / "privacy_v1" / "full" / "summary.json"

from obfuscation_attacks.evaluation.paper_profile import (  # noqa: E402
    PAPER_ARMS,
    PAPER_PROFILE_NAME,
    PreflightError,
    ProfileError,
    bind_profile,
    check_requested_budget,
    clear_faithful_entries,
    discover_local_resources,
    formal_aggregate_allowed,
    gate_battery_args,
    layers_for_depth,
    layers_for_model,
    plan_jobs,
    preflight_formal_start,
    protocol_sha256,
    register_faithful_entry,
    require_profile,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_faithful_entries()
    yield
    clear_faithful_entries()


def _complete_manifest():
    def model(name, variant, digest):
        return {
            "name": name,
            "source": "fixture-source",
            "path": "fixture-model",
            "revision": "rev-fixture",
            "variant": variant,
            "config_sha256": digest,
            "tokenizer_sha256": digest,
            "weights_sha256": digest,
            "license": "fixture-license",
        }

    return {
        "models": {
            "llama-3.2-1b-base": model("Llama-3.2-1B", "base", "a" * 64),
            "llama-3.2-3b-instruct": model("Llama-3.2-3B-Instruct", "instruct", "b" * 64),
            "qwen2.5-7b-instruct": model("Qwen2.5-7B-Instruct", "instruct", "c" * 64),
        },
        "data": {
            "split_seed": 20261107,
            "train_docs": 8192,
            "val_docs": 1024,
            "test_docs": 1000,
            "train_tokens": 1_000_000,
            "knn_texts": 1000,
            "knn_sha256": "e" * 64,
            "hashes_disjoint": True,
            "split_strategy": "char5_containment80_representative_v2", "split_sha256": "1" * 64,
            "content_sha256": "d" * 64,
        },
        "artifacts": {
            "ab": {
                "path": "fixture-artifact",
                "files_sha256": {"model_server/config.json": "f" * 64},
            },
        },
        "arms": {
            arm: {"adapter": "present", "license": "fixture", "commit": "abc123", "source_verified": True}
            for arm in PAPER_ARMS
        },
    }


def _paper():
    return require_profile("paper-v1", root=ROOT)


def _jobs(manifest=None):
    return plan_jobs(_paper(), manifest, root=ROOT)["jobs"]


def _one(jobs, *, attack, arm, model="llama-3.2-1b-base", scenario="S-public"):
    hits = [
        j for j in jobs
        if j["attack"] == attack and j["arm"] == arm and j["model"] == model and j["scenario"] == scenario
    ]
    assert hits, f"missing job {attack} / {arm} / {model} / {scenario}"
    return hits[0]


def test_protocol_file_hash_matches_loader():
    digest = protocol_sha256(ROOT)
    raw = (ROOT / "docs" / "PAPER_EVALUATION_PROTOCOL.md").read_bytes()
    import hashlib
    assert digest == hashlib.sha256(raw).hexdigest()
    profile = _paper()
    assert profile.name == PAPER_PROFILE_NAME
    assert profile.protocol_sha256 == digest
    assert "8" in json.dumps(profile.source_anchors)


def test_profile_json_is_execution_carrier_not_a_second_policy():
    doc = json.loads((ROOT / "configs" / "paper" / "paper-v1.json").read_text(encoding="utf-8"))
    assert doc["role"] == "execution_carrier"
    assert doc["normative"] is False
    assert doc["authority"] == "docs/PAPER_EVALUATION_PROTOCOL.md"
    assert doc["protocol_sha256"] == protocol_sha256(ROOT)


def test_require_profile_refuses_missing_unknown_and_drift(tmp_path):
    with pytest.raises(ProfileError, match="profile"):
        require_profile(None, root=ROOT)
    with pytest.raises(ProfileError, match="profile"):
        require_profile("", root=ROOT)
    with pytest.raises(ProfileError):
        require_profile("nope", root=ROOT)
    assert require_profile("probe", root=ROOT).name == "probe"

    proto = (ROOT / "docs" / "PAPER_EVALUATION_PROTOCOL.md").read_bytes()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PAPER_EVALUATION_PROTOCOL.md").write_bytes(proto + b"\n# tamper\n")
    (tmp_path / "configs" / "paper").mkdir(parents=True)
    good = json.loads((ROOT / "configs" / "paper" / "paper-v1.json").read_text(encoding="utf-8"))
    (tmp_path / "configs" / "paper" / "paper-v1.json").write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(ProfileError, match="drift|hash"):
        require_profile("paper-v1", root=tmp_path)

    empty = tmp_path / "empty"
    (empty / "configs" / "paper").mkdir(parents=True)
    (empty / "configs" / "paper" / "paper-v1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProfileError):
        require_profile("paper-v1", root=empty)


def test_bind_profile_rejects_mismatched_bytes():
    doc = json.loads((ROOT / "configs" / "paper" / "paper-v1.json").read_text(encoding="utf-8"))
    with pytest.raises(ProfileError, match="drift|hash"):
        bind_profile(doc, b"not-the-protocol")


def test_section8_floors_are_not_weakened():
    floors = _paper().floors
    assert floors["training_seeds"] == [17, 29, 43]
    assert floors["split_seed"] == 20261107
    assert floors["root_key_bits"] == 256
    assert floors["bootstrap_replicates"] == 2000
    assert floors["arms"] == list(PAPER_ARMS)
    assert floors["models"]["primary"]["name"] == "Llama-3.2-1B"
    assert floors["models"]["primary"]["variant"] == "base"
    assert floors["models"]["primary"]["id"] == "llama-3.2-1b-base"
    assert floors["models"]["scale"]["name"] == "Llama-3.2-3B-Instruct"
    assert floors["models"]["cross_family"]["name"] == "Qwen2.5-7B-Instruct"
    knn = floors["knn"]
    assert knn["main_texts"] == 1000 and knn["main_keys"] == 1
    assert knn["robust_texts"] == 200 and knn["robust_keys"] == 5
    assert knn["char_min"] == 20 and knn["char_max"] == 512 and knn["top_k"] == 100
    assert knn["metrics"] == ["l2", "cosine"]
    assert floors["vma_h"]["prompts"] == 200 and floors["vma_h"]["keys"] == 5
    assert floors["vma_h"]["layers_1b"] == [1, 5, 10, 15]
    assert floors["vma_h"]["prefix"] == "recovered"
    isa = floors["isa"]
    assert isa["steps"] == 50000
    assert isa["checkpoints"] == [1000, 10000, 25000, 50000]
    assert isa["extension_steps"] == [100000, 200000]
    assert isa["restarts"] == 3 and isa["val_prompts"] == 40 and isa["prompts"] == 200
    assert isa["lr_grid"] == pytest.approx([1e-4, 5e-4, 1e-3, 1e-2])
    assert isa["lambda_dm"] == pytest.approx([0, 1e-3])
    assert isa["length_strata"] == {"32-64": 50, "65-128": 50, "129-256": 50, "257-512": 50}
    assert isa["optimizer"] == "AdamW" and isa["weight_decay"] == 0 and isa["loss"] == "MSE"
    assert isa["tbs_alpha"] == pytest.approx(5 / math.pi)
    assert isa["tbs_basis"] == "full"
    assert floors["weight"]["min_types"] == 1000
    noise_tiers = floors["weight"]["noise_tiers"]
    assert len(noise_tiers) == 3
    assert [float(v) for v in noise_tiers[0]] == pytest.approx([0, 0])
    assert [float(v) for v in noise_tiers[1]] == pytest.approx([0.2, 0.01])
    assert [float(v) for v in noise_tiers[2]] == pytest.approx([1.0, 0.2])
    ima = floors["ima"]
    assert ima["train_docs"] == 8192 and ima["min_train_tokens"] == 1_000_000
    assert ima["val_docs"] == 1024 and ima["test_docs"] == 1000
    assert ima["expansion_docs"] == [16384, 32768]
    assert ima["epochs"] == [10, 20, 40] and ima["epoch_extension"] == 80 and ima["batch"] == 64
    assert floors["stream"]["requests"] == 5000 and floors["stream"]["min_tokens"] == 100_000
    assert floors["stream"]["key_pool"] == [4, 16, 64]
    assert floors["stream"]["rotate_every"] == [1, 16, 64, 256]
    assert floors["linkage"]["min_positives"] == 10000 and floors["linkage"]["min_negatives"] == 100000
    assert floors["aggregation_m"] == [1, 2, 4, 8, 16]
    assert floors["tfma"]["reference_tokens"] == 10_000_000
    assert floors["tfma"]["curve"] == [4_450_000, 10_000_000, 20_000_000]
    assert floors["tfma"]["top_k_distinct"] == 100 and floors["tfma"]["control_top1"] == pytest.approx(0.90)
    assert floors["sda"]["train_sequences"] == 60000 and floors["sda"]["beam"] == [10, 50, 100]
    assert floors["text"]["test"] == 1000 and floors["text"]["lm_max_new_tokens"] == 512
    assert floors["output"]["prompts"] == 1000 and floors["output"]["max_new_tokens"] == 256
    assert floors["output"]["entities_per_type"] == 100
    assert floors["utility"]["ppl_window"] == 512 and floors["utility"]["ppl_stride"] == 256
    assert floors["utility"]["shots"] == 0 and floors["utility"]["summary_docs"] == 1000
    assert floors["efficiency"]["concurrency"] == [1, 4, 16]
    assert floors["efficiency"]["warmup"] == 20 and floors["efficiency"]["scored_requests"] == 200
    assert floors["baseline_grids"]["epsilon"] == [1, 3, 8]
    assert floors["baseline_grids"]["obfuslm_k"] == [5, 10, 20]
    assert floors["baseline_grids"]["sentinel_glide"] == [1, 5, 10]


def test_layer_rule_does_not_copy_1b_layers_onto_other_depths():
    assert layers_for_model("llama-3.2-1b-base") == [1, 5, 10, 15]
    assert layers_for_depth(32) == [1, 11, 22, 31]
    with pytest.raises(PreflightError):
        layers_for_depth(0)
    jobs = _jobs()
    for job in jobs:
        if job["model"] == "llama-3.2-1b-base":
            continue
        if job["attack"] in {"vma_h_recovered_prefix", "isa_er"}:
            assert job.get("budget", {}).get("layers") != [1, 5, 10, 15]
            assert job["blockers"], job["attack"]


def test_reduced_budget_lists_every_shortage():
    profile = _paper()
    requested = {
        "knn_texts": 200,
        "isa_steps": [20, 100],
        "isa_prompts": 1,
        "vma_prompts": 4,
        "max_len": 64,
        "top_k": 10,
        "tfma_top100_distinct": 7,
        "n_prompts": -1,
        "layers": [],
        "training_seeds": [20261106],
        "classes": ["C99"],
        "attacks": ["not_an_attack"],
        "arms": ["plaintext", "ab"],
    }
    with pytest.raises(PreflightError) as ei:
        check_requested_budget(profile, requested)
    text = str(ei.value) + " ".join(ei.value.shortages)
    for needle in ("knn_texts", "200", "1000", "50000", "vma_prompts", "max_len", "top_k",
                   "tfma_top100_distinct", "C99", "not_an_attack", "inferdpt", "20261106"):
        assert needle in text, needle
    check_requested_budget(require_profile("probe", root=ROOT), {"knn_texts": 2, "max_len": 8})


def test_formal_start_names_missing_identity_and_instruct_substitution():
    profile = _paper()
    with pytest.raises(PreflightError) as ei:
        preflight_formal_start(profile, {})
    text = str(ei.value) + " ".join(ei.value.shortages)
    assert "sha256" in text.lower() or "revision" in text.lower() or "identity" in text.lower()
    assert "gpu" not in text.lower() or "sha256" in text.lower()

    bad = _complete_manifest()
    bad["models"]["llama-3.2-1b-base"]["variant"] = "instruct"
    with pytest.raises(PreflightError) as ei:
        preflight_formal_start(profile, bad)
    assert "instruct" in str(ei.value).lower() or "base" in str(ei.value).lower()

    short = _complete_manifest()
    short["data"]["train_docs"] = 10
    short["data"]["train_tokens"] = 100
    with pytest.raises(PreflightError) as ei:
        preflight_formal_start(profile, short)
    assert "10" in str(ei.value) and ("8192" in str(ei.value) or "1000000" in str(ei.value))


def test_complete_fixture_configures_only_registered_faithful_cells():
    profile = _paper()
    manifest = _complete_manifest()
    with pytest.raises(PreflightError, match="not_run|configured|faithful"):
        preflight_formal_start(profile, manifest)
    register_faithful_entry("knn_l2", arms={"ab"})
    preflight_formal_start(profile, manifest, requested={"attacks": ["knn_l2"], "arms": ["ab"]})
    jobs = _jobs(manifest)
    ready = _one(jobs, attack="knn_l2", arm="ab")
    assert ready["status"] == "configured"
    assert ready["faithful_entry"] is True
    assert ready["budget"]["n_texts"] == 1000
    other = _one(jobs, attack="gate_ia", arm="ab")
    assert other["status"] != "configured"
    assert other["faithful_entry"] is False


def test_plan_covers_eight_arms_and_does_not_load_models(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError(f"plan imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    plan = plan_jobs(_paper(), None, root=ROOT)
    assert plan["profile"] == "paper-v1"
    assert plan["protocol_sha256"] == protocol_sha256(ROOT)
    jobs = plan["jobs"]
    assert {j["arm"] for j in jobs} == set(PAPER_ARMS)
    assert {j["model"] for j in jobs} >= {
        "llama-3.2-1b-base", "llama-3.2-3b-instruct", "qwen2.5-7b-instruct"
    }
    assert {j["scenario"] for j in jobs} >= {"S-public", "S-deploy", "S-encoder", "S-pair"}
    attacks = {j["attack"] for j in jobs}
    for attack_id in (
        "knn_l2", "knn_cosine", "knn_l2_key_robustness", "ednn",
        "vma_h_recovered_prefix", "vma_h_observed_prefix", "isa_er", "isa_tbs",
        "vma_w", "gate_ia", "attn_ia", "least_squares", "ima_ridge", "ima_neural",
        "linkage_exact", "linkage_cosine", "linkage_knn_aligned", "linkage_recovered_text",
        "aggregation", "tfma", "tfma_after_inversion", "sda", "text_bayes", "text_lm_rerank",
        "output_pii", "utility_generation", "utility_classification", "efficiency",
    ):
        assert attack_id in attacks, attack_id
    assert not any(j.get("main_table_eligible") for j in jobs)
    assert not any(formal_aggregate_allowed(j) for j in jobs)
    assert not any(j["status"] == "configured" for j in jobs)


def test_applicability_sentinels_do_not_hide_missing_adapters():
    jobs = _jobs()
    infer_knn = _one(jobs, attack="knn_l2", arm="inferdpt")
    assert infer_knn["applicable"] is False and infer_knn["status"] == "not_applicable"
    assert "text" in infer_knn["reason"].lower() and "vector" in infer_knn["reason"].lower()

    sentinel_vma = _one(jobs, attack="vma_h_recovered_prefix", arm="sentinellms")
    assert sentinel_vma["status"] == "not_applicable"
    assert "classification" in sentinel_vma["reason"].lower() or "decoder" in sentinel_vma["reason"].lower()

    sentinel_knn = _one(jobs, attack="knn_l2", arm="sentinellms")
    assert sentinel_knn["applicable"] is True
    assert sentinel_knn["status"] in {"not_run", "resource_blocked", "unverified"}

    assert _one(jobs, attack="text_bayes", arm="inferdpt")["applicable"] is True
    assert _one(jobs, attack="gate_ia", arm="ab")["applicable"] is True
    osnip_gate = _one(jobs, attack="gate_ia", arm="osnip")
    assert osnip_gate["status"] == "not_applicable"
    assert "weight" in osnip_gate["reason"].lower() or "permutation" in osnip_gate["reason"].lower()

    least = _one(jobs, attack="least_squares", arm="aloepri")
    assert least["applicable"] is True and least["attack"] == "least_squares"
    ab_tfma = _one(jobs, attack="tfma", arm="ab")
    assert ab_tfma["status"] == "not_applicable"
    assert _one(jobs, attack="tfma", arm="aloepri")["applicable"] is True
    assert _one(jobs, attack="tfma", arm="plaintext")["applicable"] is True
    assert _one(jobs, attack="tfma", arm="inferdpt")["applicable"] is True
    assert _one(jobs, attack="tfma_after_inversion", arm="aloepri")["status"] == "not_applicable"
    composed = _one(jobs, attack="tfma_after_inversion", arm="ab")
    assert composed["applicable"] is True and composed["status"] != "configured"
    dp_out = _one(jobs, attack="output_pii", arm="dp_forward")
    assert dp_out["status"] == "not_applicable"
    assert "classification" in dp_out["reason"].lower() or "generation" in dp_out["reason"].lower()
    assert _one(jobs, attack="knn_l2", arm="plaintext")["applicable"] is True
    observed = _one(jobs, attack="vma_h_observed_prefix", arm="ab")
    recovered = _one(jobs, attack="vma_h_recovered_prefix", arm="ab")
    assert observed["attack"] != recovered["attack"]


def test_discover_local_resources_does_not_invent_a_base_hash(tmp_path):
    instruct = tmp_path / "Llama-3.2-1B-Instruct"
    instruct.mkdir()
    (instruct / "config.json").write_text("{}", encoding="utf-8")
    found = discover_local_resources([tmp_path])
    blob = json.dumps(found).lower()
    assert "instruct" in blob and "base" in blob
    assert "a" * 64 not in json.dumps(found)
    assert found.get("models", {}).get("llama-3.2-1b-base", {}).get("weights_sha256") in (None, "")


def test_aggregation_cannot_be_self_declared():
    digest = protocol_sha256(ROOT)
    assert formal_aggregate_allowed({"profile": "probe", "main_table_eligible": True}) is False
    assert formal_aggregate_allowed({"mode": "full", "main_table_eligible": True}) is False
    assert formal_aggregate_allowed({"profile": "paper-v1", "main_table_eligible": True, "protocol_sha256": digest}) is False
    assert formal_aggregate_allowed({
        "profile": "paper-v1",
        "protocol_sha256": digest,
        "audit": {"verdict": "audited_paper_scale", "auditor": "run_c01", "protocol_sha256": digest},
    }) is False
    assert formal_aggregate_allowed({
        "profile": "paper-v1",
        "protocol_sha256": digest,
        "audit": {
            "verdict": "audited_paper_scale",
            "auditor": "protocol_auditor",
            "protocol_sha256": digest,
        },
    }) is True


def test_gate_refuses_paper_start_without_falling_through():
    refused = gate_battery_args(None, plan_only=False, requested={}, root=ROOT)
    assert refused["action"] == "refuse" and refused["code"] == 2
    assert "profile" in refused["message"].lower()

    reduced = gate_battery_args(
        "paper-v1", plan_only=False, requested={"knn_texts": 200}, manifest=None, root=ROOT
    )
    assert reduced["action"] == "refuse" and reduced["code"] == 2
    assert "knn_texts" in " ".join(reduced["shortages"])

    bare = gate_battery_args("paper-v1", plan_only=False, requested={}, manifest=None, root=ROOT)
    assert bare["action"] == "refuse" and bare["code"] == 2
    assert bare["action"] != "run_paper"

    planned = gate_battery_args("paper-v1", plan_only=True, requested={}, manifest=None, root=ROOT)
    assert planned["action"] == "plan" and planned["code"] == 0
    assert {j["arm"] for j in planned["plan"]["jobs"]} == set(PAPER_ARMS)

    probe = gate_battery_args("probe", plan_only=False, requested={"knn_texts": 2}, root=ROOT)
    assert probe["action"] == "run_probe" and probe["aggregate"] is False


def test_runner_paper_profile_refuses_before_full_backend(tmp_path):
    from obfuscation_attacks.evaluation.runner import run_from_config

    out = run_from_config(
        {
            "name": "paper_reduced",
            "mode": "full",
            "profile": "paper-v1",
            "knn_texts": 200,
            "classes": ["C99"],
            "attack": {"name": "nn_knn.euclidean", "params": {}},
        },
        reports_root=str(tmp_path),
    )
    blob = json.dumps(out)
    assert out["status"] in {"refused", "preflight_failed"}
    assert "knn_texts" in blob or "200" in blob
    assert "C99" in blob
    assert "token_recovery" not in blob


def test_runner_paper_profile_refuses_a_passing_preflight_from_the_legacy_backend(monkeypatch, tmp_path):
    from obfuscation_attacks.evaluation import real_battery
    from obfuscation_attacks.evaluation.paper_runner import register_implemented_entries
    from obfuscation_attacks.evaluation.runner import run_from_config

    register_implemented_entries()
    calls = []
    monkeypatch.setattr(
        real_battery, "run_full_config",
        lambda config: calls.append(config) or {"status": "ok"},
    )
    out = run_from_config(
        {
            "name": "paper_full_legacy",
            "mode": "full",
            "profile": "paper-v1",
            "manifest": _complete_manifest(),
            "attacks": ["knn_l2"],
            "arms": ["ab"],
            "classes": ["C01"],
            "knn_texts": 1000,
            "full_backend": {"artifact": "a", "clean_model": "b", "alpaca": "c"},
        },
        reports_root=str(tmp_path),
    )
    assert out["status"] == "refused"
    assert "paper executor" in out["status_reason"]
    assert calls == []


def _load_cli():
    path = ROOT / "scripts" / "run_full_battery.py"
    spec = importlib.util.spec_from_file_location("run_full_battery_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_gates_before_model_or_alpaca_and_does_not_clobber_history(tmp_path, monkeypatch):
    module = _load_cli()
    monkeypatch.setattr(module, "ABDeployment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("deployment")))
    monkeypatch.setattr(module, "load_alpaca", lambda *a, **k: (_ for _ in ()).throw(AssertionError("alpaca")))
    before = HISTORICAL_SUMMARY.read_bytes() if HISTORICAL_SUMMARY.exists() else None

    monkeypatch.setattr(sys, "argv", ["run_full_battery.py"])
    with pytest.raises(SystemExit) as ei:
        module.main()
    assert ei.value.code == 2

    monkeypatch.setattr(sys, "argv", ["run_full_battery.py", "--profile", "paper-v1", "--knn-texts", "200"])
    with pytest.raises(SystemExit) as ei:
        module.main()
    assert ei.value.code == 2

    plan_out = tmp_path / "paper-v1.plan.json"
    monkeypatch.setattr(
        sys, "argv",
        ["run_full_battery.py", "--profile", "paper-v1", "--plan-only", "--plan-out", str(plan_out)],
    )
    try:
        module.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)
    plan = json.loads(plan_out.read_text(encoding="utf-8"))
    assert {j["arm"] for j in plan["jobs"]} == set(PAPER_ARMS)
    after = HISTORICAL_SUMMARY.read_bytes() if HISTORICAL_SUMMARY.exists() else None
    assert before == after


def test_plan_exposes_frozen_family_requirements_without_claiming_grid_expansion():
    from obfuscation_attacks.evaluation.paper_profile import require_profile, plan_jobs

    profile = require_profile("paper-v1")
    plan = plan_jobs(profile)
    assert plan["granularity"] == "attack_arm_model_scenario"
    assert plan["axis_expansion"].startswith("not_expanded")
    jobs = {job["attack"]: job for job in plan["jobs"]
            if job["arm"] == "ab" and job["model"] == "llama-3.2-1b-base"
            and job["scenario"] == "S-public"}
    for attack, family in (("isa_er", "isa"), ("ima_neural", "ima"),
                           ("tfma", "tfma"), ("sda", "sda"),
                           ("linkage_exact", "linkage"), ("output_pii", "output"),
                           ("efficiency", "efficiency")):
        assert jobs[attack]["budget"]["requirements"][family] == profile.floors[family]
    assert jobs["sda"]["budget"]["requirements"]["training_seeds"] == profile.floors["training_seeds"]
    assert jobs["aggregation"]["budget"]["requirements"]["aggregation_m"] == profile.floors["aggregation_m"]
    jobs["sda"]["budget"]["requirements"]["sda"]["beam"].append(-1)
    assert -1 not in profile.floors["sda"]["beam"]
    from obfuscation_attacks.evaluation.public_guard import forbidden_material_gaps
    assert not forbidden_material_gaps({"val_prompts": 40})
    assert forbidden_material_gaps({"val_prompts": ["private text"]})
