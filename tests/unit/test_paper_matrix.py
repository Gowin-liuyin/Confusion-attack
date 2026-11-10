from __future__ import annotations

import importlib.util
import copy
import json
import sys
from pathlib import Path

import pytest

from obfuscation_attacks.evaluation.paper_matrix import (
    _attack_axes, _defense_axes, iter_coverage, iter_expanded_specs, validate_matrix,
)
from obfuscation_attacks.evaluation.paper_profile import require_profile

ROOT = Path(__file__).resolve().parents[2]


def _profile():
    return require_profile("paper-v1", root=ROOT)


def _filters(attack, arm="ab", model="llama-3.2-1b-base", scenario="S-public"):
    return {"arms": {arm}, "models": {model}, "scenarios": {scenario}, "attacks": {attack}}


def test_stream_axes_cover_rotation_policies_priors_and_independent_streams():
    floors = _profile().floors
    axes, pending = _attack_axes("tfma", "llama-3.2-1b-base", "S-public", floors, None)
    assert pending and len(axes) == 14 * 3 * 3 * 3
    assert {row["prior"] for row in axes} == {"general", "same_domain", "matched_proxy"}
    assert {row["stream_index"] for row in axes} == {0, 1, 2}
    assert {row["rotation"]["mode"] for row in axes} == {"fixed", "fresh_per_request", "pool"}
    assert {row["reference_tokens"] for row in axes} == set(floors["tfma"]["curve"])
    sda, _ = _attack_axes("sda", "llama-3.2-1b-base", "S-public", floors, None)
    assert {row["attack_training_seed"] for row in sda} == set(floors["training_seeds"])
    assert {row["beam"] for row in sda} == set(floors["sda"]["beam"])
    assert {row["epoch_checkpoint"] for row in sda} == set(floors["sda"]["epochs"])
    assert {row["attacker_instance"] for row in sda} == {0, 1, 2}
    assert all(row["algorithm"] is None for row in sda)


def test_isa_search_separates_validation_and_conditional_extensions():
    axes, pending = _attack_axes("isa_er", "llama-3.2-1b-base", "S-public", _profile().floors, None)
    assert pending == [] and {row["layer"] for row in axes} == {1, 5, 10, 15}
    assert len([row for row in axes if row["phase"] == "validation"]) == 4 * 4 * 2
    assert {row["steps"] for row in axes if row["phase"] == "test_extension"} == {100000, 200000}
    unknown, pending = _attack_axes("isa_er", "llama-3.2-3b-instruct", "S-public", _profile().floors, None)
    assert {row["layer"] for row in unknown} == {None} and pending


def test_ima_conditions_and_defense_grids_keep_unfrozen_workpoints_visible():
    floors = _profile().floors
    defenses, pending = _defense_axes("ab", floors)
    assert len(defenses) == 9 and pending
    assert {row["defense_training_seed"] for row in defenses} == {17, 29, 43}
    public, _ = _attack_axes("ima_ridge", "llama-3.2-1b-base", "S-public", floors, None)
    paired, _ = _attack_axes("ima_ridge", "llama-3.2-1b-base", "S-pair", floors, None)
    assert "target_pairs" not in {row["information_condition"] for row in public}
    assert "target_pairs" in {row["information_condition"] for row in paired}
    assert {row["train_docs"] for row in public} == {8192, 16384, 32768}
    assert all("training_data_condition" in row for row in public)


def test_expanded_ids_change_with_axes_and_manifest_without_claiming_execution():
    profile = _profile()
    specs = list(iter_expanded_specs(profile, filters=_filters("aggregation")))
    assert len(specs) == 9 * 14 * 3 * 5
    assert len({row["plan_spec_id"] for row in specs}) == len(specs)
    assert all(row["status"] == "not_run" and row["unresolved"] for row in specs)
    bound = next(iter_expanded_specs(profile, {"data": {"source_sha256": "a" * 64}},
                                     filters=_filters("aggregation")))
    changed = next(iter_expanded_specs(profile, {"data": {"source_sha256": "b" * 64}},
                                       filters=_filters("aggregation")))
    assert bound["plan_spec_id"] != changed["plan_spec_id"]
    assert bound["identity"]["manifest_json_sha256"] is not None
    altered = copy.copy(profile)
    altered.floors = copy.deepcopy(profile.floors)
    altered.floors["stream"]["min_tokens"] += 1
    altered_spec = next(iter_expanded_specs(altered, filters=_filters("aggregation")))
    assert altered_spec["plan_spec_id"] != specs[0]["plan_spec_id"]


def test_conditional_and_independent_axes_are_frozen_in_the_execution_carrier():
    floors = _profile().floors
    text, _ = _attack_axes("text_bayes", "llama-3.2-1b-base", "S-public", floors, None)
    assert {row["epoch_checkpoint"] for row in text} == {3, 10}
    assert {row["beam"] for row in text} == {10, 50, 100}
    efficiency, _ = _attack_axes("efficiency", "llama-3.2-1b-base", "S-public", floors, None)
    assert len(efficiency) == 3 * 3 * 3
    primary, _ = _attack_axes("utility_classification", "llama-3.2-1b-base", "S-public", floors, None)
    cross, _ = _attack_axes("utility_classification", "qwen2.5-7b-instruct", "S-public", floors, None)
    assert len(primary) == 11 and len(cross) == 6
    assert {row["task"] for row in cross} < {row["task"] for row in primary}


def _write_matrix(monkeypatch, path):
    spec = importlib.util.spec_from_file_location("plan_paper_matrix_fixture", ROOT / "scripts" / "plan_paper_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["plan_paper_matrix.py", "--output", str(path), "--arm", "plaintext",
                                     "--model", "llama-3.2-1b-base", "--scenario", "S-public", "--attack", "knn_l2"])
    module.main()


def test_matrix_framing_and_coverage_refuse_incomplete_or_unbound_results(monkeypatch, tmp_path):
    path = tmp_path / "matrix.jsonl"
    _write_matrix(monkeypatch, path)
    assert validate_matrix(path)["spec_count"] == 1
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[1])["spec"]
    assert list(iter_coverage(path, []))[0]["status"] == "missing"
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"plan_spec_id": row["plan_spec_id"], "plan_identity": row["identity"]}))
    assert list(iter_coverage(path, [result]))[0]["status"] == "recorded_unverified"
    result.write_text(json.dumps({"plan_spec_id": row["plan_spec_id"], "plan_identity": {}}))
    assert list(iter_coverage(path, [result]))[0]["status"] == "identity_mismatch"
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        validate_matrix(truncated)
    changed = tmp_path / "changed.jsonl"
    changed.write_text(path.read_text(encoding="utf-8").replace('"attack":"knn_l2"', '"attack":"knn_cosine"'))
    with pytest.raises(ValueError, match="modified|digest"):
        validate_matrix(changed)


def test_comparison_keeps_unbound_files_and_never_certifies_a_match(monkeypatch, tmp_path):
    matrix = tmp_path / "matrix.jsonl"
    _write_matrix(monkeypatch, matrix)
    planned = json.loads(matrix.read_text(encoding="utf-8").splitlines()[1])["spec"]
    results = tmp_path / "results"
    (results / "matched").mkdir(parents=True)
    (results / "unbound").mkdir()
    (results / "matched" / "result.json").write_text(json.dumps({
        "plan_spec_id": planned["plan_spec_id"], "plan_identity": planned["identity"],
        "main_table_eligible": True,
    }))
    (results / "unbound" / "result.json").write_text(json.dumps({"main_table_eligible": True}))
    spec = importlib.util.spec_from_file_location("compare_paper_matrix_fixture", ROOT / "scripts" / "compare_paper_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "coverage.jsonl"
    monkeypatch.setattr(sys, "argv", ["compare_paper_matrix.py", "--matrix", str(matrix),
                                     "--results", str(results), "--output", str(output)])
    module.main()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[1]["status"] == "recorded_unverified"
    assert rows[2]["kind"] == "unbound_result"
    assert rows[-1]["counts"] == {"recorded_unverified": 1}
    assert rows[-1]["unbound_results"] == 1
