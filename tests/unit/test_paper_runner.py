"""Acceptance tests for the paper-v1 dispatcher (W7)."""

from __future__ import annotations

import pytest

from obfuscation_attacks.evaluation.paper_profile import (
    clear_faithful_entries,
    require_profile,
)
from obfuscation_attacks.evaluation.paper_runner import (
    IMPLEMENTED_ENTRIES,
    configured_jobs,
    dispatch_configured,
    register_implemented_entries,
)

pytestmark = pytest.mark.unit


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
            "knn_sha256": "e" * 64, "hashes_disjoint": True,
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
            for arm in (
                "plaintext", "aloepri", "osnip", "ab", "sentinellms",
                "obfuslm_plus", "inferdpt", "dp_forward",
            )
        },
    }


@pytest.fixture(autouse=True)
def _registry():
    clear_faithful_entries()
    yield
    clear_faithful_entries()


def test_register_only_implemented_entries():
    register_implemented_entries()
    profile = require_profile("paper-v1")
    jobs = configured_jobs(profile, _manifest(), requested={"attacks": ["knn_l2"]})
    assert jobs
    assert {job["attack"] for job in jobs} == {"knn_l2"}
    assert {job["arm"] for job in jobs} == {"plaintext", "aloepri", "ab"}
    assert all(job["model"] == "llama-3.2-1b-base" for job in jobs)
    # Only S-public is registered: the other scenarios would relabel one computation.
    assert {job["scenario"] for job in jobs} == {"S-public"}
    assert len(jobs) == 3


def test_dispatch_without_executor_is_resource_blocked():
    register_implemented_entries()
    profile = require_profile("paper-v1")
    report = dispatch_configured(
        profile, _manifest(), requested={"attacks": ["knn_l2"], "arms": ["ab"]},
    )
    assert report["configured"] == 1
    assert report["status"] == "resource_blocked"
    assert all(row["run_status"] == "resource_blocked" for row in report["results"])
    assert all(row["runner"] == "run_c01" for row in report["results"])


def test_dispatch_calls_an_injected_executor():
    register_implemented_entries()
    profile = require_profile("paper-v1")
    seen = []

    def executor(job):
        seen.append(job["job_id"])
        return {"outcome": "executed", "job_id": job["job_id"], "result_path": "fixture"}

    report = dispatch_configured(
        profile, _manifest(), requested={"attacks": ["knn_l2"], "arms": ["ab"]},
        executor=executor,
    )
    assert report["status"] == "executed"
    assert len(seen) == report["configured"] == 1
    assert all(row["run_status"] == "executed" for row in report["results"])


def test_external_arms_are_not_registered():
    register_implemented_entries()
    profile = require_profile("paper-v1")
    jobs = configured_jobs(profile, _manifest())
    arms = {job["arm"] for job in jobs}
    assert arms <= {"plaintext", "aloepri", "ab"}
    assert "sentinellms" not in arms


def test_incomplete_manifest_cannot_dispatch():
    register_implemented_entries()
    profile = require_profile("paper-v1")
    manifest = _manifest()
    del manifest["data"]["val_docs"]
    manifest["data"]["split_seed"] = 1
    jobs = configured_jobs(
        profile, manifest, requested={"attacks": ["knn_l2"], "arms": ["ab"]}
    )
    assert jobs == []
    report = dispatch_configured(
        profile, manifest, requested={"attacks": ["knn_l2"], "arms": ["ab"]},
        executor=lambda job: {"should": "not run"},
    )
    assert report["configured"] == 0
    assert report["status"] != "executed"


def _load_cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "run_full_battery.py"
    spec = importlib.util.spec_from_file_location("run_full_battery_paper_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_paper_run_dispatches_and_writes_status(tmp_path, monkeypatch):
    import json
    import sys

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    status_path = tmp_path / "status.json"
    module = _load_cli()
    monkeypatch.setattr(module, "ABDeployment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("deployment")))
    monkeypatch.setattr(module, "load_alpaca", lambda *a, **k: (_ for _ in ()).throw(AssertionError("alpaca")))
    monkeypatch.setattr(sys, "argv", [
        "run_full_battery.py", "--profile", "paper-v1",
        "--manifest", str(manifest_path), "--attacks", "knn_l2", "--arms", "ab",
        "--run-out", str(status_path),
    ])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    report = json.loads(status_path.read_text(encoding="utf-8"))
    assert report["status"] == "resource_blocked"
    assert report["configured"] == 1
    assert all(row["run_status"] == "resource_blocked" for row in report["results"])


def test_cli_execute_flag_constructs_the_real_executor(tmp_path, monkeypatch):
    import json
    import sys

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    status_path = tmp_path / "status.json"
    results_root = tmp_path / "runs"
    module = _load_cli()
    calls = []

    class FakeExecutor:
        def __init__(self, profile, manifest, **kwargs):
            calls.append({"profile": profile.name, "root_key": kwargs.get("root_key")})

        def __call__(self, job):
            return {"outcome": "executed", "job_id": job["job_id"],
                    "result_path": "fixture/result.json"}

    import obfuscation_attacks.evaluation.paper_executor as executor_module

    monkeypatch.setattr(executor_module, "PaperExecutor", FakeExecutor)
    monkeypatch.setattr(sys, "argv", [
        "run_full_battery.py", "--profile", "paper-v1",
        "--manifest", str(manifest_path), "--attacks", "knn_l2", "--arms", "ab",
        "--scenarios", "S-public", "--execute",
        "--root-key-file", str(tmp_path / "root.key"),
        "--result-root", str(results_root),
        "--run-out", str(status_path),
    ])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    report = json.loads(status_path.read_text(encoding="utf-8"))
    assert report["status"] == "executed"
    assert report["configured"] == 1
    assert calls and calls[0]["profile"] == "paper-v1"
    assert calls[0]["root_key"] is not None and len(calls[0]["root_key"]) == 32


def test_cli_without_execute_never_constructs_the_executor(tmp_path, monkeypatch):
    import json
    import sys

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    status_path = tmp_path / "status.json"
    module = _load_cli()

    class ExplodingExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("executor constructed without --execute")

    import obfuscation_attacks.evaluation.paper_executor as executor_module

    monkeypatch.setattr(executor_module, "PaperExecutor", ExplodingExecutor)
    monkeypatch.setattr(sys, "argv", [
        "run_full_battery.py", "--profile", "paper-v1",
        "--manifest", str(manifest_path), "--attacks", "knn_l2", "--arms", "ab",
        "--run-out", str(status_path),
    ])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    report = json.loads(status_path.read_text(encoding="utf-8"))
    assert report["status"] == "resource_blocked"


def test_cli_plan_with_manifest_marks_implemented_cells_configured(tmp_path, monkeypatch):
    import json
    import sys

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    module = _load_cli()
    monkeypatch.setattr(module, "ABDeployment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("deployment")))
    monkeypatch.setattr(module, "load_alpaca", lambda *a, **k: (_ for _ in ()).throw(AssertionError("alpaca")))
    monkeypatch.setattr(sys, "argv", [
        "run_full_battery.py", "--profile", "paper-v1", "--plan-only",
        "--manifest", str(manifest_path), "--plan-out", str(plan_path),
    ])
    try:
        module.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    configured = [job for job in plan["jobs"] if job["status"] == "configured"]
    assert configured
    assert {job["attack"] for job in configured} <= set(IMPLEMENTED_ENTRIES)
