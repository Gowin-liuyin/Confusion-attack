"""Acceptance tests for the paper-v1 real executor and its independent audit.

The fixtures use a tiny synthetic deployment and corpus so the call chain is
exercised without loading a 1B model; the real run is a separate acceptance step.
Every negative test mutates one required input and asserts the executor refuses
before the deployment factory is called.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from obfuscation_attacks.data.corpora import CLUSTERED_SPLIT, load_alpaca
from obfuscation_attacks.evaluation import paper_executor
from obfuscation_attacks.evaluation.export_randomness import _replay_permutation
from obfuscation_attacks.evaluation.paper_profile import require_profile
from obfuscation_attacks.evaluation.paper_executor import (
    PaperExecutor,
    texts_sha256,
)
from obfuscation_attacks.schemes.ab_deployment import Wire, WirePublic, WireTruth

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
V, D = 256, 8


@pytest.fixture(autouse=True)
def _registry():
    from obfuscation_attacks.evaluation.paper_profile import clear_faithful_entries
    from obfuscation_attacks.evaluation.paper_runner import register_implemented_entries

    clear_faithful_entries()
    register_implemented_entries()
    yield
    clear_faithful_entries()


_RNG = np.random.default_rng(0)
E = _RNG.normal(size=(V, D)).astype(np.float32)
FIXTURE_MASTER = bytes(range(32))
FIXTURE_SEED = 12345
PERM = _replay_permutation(V, list(range(V)), FIXTURE_MASTER, FIXTURE_SEED)
INV = np.empty_like(PERM)
INV[PERM] = np.arange(V)


class FakeDeployment:
    artifact = "fixture-artifact"
    vocab_size = V

    def __init__(self, *, random_first: bool = False) -> None:
        self.E = E
        self.E_clean = E
        self.E_dep = E
        self.perm = PERM
        self.inv_perm = INV
        self.random_first = random_first

    def encode(self, text):
        return np.asarray([ord(char) % V for char in text], dtype=np.int64)

    def observe(self, text, *, arm, request_id=0, key_seed=None, forward=True, **kwargs):
        ids = self.encode(text)
        permuted = self.perm[ids]
        if arm == "plaintext":
            public = WirePublic(arm=arm, request_id=request_id, seq_len=len(ids),
                                wire_token_ids=ids.copy())
        elif arm == "ab":
            if self.random_first and request_id == 0:
                z = np.full((len(ids), D), 5.0, dtype=np.float32)
            else:
                z = self.E[permuted].copy()
            public = WirePublic(arm=arm, request_id=request_id, seq_len=len(ids),
                                wire_vectors=z)
        else:
            public = WirePublic(arm=arm, request_id=request_id, seq_len=len(ids),
                                wire_token_ids=permuted.copy(),
                                hidden_states=self.E[permuted].copy()[None], layers=(1,))
        return Wire(public, WireTruth(plaintext_ids=ids, permuted_ids=permuted.copy()))

    def table(self, arm):
        return self.E


def _write_artifact(root: Path) -> dict[str, str]:
    import io

    import torch
    from safetensors.torch import save

    buffer = io.BytesIO()
    torch.save(
        {
            "perm_vocab": torch.tensor(PERM, dtype=torch.long),
            "inv_perm_vocab": torch.tensor(INV, dtype=torch.long),
            "metadata": {"alpha_e": 0.2, "variant": "fixture", "seed": FIXTURE_SEED,
                         "seed_source": "system_random", "random_source": "ChaCha20_HMAC_SHA256",
                         "randomness_mode": "system_entropy"},
            "randomness": {"random_source": "ChaCha20_HMAC_SHA256", "randomness_mode": "system_entropy"},
            "random_master_key": FIXTURE_MASTER,
        },
        buffer,
    )
    files = {
        "manifest.json": json.dumps({
            "format": "fixture",
            "components": {"B": {"recipe": "seed 17; seed 29; seed 43"}},
        }),
        "client/client_secret.pt": buffer.getvalue(),
        "osnip/encryptor.pt": b"encoder-fixture",
        "osnip/config.json": json.dumps({"width": D}),
        "model_server/config.json": json.dumps({"vocab_size": V}),
        "model_server/tokenizer.json": json.dumps({"fixture": True}),
        "model_server/tokenizer_config.json": json.dumps({"fixture": True}),
        "model_server/model.safetensors": save({"fixture": torch.zeros(1)}),
    }
    hashes = {}
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode("utf-8")
        path.write_bytes(content)
        import hashlib

        hashes[relative] = hashlib.sha256(content).hexdigest()
    return hashes


def _corpus_file(tmp_path: Path) -> tuple[Path, list[str], list[str], str]:
    import hashlib

    rows = [
        {"instruction": "Question " + hashlib.sha256(str(index).encode()).hexdigest(), "input": ""}
        for index in range(1600)
    ]
    path = tmp_path / "alpaca.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    corpus = load_alpaca(path, seed=20261107, n_knn=1000, n_robust=200, split_strategy=CLUSTERED_SPLIT)
    return path, corpus["knn"].texts(), corpus["robust"].texts(), corpus["knn"].manifest["split_sha256"]


@pytest.fixture()
def fixture_env(tmp_path, monkeypatch):
    from obfuscation_attacks.schemes import ab_deployment
    from obfuscation_attacks.evaluation import export_randomness, square_lineage, square_endpoints, encoder_lineage
    from types import SimpleNamespace

    monkeypatch.setattr(export_randomness, "_load_tokenizer",
                        lambda server: SimpleNamespace(vocab_size=V, all_special_ids=[]))
    fixture_lineage = lambda artifact, base: {"verified": True, "checked_tensors": 145}
    monkeypatch.setattr(paper_executor, "verify_square_lineage", fixture_lineage)
    monkeypatch.setattr(square_lineage, "verify_square_lineage", fixture_lineage)
    fixture_endpoints = lambda artifact, base: {"verified": True, "checked_rows": 2 * V}
    monkeypatch.setattr(paper_executor, "verify_square_endpoints", fixture_endpoints)
    monkeypatch.setattr(square_endpoints, "verify_square_endpoints", fixture_endpoints)
    fixture_encoder = lambda artifact: {"verified": True}
    monkeypatch.setattr(paper_executor, "verify_encoder_binding", fixture_encoder)
    monkeypatch.setattr(encoder_lineage, "verify_encoder_binding", fixture_encoder)

    identity = {
        "config_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "weights_sha256": "e" * 64,
        "variant_evidence": {"looks_instruct": False, "chat_template": False},
        "variant": "base",
        "name": "fixture",
        "files": {},
    }
    monkeypatch.setattr(
        paper_executor, "verify_model_identity",
        lambda path, expected_variant=None: dict(identity, path=str(path)),
    )
    monkeypatch.setattr(
        ab_deployment, "verify_model_identity",
        lambda path, expected_variant=None: dict(identity, path=str(path)),
    )
    artifact = tmp_path / "artifact"
    artifact_hashes = _write_artifact(artifact)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    alpaca, texts, robust_texts, split_sha = _corpus_file(tmp_path)
    import hashlib

    manifest = {
        "profile": "paper-v1",
        "protocol_sha256": require_profile("paper-v1").protocol_sha256,
        "models": {
            "llama-3.2-1b-base": {
                "name": "fixture", "path": str(model_dir), "source": "fixture",
                "revision": ab_deployment.local_model_revision(identity), "variant": "base", "license": "fixture",
                "config_sha256": "c" * 64, "tokenizer_sha256": "d" * 64,
                "weights_sha256": "e" * 64,
            },
        },
        "data": {
            "path": str(alpaca),
            "source_sha256": hashlib.sha256(alpaca.read_bytes()).hexdigest(),
            "knn_sha256": texts_sha256(texts),
            "split_seed": 20261107,
            "knn_texts": 1000,
            "char_min": 20,
            "char_max": 512,
            "robust_texts": 200,
            "robust_keys": 5,
            "robust_sha256": texts_sha256(robust_texts),
            "robust_char_min": 20,
            "robust_char_max": 512,
            "hashes_disjoint": True,
            "split_strategy": CLUSTERED_SPLIT,
            "split_sha256": split_sha,
        },
        "artifacts": {
            "ab": {
                "path": str(artifact),
                "files_sha256": artifact_hashes,
                "weights_sha256": paper_executor._combined_weight_sha(artifact_hashes),
                "lineage": {str(model_dir): {"verdict": "matches"}},
            },
        },
        "arms": {
            arm: {"adapter": "fixture", "source_verified": True}
            for arm in ("plaintext", "aloepri", "ab")
        },
    }
    return {
        "tmp_path": tmp_path,
        "manifest": manifest,
        "artifact": artifact,
        "model": model_dir,
        "alpaca": alpaca,
        "texts": texts,
        "robust_texts": robust_texts,
    }


def _job(attack="knn_l2", arm="ab", scenario="S-public"):
    return {
        "job_id": f"{arm}|llama-3.2-1b-base|{scenario}|{attack}",
        "attack": attack,
        "arm": arm,
        "scenario": scenario,
        "model": "llama-3.2-1b-base",
        "status": "configured",
        "applicable": True,
        "faithful_entry": True,
    }


def _executor(env, factory=None):
    calls = []

    def deployment_factory(artifact, model_path, root_key, needs_forward):
        calls.append({"artifact": artifact, "needs_forward": needs_forward})
        dep = (factory or FakeDeployment)()
        dep.artifact = artifact
        return dep

    executor = PaperExecutor(
        require_profile("paper-v1"),
        env["manifest"],
        root_key=b"k" * 32,
        result_root=env["tmp_path"] / "runs",
        evidence_root=env["tmp_path"] / "runs" / "evidence",
        deployment_factory=deployment_factory,
        code_identity={"commit": "fixture"},
    )
    return executor, calls


def _load_audit_module():
    path = ROOT / "scripts" / "audit_paper_run.py"
    spec = importlib.util.spec_from_file_location("audit_paper_run_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_executor_is_callable_by_the_dispatcher(fixture_env):
    from obfuscation_attacks.evaluation.paper_runner import dispatch_configured

    executor, calls = _executor(fixture_env)
    assert callable(executor)
    report = dispatch_configured(
        require_profile("paper-v1"),
        fixture_env["manifest"],
        requested={"attacks": ["knn_l2"], "arms": ["ab"], "scenarios": ["S-public"]},
        executor=executor,
    )
    assert report["status"] == "executed"
    assert report["configured"] == 1
    assert calls


def test_executor_refuses_ab_without_root_key(fixture_env):
    executor, _ = _executor(fixture_env)
    executor.root_key = None
    out = executor.run(_job(arm="ab"))
    assert out["outcome"] == "failed"
    assert "root key" in out["reason"]


def test_executor_runs_real_chain_and_writes_evidence(fixture_env):
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "executed"
    assert out["positive_control"]["top1"] == 1.0
    result_path = Path(out["result_path"])
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "paper-v1"
    assert payload["protocol_sha256"] == require_profile("paper-v1").protocol_sha256
    assert payload["n_prompts"] == 1000
    rows = payload["diagnostics"]["per_unit_predictions"]
    assert len(rows) == 1000
    assert sum(row["count"] for row in rows) == payload["n_tokens"]
    assert payload["diagnostics"]["evidence"]["sha256"]
    assert payload["identity"]["model"]["variant"] == "base"
    assert payload["identity"]["key_schedule"]["root_commitment"]
    assert "oracle permutation" in payload["metrics"]["frequency_buckets_scope"]
    assert calls and calls[0]["needs_forward"] is False


def test_executor_runs_key_robustness_200x5(fixture_env, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_load_table", lambda root, name="model.embed_tokens.weight": E)
    monkeypatch.setattr(audit, "_build_wire_deployment", lambda identity, arm, root_key: FakeDeployment())
    import transformers

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) % V for char in text]}

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakeTokenizer()),
    )
    executor, _ = _executor(fixture_env)
    job = _job(attack="knn_l2_key_robustness", arm="ab")
    out = executor.run(job)
    assert out["outcome"] == "executed", out
    assert out["positive_control"]["top1"] == 1.0
    path = Path(out["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["attack_id"] == "knn_l2_key_robustness"
    assert payload["n_prompts"] == 200
    assert payload["n_keys"] == 5
    assert len(payload["diagnostics"]["per_unit_predictions"]) == 1000
    assert sum(row["count"] for row in payload["diagnostics"]["per_unit_predictions"]) == payload["n_tokens"]
    assert payload["identity"]["data"]["role"] == "robust"
    assert payload["identity"]["data"]["n_texts"] == 200
    profile = require_profile("paper-v1")
    result = audit._audit_result(
        path, profile, profile.protocol_sha256, root_key=b"k" * 32
    )
    assert result["verdict"] == "audited_paper_scale", (result["gaps"], result["base_audit"])
    assert result["wire_recompute"]["n_rows"] == payload["n_tokens"]


def test_one_executor_keeps_role_specific_identity(fixture_env):
    executor, _ = _executor(fixture_env)
    main = executor.run(_job(attack="knn_l2", arm="ab"))
    robust = executor.run(_job(attack="knn_l2_key_robustness", arm="ab"))
    assert main["outcome"] == "executed" and robust["outcome"] == "executed"
    for outcome, role, n_texts in ((main, "knn", 1000), (robust, "robust", 200)):
        path = Path(outcome["result_path"])
        if not path.is_absolute():
            path = ROOT / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["identity"]["data"]["role"] == role
        assert payload["identity"]["data"]["n_texts"] == n_texts


def test_second_run_cannot_overwrite_the_first(fixture_env):
    executor, _ = _executor(fixture_env)
    first = executor.run(_job())
    path = Path(first["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    original = path.read_bytes()
    second = executor.run(_job())
    assert first["outcome"] == "executed"
    assert second["outcome"] == "failed"
    assert "already has a record" in second["reason"]
    assert path.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == first["run_id"]


def test_incomplete_manifest_is_refused_before_any_model_factory(fixture_env):
    del fixture_env["manifest"]["artifacts"]["ab"]["files_sha256"]
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "verification refused" in out["reason"]
    assert calls == []


def test_artifact_hash_drift_is_refused(fixture_env):
    fixture_env["manifest"]["artifacts"]["ab"]["files_sha256"]["osnip/encryptor.pt"] = "0" * 64
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "sha256 drift" in out["reason"] or "drift" in out["reason"]
    assert calls == []


def test_model_hash_and_variant_drift_are_refused(fixture_env):
    fixture_env["manifest"]["models"]["llama-3.2-1b-base"]["weights_sha256"] = "1" * 64
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed" and calls == []

    fixture_env["manifest"]["models"]["llama-3.2-1b-base"]["weights_sha256"] = "e" * 64
    fixture_env["manifest"]["models"]["llama-3.2-1b-base"]["variant"] = "instruct"
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed" and calls == []


def test_protocol_hash_drift_is_refused(fixture_env):
    fixture_env["manifest"]["protocol_sha256"] = "9" * 64
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "protocol_sha256 drift" in out["reason"]
    assert calls == []


def test_data_count_and_content_drift_are_refused(fixture_env):
    fixture_env["manifest"]["data"]["knn_texts"] = 200
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed" and calls == []

    fixture_env["manifest"]["data"]["knn_texts"] = 1000
    fixture_env["manifest"]["data"]["knn_sha256"] = "2" * 64
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "knn_sha256 drift" in out["reason"] and calls == []


def test_executor_refuses_missing_cluster_proof_before_loading_model(fixture_env):
    del fixture_env["manifest"]["data"]["split_strategy"]
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed" and calls == []
    assert "clustered split" in out["reason"]


def test_executor_recomputes_isolation_instead_of_trusting_manifest(fixture_env):
    executor, calls = _executor(fixture_env)
    loader = executor.data_loader

    def contaminated(*args, **kwargs):
        corpus = loader(*args, **kwargs)
        corpus["vma"].rows.append(corpus["reference"].rows[0])
        return corpus

    executor.data_loader = contaminated
    out = executor.run(_job())
    assert out["outcome"] == "failed" and calls == []
    assert "near-duplicate clusters cross role boundaries" in out["reason"]


def test_unsupported_arm_attack_or_scenario_is_refused(fixture_env):
    executor, calls = _executor(fixture_env)
    for job in (_job(arm="osnip"), _job(attack="vma_h_recovered_prefix"),
                _job(scenario="S-deploy")):
        out = executor.run(job)
        assert out["outcome"] == "failed"
    assert calls == []


def test_unverified_remote_revision_is_refused(fixture_env):
    executor, _ = _executor(fixture_env)
    model = dict(fixture_env["manifest"]["models"]["llama-3.2-1b-base"])
    model["revision"] = "a" * 40
    gaps = []
    executor._verify_model("llama-3.2-1b-base", model, gaps)
    assert any("revision is not the verified local content identity" in gap for gap in gaps)


def test_attack_predictions_do_not_read_truth_or_permutation():
    from obfuscation_attacks.evaluation.real_battery import knn_cell

    texts = ["abcdefghij"]
    dep = FakeDeployment()
    cell = knn_cell(dep, texts, arm="ab", metric="l2", forward=False,
                    oracle_diagnostics=False)
    truth = cell["truth_perm"].copy()
    cell2 = knn_cell(dep, texts, arm="ab", metric="l2", forward=False,
                     oracle_diagnostics=False)
    assert np.array_equal(cell["candidates"], cell2["candidates"])

    class NoPerm(FakeDeployment):
        @property
        def inv_perm(self):
            raise AssertionError("attack-side KNN read the secret permutation")

        @inv_perm.setter
        def inv_perm(self, value):
            self._ignored_inv = value

    no_perm = NoPerm()
    cell3 = knn_cell(no_perm, texts, arm="ab", metric="l2", forward=False,
                     oracle_diagnostics=False)
    assert np.array_equal(cell3["candidates"], cell["candidates"])
    assert np.array_equal(truth, cell3["truth_perm"])


def test_norm_attack_reports_recovered_mapping_separately():
    from obfuscation_attacks.evaluation.real_battery import knn_cell

    dep = FakeDeployment()
    plain = knn_cell(dep, ["abcdefghij"], arm="ab", metric="norm", forward=False,
                     oracle_diagnostics=False, recovered_mapping={})
    assert plain["composed"] is None
    assert plain["end_to_end_recovered_mapping"]["top1"]["numerator"] == 0
    assert plain["end_to_end_recovered_mapping"]["top1"]["denominator"] == len(plain["truth_plain"])
    assert plain["stage1"]["top1"]["denominator"] == len(plain["truth_perm"])


def test_uncovered_positions_stay_in_the_denominator():
    from obfuscation_attacks.evaluation.real_battery import asr_from_ranks, truth_rank

    candidates = np.tile(np.arange(100, dtype=np.int64), (3, 1))
    truth = np.array([5, 200, 3])
    ranks = truth_rank(candidates, truth)
    assert ranks.tolist() == [6, 101, 4]
    metrics = asr_from_ranks(ranks, ks=(1, 5, 10, 100))
    assert metrics["top1"] == {"value": 0.0, "numerator": 0, "denominator": 3}
    assert metrics["top100"] == {"value": 2 / 3, "numerator": 2, "denominator": 3}

    class MissDeployment(FakeDeployment):
        def observe(self, text, *, arm, request_id=0, **kwargs):
            ids = np.array([1, 2], dtype=np.int64)
            public = WirePublic(arm=arm, request_id=request_id, seq_len=2,
                                wire_vectors=np.full((2, D), 5.0, dtype=np.float32))
            return Wire(public, WireTruth(plaintext_ids=ids,
                                          permuted_ids=np.array([100, 101])))

    from obfuscation_attacks.evaluation.real_battery import knn_cell

    cell = knn_cell(MissDeployment(), ["ab"], arm="ab", metric="l2", forward=False,
                    oracle_diagnostics=False)
    assert cell["stage1"]["top1"]["denominator"] == 2
    assert cell["per_unit"][0]["count"] == 2


def test_audit_rejects_evidence_from_another_arm(tmp_path):
    from obfuscation_attacks.evaluation.real_battery import _arrays_digest

    audit = _load_audit_module()
    queries = np.ones((1, 2), dtype=np.float32)
    lengths = np.array([1], dtype=np.int64)
    evidence_path = tmp_path / "query.npz"
    np.savez(evidence_path, queries=queries, lengths=lengths, arm="aloepri", metric="l2")
    payload = {"arm": "ab", "target": "model/ab", "diagnostics": {
        "metric": "l2", "per_unit_predictions": [{"count": 1}],
        "evidence": {"path": str(evidence_path), "shape": [1, 2],
                     "arrays_sha256": _arrays_digest(queries, lengths, "aloepri", "l2")},
    }}
    gaps, _ = audit._gaps_from_evidence(payload, {})
    assert any("arm in query matrix" in gap for gap in gaps)


def test_audit_hash_detects_in_place_change_even_with_restored_mtime(tmp_path):
    import os

    audit = _load_audit_module()
    path = tmp_path / "weights"
    path.write_bytes(b"old")
    original = path.stat()
    first = audit._cached_sha(path)
    path.write_bytes(b"new")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert audit._cached_sha(path) != first


def test_audit_corpus_cache_is_bound_to_file_content(tmp_path, monkeypatch):
    import os
    from obfuscation_attacks.data import corpora

    audit = _load_audit_module()
    calls = []
    monkeypatch.setattr(corpora, "load_alpaca", lambda *a, **k: calls.append(1) or object())
    path = tmp_path / "corpus.json"
    path.write_bytes(b"old")
    original = path.stat()
    kwargs = dict(seed=17, n_knn=1, n_robust=0, split_strategy="exact_v1")
    first = audit._rebuild_corpus(path, **kwargs)
    assert audit._rebuild_corpus(path, **kwargs) is first and len(calls) == 1
    path.write_bytes(b"new")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert audit._rebuild_corpus(path, **kwargs) is not first and len(calls) == 2


def test_audit_reports_unrebuildable_corpus_as_a_gap(tmp_path):
    audit = _load_audit_module()
    path = tmp_path / "corrupt.json"
    path.write_text("not json", encoding="utf-8")
    gaps, _ = audit._gaps_from_hashes({}, {"data": {"path": str(path), "split_seed": 20261107}})
    assert any("cannot rebuild or verify frozen split" in gap for gap in gaps)


def test_independent_audit_recomputes_and_rejects_tampering(fixture_env, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_load_table", lambda root, name="model.embed_tokens.weight": E)
    monkeypatch.setattr(audit, "_build_wire_deployment", lambda identity, arm, root_key: FakeDeployment())

    import transformers

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) % V for char in text]}

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakeTokenizer()),
    )
    executor, _ = _executor(fixture_env)
    out = executor.run(_job())
    path = Path(out["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    profile = require_profile("paper-v1")
    digest = profile.protocol_sha256

    clean = audit._audit_result(path, profile, digest, root_key=b"k" * 32)
    assert clean["verdict"] == "audited_paper_scale", (clean["gaps"], clean["base_audit"])

    payload = json.loads(path.read_text(encoding="utf-8"))

    def _write_mutated(mutator, name):
        doc = json.loads(json.dumps(payload))
        mutator(doc)
        target = fixture_env["tmp_path"] / name
        target.write_text(json.dumps(doc), encoding="utf-8")
        return target

    tampered_rank = _write_mutated(
        lambda doc: doc["diagnostics"]["per_unit_predictions"][0]["rank"].__setitem__(0, 2),
        "tampered_rank.json",
    )
    assert audit._audit_result(tampered_rank, profile, digest, root_key=b"k" * 32)["verdict"] == "evidence_mismatch"

    tampered_count = _write_mutated(
        lambda doc: doc.__setitem__("n_tokens", doc["n_tokens"] + 1),
        "tampered_count.json",
    )
    assert audit._audit_result(tampered_count, profile, digest, root_key=b"k" * 32)["verdict"] == "evidence_mismatch"

    tampered_hash = _write_mutated(
        lambda doc: doc.__setitem__("data_sha256", "0" * 64),
        "tampered_hash.json",
    )
    assert audit._audit_result(tampered_hash, profile, digest, root_key=b"k" * 32)["verdict"] == "evidence_mismatch"

    tampered_control = _write_mutated(
        lambda doc: doc.__setitem__(
            "positive_control", {"name": "clean_embedding_retrieval", "top1": 0.5}
        ),
        "tampered_control.json",
    )
    result = audit._audit_result(tampered_control, profile, digest, root_key=b"k" * 32)
    assert result["verdict"] in {"control_failed", "evidence_mismatch"}

    evidence_path = Path(payload["diagnostics"]["evidence"]["path"])
    if not evidence_path.is_absolute():
        evidence_path = ROOT / evidence_path
    original = evidence_path.read_bytes()
    evidence_path.write_bytes(original[:-8] + b"tampered")
    try:
        result = audit._audit_result(path, profile, digest, root_key=b"k" * 32)
        assert result["verdict"] == "evidence_mismatch"
    finally:
        evidence_path.write_bytes(original)


def test_audit_rejects_denominator_laundering(fixture_env, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_load_table", lambda root, name="model.embed_tokens.weight": E)
    monkeypatch.setattr(audit, "_build_wire_deployment", lambda identity, arm, root_key: FakeDeployment())
    import transformers

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) % V for char in text]}

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakeTokenizer()),
    )
    executor, _ = _executor(fixture_env)
    out = executor.run(_job())
    path = Path(out["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    profile = require_profile("paper-v1")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc["diagnostics"]["per_unit_predictions"]
    first = rows[0]
    kept = {
        "unit": 0, "hits": first["hits"], "count": first["count"],
        "rank": first["rank"], "predicted": first["predicted"], "truth": first["truth"],
    }
    placeholders = [
        {"unit": index, "hits": 0, "count": 0, "rank": [], "predicted": [], "truth": []}
        for index in range(1, 1000)
    ]
    doc["diagnostics"]["per_unit_predictions"] = [kept] + placeholders
    doc["n_tokens"] = kept["count"]
    for label in ("top1", "top5", "top10", "top100"):
        k = int(label[3:])
        hits = sum(1 for value in kept["rank"] if value <= k)
        block = {"value": hits / kept["count"], "numerator": hits, "denominator": kept["count"]}
        doc["metrics"]["stage1_permuted_id"][label] = block
        doc["metrics"]["composed_with_oracle_permutation"][label] = dict(block)
    tampered = fixture_env["tmp_path"] / "laundered.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    result = audit._audit_result(tampered, profile, profile.protocol_sha256, root_key=b"k" * 32)
    assert result["verdict"] == "evidence_mismatch"
    assert any("frozen tokenisation" in gap for gap in result["gaps"])


def test_audit_rejects_a_forged_control(fixture_env, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_load_table", lambda root, name="model.embed_tokens.weight": E)
    monkeypatch.setattr(audit, "_build_wire_deployment", lambda identity, arm, root_key: FakeDeployment())
    import transformers

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) % V for char in text]}

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakeTokenizer()),
    )
    executor, _ = _executor(fixture_env)
    out = executor.run(_job())
    path = Path(out["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    profile = require_profile("paper-v1")
    fake_run = "run-fakecontrol"
    fake_path = (
        fixture_env["tmp_path"] / "runs" / "C01" / "nn_knn.euclidean"
        / "plaintext" / fake_run / "result.json"
    )
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text(json.dumps({
        "attack": "nn_knn.euclidean", "arm": "plaintext",
        "profile": "probe", "diagnostics": {"per_unit_predictions": []},
    }), encoding="utf-8")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["positive_control"] = {"name": "clean_embedding_retrieval", "top1": 1.0,
                               "run_id": fake_run}
    tampered = fixture_env["tmp_path"] / "forged_control.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    result = audit._audit_result(tampered, profile, profile.protocol_sha256, root_key=b"k" * 32)
    assert result["verdict"] == "evidence_mismatch"
    assert any("control" in gap for gap in result["gaps"])


def test_audit_rejects_composed_tampering(fixture_env, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_load_table", lambda root, name="model.embed_tokens.weight": E)
    monkeypatch.setattr(audit, "_build_wire_deployment", lambda identity, arm, root_key: FakeDeployment())
    executor, _ = _executor(fixture_env)
    out = executor.run(_job())
    path = Path(out["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    profile = require_profile("paper-v1")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["metrics"]["composed_with_oracle_permutation"]["top1"] = {
        "value": 0.0, "numerator": 0, "denominator": doc["n_tokens"]
    }
    tampered = fixture_env["tmp_path"] / "composed.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    result = audit._audit_result(tampered, profile, profile.protocol_sha256, root_key=b"k" * 32)
    assert result["verdict"] == "evidence_mismatch"
    assert any("composed" in gap for gap in result["gaps"])


def test_executor_refuses_undeclared_artifact_file(fixture_env):
    (fixture_env["artifact"] / "client" / "extra.bin").write_bytes(b"planted")
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "no declared hash" in out["reason"]
    assert calls == []


def test_executor_refuses_old_or_forged_export_even_with_matching_hashes(fixture_env):
    import hashlib
    import torch

    secret_path = fixture_env["artifact"] / "client" / "client_secret.pt"
    secret = torch.load(secret_path, map_location="cpu", weights_only=True)
    del secret["random_master_key"]
    torch.save(secret, secret_path)
    artifact = fixture_env["manifest"]["artifacts"]["ab"]
    artifact["files_sha256"]["client/client_secret.pt"] = hashlib.sha256(secret_path.read_bytes()).hexdigest()
    artifact["weights_sha256"] = paper_executor._combined_weight_sha(artifact["files_sha256"])
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "export randomness verification failed" in out["reason"]
    assert calls == []


def test_executor_does_not_trust_manifest_lineage_label(fixture_env, monkeypatch):
    def mismatch(artifact, base):
        raise ValueError("decoder tensor does not match transformed base")

    monkeypatch.setattr(paper_executor, "verify_square_lineage", mismatch)
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "decoder lineage verification failed" in out["reason"]
    assert calls == []


def test_executor_rejects_unwired_expanded_plan_axes(fixture_env):
    executor, calls = _executor(fixture_env)
    job = {**_job(), "plan_spec_id": "plan-" + "a" * 64}
    result = executor.run(job)
    assert result["outcome"] == "failed"
    assert "expanded matrix axes are not yet wired" in result["reason"]
    assert calls == []


def test_executor_rejects_mismatched_endpoint_even_when_decoder_matches(fixture_env, monkeypatch):
    def mismatch(artifact, base):
        raise ValueError("noisy endpoint does not match export replay")

    monkeypatch.setattr(paper_executor, "verify_square_endpoints", mismatch)
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "endpoint lineage verification failed" in out["reason"]
    assert calls == []


def test_executor_rejects_unbound_encoder(fixture_env, monkeypatch):
    def mismatch(artifact):
        raise ValueError("encoder training server contents do not match")
    monkeypatch.setattr(paper_executor, "verify_encoder_binding", mismatch)
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "encoder binding verification failed" in out["reason"]
    assert calls == []


def test_independent_audit_rechecks_encoder_binding(fixture_env, monkeypatch):
    audit = _load_audit_module()
    def mismatch(artifact):
        raise ValueError("encoder does not match")
    monkeypatch.setattr(audit, "verify_encoder_binding", mismatch)
    gaps, _ = audit._gaps_from_hashes({"arm": "ab"}, {"artifact": {"path": str(fixture_env["artifact"])},
                                                       "model": {"path": str(fixture_env["model"])}})
    assert any("encoder binding verification failed" in gap for gap in gaps)


def test_executor_rechecks_artifact_after_first_verified_job(fixture_env):
    import os

    executor, calls = _executor(fixture_env)
    executor.verify(_job())
    path = fixture_env["artifact"] / "osnip" / "encryptor.pt"
    stat = path.stat()
    path.write_bytes(b"changed-in-place")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(paper_executor.ExecutorRefusal, match="artifact.*drift"):
        executor.verify(_job())
    assert calls == []


def test_executor_rechecks_code_after_first_verified_job(fixture_env, monkeypatch):
    executor, _ = _executor(fixture_env)
    executor.code_identity = {"files_sha256": {"src/example.py": "first"}}
    monkeypatch.setattr(paper_executor, "_git_identity",
                        lambda: {"files_sha256": {"src/example.py": "first"}})
    executor.verify(_job())
    monkeypatch.setattr(paper_executor, "_git_identity",
                        lambda: {"files_sha256": {"src/example.py": "changed"}})
    with pytest.raises(paper_executor.ExecutorRefusal, match="code drift"):
        executor.verify(_job())


def test_independent_audit_rechecks_export_randomness(fixture_env):
    import torch

    audit = _load_audit_module()
    path = fixture_env["artifact"] / "client" / "client_secret.pt"
    secret = torch.load(path, map_location="cpu", weights_only=True)
    del secret["random_master_key"]
    torch.save(secret, path)
    gaps, _ = audit._gaps_from_hashes({}, {"artifact": {"path": str(fixture_env["artifact"])}})
    assert any("export randomness verification failed" in gap for gap in gaps)


def test_independent_audit_rechecks_decoder_lineage(fixture_env, monkeypatch):
    audit = _load_audit_module()

    def mismatch(artifact, base):
        raise ValueError("decoder tensor does not match transformed base")

    monkeypatch.setattr(audit, "verify_square_lineage", mismatch)
    gaps, _ = audit._gaps_from_hashes({}, {"artifact": {"path": str(fixture_env["artifact"])},
                                            "model": {"path": str(fixture_env["model"])}})
    assert any("decoder lineage verification failed" in gap for gap in gaps)


def test_independent_audit_rechecks_endpoint_lineage(fixture_env, monkeypatch):
    audit = _load_audit_module()

    def mismatch(artifact, base):
        raise ValueError("noisy endpoint does not match export replay")

    monkeypatch.setattr(audit, "verify_square_endpoints", mismatch)
    gaps, _ = audit._gaps_from_hashes({}, {"artifact": {"path": str(fixture_env["artifact"])},
                                            "model": {"path": str(fixture_env["model"])}})
    assert any("endpoint lineage verification failed" in gap for gap in gaps)


def test_executor_refuses_lineage_mismatch_and_missing_protocol(fixture_env):
    fixture_env["manifest"]["artifacts"]["ab"]["lineage"] = {
        str(fixture_env["model"]): {"verdict": "same_family_excess_residual"}
    }
    executor, calls = _executor(fixture_env)
    assert executor.run(_job())["outcome"] == "failed"
    assert calls == []

    fixture_env["manifest"]["artifacts"]["ab"]["lineage"] = {
        str(fixture_env["model"]): {"verdict": "matches"}
    }
    del fixture_env["manifest"]["protocol_sha256"]
    executor, calls = _executor(fixture_env)
    out = executor.run(_job())
    assert out["outcome"] == "failed"
    assert "protocol_sha256" in out["reason"]
    assert calls == []


def test_run_identity_changes_with_binding():
    from obfuscation_attacks.evaluation.run_store import make_run_id

    base = dict(profile="paper-v1", model="m/p", arm="p", scenario="S-public",
                attack="nn_knn.euclidean", workpoint="full", seed=1, data_sha256="d" * 64)
    first = make_run_id(**base, binding={"key_root_commitment": "a" * 64})
    second = make_run_id(**base, binding={"key_root_commitment": "b" * 64})
    assert first != second
    assert make_run_id(**base) != first


def test_identity_binding_separates_splits_and_code_revisions():
    from obfuscation_attacks.evaluation.real_battery import identity_binding

    base = {
        "model": {"weights_sha256": "1" * 64, "tokenizer_sha256": "2" * 64,
                  "config_sha256": "3" * 64},
        "artifact": {"weights_sha256": "4" * 64,
                     "files_sha256": {"client/client_secret.pt": "5" * 64}},
        "code": {"commit": "abc", "n_uncommitted": 0,
                 "files_sha256": {"src/x.py": "6" * 64}},
        "key_schedule": {"root_commitment": "7" * 64},
        "data": {"role": "knn", "n_texts": 1000, "knn_sha256": "a" * 64},
    }
    first = identity_binding(base, "p" * 64)["data_identity_sha256"]
    changed_data = {**base, "data": {**base["data"], "knn_sha256": "b" * 64}}
    assert identity_binding(changed_data, "p" * 64)["data_identity_sha256"] != first
    changed_code = {
        **base,
        "code": {**base["code"], "files_sha256": {"src/x.py": "8" * 64}},
    }
    assert identity_binding(changed_code, "p" * 64)["code_digest"] != identity_binding(
        base, "p" * 64
    )["code_digest"]


def test_audit_checks_each_top_k_not_only_top100():
    audit = _load_audit_module()
    stage = {
        "top1": {"value": 0.75, "numerator": 3, "denominator": 4},
        "top5": {"value": 0.75, "numerator": 3, "denominator": 4},
        "top10": {"value": 1.0, "numerator": 4, "denominator": 4},
        "top100": {"value": 1.0, "numerator": 4, "denominator": 4},
    }
    payload = {
        "n_tokens": 4,
        "n_prompts": 1,
        "attack": "nn_knn.euclidean",
        "attack_id": "knn_l2",
        "arm": "ab",
        "target": "llama-3.2-1b-base/ab",
        "diagnostics": {
            "per_unit_predictions": [{
                "unit": 0, "hits": 3, "count": 4, "rank": [1, 1, 1, 6],
                "predicted": [1, 2, 3, 9], "truth": [1, 2, 3, 4],
            }]
        },
        "metrics": {
            "stage1_permuted_id": stage,
            "composed_with_oracle_permutation": json.loads(json.dumps(stage)),
        },
    }
    gaps, _ = audit._gaps_from_ranks(payload)
    assert not any("top5" in gap or "top10" in gap or "top100" in gap for gap in gaps)

    forged = json.loads(json.dumps(payload))
    forged["metrics"]["stage1_permuted_id"]["top5"]["numerator"] = 4
    forged_gaps, _ = audit._gaps_from_ranks(forged)
    assert any("top5" in gap for gap in forged_gaps)


def test_audit_rejects_forged_permuted_truth(fixture_env, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_load_table", lambda root, name="model.embed_tokens.weight": E)
    monkeypatch.setattr(audit, "_build_wire_deployment", lambda identity, arm, root_key: FakeDeployment(random_first=True))
    import transformers

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) % V for char in text]}

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakeTokenizer()),
    )
    executor, _ = _executor(
        fixture_env, factory=lambda: FakeDeployment(random_first=True)
    )
    out = executor.run(_job())
    path = Path(out["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    profile = require_profile("paper-v1")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for row in doc["diagnostics"]["per_unit_predictions"]:
        row["rank"] = [1] * row["count"]
        row["hits"] = row["count"]
        row["truth"] = list(row["predicted"])
    total = doc["n_tokens"]
    for label in ("top1", "top5", "top10", "top100"):
        block = {"value": 1.0, "numerator": total, "denominator": total}
        doc["metrics"]["stage1_permuted_id"][label] = block
        doc["metrics"]["composed_with_oracle_permutation"][label] = dict(block)
    tampered = fixture_env["tmp_path"] / "forged_truth.json"
    tampered.write_text(json.dumps(doc), encoding="utf-8")
    result = audit._audit_result(tampered, profile, profile.protocol_sha256, root_key=b"k" * 32)
    assert result["verdict"] == "evidence_mismatch"
    assert any("permuted truth" in gap for gap in result["gaps"])



def _robust_run(fixture_env, monkeypatch):
    audit = _load_audit_module()
    monkeypatch.setattr(audit, "_load_table", lambda root, name="model.embed_tokens.weight": E)
    monkeypatch.setattr(audit, "_build_wire_deployment", lambda identity, arm, root_key: FakeDeployment())
    import transformers

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(char) % V for char in text]}

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakeTokenizer()),
    )
    executor, _ = _executor(fixture_env)
    job = _job(attack="knn_l2_key_robustness", arm="ab")
    out = executor.run(job)
    path = Path(out["result_path"])
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return audit, path, payload, require_profile("paper-v1"), out


def test_audit_rejects_text_key_grouping_forgery(fixture_env, monkeypatch):
    audit, path, payload, profile, _ = _robust_run(fixture_env, monkeypatch)
    rows = payload["diagnostics"]["per_unit_predictions"]
    # rows 0 and 5 both have count 41; swap their text/key grouping only
    rows[0]["text"], rows[5]["text"] = rows[5]["text"], rows[0]["text"]
    tampered = fixture_env["tmp_path"] / "grouping.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    result = audit._audit_result(tampered, profile, profile.protocol_sha256, root_key=b"k" * 32)
    assert result["verdict"] == "evidence_mismatch"
    assert any("text/key grouping" in gap for gap in result["gaps"])


def test_audit_rejects_ci_tampering(fixture_env, monkeypatch):
    audit, path, payload, profile, _ = _robust_run(fixture_env, monkeypatch)
    original = json.loads(json.dumps(payload))

    lo_doc = json.loads(json.dumps(original))
    lo_doc["ci"]["top1_micro"]["lo"] = float(lo_doc["ci"]["top1_micro"]["lo"]) - 0.01
    tampered = fixture_env["tmp_path"] / "ci.json"
    tampered.write_text(json.dumps(lo_doc), encoding="utf-8")
    result = audit._audit_result(tampered, profile, profile.protocol_sha256, root_key=b"k" * 32)
    assert result["verdict"] == "evidence_mismatch"
    assert any("ci: micro.lo" in gap for gap in result["gaps"])

    cluster_doc = json.loads(json.dumps(original))
    cluster_doc["ci"]["top1_micro"]["n_clusters"] = 999
    tampered = fixture_env["tmp_path"] / "ci_clusters.json"
    tampered.write_text(json.dumps(cluster_doc), encoding="utf-8")
    result = audit._audit_result(tampered, profile, profile.protocol_sha256, root_key=b"k" * 32)
    assert result["verdict"] == "evidence_mismatch"
    assert any("ci: micro.n_clusters" in gap for gap in result["gaps"])


def test_verify_data_requires_main_knn_floor(fixture_env):
    executor, _ = _executor(fixture_env)
    bad = json.loads(json.dumps(fixture_env["manifest"]))
    bad["data"]["knn_texts"] = 0
    executor.manifest = bad
    shortages: list[str] = []
    executor._verify_data(shortages)
    assert any("knn_texts" in item for item in shortages)


def test_preflight_requires_robust_keys_and_main_set(fixture_env):
    from obfuscation_attacks.evaluation.paper_profile import PreflightError, preflight_formal_start

    profile = require_profile("paper-v1")
    good = fixture_env["manifest"]
    preflight_formal_start(
        profile, good,
        {"attacks": ["knn_l2_key_robustness"], "arms": ["ab"], "scenarios": ["S-public"]},
        models={"llama-3.2-1b-base"},
    )
    bad_keys = json.loads(json.dumps(good))
    bad_keys["data"]["robust_keys"] = 4
    with pytest.raises(PreflightError, match="robust_keys"):
        preflight_formal_start(
            profile, bad_keys,
            {"attacks": ["knn_l2_key_robustness"], "arms": ["ab"], "scenarios": ["S-public"]},
            models={"llama-3.2-1b-base"},
        )
    bad_main = json.loads(json.dumps(good))
    bad_main["data"]["knn_texts"] = 0
    with pytest.raises(PreflightError):
        preflight_formal_start(
            profile, bad_main,
            {"attacks": ["knn_l2_key_robustness"], "arms": ["ab"], "scenarios": ["S-public"]},
            models={"llama-3.2-1b-base"},
        )


def test_control_failure_is_reported_separately(fixture_env, monkeypatch):
    class BrokenControl(FakeDeployment):
        def observe(self, text, *, arm, request_id=0, **kwargs):
            wire = super().observe(text, arm=arm, request_id=request_id, **kwargs)
            if arm == "plaintext":
                wire.public.wire_token_ids = (wire.public.wire_token_ids + 1) % V
            return wire

    executor, _ = _executor(fixture_env, factory=BrokenControl)
    out = executor.run(_job())
    assert out["outcome"] == "control_failed"
    assert out["positive_control_passed"] is False


def test_wire_recompute_without_a_key_does_not_crash_for_aloeori():
    audit = _load_audit_module()
    gaps, _ = audit._gaps_from_wire({"arm": "aloepri"}, {"artifact": {"path": "."}}, None)
    assert isinstance(gaps, list)


def test_failure_records_are_public_guarded(fixture_env):
    executor, _ = _executor(fixture_env)
    with pytest.raises(ValueError, match="truth/secret"):
        executor._failed({"job_id": "bad", "truth": [1, 2]}, "boom")
    failures = fixture_env["tmp_path"] / "runs" / "failures"
    assert not list(failures.glob("*.json")) if failures.exists() else True

    ok = executor._failed({"job_id": "ok"}, "execution error: boom")
    record = json.loads(Path(ok["error_path"]).read_text(encoding="utf-8"))
    assert "truth" not in record
    assert "plaintext_ids" not in record
    assert "permuted_ids" not in record
    assert "root_key" not in record
