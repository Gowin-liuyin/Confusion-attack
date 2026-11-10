"""Regression tests for the legacy-probe resume/aggregation and probe guard.

Defect 1: ``load_done`` treated every JSONL row with a ``job_id`` as completed,
so a failed or partial record silently suppressed the retry and the unit left
its per-(layer, steps) denominator. A record now counts as done only with a
successful status and result fields; failure rows keep the job identity.

Defect 2: ``rm_isa_full.py`` is a fixed-hyperparameter probe and must not be
runnable or aggregatable as the formal paper-v1 ISA executor (``run_c05``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import rm_common  # noqa: E402
import rm_isa_full  # noqa: E402
import rm_sda2023_full  # noqa: E402
import rm_sda_full  # noqa: E402
import rm_summary  # noqa: E402
import rm_vma_weight_full  # noqa: E402

pytestmark = pytest.mark.unit


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _job() -> dict:
    return {"job_id": "isa_er:0:L2", "method": "er", "pi": 0, "layer": 2,
            "steps": 50000, "model": "llama-3.2-1b",
            "model_path": r"F:\Models\Llama-3.2-1B-Instruct"}


def _ok_record(**extra) -> dict:
    rec = {
        "job_id": "isa_er:0:L2",
        "family": "isa",
        "variant": "er",
        "method": "er",
        "model": "llama-3.2-1b",
        "prompt_idx": 0,
        "layer": 2,
        "steps": 50000,
        "ttrsr": 0.6,
        "asr1": 0.4,
        "asr5": 0.8,
    }
    rec.update(extra)
    return rec


def test_successful_record_is_done(tmp_path: Path) -> None:
    log = _write(tmp_path / "isa_full.jsonl", [_ok_record()])
    assert rm_common.load_done(log) == {"isa_er:0:L2"}
    assert rm_common.load_done(
        log, required_fields=("layer", "steps", "ttrsr")
    ) == {"isa_er:0:L2"}


def test_truncated_final_line_does_not_abort_resume(tmp_path: Path) -> None:
    first = _ok_record()
    second = _ok_record(job_id="isa_er:1:L2", prompt_idx=1)
    truncated = '{"job_id": "isa_er:2:L2", "status": "ok", "tt'
    log = tmp_path / "isa_full.jsonl"
    log.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n" + truncated,
        encoding="utf-8",
    )
    stats: dict = {}
    done = rm_common.load_done(
        log, required_fields=("layer", "steps", "ttrsr"), stats=stats)
    assert done == {"isa_er:0:L2", "isa_er:1:L2"}
    assert stats["skipped_lines"] == 1
    # The malformed tail leaves its job pending; it is not marked done.
    assert "isa_er:2:L2" not in done
    # Without stats the loader still does not raise.
    assert rm_common.load_done(log) == {"isa_er:0:L2", "isa_er:1:L2"}


def test_invalid_utf8_tail_does_not_abort_resume(tmp_path: Path) -> None:
    log = tmp_path / "isa_full.jsonl"
    log.write_bytes((json.dumps(_ok_record()) + "\n").encode("utf-8") + b"\xff\xfe")
    stats: dict = {}
    done = rm_common.load_done(log, stats=stats)
    assert done == {"isa_er:0:L2"}
    assert stats["decode_errors"] == 1
    assert stats["skipped_lines"] == 1


def test_failed_record_is_retried(tmp_path: Path) -> None:
    rec = rm_isa_full._failure_record(_job(), "failed", "capability denied")
    assert rec["status"] == "failed"
    log = _write(tmp_path / "isa_full.jsonl", [rec])
    assert rm_common.load_done(log) == set()
    assert rm_common.load_done(log, required_fields=("layer", "steps", "ttrsr")) == set()


def test_legacy_enum_failure_status_is_retried(tmp_path: Path) -> None:
    # Old writers used str(res.status), i.e. "AttackStatus.FAILED".
    rec = {"job_id": "isa_er:0:L2", "status": "AttackStatus.FAILED"}
    log = _write(tmp_path / "isa_full.jsonl", [rec])
    assert rm_common.load_done(log) == set()


def test_partial_record_missing_identity_is_retried(tmp_path: Path) -> None:
    partial = {"job_id": "isa_er:0:L2", "status": "ok", "ttrsr": 0.6}
    log = _write(tmp_path / "isa_full.jsonl", [partial])
    assert rm_common.load_done(log, required_fields=("layer", "steps", "ttrsr")) == set()
    # Configuration fields alone are not result data: a row with status "ok"
    # but no metric is still a partial write and stays pending.
    config_only = {"job_id": "isa_er:0:L2", "status": "ok", "layer": 2, "steps": 50000}
    assert rm_common.load_done(_write(tmp_path / "config.jsonl", [config_only])) == set()
    # A bare identity row is not a completed unit either.
    bare = _write(tmp_path / "bare.jsonl", [{"job_id": "isa_er:0:L2"}])
    assert rm_common.load_done(bare) == set()


def test_retry_after_failure_is_resumed_as_done(tmp_path: Path) -> None:
    failed = rm_isa_full._failure_record(_job(), "failed", "boom")
    log = _write(tmp_path / "isa_full.jsonl", [failed, _ok_record()])
    assert rm_common.load_done(log, required_fields=("layer", "steps", "ttrsr")) == {
        "isa_er:0:L2"
    }


def test_failed_unit_keeps_its_denominator_group() -> None:
    records = [_ok_record(), rm_isa_full._failure_record(_job(), "failed", "boom")]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r.get("variant"), r.get("layer"), r.get("steps"))].append(r)
    group = groups[("er", 2, 50000)]
    # The failed unit stays in the (variant, layer, steps) denominator ...
    assert len(group) == 2
    # ... while only the successful record contributes recovered metrics.
    assert sum(rm_common.is_success_record(r) for r in group) == 1
    failure = group[1]
    assert failure["method"] == "er" and failure["prompt_idx"] == 0
    assert failure["probe_only"] is True and failure["profile"] == "probe"


def test_failure_record_is_marked_probe_only() -> None:
    rec = rm_isa_full._failure_record(_job(), "failed", "boom", seconds=1.234)
    assert rec["probe_only"] is True
    assert rec["profile"] == "probe"
    assert rec["layer"] == 2 and rec["steps"] == 50000


def test_paper_v1_profile_refused(capsys: pytest.CaptureFixture[str]) -> None:
    code = rm_isa_full.main(["--profile", "paper-v1", "--prompts", "0", "--steps", "10"])
    assert code == 2
    err = capsys.readouterr().err
    assert "probe" in err and "run_c05" in err


def test_only_probe_profile_is_accepted() -> None:
    assert rm_isa_full.profile_refusal("probe") is None
    assert rm_isa_full.profile_refusal("paper-v1") is not None
    assert rm_isa_full.profile_refusal("full") is not None


def test_deployment_artifact_model_is_refused(tmp_path: Path,
                                              capsys: pytest.CaptureFixture[str]) -> None:    # An ab-style artifact layout: model_server next to client/ and osnip/.
    artifact = tmp_path / "ab_runnable"
    (artifact / "model_server").mkdir(parents=True)
    (artifact / "client").mkdir()
    (artifact / "osnip").mkdir()
    (artifact / "client" / "client_secret.pt").write_bytes(b"placeholder")
    model_dir = str(artifact / "model_server")

    # Pointing the plaintext-caliber probe at the deployed model must be
    # refused before any job runs: the scoring space would silently mismatch.
    code = rm_isa_full.main(["--model", model_dir, "--steps", "10",
                             "--methods", "er:2", "--prompts", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "deployment artifact" in err and "run_c05" in err

    # A normal plaintext model directory (no artifact fingerprint) is accepted.
    assert rm_isa_full.deployment_artifact_refusal(model_dir) is not None
    assert rm_isa_full.deployment_artifact_refusal(
        str(tmp_path / "Llama-3.2-1B")) is None


def test_vma_weight_failure_record_keeps_identity(tmp_path: Path) -> None:
    rec = rm_vma_weight_full.failure_record(
        3, 1.0, "failed", "weight attack refused", layers=[4, 8], seconds=2.5)
    assert rec["job_id"] == "vma_weight:1.0:3"
    assert rec["family"] == "vma" and rec["variant"] == "weight_rowsort"
    assert rec["seed"] == 3 and rec["alpha_e"] == 1.0
    assert rec["matrices"] == 4 and rec["matrix_shape"] == "512x2048"
    assert rec["status"] == "failed"
    assert rec["status_reason"] == "weight attack refused"
    assert rec["probe_only"] is True and rec["seconds"] == 2.5
    log = _write(tmp_path / "vma_weight_full.jsonl", [rec])
    assert rm_common.load_done(log) == set()
    assert rm_common.load_done(
        log, required_fields=("variant", "seed", "alpha_e", "z1_top1")) == set()
    from obfuscation_attacks.data.contracts import AttackStatus

    assert rm_common.status_value(AttackStatus.FAILED) == "failed"
    assert rm_common.status_value(AttackStatus.OK) == "ok"


def test_summary_preserves_each_attempt_without_inferring_identity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    failed = rm_isa_full._failure_record(_job(), "failed", "boom")
    ok = _ok_record()
    (tmp_path / "isa_full.jsonl").write_text(
        json.dumps(failed) + "\n" + json.dumps(ok) + "\n", encoding="utf-8")
    assert rm_summary.main() == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    block = summary["isa_full.jsonl"]
    assert len(block) == 2
    row = next(r for r in block.values() if r["source_record_index"] == 2)
    assert next(r for r in block.values() if r["source_record_index"] == 1)["non_ok_status"]
    assert row["n_jobs"] == 1
    assert "non_ok_status" not in row
    assert row["ttrsr"]["n"] == 1
    assert row["ttrsr"]["mean"] == pytest.approx(0.6)


def test_summary_never_merges_legacy_job_id_collisions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    rows = [
        {"job_id": "legacy", "family": "vma", "variant": "hidden", "model": model,
         "layer": layer, "steps": steps, "asr1": score, "numerator": numerator,
         "denominator": denominator}
        for model, layer, steps, score, numerator, denominator in (
            ("a", 1, 10, 0.5, 1, 2), ("b", 1, 10, 0.9, 9, 10),
            ("a", 2, 10, 0.1, 1, 10), ("a", 1, 20, 0.2, 2, 10))
    ]
    (tmp_path / "vma_full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert rm_summary.main() == 0
    block = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["vma_full.jsonl"]
    assert len(block) == 4
    for row in block.values():
        original = rows[row["source_record_index"] - 1]
        assert row["model"] == original["model"]
        assert row["layer"] == original["layer"]
        assert row["steps"] == original["steps"]
        assert row["numerator"] == original["numerator"]
        assert row["denominator"] == original["denominator"]
        assert row["asr1"]["mean"] == original["asr1"]


def test_summary_does_not_score_failed_or_nonfinite_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    rows = [{"job_id": "failed", "status": "failed", "asr1": 1.0},
            {"job_id": "nonfinite", "status": "ok", "asr1": float("nan")}]
    (tmp_path / "vma_full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert rm_summary.main() == 0
    block = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["vma_full.jsonl"]
    assert all("asr1" not in row for row in block.values())


def test_summary_tolerates_truncated_final_line(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    (tmp_path / "isa_full.jsonl").write_text(
        json.dumps(_ok_record()) + "\n" + '{"job_id": "isa_er:1:L2", "tt',
        encoding="utf-8")
    assert rm_summary.main() == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["_read_stats"]["isa_full.jsonl"]["skipped_lines"] == 1
    row = next(iter(summary["isa_full.jsonl"].values()))
    assert row["n_jobs"] == 1
    assert row["ttrsr"]["n"] == 1
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Malformed lines skipped: 1" in md


SDA_REQUIRED = ("variant", "seed", "num_train_pairs", "epochs", "beam_width",
                "corpus_sha256", "ttrsr")


def _sda_args(num_train: int = 100000, epochs: int = 8, beam: int = 10):
    return argparse.Namespace(num_train=num_train, epochs=epochs, beam=beam)


def _sda_record(args, chash: str, seed: int = 0) -> dict:
    return {
        "job_id": rm_sda_full.job_id(args, chash, seed),
        "family": "sda", "method": "sda2021", "variant": "original_2021",
        "seed": seed, "num_train_pairs": args.num_train, "epochs": args.epochs,
        "beam_width": args.beam, "corpus_sha256": chash, "ttrsr": 0.6,
    }


def test_sda_job_id_tracks_training_config_and_corpus(tmp_path: Path) -> None:
    base = _sda_args()
    bigger = _sda_args(num_train=2_000_000)
    jid = rm_sda_full.job_id(base, "a" * 64, 0)
    jid_bigger = rm_sda_full.job_id(bigger, "a" * 64, 0)
    jid_corpus = rm_sda_full.job_id(base, "b" * 64, 0)
    assert jid.startswith("sda2021:")
    assert len({jid, jid_bigger, jid_corpus}) == 3
    log = _write(tmp_path / "sda_full.jsonl", [_sda_record(base, "a" * 64)])
    done = rm_common.load_done(log, required_fields=SDA_REQUIRED)
    # Identical config + corpus -> skipped as already done ...
    assert done == {jid}
    # ... while a bigger training budget or a changed corpus is a new job.
    assert jid_bigger not in done
    assert jid_corpus not in done


def test_sda_failure_record_keeps_config_and_corpus(tmp_path: Path) -> None:
    args = _sda_args()
    rec = rm_sda_full.failure_record(
        0, "c" * 64, "failed", "training diverged", args, seconds=1.5)
    assert rec["job_id"] == rm_sda_full.job_id(args, "c" * 64, 0)
    assert rec["method"] == "sda2021"
    assert rec["num_train_pairs"] == 100000
    assert rec["epochs"] == 8 and rec["beam_width"] == 10
    assert rec["corpus_sha256"] == "c" * 64
    assert rec["status"] == "failed" and rec["probe_only"] is True
    log = _write(tmp_path / "sda_full.jsonl", [rec])
    assert rm_common.load_done(log, required_fields=SDA_REQUIRED) == set()


SDA2023_REQUIRED = ("variant", "seed", "cipher_type", "num_train_pairs", "epochs",
                    "beam_width", "corpus_sha256", "ttrsr")


def _sda2023_record(args, chash: str, seed: int = 0,
                    homophonic: bool = False) -> dict:
    return {
        "job_id": rm_sda2023_full.job_id(args, chash, seed, homophonic),
        "family": "sda", "method": "sda2023", "variant": "original_2023",
        "seed": seed, "cipher_type": "homophonic" if homophonic else "1:1",
        "num_train_pairs": args.num_train, "epochs": args.epochs,
        "beam_width": args.beam, "corpus_sha256": chash, "ttrsr": 0.5,
        "losses": {"total": 3.0, "src": 1.0, "tgt": 2.0},
        "missing_loss": False, "missing_loss_reason": None,
    }


def test_sda2023_homophonic_key_builds() -> None:
    plain = np.asarray([0, 0, 1, 2, 2, 2], dtype=int)
    key, cipher = rm_sda2023_full.build_cipher(plain, 3, 0, homophonic=True)
    assert key.homophonic is True
    assert not key.is_one_to_one()
    assert cipher.size == plain.size
    assert set(cipher.tolist()) <= set(range(4 * 3))
    # The 1:1 condition remains a plain A-symbol permutation.
    key1, cipher1 = rm_sda2023_full.build_cipher(plain, 3, 0, homophonic=False)
    assert key1.is_one_to_one()
    assert set(cipher1.tolist()) <= set(range(3))
    sparse, _ = rm_sda2023_full.build_cipher(np.asarray([2, 2, 2, 7]), 1000, 0, homophonic=True)
    assert set(sparse.plain_to_cipher) == {2, 7}
    assert len(sparse.plain_to_cipher[2]) == 3


def test_sda2023_job_id_tracks_config_and_corpus(tmp_path: Path) -> None:
    base = _sda_args()
    bigger = _sda_args(num_train=200000)
    jid = rm_sda2023_full.job_id(base, "a" * 64, 0, False)
    jid_bigger = rm_sda2023_full.job_id(bigger, "a" * 64, 0, False)
    jid_corpus = rm_sda2023_full.job_id(base, "b" * 64, 0, False)
    jid_homo = rm_sda2023_full.job_id(base, "a" * 64, 0, True)
    assert jid.startswith("sda2023:")
    assert len({jid, jid_bigger, jid_corpus, jid_homo}) == 4
    log = _write(
        tmp_path / "sda2023_full.jsonl", [_sda2023_record(base, "a" * 64)])
    done = rm_common.load_done(log, required_fields=SDA2023_REQUIRED)
    # Identical config + corpus -> skipped as already done ...
    assert done == {jid}
    # ... while changed config, corpus or cipher condition is a new job.
    assert jid_bigger not in done
    assert jid_corpus not in done
    assert jid_homo not in done


def test_sda2023_failure_record_keeps_config_and_corpus(tmp_path: Path) -> None:
    args = _sda_args()
    rec = rm_sda2023_full.failure_record(
        0, True, "d" * 64, "failed", "training diverged", args, seconds=3.5)
    assert rec["job_id"] == rm_sda2023_full.job_id(args, "d" * 64, 0, True)
    assert rec["method"] == "sda2023"
    assert rec["cipher_type"] == "homophonic"
    assert rec["num_train_pairs"] == 100000
    assert rec["corpus_sha256"] == "d" * 64
    assert rec["status"] == "failed" and rec["probe_only"] is True
    log = _write(tmp_path / "sda2023_full.jsonl", [rec])
    assert rm_common.load_done(log, required_fields=SDA2023_REQUIRED) == set()


ISA_REQUIRED = ("layer", "steps", "ttrsr", "method", "prompt_idx", "variant",
                "model")


def test_isa_job_identity_includes_the_optimizer_budget(tmp_path: Path) -> None:
    small = rm_isa_full._isa_job_id("er", 0, 2, 1000, "model-a")
    paper = rm_isa_full._isa_job_id("er", 0, 2, 50000, "model-a")
    assert small != paper
    log = _write(
        tmp_path / "isa_full.jsonl",
        [_ok_record(job_id=small, steps=1000, model="model-a")],
    )
    done = rm_common.load_done(log, required_fields=ISA_REQUIRED)
    assert small in done
    # The completed 1,000-step probe must not satisfy the 50,000-step budget.
    assert paper not in done


def test_isa_job_identity_includes_the_model(tmp_path: Path) -> None:
    model_a = rm_isa_full.model_identity(r"F:\Models\Llama-3.2-1B-Instruct")
    model_b = rm_isa_full.model_identity(r"F:\Models\Llama-3.2-1B-Base")
    assert model_a != model_b
    jid_a = rm_isa_full._isa_job_id("er", 0, 2, 50000, model_a)
    jid_b = rm_isa_full._isa_job_id("er", 0, 2, 50000, model_b)
    assert jid_a != jid_b
    log = _write(
        tmp_path / "isa_full.jsonl",
        [_ok_record(job_id=jid_a, steps=50000, model=model_a)],
    )
    done = rm_common.load_done(log, required_fields=ISA_REQUIRED)
    assert done == {jid_a}
    # A job completed on model A must not be resumed as model B.
    assert jid_b not in done


def test_summary_separates_sda_configs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    base = _sda_args()
    bigger = _sda_args(num_train=2_000_000)
    rows = [
        _sda_record(base, "a" * 64, seed=0),
        _sda_record(base, "a" * 64, seed=1),      # same setting, different seed
        _sda_record(bigger, "a" * 64, seed=0),    # different training budget
        _sda_record(base, "b" * 64, seed=0),      # changed corpus
    ]
    (tmp_path / "sda_full.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert rm_summary.main() == 0
    block = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8"))["sda_full.jsonl"]
    assert len(block) == 4
    assert all(v["n_jobs"] == 1 for v in block.values())




def test_rm_tfma_chance_is_top1_over_the_full_vocabulary() -> None:
    import numpy as np

    import rm_tfma_full

    perm = np.random.default_rng(0).permutation(1000)
    plain = np.arange(1000)
    observed = sorted({int(c) for c in perm[plain]})
    cut, cipher_g, obs_g = rm_tfma_full._truncate_to_symbol_cap(perm, plain, 100)
    assert len(obs_g) <= 100
    assert len(obs_g) < len(observed)          # truncation actually happened
    # The query set is capped, but top-1 chance ranks all V reference symbols.
    assert rm_tfma_full.chance_top1(1000) == pytest.approx(1.0 / 1000)
    assert rm_tfma_full.chance_top1(1000) < 1.0 / len(obs_g)


def test_tfma_job_id_tracks_model_config_and_corpus(tmp_path: Path) -> None:
    import rm_tfma_full

    model_a = rm_tfma_full.model_identity(
        r"F:\Models\Llama-3.2-1B-Instruct", 128000)
    model_b = rm_tfma_full.model_identity(
        r"F:\Models\Llama-3.2-3B-Instruct", 128000)
    assert model_a != model_b
    # The tokenizer vocabulary is part of the identity even for the same path.
    assert rm_tfma_full.model_identity(
        r"F:\Models\Llama-3.2-1B-Instruct", 152064) != model_a
    jid = rm_tfma_full.job_id("independent", 0, 2500, 100, "a" * 64, model_a)
    jid_cap = rm_tfma_full.job_id("independent", 0, 5000, 100, "a" * 64, model_a)
    jid_corpus = rm_tfma_full.job_id("independent", 0, 2500, 100, "b" * 64, model_a)
    jid_variant = rm_tfma_full.job_id(
        "global_assignment", 0, 2500, 100, "a" * 64, model_a,
        cap_used=524, budget_bytes=268435456)
    jid_model = rm_tfma_full.job_id("independent", 0, 2500, 100, "a" * 64, model_b)
    assert len({jid, jid_cap, jid_corpus, jid_variant, jid_model}) == 5
    rec = {
        "job_id": jid, "family": "tfma", "variant": "independent", "seed": 0,
        "status": "ok", "model": model_a, "corpus_sha256": "a" * 64,
        "max_global_symbols": 2500, "top_k": 100,
        "cap_used": None, "memory_budget_bytes": None, "mapping_top1": 0.1,
    }
    fields = ("variant", "seed", "model", "corpus_sha256",
              "max_global_symbols", "top_k", "cap_used",
              "memory_budget_bytes", "mapping_top1")
    done = rm_common.load_done(
        _write(tmp_path / "tfma_full.jsonl", [rec]), required_fields=fields)
    assert done == {jid}
    # A changed cap, corpus, variant or model is a different job.
    assert jid_cap not in done
    assert jid_corpus not in done
    assert jid_variant not in done
    assert jid_model not in done
    # A partial row without the model/config/corpus identity is retried.
    partial = {k: v for k, v in rec.items()
               if k not in ("model", "corpus_sha256", "max_global_symbols")}
    assert rm_common.load_done(
        _write(tmp_path / "partial.jsonl", [partial]),
        required_fields=fields) == set()


def test_sda2023_loss_diagnostics_reads_real_keys() -> None:
    class _FakeResult:
        diagnostics = {
            "final_train_loss_total": 3.5,
            "final_train_loss_src": 1.25,
            "final_train_loss_tgt": 2.25,
        }

    info = rm_sda2023_full.loss_diagnostics(_FakeResult())
    assert info["losses"] == {"total": 3.5, "src": 1.25, "tgt": 2.25}
    assert info["missing_loss_reason"] is None
    # Legacy diagnostic names remain a fallback.
    legacy = rm_sda2023_full.loss_diagnostics({"final_total_loss": 4.0})
    assert legacy["losses"]["total"] == 4.0
    # A non-finite or absent total is reported, not silently stored as null.
    nan = rm_sda2023_full.loss_diagnostics({"final_train_loss_total": float("nan")})
    assert nan["missing_loss_reason"]
    absent = rm_sda2023_full.loss_diagnostics(
        {"method": "causal_lm_recurrence_2023"})
    assert "final_train_loss_total" in absent["missing_loss_reason"]


def test_rm_isa_target_matches_the_forwarded_positions() -> None:
    import rm_isa_full

    ids = list(range(10))
    assert rm_isa_full._CONTENT_START == 0
    assert rm_isa_full._content_slice(ids, 5) == [0, 1, 2, 3, 4]
    # The target slice uses the same start as the forwarded embedding prefix.
    start = rm_isa_full._CONTENT_START
    assert ids[start:start + 5] == rm_isa_full._content_slice(ids, 5)


def test_sda_success_record_marks_probe_and_resource_note() -> None:
    args = _sda_args()
    rec = rm_sda_full.success_record(
        args, "a" * 64, 0, alphabet_size=40, cipher_len=256,
        meter={"asr@1": 0.4, "ttrsr": 0.5, "symbol_error_rate": 0.6,
               "bleu4": 0.2},
        original="abc", recovered="abd",
        diagnostics={"final_train_loss": 1.5}, seconds=12.34)
    assert rec["probe_only"] is True
    assert rec["profile"] == "probe"
    assert rec["resource_note"] == rm_sda_full.RESOURCE_NOTE
    assert rec["bleu4"] == 0.2
    assert rec["job_id"] == rm_sda_full.job_id(args, "a" * 64, 0)


def test_sda2023_success_record_marks_probe_and_resource_note() -> None:
    args = _sda_args()
    loss_info = {"losses": {"total": 3.0, "src": 1.0, "tgt": 2.0},
                 "missing_loss_reason": None}
    rec = rm_sda2023_full.success_record(
        args, "a" * 64, 0, True, alphabet_size=40, cipher_len=256,
        meter={"asr@1": 0.4, "ttrsr": 0.5},
        original="abc", recovered="abd", loss_info=loss_info, seconds=5.0)
    assert rec["probe_only"] is True
    assert rec["profile"] == "probe"
    assert rec["resource_note"] == rm_sda2023_full.RESOURCE_NOTE
    assert rec["losses"]["total"] == 3.0
    assert rec["cipher_type"] == "homophonic"


def test_summary_metrics_include_every_produced_metric() -> None:
    # rm_sda_full writes bleu4; the TFMA/VMA runners write the others.
    assert {"bleu4", "mapping_asr100", "z1_top10", "mean_true_rank"} <= set(
        rm_summary.METRICS)


def test_summary_averages_bleu4(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    rec = _sda_record(_sda_args(), "a" * 64)
    rec["bleu4"] = 0.42
    (tmp_path / "sda_full.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8")
    assert rm_summary.main() == 0
    block = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8"))["sda_full.jsonl"]
    row = next(iter(block.values()))
    assert row["bleu4"]["mean"] == pytest.approx(0.42)
    assert row["bleu4"]["n"] == 1


class _FakeTok:
    def __init__(self, ids):
        self._ids = torch.tensor([ids], dtype=torch.long)
        self.add_special_tokens = None

    def __call__(self, text, *, add_special_tokens=True, return_tensors=None):
        self.add_special_tokens = add_special_tokens
        return {"input_ids": self._ids}


class _FakeInnerModel:
    def __init__(self):
        self.last_ids = None

    def __call__(self, input_ids, **kwargs):
        self.last_ids = input_ids
        h = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(hidden_states=[h] * 4)


class _FakeModel:
    def __init__(self, vocab: int = 128256):
        self.config = SimpleNamespace(vocab_size=vocab)
        self.model = _FakeInnerModel()


class _FakeHandle:
    def __init__(self, vocab: int = 128256):
        self._E = torch.zeros(vocab, 1)

    def embedding_matrix(self):
        return self._E


def test_tokenize_content_disables_special_tokens() -> None:
    tok = _FakeTok([791, 4062, 14198, 39935, 35308])
    ids = rm_isa_full.tokenize_content(tok, "The quick brown fox jumps")
    assert tok.add_special_tokens is False
    assert ids.tolist() == [791, 4062, 14198, 39935, 35308]
    content = rm_isa_full._content_slice(ids, 5)
    assert len(content) == 5
    assert 128000 not in content.tolist()


LLAMA_DIR = Path(r"F:\Models\Llama-3.2-1B-Instruct")


@pytest.mark.skipif(not LLAMA_DIR.is_dir(),
                    reason="local Llama tokenizer not available")
def test_llama_content_slice_has_five_ids_without_bos() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(LLAMA_DIR), local_files_only=True)
    ids = rm_isa_full.tokenize_content(tok, "The quick brown fox jumps")
    content = rm_isa_full._content_slice(ids, 5)
    assert content.tolist() == ids.tolist()[:5]
    assert len(content) == 5
    assert 128000 not in content.tolist()


def test_build_sample_scores_the_forwarded_content_rows(monkeypatch) -> None:
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
    ids = [791, 4062, 14198, 39935, 35308, 1234, 5678]
    tok = _FakeTok(ids)
    model = _FakeModel()
    sample, content, forwarded = rm_isa_full.build_sample(
        tok, 0, 2, _FakeHandle(), model)
    assert tok.add_special_tokens is False
    assert forwarded.tolist() == ids
    assert model.model.last_ids.tolist() == [ids]   # shape [1, T]
    assert content.tolist() == ids[:5]
    assert sample.plaintext_token_ids.tolist() == ids[:5]
    # The target rows are read from exactly the forwarded content positions.
    assert sample.obfuscated_hidden_states[:, 0].tolist() == [
        float(x) for x in ids[:5]]


def test_summary_markdown_header_matches_row_columns(monkeypatch,
                                                     tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    ok = {
        "job_id": "vma_weight:0.0:0", "family": "vma",
        "variant": "weight_rowsort", "seed": 0, "alpha_e": 0.0,
        "z1_top1": 0.9, "verified": True, "z1_exact": True, "seconds": 1.0,
    }
    failed = {
        "job_id": "vma_weight:1.0:0", "family": "vma",
        "variant": "weight_rowsort", "seed": 0, "alpha_e": 1.0,
        "status": "failed", "status_reason": "boom", "seconds": 0.5,
    }
    (tmp_path / "vma_weight_full.jsonl").write_text(
        json.dumps(ok) + "\n" + json.dumps(failed) + "\n", encoding="utf-8")
    assert rm_summary.main() == 0
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    table = [line for line in md.splitlines() if line.startswith("|")]
    header, separator, data_rows = table[0], table[1], table[2:]
    assert "verified" in header
    assert "z1_exact" in header
    assert "non_ok_status" in header
    width = header.count("|")
    assert separator.count("|") == width
    assert data_rows
    assert all(row.count("|") == width for row in data_rows)


def _tfma_record(seed: int = 0, max_symbols: int = 2500, top_k: int = 100,
                 chash: str = "a" * 64, model: str = "model-a") -> dict:
    import rm_tfma_full

    return {
        "job_id": rm_tfma_full.job_id(
            "independent", seed, max_symbols, top_k, chash, model),
        "family": "tfma", "variant": "independent", "seed": seed,
        "status": "ok", "model": model, "max_global_symbols": max_symbols,
        "top_k": top_k, "corpus_sha256": chash, "mapping_top1": 0.1,
    }


def test_summary_separates_tfma_configs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    rows = [
        _tfma_record(seed=0),
        _tfma_record(seed=1),                      # same config, different seed
        _tfma_record(seed=0, max_symbols=5000),    # different query cap
        _tfma_record(seed=0, chash="b" * 64),      # changed corpus
        _tfma_record(seed=0, model="model-b"),     # different model/tokenizer
    ]
    (tmp_path / "tfma_full.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert rm_summary.main() == 0
    block = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8"))["tfma_full.jsonl"]
    assert len(block) == 5
    assert all(v["n_jobs"] == 1 for v in block.values())


def test_tfma_truncation_rejects_nonpositive_cap() -> None:
    import rm_tfma_full

    perm = np.random.default_rng(0).permutation(1000)
    plain = np.arange(1000)
    for cap in (0, -5):
        with pytest.raises(ValueError, match=">= 1"):
            rm_tfma_full._truncate_to_symbol_cap(perm, plain, cap)
    with pytest.raises(ValueError, match="must be >= 1"):
        rm_tfma_full.bounded_symbol_cap(0, 1000)


def test_tfma_truncation_has_an_iteration_bound() -> None:
    import rm_tfma_full

    perm = np.random.default_rng(0).permutation(1000)
    with pytest.raises(RuntimeError, match="iterations"):
        rm_tfma_full._truncate_to_symbol_cap(
            perm, np.arange(1000), 1, max_iterations=0)


def test_tfma_global_cap_is_memory_bounded() -> None:
    import rm_tfma_full

    info = rm_tfma_full.bounded_symbol_cap(2500, 128256)
    budget_cap = (rm_tfma_full.DEFAULT_MEMORY_BUDGET_BYTES - 64 * 128256) // (128256 * 32 + 64)
    assert info["cap_requested"] == 2500
    assert info["cap_used"] == min(2500, budget_cap)
    assert info["cap_used"] < 2500
    assert info["cap_reduced"] is True
    assert "working set" in info["cap_reason"]
    assert info["memory_budget_bytes"] == rm_tfma_full.DEFAULT_MEMORY_BUDGET_BYTES
    fits = rm_tfma_full.bounded_symbol_cap(100, 1000)
    assert fits["cap_used"] == 100
    assert fits["cap_reduced"] is False


def test_tfma_global_cap_refuses_an_unusable_budget() -> None:
    import rm_tfma_full

    with pytest.raises(rm_tfma_full.ResourceBlockedError, match="resource-blocked"):
        rm_tfma_full.bounded_symbol_cap(2500, 128256, budget_bytes=128256 * 4)
    with pytest.raises(rm_tfma_full.ResourceBlockedError, match="resource-blocked"):
        rm_tfma_full.bounded_symbol_cap(1, 1000)


def test_tfma_global_resource_limit_does_not_shorten_test_stream():
    import rm_tfma_full

    class Tokenizer:
        vocab_size = 10
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])}

    info = {"cap_requested": 2, "cap_used": 2, "cap_reduced": False,
            "cap_reason": "fixture", "memory_budget_bytes": 4096}
    rows = rm_tfma_full.run_seed(Tokenizer(), 0, "fixture", "a" * 64, "model", "local", info,
                                 max_global_symbols=2)
    independent, global_result = rows
    assert independent["cipher_tokens_used"] == 5
    assert global_result["status"] == "resource_blocked"
    assert global_result["text_tokens"] == 5 and global_result["n_symbols"] == 5
    assert "mapping_top1" not in global_result


TFMA_CAP_FIELDS = ("variant", "seed", "model", "corpus_sha256",
                   "max_global_symbols", "top_k", "cap_used",
                   "memory_budget_bytes", "mapping_top1")


def _tfma_global_record(seed: int, cap_used: int,
                        budget_bytes: int) -> dict:
    import rm_tfma_full

    return {
        "job_id": rm_tfma_full.job_id(
            "global_assignment", seed, 2500, 100, "a" * 64, "model-a",
            cap_used=cap_used, budget_bytes=budget_bytes),
        "family": "tfma", "variant": "global_assignment", "seed": seed,
        "status": "ok", "model": "model-a", "max_global_symbols": 2500,
        "top_k": 100, "corpus_sha256": "a" * 64, "cap_used": cap_used,
        "memory_budget_bytes": budget_bytes, "mapping_top1": 0.1,
    }


def test_tfma_job_id_tracks_the_budget_derived_cap(tmp_path: Path) -> None:
    import rm_tfma_full

    jid_a = rm_tfma_full.job_id(
        "global_assignment", 0, 2500, 100, "a" * 64, "model-a",
        cap_used=524, budget_bytes=268435456)
    jid_b = rm_tfma_full.job_id(
        "global_assignment", 0, 2500, 100, "a" * 64, "model-a",
        cap_used=262, budget_bytes=134217728)
    assert jid_a != jid_b
    # Identical request/cap/budget stays the same job.
    assert jid_a == rm_tfma_full.job_id(
        "global_assignment", 0, 2500, 100, "a" * 64, "model-a",
        cap_used=524, budget_bytes=268435456)
    log = _write(
        tmp_path / "tfma_full.jsonl",
        [_tfma_global_record(0, 524, 268435456)],
    )
    done = rm_common.load_done(log, required_fields=TFMA_CAP_FIELDS)
    assert done == {jid_a}
    # A different memory budget/cap must not be resumed as done.
    assert jid_b not in done


def test_summary_separates_tfma_budgets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm_summary, "OUTDIR", tmp_path)
    rows = [
        _tfma_global_record(0, 524, 268435456),
        _tfma_global_record(1, 524, 268435456),   # same budget, different seed
        _tfma_global_record(0, 262, 134217728),   # half budget -> smaller cap
    ]
    (tmp_path / "tfma_full.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert rm_summary.main() == 0
    block = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8"))["tfma_full.jsonl"]
    assert len(block) == 3
    assert all(v["n_jobs"] == 1 for v in block.values())


SDA2023_REQUIRED_FIELDS = ("variant", "seed", "cipher_type", "num_train_pairs",
                           "epochs", "beam_width", "corpus_sha256", "ttrsr")


def test_sda2023_resume_requires_loss_evidence(tmp_path: Path) -> None:
    import rm_sda2023_full

    base = _sda2023_record(_sda_args(), "a" * 64)
    finite = _write(tmp_path / "finite.jsonl", [base])
    assert rm_common.load_done(
        finite, required_fields=SDA2023_REQUIRED_FIELDS,
        predicate=rm_sda2023_full.has_loss_evidence) == {base["job_id"]}

    null_loss = dict(
        base, losses={"total": None, "src": None, "tgt": None},
        missing_loss=False, missing_loss_reason=None)
    log_null = _write(tmp_path / "null.jsonl", [null_loss])
    # Field presence alone would resume it as done (the old defect) ...
    assert rm_common.load_done(
        log_null, required_fields=SDA2023_REQUIRED_FIELDS) == {base["job_id"]}
    # ... but without finite loss evidence or an explicit reason it is retried.
    assert rm_common.load_done(
        log_null, required_fields=SDA2023_REQUIRED_FIELDS,
        predicate=rm_sda2023_full.has_loss_evidence) == set()

    # An explicit missing_loss reason explains the absence and is accepted.
    explained = dict(
        null_loss, missing_loss=True,
        missing_loss_reason="SDA2023 result carries no final_train_loss_total")
    log_explained = _write(tmp_path / "explained.jsonl", [explained])
    assert rm_common.load_done(
        log_explained, required_fields=SDA2023_REQUIRED_FIELDS,
        predicate=rm_sda2023_full.has_loss_evidence) == {base["job_id"]}

    # A non-finite total with no reason is likewise retried.
    nan_loss = dict(null_loss, losses={"total": float("nan"), "src": 1.0,
                                       "tgt": 2.0})
    log_nan = _write(tmp_path / "nan.jsonl", [nan_loss])
    assert rm_common.load_done(
        log_nan, required_fields=SDA2023_REQUIRED_FIELDS,
        predicate=rm_sda2023_full.has_loss_evidence) == set()


def test_tfma_model_identity_tracks_weight_lineage(tmp_path: Path) -> None:
    import struct

    import rm_tfma_full

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "llama"}',
                                           encoding="utf-8")

    def write_weights(lineage: str) -> None:
        header = {
            "__metadata__": {"lineage": lineage, "format": "pt"},
            "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        payload = json.dumps(header).encode("utf-8")
        payload += b" " * (-len(payload) % 8)  # safetensors 8-byte padding
        (model_dir / "model.safetensors").write_bytes(
            struct.pack("<Q", len(payload)) + payload + b"\x00" * 4)

    write_weights("v1")
    first = rm_tfma_full.model_identity(str(model_dir), 100)
    assert rm_tfma_full.model_identity(str(model_dir), 100) == first
    # Same path/config/vocab, different weight header/metadata -> new identity.
    write_weights("v2")
    second = rm_tfma_full.model_identity(str(model_dir), 100)
    assert second != first
    jid_first = rm_tfma_full.job_id(
        "global_assignment", 0, 2500, 100, "a" * 64, first,
        cap_used=2, budget_bytes=1000)
    jid_second = rm_tfma_full.job_id(
        "global_assignment", 0, 2500, 100, "a" * 64, second,
        cap_used=2, budget_bytes=1000)
    assert jid_first != jid_second
    rec = {
        "job_id": jid_first, "family": "tfma", "variant": "global_assignment",
        "seed": 0, "status": "ok", "model": first,
        "max_global_symbols": 2500, "top_k": 100, "cap_used": 2,
        "memory_budget_bytes": 1000, "corpus_sha256": "a" * 64,
        "mapping_top1": 0.1,
    }
    done = rm_common.load_done(
        _write(tmp_path / "tfma_full.jsonl", [rec]),
        required_fields=TFMA_CAP_FIELDS)
    assert done == {jid_first}
    assert jid_second not in done


def test_sda_probe_job_identity_tracks_implementation(monkeypatch):
    args = _sda_args()
    for module, homophonic in ((rm_sda_full, None), (rm_sda2023_full, False)):
        def make_id():
            if homophonic is None:
                return module.job_id(args, "a" * 64, 0)
            return module.job_id(args, "a" * 64, 0, homophonic)
        previous = make_id()
        original = module.sda_probe_identity
        def changed(*a, **kw):
            result = original(*a, **kw)
            result["code_sha256"] = "f" * 64
            return result
        monkeypatch.setattr(module, "sda_probe_identity", changed)
        assert make_id() != previous
        monkeypatch.setattr(module, "sda_probe_identity", original)


def test_sda_scripts_provision_checkpoints_for_disk_pair_storage(monkeypatch, tmp_path):
    """num_train >= 50,000 makes fit_sda require a provenance path; the probe
    scripts must pass one (and resume an interrupted job) or every queued
    paper-floor run crashes at launch with ValueError."""
    from types import SimpleNamespace

    captured = {}

    class _FakeResult:
        # recovered_token_ids None -> the script takes its failure-record path,
        # so the test observes the constructor kwargs without invoking the scorer.
        recovered_token_ids = None
        status = None
        status_reason = ""
        diagnostics = {}

    def make_fake(key, extra):
        def fake(**params):
            captured[key] = params
            for name, value in extra.items():
                assert params.get(name) == value, f"{name} not forwarded"
            return SimpleNamespace(run=lambda ctx: _FakeResult())

        return fake

    import obfuscation_attacks.attacks.sda.sda2021_seq2seq as seq2seq_mod
    import obfuscation_attacks.attacks.sda.sda2023_recurrence_clm as clm_mod

    monkeypatch.setattr(rm_sda_full, "OUTDIR", tmp_path)
    monkeypatch.setattr(rm_sda2023_full, "OUTDIR", tmp_path)
    text = "the quick brown fox jumps over the lazy dog. " * 40
    chash = "c" * 64

    args_2021 = _sda_args()
    args_2021.num_train = 100000
    jid_2021 = rm_sda_full.job_id(args_2021, chash, 0)
    monkeypatch.setattr(
        seq2seq_mod, "SDA2021",
        make_fake("sda2021", {"checkpoint_path": str(rm_sda_full.checkpoint_path(jid_2021)),
                              "resume": False}))
    rec = rm_sda_full.run("model", 0, args_2021, text, chash)
    assert rec["num_train_pairs"] == 100000
    assert captured["sda2021"]["checkpoint_path"].startswith(str(tmp_path))
    assert not captured["sda2021"]["resume"]

    args_2023 = SimpleNamespace(num_train=60000, epochs=6, beam=10, model="")
    jid_2023 = rm_sda2023_full.job_id(args_2023, chash, 0, homophonic=True)
    expected = rm_sda2023_full.checkpoint_path(jid_2023)
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"interrupted")  # a crashed job must resume, not restart
    monkeypatch.setattr(clm_mod, "SDA2023",
                        make_fake("sda2023", {"checkpoint_path": str(expected),
                                              "resume": True}))
    rec = rm_sda2023_full.run(0, True, args_2023, text, chash)
    assert rec["num_train_pairs"] == 60000
    assert captured["sda2023"]["resume"] is True
