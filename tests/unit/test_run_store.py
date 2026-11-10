"""Acceptance tests for immutable run storage (F09)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from obfuscation_attacks.evaluation.run_store import (
    content_sha256,
    make_run_id,
    write_immutable,
)

pytestmark = pytest.mark.unit


def _payload(**overrides) -> dict:
    payload = {
        "schema_valid": True,
        "profile": "paper-v1",
        "attack": "knn_l2",
        "status": "completed",
        "executed_steps": 1,
        "metrics": {"top1": {"value": 1.0, "numerator": 1, "denominator": 1}},
    }
    payload.update(overrides)
    return payload


def test_content_hash_uses_file_bytes(tmp_path: Path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"real bytes")
    assert content_sha256(path) == hashlib.sha256(b"real bytes").hexdigest()
    assert content_sha256([path]) == content_sha256(path)
    assert content_sha256("real bytes") != content_sha256(path)


def test_run_id_changes_with_every_identity_field():
    base = dict(
        profile="paper-v1", model="llama-3.2-1b-base", arm="ab", scenario="S-public",
        attack="knn_l2", workpoint="w0", seed=17, data_sha256="d" * 64,
    )
    first = make_run_id(**base)
    assert first == make_run_id(**base)
    for field, value in (
        ("seed", 29), ("arm", "aloepri"), ("scenario", "S-deploy"),
        ("attack", "knn_cosine"), ("data_sha256", "e" * 64),
    ):
        changed = dict(base)
        changed[field] = value
        assert make_run_id(**changed) != first, field


def test_completed_run_cannot_be_overwritten(tmp_path: Path):
    run_id = "run-fixed"
    first = write_immutable(tmp_path, run_id, _payload(), status="completed")
    assert first.is_file()
    with pytest.raises(FileExistsError):
        write_immutable(tmp_path, run_id, _payload(), status="completed")


def test_unverified_run_cannot_be_overwritten(tmp_path: Path):
    write_immutable(tmp_path, "run-unverified", _payload(), status="not_verified")
    with pytest.raises(FileExistsError):
        write_immutable(tmp_path, "run-unverified", _payload(), status="completed")


def test_different_run_ids_coexist(tmp_path: Path):
    write_immutable(tmp_path, "run-a", _payload())
    write_immutable(tmp_path, "run-b", _payload())
    assert (tmp_path / "run-a" / "result.json").is_file()
    assert (tmp_path / "run-b" / "result.json").is_file()


def test_invalid_schema_is_refused(tmp_path: Path):
    with pytest.raises(ValueError):
        write_immutable(tmp_path, "run-bad", _payload(schema_valid=False))


def test_resume_cannot_claim_completion_with_zero_steps(tmp_path: Path):
    with pytest.raises(ValueError):
        write_immutable(tmp_path, "run-zero", _payload(executed_steps=0), status="completed")


def test_atomic_write_leaves_no_temp_file(tmp_path: Path):
    write_immutable(tmp_path, "run-atomic", _payload())
    leftovers = list((tmp_path / "run-atomic").glob("*.tmp"))
    assert leftovers == []
