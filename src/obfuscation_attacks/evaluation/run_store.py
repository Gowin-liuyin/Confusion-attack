"""Immutable, content-bound run storage (protocol §4.4, §6; F09).

A run identity covers model, arm, scenario, attack, workpoint, seed and the data
content hash. A completed result is written exactly once with an atomic rename;
a later attempt with the same identity is refused instead of silently replacing
the measurement. ``content_sha256`` hashes real file bytes, not a description.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

_CHUNK = 1 << 20


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hash_paths(paths: list[Path]) -> str:
    if len(paths) == 1:
        return _file_sha256(paths[0])
    combined = hashlib.sha256()
    for path in paths:
        combined.update(_file_sha256(path).encode("ascii"))
    return combined.hexdigest()


def content_sha256(obj: Any) -> str:
    """Hash file bytes for paths/lists of paths, canonical JSON for everything else."""
    if isinstance(obj, Path):
        return _file_sha256(obj)
    if isinstance(obj, (list, tuple)) and obj and all(
        isinstance(item, (str, Path)) and Path(item).is_file() for item in obj
    ):
        return _hash_paths([Path(item) for item in obj])
    if isinstance(obj, str) and Path(obj).is_file():
        return _file_sha256(Path(obj))
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def make_run_id(
    *,
    profile: str,
    model: str,
    arm: str,
    scenario: str,
    attack: str,
    workpoint: str,
    seed: int | str,
    data_sha256: str,
    binding: Mapping[str, Any] | None = None,
) -> str:
    identity = {
        "profile": str(profile),
        "model": str(model),
        "arm": str(arm),
        "scenario": str(scenario),
        "attack": str(attack),
        "workpoint": str(workpoint),
        "seed": int(seed) if isinstance(seed, int) or str(seed).isdigit() else str(seed),
        "data_sha256": str(data_sha256),
    }
    if binding:
        identity["binding"] = {
            str(key): (None if value is None else str(value))
            for key, value in sorted(binding.items())
        }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"run-{digest[:32]}"


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_immutable(
    root: Path | str,
    run_id: str,
    payload: dict,
    *,
    status: str = "completed",
) -> Path:
    """Atomically write one run; refuse to replace an existing completed run."""
    if payload.get("schema_valid") is not True:
        raise ValueError("refusing to write a payload whose schema_valid is not True")
    if status == "completed":
        steps = payload.get("executed_steps", 0)
        if isinstance(steps, bool) or not isinstance(steps, (int, float)) or steps <= 0:
            raise ValueError(
                "a completed run must report executed_steps > 0; "
                "zero-step resumes are not completion"
            )
    destination = Path(root) / str(run_id) / "result.json"
    if destination.exists():
        raise FileExistsError(
            f"run {run_id!r} already has a record at {destination}; "
            "a new measurement needs a new run identity"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["run_id"] = str(run_id)
    record["status"] = status
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_completed(root: Path | str, run_id: str) -> dict | None:
    path = Path(root) / str(run_id) / "result.json"
    record = _read(path)
    if record is None or record.get("status") != "completed":
        return None
    return record
