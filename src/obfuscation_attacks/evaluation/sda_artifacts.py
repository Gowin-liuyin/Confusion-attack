"""Isolated public SDA predictions and scoring-side verification."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from collections.abc import Mapping

from . import sda_public
from .public_guard import assert_public_payload
from .sda_stream import _checkpoint_sha256, _sda_code_identity, load_sda_checkpoint
from .sda_token_scoring import score_sda_token_rows


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _artifact_digest(artifact: dict) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in artifact.items() if k != "artifact_sha256"})).hexdigest()


def _runtime_identity() -> dict:
    from . import sda_group
    return {"training_code_and_protocol": _sda_code_identity(),
            "public_decoder_sha256": _checkpoint_sha256(sda_public.__file__),
            "group_decoder_sha256": _checkpoint_sha256(sda_group.__file__),
            "artifact_writer_sha256": _checkpoint_sha256(__file__)}


def _freeze_observations(observations: dict) -> dict:
    if set(observations) != {"surface", "rows", "public_uploads_sha256", "frontend", "processed_observations_sha256"}:
        raise ValueError("unexpected fields in public observation artifact")
    if not isinstance(observations["frontend"], dict) or set(observations["frontend"]) != {
            "vector_method", "distance_threshold", "candidate_k", "candidate_table_sha256",
            "candidate_table_shape", "metric", "query_chunk", "text_alignment", "status"}:
        raise ValueError("unexpected fields in public observation frontend")
    if any(set(row) != {"request_id", "group_id", "surface", "symbols", "candidates", "source_length"}
           for row in observations["rows"]):
        raise ValueError("unexpected fields in public observation rows")
    frozen = json.loads(_canonical(observations))
    assert_public_payload(frozen, label="SDA public observations")
    if frozen.get("processed_observations_sha256") != sda_public._observation_sha256(frozen):
        raise ValueError("processed SDA observations do not match their fingerprint")
    ids = [row["request_id"] for row in frozen["rows"]]
    if any(isinstance(i, bool) or not isinstance(i, int) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("public observation request IDs must be unique integers")
    if any(row["surface"] != frozen["surface"] for row in frozen["rows"]):
        raise ValueError("public observation surfaces do not match")
    if frozen["rows"] and frozen["surface"] not in {"ids", "vectors", "text"}:
        raise ValueError("unsupported public observation surface")
    for row in frozen["rows"]:
        length, symbols, group = row["source_length"], row["symbols"], row["group_id"]
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError("public observation source length must be a positive integer")
        if group is not None and not isinstance(group, str):
            raise ValueError("public observation group must be a string or None")
        if not isinstance(symbols, list) or not symbols or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63
                for value in symbols):
            raise ValueError("public observation symbols must be nonempty non-negative int64 IDs")
        if row["surface"] != "text" and len(symbols) != length:
            raise ValueError("public observation symbols and source length disagree")
        candidates = row["candidates"]
        if candidates is not None:
            if not isinstance(candidates, list) or len(candidates) != len(symbols):
                raise ValueError("public observation candidates do not match symbol count")
            for symbol, choices in zip(symbols, candidates):
                if (not isinstance(choices, list) or not choices or choices[0] != symbol
                        or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**63 for x in choices)
                        or len(set(choices)) != len(choices)):
                    raise ValueError("invalid public observation candidates")
    return frozen


def _publish_artifact(target: Path, artifact: dict) -> dict:
    artifact = json.loads(_canonical(artifact))
    artifact["artifact_sha256"] = _artifact_digest(artifact)
    assert_public_payload(artifact, label="SDA public prediction artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(artifact))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return artifact


def write_sda_predictions(observations: dict, checkpoint_path, output_path, *, beam_width: int,
                          device: str = "cpu", group_joint: bool = False, use_cache: bool = False,
                          decode_mode: str = "raw") -> dict:
    """Decode frozen public observations without accepting private scoring data.

    The output is an engineering artifact, not a formal-eligibility decision.
    The destination must not already exist; publication is exclusive and atomic.
    """
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"SDA prediction artifact already exists: {target}")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if not isinstance(group_joint, bool) or not isinstance(use_cache, bool):
        raise ValueError("group_joint and use_cache must be booleans")
    if decode_mode not in {"raw", "constrained"}:
        raise ValueError("decode mode must be raw or constrained")
    observations = _freeze_observations(observations)
    before = _checkpoint_sha256(checkpoint_path)
    runtime = _runtime_identity()
    bundle = load_sda_checkpoint(checkpoint_path, device=device)
    start = time.perf_counter()
    decoded = sda_public.decode_sda_observations(bundle, observations, beam_width=beam_width,
                                                  group_joint=group_joint, use_cache=use_cache, decode_mode=decode_mode)
    decode_seconds = time.perf_counter() - start
    sda_public.decoded_sda_tokens(decoded)  # Validate request identities and selected predictions.
    if group_joint:
        sda_public.decoded_sda_group_tokens(decoded)
    if _checkpoint_sha256(checkpoint_path) != before or _runtime_identity() != runtime:
        raise ValueError("checkpoint or implementation changed during SDA decoding")
    artifact = {
        "format": "sda_public_prediction_v1", "profile": "probe",
        "checkpoint_sha256": before, "checkpoint_identity": bundle["checkpoint_identity"],
        "checkpoint_epoch": bundle["checkpoint_epoch"],
        "selected_epoch": bundle["validation_checkpoint"].get("best_epoch"),
        "optimizer_steps": bundle["optimizer_steps"], "examples_seen": bundle["examples_seen"],
        "training_pairs": bundle["pairs"], "decode_seconds": decode_seconds,
        "method": bundle["method"], "model_config": vars(bundle["cfg"]),
        "length_mode": bundle["length_mode"], "space_token_id": bundle.get("space_token_id"),
        "beam_width": beam_width, "device": device, "group_joint": group_joint, "decode_mode": decode_mode,
        "cache_mode": "kv_incremental" if use_cache else "full_prefix",
        "runtime_identity": runtime, "decoded": decoded,
        "source_lengths": [{"request_id": row["request_id"], "length": row["source_length"]}
                           for row in observations["rows"]],
        "formal_status": "not_assessed",
    }
    artifact = _publish_artifact(target, artifact)
    return {"path": str(target), "artifact_sha256": artifact["artifact_sha256"],
            "checkpoint_sha256": before, "processed_observations_sha256": decoded["processed_observations_sha256"],
            "n_requests": len(decoded["rows"]), "formal_status": "not_assessed"}


def write_sda_candidate_predictions(observations: dict, model, tokenizer, decode_candidate_ids,
                                    checkpoint_path, output_path, *, expected_checkpoint_sha256: str,
                                    candidate_tokenizer_sha256: str, classifier_tokenizer_sha256: str,
                                    candidate_id_space: str, beam_width: int, batch_size: int = 128,
                                    device: str = "cpu") -> dict:
    """Freeze candidate-beam predictions from a loaded public-data classifier probe."""
    from ..attacks.sda import candidate_classifier, candidate_search
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"SDA prediction artifact already exists: {target}")
    observations = _freeze_observations(observations)
    runtime = {**_runtime_identity(), "candidate_classifier_sha256": _checkpoint_sha256(candidate_classifier.__file__),
               "candidate_search_sha256": _checkpoint_sha256(candidate_search.__file__)}
    loaded = candidate_classifier.load_candidate_classifier(
        model, checkpoint_path, expected_checkpoint_sha256=expected_checkpoint_sha256,
        candidate_tokenizer_sha256=candidate_tokenizer_sha256,
        classifier_tokenizer_sha256=classifier_tokenizer_sha256, device=device)
    scorer = candidate_search.BertCandidateScorer(model, tokenizer, decode_candidate_ids, device=device)
    start = time.perf_counter()
    decoded = sda_public.decode_candidate_observations(
        observations, scorer, beam_width=beam_width, batch_size=batch_size,
        scorer_sha256=expected_checkpoint_sha256, candidate_id_space=candidate_id_space)
    seconds = time.perf_counter() - start
    current_runtime = {**_runtime_identity(), "candidate_classifier_sha256": _checkpoint_sha256(candidate_classifier.__file__),
                       "candidate_search_sha256": _checkpoint_sha256(candidate_search.__file__)}
    if (_checkpoint_sha256(checkpoint_path) != expected_checkpoint_sha256 or runtime != current_runtime
            or candidate_classifier._model_digest(model) != loaded["selected_model_sha256"]):
        raise ValueError("candidate classifier or implementation changed during decoding")
    artifact = _publish_artifact(target, {
        "format": "sda_public_prediction_v1", "profile": "probe", "formal_status": "not_assessed",
        "method": "candidate_prefix_scoring_probe", "length_mode": "fixed",
        "checkpoint_sha256": expected_checkpoint_sha256, "selected_epoch": loaded["selected_epoch"],
        "selected_model_sha256": loaded["selected_model_sha256"], "runtime_identity": runtime,
        "candidate_tokenizer_sha256": candidate_tokenizer_sha256,
        "classifier_tokenizer_sha256": classifier_tokenizer_sha256,
        "tokenizer_commitment_status": "caller_supplied_unverified", "device": device,
        "beam_width": beam_width, "batch_size": batch_size, "decode_seconds": seconds, "decoded": decoded,
        "source_lengths": [{"request_id": row["request_id"], "length": row["source_length"]}
                           for row in observations["rows"]]})
    return {"path": str(target), "artifact_sha256": artifact["artifact_sha256"],
            "checkpoint_sha256": expected_checkpoint_sha256,
            "processed_observations_sha256": decoded["processed_observations_sha256"],
            "n_requests": len(decoded["rows"]), "formal_status": "not_assessed"}


def _verified_prediction_artifact(path, expected_artifact_sha256, expected_observations_sha256):
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict) or artifact.get("format") != "sda_public_prediction_v1":
        raise ValueError("unsupported SDA prediction artifact")
    if artifact.get("artifact_sha256") != _artifact_digest(artifact):
        raise ValueError("SDA prediction artifact fingerprint mismatch")
    if not isinstance(expected_artifact_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_artifact_sha256):
        raise ValueError("expected artifact fingerprint must be a SHA-256 hex digest")
    if artifact["artifact_sha256"] != expected_artifact_sha256:
        raise ValueError("SDA prediction artifact differs from the frozen expected fingerprint")
    if not isinstance(expected_observations_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_observations_sha256):
        raise ValueError("expected observations fingerprint must be a SHA-256 hex digest")
    if artifact["decoded"].get("processed_observations_sha256") != expected_observations_sha256:
        raise ValueError("SDA prediction artifact does not match the expected observations")
    return artifact


def _validate_source_lengths(source_lengths, request_ids, references):
    if not isinstance(source_lengths, list) or len(source_lengths) != len(request_ids):
        raise ValueError("prediction artifact lacks complete frozen source lengths; regenerate it for positional scoring")
    lengths = {}
    for row in source_lengths:
        if (not isinstance(row, dict) or set(row) != {"request_id", "length"} or
                isinstance(row["request_id"], bool) or not isinstance(row["request_id"], int) or
                isinstance(row["length"], bool) or not isinstance(row["length"], int) or row["length"] <= 0 or
                row["request_id"] in lengths):
            raise ValueError("invalid frozen source lengths")
        lengths[row["request_id"]] = row["length"]
    if set(lengths) != request_ids:
        raise ValueError("frozen source lengths do not match request IDs")
    try:
        mismatch = any(len(references[request_id]) != length for request_id, length in lengths.items())
    except TypeError as exc:
        raise ValueError("reference sequences must have a length") from exc
    if mismatch:
        raise ValueError("reference length does not match its frozen source length")


def score_sda_prediction_artifact(path, references: Mapping[int, object], *,
                                  expected_artifact_sha256: str,
                                  expected_observations_sha256: str,
                                  clusters: Mapping[int, Mapping[str, str]] | None = None, seed: int = 0,
                                  prediction_variant: str = "per_request_selected") -> dict:
    """Verify a frozen prediction artifact, then score using private references."""
    artifact = _verified_prediction_artifact(path, expected_artifact_sha256, expected_observations_sha256)
    if artifact["decoded"]["frontend"].get("text_alignment") is not None:
        raise ValueError("text surface requires separate character-alignment scoring")
    if artifact.get("length_mode") != "fixed":
        raise ValueError("variable-length predictions require separate alignment scoring")
    request_ids = set(sda_public.decoded_sda_tokens(artifact["decoded"]))
    if not isinstance(references, Mapping) or set(references) != request_ids:
        raise ValueError("scoring reference request IDs must exactly match the frozen predictions")
    _validate_source_lengths(artifact.get("source_lengths"), request_ids, references)
    if prediction_variant == "per_request_selected":
        predictions = sda_public.decoded_sda_tokens(artifact["decoded"])
    elif prediction_variant == "group_joint":
        if artifact.get("group_joint") is not True:
            raise ValueError("artifact does not contain joint group predictions")
        predictions = sda_public.decoded_sda_group_tokens(artifact["decoded"])
    else:
        raise ValueError("unsupported SDA prediction variant")
    result = score_sda_token_rows(references, predictions, clusters=clusters, seed=seed)
    return {"artifact_sha256": artifact["artifact_sha256"],
            "processed_observations_sha256": expected_observations_sha256,
            "prediction_variant": prediction_variant, "scoring": result, "formal_status": "not_assessed"}


def score_sda_text_prediction_artifact(path, references: Mapping[int, str], decode_tokens, *,
                                       expected_artifact_sha256: str, expected_observations_sha256: str,
                                       max_cells: int = 2_000_000) -> dict:
    """Score frozen attacker predictions against isolated reference strings."""
    from .sda_text_scoring import score_sda_text_rows
    artifact = _verified_prediction_artifact(path, expected_artifact_sha256, expected_observations_sha256)
    if not callable(decode_tokens):
        raise ValueError("an explicit public token decoder is required")
    predictions = sda_public.decoded_sda_text(artifact["decoded"], decode_tokens)
    if not isinstance(references, Mapping) or set(references) != set(predictions):
        raise ValueError("scoring reference request IDs must exactly match the frozen predictions")
    return {"artifact_sha256": artifact["artifact_sha256"],
            "processed_observations_sha256": expected_observations_sha256,
            "prediction_variant": "per_request_selected", "decoder_provenance": "caller_supplied_unverified",
            "scoring": score_sda_text_rows(references, predictions, max_cells=max_cells),
            "formal_status": "not_assessed"}


def write_sda_prediction_stream(observations: dict, checkpoint_path, output_path, *, beam_width: int,
                                device: str = "cpu", use_cache: bool = False,
                                group_joint: bool = False, decode_mode: str = "raw") -> dict:
    """Write per-request or group-joint predictions to an immutable JSONL file."""
    import torch
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"SDA prediction stream already exists: {target}")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if not isinstance(use_cache, bool) or not isinstance(group_joint, bool):
        raise ValueError("use_cache and group_joint must be booleans")
    if decode_mode not in {"raw", "constrained"}:
        raise ValueError("decode mode must be raw or constrained")
    if group_joint and decode_mode != "raw":
        raise ValueError("group joint search is already constrained; decode_mode applies only to per-request search")
    observations = _freeze_observations(observations)
    if not observations["rows"]:
        raise ValueError("SDA prediction stream needs at least one request")
    checkpoint_sha = _checkpoint_sha256(checkpoint_path)
    runtime = _runtime_identity()
    bundle = load_sda_checkpoint(checkpoint_path, device=device)
    if use_cache and bundle["method"] != "sda2023":
        raise ValueError("KV cache is currently supported only for SDA2023")
    if group_joint and bundle["length_mode"] != "fixed":
        raise ValueError("group joint decoding requires fixed length")
    if decode_mode == "constrained" and bundle["length_mode"] != "fixed":
        raise ValueError("constrained decoding requires fixed length")
    header = {"type": "header", "format": "sda_public_prediction_jsonl_v1", "profile": "probe",
              "checkpoint_sha256": checkpoint_sha, "checkpoint_identity": bundle["checkpoint_identity"],
              "checkpoint_epoch": bundle["checkpoint_epoch"], "method": bundle["method"],
              "selected_epoch": bundle["validation_checkpoint"].get("best_epoch"),
              "optimizer_steps": bundle["optimizer_steps"], "examples_seen": bundle["examples_seen"],
              "training_seed": bundle["training_seed"],
              "training_pairs": bundle["pairs"], "model_config": vars(bundle["cfg"]),
              "length_mode": bundle["length_mode"], "space_token_id": bundle.get("space_token_id"),
              "beam_width": beam_width, "device": device,
              "decode_mode": "group_constrained" if group_joint else decode_mode,
              "cache_mode": "kv_incremental" if use_cache else "full_prefix",
              "public_uploads_sha256": observations["public_uploads_sha256"],
              "processed_observations_sha256": observations["processed_observations_sha256"],
              "frontend": observations["frontend"], "runtime_identity": runtime,
              "source_lengths": [{"request_id": row["request_id"], "length": row["source_length"]}
                                 for row in observations["rows"]],
              "n_requests": len(observations["rows"]), "prediction_variant": "group_joint" if group_joint else "per_request",
              "formal_status": "not_assessed"}
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    successful = failed = 0
    content_digest = hashlib.sha256()
    started = time.perf_counter()
    known_errors = {"cipher exceeds configured max_cipher_len": "cipher_exceeds_configured_length",
                    "beam search has no finite candidates": "no_finite_beam_candidates"}
    try:
        with os.fdopen(fd, "wb") as handle:
            def write_content(record):
                assert_public_payload(record, label="SDA prediction stream")
                line = _canonical(record) + b"\n"
                handle.write(line)
                content_digest.update(line)
            write_content(header)
            groups = {}
            for row in observations["rows"]:
                label = (("group", row["group_id"]) if row["group_id"] is not None else
                         ("request", row["request_id"])) if group_joint else ("request", row["request_id"])
                groups.setdefault(label, []).append(row)
            for label, members in groups.items():
                try:
                    if group_joint:
                        from .sda_group import decode_sda_group
                        decoded = decode_sda_group(bundle, [row["symbols"] for row in members],
                                                   beam_width=beam_width, use_cache=use_cache)
                        if decoded["status"] != "ok":
                            raise ValueError("joint decoding infeasible")
                        parts, offset = [], 0
                        for row in members:
                            length = len(row["symbols"])
                            parts.append({"request_id": row["request_id"], "tokens": decoded["tokens"][offset:offset + length]})
                            offset += length
                        record = {"type": "group_prediction", "group": list(label),
                                  "request_ids": [row["request_id"] for row in members],
                                  "status": "ok", "decode": decoded, "rows": parts}
                        sda_public.decoded_sda_group_tokens({"mode": "neural",
                            "rows": [{"request_id": row["request_id"], "decode": {"tokens": []}} for row in members],
                            "group_joint": [record]})
                    else:
                        row = members[0]
                        decoded = sda_public.decode_sda(bundle, row["symbols"], beam_width=beam_width,
                                                        use_cache=use_cache, mode=decode_mode)
                        sda_public.decoded_sda_tokens({"mode": "neural", "rows": [{"request_id": row["request_id"],
                                                                                     "decode": decoded}]})
                        record = {"type": "prediction", "request_id": row["request_id"], "status": "ok", "decode": decoded}
                except (MemoryError, torch.cuda.OutOfMemoryError):
                    raise
                except Exception as exc:
                    failed += len(members)
                    record = {"type": "group_prediction" if group_joint else "prediction", "status": "failed",
                              "error_type": type(exc).__name__, "reason_code": known_errors.get(str(exc),
                                  "joint_infeasible" if str(exc) == "joint decoding infeasible" else "decoder_exception"),
                              "error_detail": "omitted"}
                    if group_joint:
                        record.update({"group": list(label), "request_ids": [row["request_id"] for row in members]})
                    else:
                        record["request_id"] = members[0]["request_id"]
                else:
                    successful += len(members)
                write_content(record)
            if _checkpoint_sha256(checkpoint_path) != checkpoint_sha or _runtime_identity() != runtime:
                raise ValueError("checkpoint or implementation changed during SDA decoding")
            footer = {"type": "footer", "n_requests": len(observations["rows"]),
                      "succeeded": successful, "failed": failed, "content_sha256": content_digest.hexdigest(),
                      "decode_seconds": time.perf_counter() - started}
            handle.write(_canonical(footer) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        digest = _checkpoint_sha256(temporary)
        os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(target), "artifact_sha256": digest, "checkpoint_sha256": checkpoint_sha,
            "processed_observations_sha256": observations["processed_observations_sha256"],
            "n_requests": len(observations["rows"]), "succeeded": successful, "failed": failed,
            "prediction_variant": header["prediction_variant"],
            "formal_status": "not_assessed"}


def score_sda_prediction_stream(path, references: Mapping[int, object], *, expected_artifact_sha256: str,
                                expected_observations_sha256: str,
                                clusters: Mapping[int, Mapping[str, str]] | None = None, seed: int = 0) -> dict:
    """Verify the complete frozen JSONL stream before scoring its successful rows."""
    for value in (expected_artifact_sha256, expected_observations_sha256):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("expected fingerprints must be SHA-256 hex digests")
    if _checkpoint_sha256(path) != expected_artifact_sha256:
        raise ValueError("SDA prediction stream differs from the frozen expected fingerprint")
    digest, content_digest = hashlib.sha256(), hashlib.sha256()
    predictions, request_ids, header, footer, failed = {}, set(), None, None, 0
    with open(path, "rb") as handle:
        for raw in handle:
            digest.update(raw)
            record = json.loads(raw)
            if not isinstance(record, dict) or footer is not None:
                raise ValueError("invalid SDA prediction stream record order")
            kind = record.get("type")
            if header is None:
                if kind != "header" or record.get("format") != "sda_public_prediction_jsonl_v1":
                    raise ValueError("unsupported SDA prediction stream header")
                header = record
                if header.get("processed_observations_sha256") != expected_observations_sha256:
                    raise ValueError("SDA prediction stream does not match expected observations")
                if header.get("frontend", {}).get("text_alignment") is not None:
                    raise ValueError("text surface requires separate character-alignment scoring")
                if header.get("length_mode") != "fixed":
                    raise ValueError("variable-length predictions require separate alignment scoring")
                if header.get("prediction_variant") not in {"per_request", "group_joint"}:
                    raise ValueError("unsupported SDA stream prediction variant")
                content_digest.update(raw)
            elif kind == "prediction":
                if header["prediction_variant"] != "per_request":
                    raise ValueError("prediction record does not match stream variant")
                request_id = record.get("request_id")
                if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id in request_ids:
                    raise ValueError("duplicate or invalid SDA prediction request ID")
                request_ids.add(request_id)
                if record.get("status") == "ok":
                    predictions.update(sda_public.decoded_sda_tokens({"mode": "neural", "rows": [record]}))
                elif record.get("status") == "failed":
                    failed += 1
                else:
                    raise ValueError("invalid SDA prediction status")
                content_digest.update(raw)
            elif kind == "group_prediction":
                if header["prediction_variant"] != "group_joint":
                    raise ValueError("prediction record does not match stream variant")
                members = record.get("request_ids")
                if (not isinstance(members, list) or not members or
                        any(isinstance(i, bool) or not isinstance(i, int) for i in members) or
                        len(set(members)) != len(members) or request_ids.intersection(members)):
                    raise ValueError("duplicate or invalid SDA group request IDs")
                request_ids.update(members)
                if record.get("status") == "ok":
                    predictions.update(sda_public.decoded_sda_group_tokens({"mode": "neural",
                        "rows": [{"request_id": i, "decode": {"tokens": []}} for i in members],
                        "group_joint": [record]}))
                elif record.get("status") == "failed":
                    if "rows" in record or "decode" in record:
                        raise ValueError("failed group cannot contain predictions")
                    failed += len(members)
                else:
                    raise ValueError("invalid SDA group prediction status")
                content_digest.update(raw)
            elif kind == "footer":
                footer = record
            else:
                raise ValueError("invalid SDA prediction stream record")
    if (digest.hexdigest() != expected_artifact_sha256 or header is None or footer is None
            or footer.get("content_sha256") != content_digest.hexdigest()
            or footer.get("n_requests") != len(request_ids) or header.get("n_requests") != len(request_ids)
            or footer.get("failed") != failed or footer.get("succeeded") != len(predictions)):
        raise ValueError("incomplete or changed SDA prediction stream")
    if not isinstance(references, Mapping) or set(references) != request_ids:
        raise ValueError("scoring reference request IDs must exactly match the frozen stream")
    _validate_source_lengths(header.get("source_lengths"), request_ids, references)
    scoring = score_sda_token_rows(references, predictions, clusters=clusters, seed=seed)
    return {"artifact_sha256": expected_artifact_sha256,
            "processed_observations_sha256": expected_observations_sha256,
            "prediction_variant": header["prediction_variant"], "failed_requests": failed,
            "scoring": scoring, "formal_status": "not_assessed"}
