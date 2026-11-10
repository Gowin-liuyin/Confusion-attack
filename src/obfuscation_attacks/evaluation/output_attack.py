"""Replay an output attack using only stored server-visible generation steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def recover_output_from_trace(path, *, recovered_mapping: Mapping[int, int] | None = None,
                              expected_sha256: str | None = None) -> dict:
    """Decode greedy server logits through an independently recovered mapping.

    Ambiguous ties and unmapped symbols remain -1. Client truth, actual secret
    permutations and the scoring-side continuation are neither accepted nor read.
    """
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("public generation manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("format") != "server_generation_trace_v2":
        raise ValueError("unsupported public generation trace")
    if manifest.get("greedy") is not True:
        raise ValueError("logit argmax cannot reproduce client sampling")
    arm = manifest.get("arm")
    if arm not in {"plaintext", "aloepri", "ab"}:
        raise ValueError("unsupported generation arm")
    if arm != "plaintext" and recovered_mapping is None:
        raise ValueError("protected output requires an attacker-recovered mapping")
    mapping = {} if recovered_mapping is None else {int(k): int(v) for k, v in recovered_mapping.items()}
    steps = manifest.get("steps")
    if not isinstance(steps, list) or manifest.get("n_steps") != len(steps):
        raise ValueError("trace step count mismatch")
    predictions, ambiguity, uncovered = [], 0, 0
    for index, step in enumerate(steps):
        if step.get("index") != index:
            raise ValueError("trace step indices are not contiguous")
        arrays = step.get("arrays")
        if not isinstance(arrays, dict) or not {"last_logits", "server_logits"} <= set(arrays):
            raise ValueError("trace lacks replayable server logits")
        allowed = {"last_logits", "server_logits", "wire_vectors", "wire_token_ids", "hidden_states"}
        if set(arrays) - allowed:
            raise ValueError("unknown server trace array")
        loaded = {}
        for name, info in arrays.items():
            if not isinstance(info, dict) or info.get("path") != f"step-{index:06d}/{name}.npy":
                raise ValueError("invalid trace chunk name")
            chunk = path.parent / info["path"]
            digest_chunk = hashlib.sha256()
            with chunk.open("rb") as handle:
                for data in iter(lambda: handle.read(1 << 20), b""):
                    digest_chunk.update(data)
            if digest_chunk.hexdigest() != info.get("sha256"):
                raise ValueError("public generation chunk hash mismatch")
            array = np.load(chunk, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != info.get("shape") or str(array.dtype) != info.get("dtype"):
                raise ValueError("public generation array metadata mismatch")
            loaded[name] = array
        logits = loaded["last_logits"]
        server = loaded["server_logits"]
        length = int(step.get("seq_len", -1))
        if (logits.ndim != 1 or not len(logits) or logits.dtype.kind != "f"
                or not np.isfinite(logits).all() or server.ndim != 2
                or server.shape != (length, len(logits))
                or not np.array_equal(server[-1], logits)):
            raise ValueError("inconsistent server logits in public trace")
        if (step.get("logits_perm_shape") != list(logits.shape)
                or step.get("logits_perm_sha256") != hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest()):
            raise ValueError("server logit summary mismatch")
        upload = "wire_vectors" if arm == "ab" else "wire_token_ids"
        expected_ndim = 2 if arm == "ab" else 1
        if upload not in loaded or loaded[upload].ndim != expected_ndim or loaded[upload].shape[0] != length:
            raise ValueError("public trace lacks its uploaded prefix")
        if arm == "ab" and step.get("wire_vectors_shape") != list(loaded[upload].shape):
            raise ValueError("public vector summary mismatch")
        if arm != "ab" and not np.array_equal(step.get("wire_token_ids"), loaded[upload]):
            raise ValueError("public token summary mismatch")
        maxima = np.flatnonzero(logits == logits.max())
        ambiguity += int(len(maxima) > 1)
        if arm == "plaintext":
            predicted = int(maxima[0])
        else:
            mapped = [mapping.get(int(symbol)) for symbol in maxima]
            if any(value is None or value < 0 or value >= len(logits) for value in mapped):
                predicted = -1
                uncovered += 1
            else:
                predicted = min(mapped)
        predictions.append(predicted)
    eos = manifest.get("eos_id")
    removed_eos = bool(predictions and eos is not None and predictions[-1] == int(eos))
    if removed_eos:
        predictions.pop()
    return {"status": "executed", "recovered_ids": predictions, "n_steps": len(steps), "uncovered_steps": uncovered,
            "tied_steps": ambiguity, "predicted_eos_removed": removed_eos,
            "manifest_sha256": digest, "input": "server logits and attacker-recovered mapping",
            "mapping_provenance": "caller_supplied_unverified" if arm != "plaintext" else "identity"}
