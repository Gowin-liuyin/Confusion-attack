"""Real stepwise generation protocol (protocol §1, §8.3; F07/W6).

The client keeps the exact token prefix, re-encodes and re-uploads it every step,
the server returns logits in its own vocabulary, and the client samples the next
token after the inverse permutation. Every step is recorded so the output audit
uses the real trajectory, never a pre-written answer.

``GenerationRecord.public_trace()`` is the server-visible part; the client-chosen
token, the clean-space logits and the continuation are scoring-side data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass
class GenerationStep:
    index: int
    wire_public: Optional[object]
    logits_perm: np.ndarray
    logits_client: np.ndarray
    chosen_id: int
    greedy: bool
    group_id: Optional[str] = None
    epoch: int = 0
    public_snapshot: Optional[dict] = None

    def public_view(self) -> dict:
        if self.public_snapshot is not None:
            return dict(self.public_snapshot)
        wire = self.wire_public
        token_ids = getattr(wire, "wire_token_ids", None)
        vectors = getattr(wire, "wire_vectors", None)
        logits = np.ascontiguousarray(self.logits_perm)
        return {
            "index": int(self.index),
            "group_id": self.group_id,
            "epoch": int(self.epoch),
            "seq_len": int(getattr(wire, "seq_len", 0) or 0),
            "wire_token_ids": None if token_ids is None else [int(x) for x in token_ids],
            "wire_vectors_shape": None if vectors is None else list(np.asarray(vectors).shape),
            "logits_perm_shape": list(logits.shape),
            "logits_perm_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        }


@dataclass
class GenerationRecord:
    arm: str
    prompt_ids: list[int]
    continuation_ids: list[int]
    steps: list[GenerationStep]
    greedy: bool
    eos_id: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def full_ids(self) -> list[int]:
        return list(self.prompt_ids) + list(self.continuation_ids)

    def public_trace(self) -> dict:
        return {
            "arm": self.arm,
            "prompt_len": len(self.prompt_ids),
            "n_steps": len(self.steps),
            "greedy": self.greedy,
            "eos_id": self.eos_id,
            "steps": [step.public_view() for step in self.steps],
            "meta": dict(self.meta),
        }

    def scoring_trace(self) -> dict:
        return {
            **self.public_trace(),
            "continuation_len": len(self.continuation_ids),
            "chosen_ids": [int(step.chosen_id) for step in self.steps],
            "continuation_ids": [int(token) for token in self.continuation_ids],
        }

    def save_trace(self, path, *, include_truth: bool = False, scoring_dir=None) -> dict:
        """Write the public JSON trace and the server-visible arrays to an npz file."""
        if any(step.public_snapshot is not None for step in self.steps):
            raise ValueError("streamed generation must use its existing public trace")
        base = Path(path)
        scoring_path = None
        if include_truth:
            if scoring_dir is None:
                raise ValueError("scoring_dir is required for client-side truth")
            scoring_root = Path(scoring_dir).resolve()
            if scoring_root == base.parent.resolve() or scoring_root.is_relative_to(base.parent.resolve()):
                raise ValueError("scoring_dir must be outside the public trace directory")
            scoring_root.mkdir(parents=True, exist_ok=True)
            scoring_path = scoring_root / (base.stem + "_scoring.json")
        base.parent.mkdir(parents=True, exist_ok=True)
        array_path = base.with_suffix(".npz")
        public_path = base.with_suffix(".json")
        if array_path.exists() or public_path.exists() or (scoring_path is not None and scoring_path.exists()):
            raise FileExistsError("trace already exists")
        arrays: dict[str, np.ndarray] = {}
        for step in self.steps:
            arrays[f"step{step.index}_logits_perm"] = np.asarray(
                step.logits_perm, dtype=np.float32
            )
            wire = step.wire_public
            vectors = getattr(wire, "wire_vectors", None)
            if vectors is not None:
                arrays[f"step{step.index}_vectors"] = np.asarray(vectors, dtype=np.float32)
            token_ids = getattr(wire, "wire_token_ids", None)
            if token_ids is not None:
                arrays[f"step{step.index}_token_ids"] = np.asarray(token_ids, dtype=np.int64)
        with array_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        with public_path.open("x", encoding="utf-8") as handle:
            json.dump(self.public_trace(), handle, indent=2, ensure_ascii=False)
        result = {
            "public_trace": str(public_path),
            "arrays": str(array_path),
            "scoring_trace": None,
        }
        if include_truth:
            with scoring_path.open("x", encoding="utf-8") as handle:
                json.dump(self.scoring_trace(), handle, indent=2, ensure_ascii=False)
            result["scoring_trace"] = str(scoring_path)
        return result


class PublicGenerationWriter:
    """Write one server-visible step per chunk, with an immutable final manifest."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.steps: list[dict] = []

    def add(self, step: GenerationStep) -> None:
        if int(step.index) != len(self.steps):
            raise ValueError("generation steps must be contiguous")
        wire = step.wire_public
        arrays = {"last_logits": np.asarray(step.logits_perm)}
        for name, attribute in (("server_logits", "logits_perm"), ("wire_vectors", "wire_vectors"),
                                ("wire_token_ids", "wire_token_ids"), ("hidden_states", "hidden_states")):
            value = getattr(wire, attribute, None)
            if value is not None:
                array = np.asarray(value)
                if array.dtype.hasobject:
                    raise ValueError("public trace cannot contain object arrays")
                arrays[name] = array
        length = int(getattr(wire, "seq_len", -1))
        server = arrays.get("server_logits")
        last = arrays["last_logits"]
        if (server is None or server.ndim != 2 or server.shape[0] != length
                or length < 1 or last.ndim != 1 or last.dtype.kind != "f"
                or not np.isfinite(last).all() or not np.array_equal(server[-1], last)):
            raise ValueError("public trace requires consistent full server logits")
        arm = getattr(wire, "arm", None)
        if arm == "ab" and ("wire_vectors" not in arrays or arrays["wire_vectors"].ndim != 2
                            or arrays["wire_vectors"].shape[0] != length):
            raise ValueError("AB public trace requires the uploaded vector sequence")
        if arm in {"plaintext", "aloepri"} and ("wire_token_ids" not in arrays
                or arrays["wire_token_ids"].ndim != 1 or len(arrays["wire_token_ids"]) != length):
            raise ValueError("token-based public trace requires the uploaded token sequence")
        if arm not in {"plaintext", "aloepri", "ab"}:
            raise ValueError("unsupported public trace arm")
        step_dir = self.directory / f"step-{step.index:06d}"
        step_dir.mkdir(exist_ok=False)
        files = {}
        for name, array in arrays.items():
            path = step_dir / f"{name}.npy"
            with path.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            files[name] = {"path": f"{step_dir.name}/{path.name}", "sha256": digest.hexdigest(),
                           "shape": list(array.shape), "dtype": str(array.dtype)}
        snapshot = step.public_view()
        snapshot["arrays"] = files
        snapshot["request_id"] = int(getattr(wire, "request_id", step.index))
        snapshot["session_id"] = getattr(wire, "session_id", None)
        snapshot["layers"] = list(getattr(wire, "layers", ()) or ())
        self.steps.append(snapshot)
        step.public_snapshot = snapshot
        step.wire_public = None
        step.logits_perm = np.empty(0, dtype=np.float32)
        step.logits_client = np.empty(0, dtype=np.float32)

    def finish(self, record: GenerationRecord) -> dict:
        if len(record.steps) != len(self.steps):
            raise ValueError("trace step count does not match generation")
        manifest = {**record.public_trace(), "format": "server_generation_trace_v2", "steps": self.steps}
        path = self.directory / "manifest.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_steps": len(self.steps), "format": manifest["format"]}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def generate_record(
    deployment,
    prompt_ids: Sequence[int],
    *,
    arm: str,
    max_new_tokens: int = 256,
    greedy: bool = True,
    seed: int = 0,
    key_index: int = 0,
    session_id: Optional[str] = None,
    public_writer: Optional[PublicGenerationWriter] = None,
) -> GenerationRecord:
    """Generate token by token through the deployment's real message path."""
    if arm not in {"plaintext", "aloepri", "ab"}:
        raise ValueError(f"generation is not implemented for arm {arm!r}")
    rng = np.random.default_rng(seed)
    ids = [int(token) for token in prompt_ids]
    if not ids or int(max_new_tokens) < 0:
        raise ValueError("generation needs a non-empty prompt and non-negative output budget")
    prompt_len = len(ids)
    steps: list[GenerationStep] = []
    eos_id = getattr(deployment.tokenizer, "eos_token_id", None)
    for step in range(int(max_new_tokens)):
        wire = deployment.observe_ids(
            ids, arm=arm, request_id=step, key_index=key_index, epoch=0,
            session_id=session_id,
        )
        if bool(getattr(wire.public, "truncated", False)) or int(wire.public.seq_len) != len(ids):
            raise ValueError("generation refuses a truncated or incomplete uploaded prefix")
        logits_perm = np.asarray(wire.public.logits_perm[-1])
        if arm in {"ab", "aloepri"}:
            # The server emits logits indexed by the *deployment* token id; the
            # client's clean token c lives at deployment id perm[c], so map with
            # perm (not its inverse). ab_runtime.py uses the same
            # ``logits_perm[..., perm_vocab]`` direction.
            perm = getattr(deployment, "perm", None)
            if perm is None:
                perm = np.argsort(np.asarray(deployment.inv_perm))
            logits_client = logits_perm[np.asarray(perm)]
        else:
            logits_client = logits_perm
        if greedy:
            chosen = int(np.argmax(logits_client))
        else:
            chosen = int(rng.choice(logits_client.size, p=_softmax(logits_client)))
        step_record = GenerationStep(
                index=step,
                wire_public=wire.public,
                logits_perm=logits_perm,
                logits_client=logits_client,
                chosen_id=chosen,
                greedy=bool(greedy),
                group_id=getattr(wire.public, "group_id", None),
                epoch=int(getattr(wire.public, "epoch", 0) or 0),
            )
        if public_writer is not None:
            public_writer.add(step_record)
        steps.append(step_record)
        ids.append(chosen)
        if eos_id is not None and chosen == int(eos_id):
            break
    continuation = ids[prompt_len:]
    if eos_id is not None and continuation and continuation[-1] == int(eos_id):
        continuation = continuation[:-1]
    return GenerationRecord(
        arm=arm,
        prompt_ids=[int(token) for token in prompt_ids],
        continuation_ids=continuation,
        steps=steps,
        greedy=bool(greedy),
        eos_id=eos_id,
        meta={
            "message_path": "full-prefix re-encode/re-upload",
            "max_new_tokens": int(max_new_tokens),
            "key_index_private": True,
            "group_ids": sorted({step.group_id for step in steps if step.group_id}),
        },
    )
