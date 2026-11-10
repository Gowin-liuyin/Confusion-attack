"""Public-auxiliary BERT scoring model training, not a verified paper reproduction."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from ...evaluation.sda_stream import _checkpoint_sha256, _document_record, _sda_code_identity, check_document_isolation, document_sha256
from . import candidate_search
from ._neural import validate_sda_device


def _examples(documents, candidate_sets, rng):
    if not documents or len(documents) != len(candidate_sets):
        raise ValueError("public documents and candidate lattices must be non-empty and aligned")
    examples, unchanged = [], 0
    for tokens, lattice in zip(documents, candidate_sets):
        tokens = list(tokens)
        if not tokens or len(tokens) != len(lattice):
            raise ValueError("candidate positions must match each non-empty public document")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**63
               for value in tokens):
            raise ValueError("public document token IDs must be non-negative integers")
        tokens = [int(value) for value in tokens]
        negative, alternatives = [], []
        for position, choices in enumerate(lattice):
            values = list(choices)
            if (not values or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v < 2**63
                                  for v in values) or len(set(values)) != len(values)):
                raise ValueError("candidate sets must contain distinct non-negative integer IDs")
            negative.append(int(values[int(rng.integers(len(values)))]))
            alternatives.append([int(value) for value in values if value != tokens[position]])
        if negative == tokens:
            positions = [i for i, values in enumerate(alternatives) if values]
            if not positions:
                raise ValueError("candidate lattice cannot produce a distinct negative example")
            position = positions[int(rng.integers(len(positions)))]
            choices = alternatives[position]
            negative[position] = choices[int(rng.integers(len(choices)))]
            unchanged += 1
        examples.extend(((tokens, 1), (negative, 0)))
    return examples, unchanged


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous()
        for part in (name.encode(), str(array.dtype).encode(), str(tuple(array.shape)).encode(),
                     array.reshape(-1).view(torch.uint8).numpy().tobytes()):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return digest.hexdigest()


def train_candidate_classifier(model, tokenizer, decode_candidate_ids, *,
                               train_documents, train_candidates, validation_documents, validation_candidates,
                               candidate_tokenizer_sha256: str, classifier_tokenizer_sha256: str,
                               checkpoint_path, epochs: int, batch_size: int, learning_rate: float,
                               seed: int, validation_seed: int = 0, device: str = "cpu",
                               training_checkpoint_path=None, resume: bool = False) -> dict:
    """Train on attacker-owned positive/negative examples and save validation-selected weights.

    All supplied documents and candidates must be public auxiliary data. Caller
    tokenizer commitments and data-source ownership remain independently unverified.
    The inference export is immutable. An optional separate training checkpoint
    stores optimizer and RNG state at each completed epoch for resumption.
    """
    validate_sda_device(device)
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("seed", seed), ("validation_seed", validation_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float, np.floating)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    learning_rate = float(learning_rate)
    for digest in (candidate_tokenizer_sha256, classifier_tokenizer_sha256):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("tokenizer commitments must be SHA-256 hex strings")
    if not callable(decode_candidate_ids) or getattr(model.config, "model_type", None) != "bert" or getattr(model.config, "num_labels", None) != 2:
        raise ValueError("a candidate decoder and binary BERT sequence classifier are required")
    target = Path(checkpoint_path)
    if target.exists():
        raise FileExistsError(f"classifier checkpoint already exists: {target}")
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean")
    training_checkpoint = Path(training_checkpoint_path) if training_checkpoint_path is not None else None
    if resume and training_checkpoint is None:
        raise ValueError("resume requires a training checkpoint path")
    if training_checkpoint is not None and training_checkpoint.resolve() == target.resolve():
        raise ValueError("training checkpoint and inference export must be different files")
    if training_checkpoint is not None and training_checkpoint.exists() and not resume:
        raise FileExistsError("training checkpoint exists; resume or choose another path")
    if resume and not training_checkpoint.is_file():
        raise FileNotFoundError("training checkpoint does not exist")
    train_documents = [list(doc) for doc in train_documents]
    validation_documents = [list(doc) for doc in validation_documents]
    train_candidates = [[list(choices) for choices in document] for document in train_candidates]
    validation_candidates = [[list(choices) for choices in document] for document in validation_candidates]
    roles = {role: [_document_record({"source": role, "index": i, "tokens": doc}, role=role)
                    for i, doc in enumerate(docs)]
             for role, docs in (("train", train_documents), ("validation", validation_documents))}
    isolation = check_document_isolation(roles)
    if not isolation["passed"]:
        raise ValueError("public training and validation documents overlap or cannot be isolated")
    rng = np.random.default_rng(seed)
    train, unchanged_train = _examples(train_documents, train_candidates, rng)
    validation, unchanged_validation = _examples(validation_documents, validation_candidates,
                                                 np.random.default_rng(validation_seed))
    def decoded_hashes(examples):
        token_hashes, text_hashes, sequence_hash = set(), set(), hashlib.sha256()
        for tokens, label in examples:
            text = decode_candidate_ids(tokens)
            if not isinstance(text, str):
                raise TypeError("candidate token decoder must return strings")
            token_hashes.add(document_sha256(tokens))
            encoded = text.encode("utf-8")
            text_hashes.add(hashlib.sha256(encoded).hexdigest())
            sequence_hash.update(len(encoded).to_bytes(8, "little"))
            sequence_hash.update(encoded)
            sequence_hash.update(bytes([label]))
        return token_hashes, text_hashes, sequence_hash.hexdigest()
    train_token_hashes, train_text_hashes, train_decoded_sha256 = decoded_hashes(train)
    validation_token_hashes, validation_text_hashes, validation_decoded_sha256 = decoded_hashes(validation)
    if train_token_hashes & validation_token_hashes or train_text_hashes & validation_text_hashes:
        raise ValueError("generated public training and validation examples overlap")
    for examples in (train, validation):
        positive_texts = {decode_candidate_ids(tokens) for tokens, label in examples if label == 1}
        negative_texts = {decode_candidate_ids(tokens) for tokens, label in examples if label == 0}
        if positive_texts & negative_texts:
            raise ValueError("decoded positive and negative examples have conflicting labels")
    def lattice_hash(lattice):
        data = [[[int(token) for token in choices] for choices in document] for document in lattice]
        return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()
    provenance = {"train_document_sha256": [document_sha256(doc) for doc in train_documents],
                  "validation_document_sha256": [document_sha256(doc) for doc in validation_documents],
                  "train_candidate_sha256": lattice_hash(train_candidates),
                  "validation_candidate_sha256": lattice_hash(validation_candidates),
                  "candidate_tokenizer_sha256": candidate_tokenizer_sha256,
                  "classifier_tokenizer_sha256": classifier_tokenizer_sha256,
                  "tokenizer_commitment_status": "caller_supplied_unverified",
                  "source_ownership_status": "caller_supplied_unverified",
                  "initial_model_source_status": "weight_hash_only_pretraining_unverified",
                  "negative_sampling_rule": "uniform independent choice per candidate set; unchanged sample corrected at a random position with an alternative",
                  "train_documents": len(train_documents), "validation_documents": len(validation_documents),
                  "generated_cross_role_exact_overlap": 0,
                  "train_decoded_sha256": train_decoded_sha256,
                  "validation_decoded_sha256": validation_decoded_sha256,
                  "unchanged_negative_train": 0, "unchanged_negative_validation": 0,
                  "corrected_unchanged_train": unchanged_train, "corrected_unchanged_validation": unchanged_validation,
                  "validation_seed": validation_seed,
                  "validation_negative_sha256": hashlib.sha256(b"".join(
                      bytes.fromhex(document_sha256(tokens)) for tokens, label in validation if label == 0)).hexdigest(),
                  "isolation": isolation}
    def tokenized_input_sha256(examples):
        digest = hashlib.sha256()
        input_labels = {}
        for start in range(0, len(examples), batch_size):
            group = examples[start:start + batch_size]
            texts = [decode_candidate_ids(tokens) for tokens, _ in group]
            if any(not isinstance(text, str) for text in texts):
                raise TypeError("candidate token decoder must return strings")
            inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            if "input_ids" not in inputs or "attention_mask" not in inputs:
                raise ValueError("classifier tokenizer must provide input_ids and attention_mask")
            if (not isinstance(inputs["input_ids"], torch.Tensor) or not isinstance(inputs["attention_mask"], torch.Tensor)
                    or inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != len(group)
                    or inputs["attention_mask"].shape != inputs["input_ids"].shape):
                raise ValueError("classifier tokenizer must return aligned batch tensors")
            if inputs["input_ids"].shape[1] > int(model.config.max_position_embeddings):
                raise ValueError("classifier input exceeds positional capacity; no truncation was performed")
            for key, tensor in sorted(inputs.items()):
                if not isinstance(tensor, torch.Tensor) or tensor.shape != inputs["input_ids"].shape:
                    raise ValueError("classifier tokenizer outputs must be aligned batch tensors")
                value = tensor.detach().cpu().contiguous()
                for part in (str(key).encode(), str(value.dtype).encode(), str(tuple(value.shape)).encode(),
                             value.reshape(-1).view(torch.uint8).numpy().tobytes()):
                    digest.update(len(part).to_bytes(8, "little"))
                    digest.update(part)
            for i, (_, label) in enumerate(group):
                mask = inputs["attention_mask"][i].bool()
                item = hashlib.sha256()
                for key, tensor in sorted(inputs.items()):
                    value = tensor[i][mask].detach().cpu().contiguous()
                    for part in (str(key).encode(), str(value.dtype).encode(), value.reshape(-1).view(torch.uint8).numpy().tobytes()):
                        item.update(len(part).to_bytes(8, "little"))
                        item.update(part)
                fingerprint = item.hexdigest()
                if fingerprint in input_labels and input_labels[fingerprint] != label:
                    raise ValueError("tokenized positive and negative examples have conflicting labels")
                input_labels[fingerprint] = label
        return digest.hexdigest(), set(input_labels)
    provenance["train_tokenized_input_sha256"], train_inputs = tokenized_input_sha256(train)
    provenance["validation_tokenized_input_sha256"], validation_inputs = tokenized_input_sha256(validation)
    if train_inputs & validation_inputs:
        raise ValueError("tokenized public training and validation examples overlap")
    provenance["tokenized_input_scope"] = "ordered batches before training; tokenizer implementation remains unverified"
    initial_sha256 = _model_digest(model)
    def runtime_identity():
        return {"training_source_sha256": _checkpoint_sha256(__file__),
                "search_source_sha256": _checkpoint_sha256(candidate_search.__file__),
                "sda_code_and_protocol": _sda_code_identity()}
    runtime = runtime_identity()
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01, amsgrad=False)
    config = json.loads(json.dumps(model.config.to_dict()))
    history, best_state, best_loss, best_epoch, steps = [], None, math.inf, None, 0
    gpu_devices = ([{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))}
                    for i in range(torch.cuda.device_count())] if str(device).startswith("cuda") else [])
    identity_material = {"initial_model_sha256": initial_sha256, "model_config": config,
                         "provenance": provenance, "seed": seed, "batch_size": batch_size,
                         "learning_rate": learning_rate, "device": str(device), "torch_version": str(torch.__version__),
                         "numpy_version": str(np.__version__), "python_version": platform.python_version(),
                         "cuda_version": str(torch.version.cuda), "runtime_identity": runtime,
                         "execution_environment": {"platform": platform.platform(), "processor": platform.processor(),
                                                   "torch_threads": torch.get_num_threads(),
                                                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                                   "cudnn_version": torch.backends.cudnn.version(), "cuda_devices": gpu_devices}}
    identity = hashlib.sha256(json.dumps(identity_material, sort_keys=True, allow_nan=False).encode()).hexdigest()
    start_epoch, current_training_sha, initial_steps = 0, None, 0
    if resume:
        current_training_sha = _checkpoint_sha256(training_checkpoint)
        state = torch.load(training_checkpoint, map_location="cpu", weights_only=True)
        if _checkpoint_sha256(training_checkpoint) != current_training_sha:
            raise ValueError("classifier training checkpoint changed during loading")
        if (state.get("format") != "candidate_classifier_training_v1" or state.get("identity") != identity
                or state.get("identity_material") != identity_material):
            raise ValueError("classifier training checkpoint identity mismatch")
        start_epoch = state["epoch"]
        history, steps = state["history"], state["optimizer_steps"]
        if (isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or not 0 < start_epoch <= epochs
                or isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
                or not isinstance(history, list) or len(history) != start_epoch
                or any(not isinstance(row, dict) or row.get("epoch") != i + 1
                       or isinstance(row.get("optimizer_steps"), bool)
                       or not isinstance(row.get("optimizer_steps"), int) or row["optimizer_steps"] <= 0
                       or isinstance(row.get("validation_nll"), bool) or not isinstance(row.get("validation_nll"), (int, float))
                       or not math.isfinite(row["validation_nll"])
                       for i, row in enumerate(history))
                or steps != sum(row["optimizer_steps"] for row in history)):
            raise ValueError("classifier training checkpoint has invalid epoch or step history")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_state, best_loss, best_epoch = state["best_state"], state["best_loss"], state["best_epoch"]
        if (not isinstance(best_state, dict) or isinstance(best_epoch, bool) or not isinstance(best_epoch, int)
                or not 1 <= best_epoch <= start_epoch or isinstance(best_loss, bool) or not isinstance(best_loss, (int, float))
                or not math.isfinite(best_loss) or best_loss != history[best_epoch - 1]["validation_nll"]
                or best_epoch != min(history, key=lambda row: row["validation_nll"])["epoch"]):
            raise ValueError("classifier training checkpoint has invalid validation selection")
        rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        initial_steps = steps
    started = time.perf_counter()

    def save_training_checkpoint(epoch):
        nonlocal current_training_sha
        if training_checkpoint is None:
            return
        if runtime_identity() != runtime:
            raise ValueError("candidate classifier implementation or protocol changed during training")
        training_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if current_training_sha is None:
            if training_checkpoint.exists():
                raise FileExistsError("training checkpoint appeared during execution")
            parent_file = None
        else:
            if not training_checkpoint.is_file() or _checkpoint_sha256(training_checkpoint) != current_training_sha:
                raise ValueError("training checkpoint changed during execution")
            parent_file = training_checkpoint.name + ".parent-" + current_training_sha
            archive = training_checkpoint.parent / parent_file
            if not archive.exists():
                try:
                    destination = archive.open("xb")
                except FileExistsError:
                    pass
                else:
                    try:
                        with destination, training_checkpoint.open("rb") as source:
                            shutil.copyfileobj(source, destination)
                    except Exception:
                        archive.unlink(missing_ok=True)
                        raise
            if _checkpoint_sha256(archive) != current_training_sha:
                raise ValueError("classifier checkpoint parent archive hash mismatch")
        record = {"format": "candidate_classifier_training_v1", "identity": identity,
                  "identity_material": identity_material, "parent_sha256": current_training_sha,
                  "parent_file": parent_file, "epoch": epoch, "optimizer_steps": steps,
                  "history": history, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                  "best_state": best_state, "best_epoch": best_epoch, "best_loss": best_loss,
                  "numpy_rng_state": rng.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                  "cuda_rng_state": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
                  "single_writer_required": True}
        fd, temporary = tempfile.mkstemp(prefix=training_checkpoint.name + ".", suffix=".tmp",
                                         dir=training_checkpoint.parent)
        os.close(fd)
        try:
            torch.save(record, temporary)
            if current_training_sha is None:
                os.link(temporary, training_checkpoint)
            else:
                if _checkpoint_sha256(training_checkpoint) != current_training_sha:
                    raise ValueError("training checkpoint changed before publication")
                os.replace(temporary, training_checkpoint)
            current_training_sha = _checkpoint_sha256(training_checkpoint)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def batches(examples, order):
        for start in range(0, len(order), batch_size):
            group = [examples[int(i)] for i in order[start:start + batch_size]]
            texts = [decode_candidate_ids(tokens) for tokens, _ in group]
            if any(not isinstance(text, str) for text in texts):
                raise TypeError("candidate token decoder must return strings")
            inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if "attention_mask" not in inputs:
                raise ValueError("classifier tokenizer must provide attention_mask for token accounting")
            if inputs["input_ids"].shape[1] > int(model.config.max_position_embeddings):
                raise ValueError("classifier input exceeds positional capacity; no truncation was performed")
            labels = torch.tensor([label for _, label in group], dtype=torch.long, device=device)
            yield inputs, labels

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        train_sum, train_n, train_tokens, epoch_steps = 0.0, 0, 0, 0
        for inputs, labels in batches(train, rng.permutation(len(train))):
            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite candidate classifier loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite candidate classifier gradient")
            optimizer.step()
            steps += 1
            epoch_steps += 1
            train_sum += float(loss.detach()) * len(labels)
            train_n += len(labels)
            train_tokens += int(inputs["attention_mask"].sum())
        model.eval()
        val_sum, val_n, hits = 0.0, 0, 0
        with torch.inference_mode():
            for inputs, labels in batches(validation, np.arange(len(validation))):
                logits = model(**inputs).logits
                loss = F.cross_entropy(logits, labels, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite candidate classifier validation loss")
                val_sum += float(loss)
                val_n += len(labels)
                hits += int((logits.argmax(dim=1) == labels).sum())
        val_loss = val_sum / val_n
        history.append({"epoch": epoch, "train_nll": train_sum / train_n, "train_examples": train_n,
                        "train_model_input_tokens": train_tokens, "optimizer_steps": epoch_steps,
                        "validation_nll": val_loss, "validation_hits": hits, "validation_examples": val_n})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        save_training_checkpoint(epoch)
    model.load_state_dict(best_state)
    model.eval()
    if runtime_identity() != runtime:
        raise ValueError("candidate classifier implementation or protocol changed during training")
    checkpoint = {"format": "candidate_classifier_probe_v1", "model_config": config,
                  "model_state": best_state, "initial_model_sha256": initial_sha256,
                  "selected_model_sha256": _model_digest(model), "selected_epoch": best_epoch,
                  "history": history, "optimizer_steps": steps, "seed": seed,
                  "optimizer": {"name": "torch.optim.AdamW", "lr": learning_rate, "betas": [0.9, 0.999],
                                "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False},
                  "torch_version": str(torch.__version__), "device": str(device),
                  "batch_size": batch_size, "provenance": provenance, "resume_supported": False,
                  "training_checkpoint_sha256": current_training_sha,
                  "training_checkpoint_identity": identity if training_checkpoint is not None else None,
                  "runtime_identity": runtime,
                  "formal_status": "not_assessed", "algorithm_fidelity": "unverified_engineering_variant"}
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        digest = _checkpoint_sha256(temporary)
        os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"checkpoint_sha256": digest, "selected_model_sha256": checkpoint["selected_model_sha256"],
            "initial_model_sha256": initial_sha256, "selected_epoch": best_epoch, "history": history,
            "optimizer_steps": steps, "new_optimizer_steps": steps - initial_steps,
            "training_start_epoch": start_epoch, "training_checkpoint_sha256": current_training_sha,
            "training_checkpoint_identity": identity if training_checkpoint is not None else None,
            "training_resume_supported": training_checkpoint is not None,
            "training_concurrency": "single_writer_required" if training_checkpoint is not None else None,
            "provenance": provenance, "training_and_checkpoint_seconds": time.perf_counter() - started,
            "formal_status": "not_assessed", "resume_supported": False}


def load_candidate_classifier(model, checkpoint_path, *, expected_checkpoint_sha256: str,
                              candidate_tokenizer_sha256: str, classifier_tokenizer_sha256: str,
                              device: str = "cpu") -> dict:
    """Load a frozen probe classifier after checking its externally saved identity."""
    if not isinstance(expected_checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_sha256):
        raise ValueError("expected checkpoint must be a SHA-256 hex digest")
    if _checkpoint_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("candidate classifier checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if _checkpoint_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("candidate classifier checkpoint changed during loading")
    if checkpoint.get("format") != "candidate_classifier_probe_v1":
        raise ValueError("unsupported candidate classifier checkpoint")
    expected_runtime = {"training_source_sha256": _checkpoint_sha256(__file__),
                        "search_source_sha256": _checkpoint_sha256(candidate_search.__file__),
                        "sda_code_and_protocol": _sda_code_identity()}
    if checkpoint.get("runtime_identity") != expected_runtime:
        raise ValueError("candidate classifier implementation or protocol changed")
    provenance = checkpoint["provenance"]
    if (provenance["candidate_tokenizer_sha256"] != candidate_tokenizer_sha256 or
            provenance["classifier_tokenizer_sha256"] != classifier_tokenizer_sha256):
        raise ValueError("candidate classifier tokenizer commitment mismatch")
    if json.loads(json.dumps(model.config.to_dict())) != checkpoint["model_config"]:
        raise ValueError("candidate classifier architecture mismatch")
    model.load_state_dict(checkpoint["model_state"])
    if _model_digest(model) != checkpoint["selected_model_sha256"]:
        raise ValueError("candidate classifier selected weights hash mismatch")
    model.to(device).eval()
    return {"checkpoint_sha256": expected_checkpoint_sha256, "selected_epoch": checkpoint["selected_epoch"],
            "selected_model_sha256": checkpoint["selected_model_sha256"],
            "formal_status": "not_assessed", "resume_supported": False}
