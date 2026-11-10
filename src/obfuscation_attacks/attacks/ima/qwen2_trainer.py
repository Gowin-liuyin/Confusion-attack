"""Batched Qwen2 inversion training on attacker-owned, isolated document pairs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _array_hash(array) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _prepare(pairs, *, obf_dim: int, vocab_size: int, role: str):
    result = []
    for memory, target in pairs:
        memory = np.asarray(memory, dtype=np.float32)
        target = np.asarray(target)
        if (memory.ndim != 2 or memory.shape[1] != obf_dim or target.ndim != 1
                or target.dtype.kind not in "iu"
                or len(target) == 0 or len(target) != len(memory)
                or not np.isfinite(memory).all()
                or np.any((target < 0) | (target >= vocab_size))):
            raise ValueError(f"invalid {role} document pair")
        result.append((memory, target.astype(np.int64, copy=False)))
    return result


def _identity(model, train, validation, reference, *, objective, seed, lr, batch_size, weight_decay, device):
    import transformers

    root = Path(__file__).resolve().parents[4]
    model_file = Path(__file__).with_name("qwen2_aloepri.py")
    sources = (Path(__file__), model_file, root / "docs" / "PAPER_EVALUATION_PROTOCOL.md")
    initial = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        initial.update(name.encode())
        initial.update(_array_hash(tensor.detach().cpu().numpy()).encode())
    return {
        "train": [[_array_hash(x), _array_hash(y)] for x, y in train],
        "validation": [[_array_hash(x), _array_hash(y)] for x, y in validation],
        "reference": _array_hash(reference) if reference is not None else None,
        "initial_model": initial.hexdigest(), "objective": objective, "seed": seed,
        "lr": lr, "batch_size": batch_size, "weight_decay": weight_decay,
        "device": str(device), "torch": str(torch.__version__), "numpy": str(np.__version__),
        "python": platform.python_version(), "cuda": str(torch.version.cuda),
        "transformers": str(transformers.__version__),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "sources": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources},
    }


def _batches(pairs, order, batch_size, device):
    for start in range(0, len(order), batch_size):
        chunk = [pairs[int(i)] for i in order[start:start + batch_size]]
        length = max(len(target) for _, target in chunk)
        memory = torch.zeros((len(chunk), length, chunk[0][0].shape[1]), device=device)
        targets = torch.zeros((len(chunk), length), dtype=torch.long, device=device)
        mask = torch.zeros((len(chunk), length), dtype=torch.bool, device=device)
        for index, (observed, truth) in enumerate(chunk):
            n = len(truth)
            memory[index, :n] = torch.as_tensor(observed, device=device)
            targets[index, :n] = torch.as_tensor(truth, device=device)
            mask[index, :n] = True
        yield memory, targets, mask


def _nearest_ids(predictions, reference, *, block=4096, query_block=256):
    """Exact full-table nearest neighbours with stable lowest-id ties."""
    best_distance = torch.full((len(predictions),), float("inf"), device=predictions.device)
    best_id = torch.zeros(len(predictions), dtype=torch.long, device=predictions.device)
    q = predictions.float()
    for start in range(0, len(reference), block):
        table = torch.as_tensor(reference[start:start + block], device=q.device)
        table_norm = table.square().sum(dim=1).unsqueeze(0)
        for query_start in range(0, len(q), query_block):
            query_slice = slice(query_start, query_start + query_block)
            query = q[query_slice]
            distance = query.square().sum(dim=1, keepdim=True) + table_norm - 2 * query @ table.T
            values, indices = distance.min(dim=1)
            improve = values < best_distance[query_slice]
            best_distance[query_slice] = torch.where(improve, values, best_distance[query_slice])
            best_id[query_slice] = torch.where(improve, indices + start, best_id[query_slice])
    return best_id


def fit_qwen2_inverter(model, train_pairs, *, objective: str, reference_embeddings=None,
                       validation_pairs=(), epochs: int, batch_size: int = 1, lr: float = 1e-3,
                       weight_decay: float = 0.0, seed: int = 0, device: str = "cpu",
                       checkpoint_path=None, resume: bool = False) -> dict:
    """Fit one learning rate; select the best epoch using held-out token recovery.

    The checkpoint retains the latest optimizer/model for continuation separately
    from the validation-selected model returned for inference. This routine does
    not establish formal budget, document independence or deployment provenance.
    """
    if objective not in {"embedding", "token_index"}:
        raise ValueError("unknown inversion objective")
    epochs, batch_size, seed = int(epochs), int(batch_size), int(seed)
    lr, weight_decay = float(lr), float(weight_decay)
    if epochs < 0 or batch_size < 1 or not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("invalid training configuration")
    device = torch.device(device)
    train = _prepare(train_pairs, obf_dim=model.obf_dim, vocab_size=model.vocab_size, role="training")
    validation = _prepare(validation_pairs, obf_dim=model.obf_dim, vocab_size=model.vocab_size, role="validation")
    if not train:
        raise ValueError("training pairs are empty")
    if {_array_hash(y) for _, y in train} & {_array_hash(y) for _, y in validation}:
        raise ValueError("training and validation documents overlap")
    reference = None if reference_embeddings is None else np.asarray(reference_embeddings, dtype=np.float32)
    if objective == "embedding":
        if (reference is None or reference.ndim != 2 or len(reference) != model.vocab_size
                or reference.shape[1] != model.plain_embed_dim or not np.isfinite(reference).all()):
            raise ValueError("embedding objective requires a finite full-vocabulary reference table")
    identity_material = _identity(model, train, validation, reference, objective=objective,
                                  seed=seed, lr=lr, batch_size=batch_size,
                                  weight_decay=weight_decay, device=device)
    identity = hashlib.sha256(json.dumps(identity_material, sort_keys=True).encode()).hexdigest()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if resume and checkpoint is None:
        raise ValueError("resume requires checkpoint_path")
    if checkpoint is not None and checkpoint.exists() and not resume:
        raise FileExistsError("checkpoint exists; use resume or a new path")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    history, best_state, best_epoch, best_accuracy, best_loss = [], None, None, None, None
    steps, examples, tokens, start_epoch = 0, 0, 0, 0
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state.get("format") != "qwen2_ima_training_v1" or state.get("identity") != identity:
            raise ValueError("checkpoint data, configuration, implementation or environment mismatch")
        start_epoch = int(state["epoch"])
        if start_epoch > epochs:
            raise ValueError("checkpoint has more epochs than requested")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["generator_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        history, best_state = state["history"], state["best_model_state"]
        best_epoch, best_accuracy, best_loss = state["best_epoch"], state["best_accuracy"], state["best_loss"]
        steps, examples, tokens = state["steps"], state["examples_seen"], state["tokens_seen"]

    def run_batch(memory, targets, mask, *, score=False):
        if objective == "token_index":
            predicted = model.predict_token_logits(memory, attention_mask=mask)
            loss = F.cross_entropy(predicted[mask], targets[mask], reduction="sum")
            ids = predicted[mask].argmax(dim=-1) if score else None
        else:
            predicted = model.predict_embeddings(memory, attention_mask=mask)[mask]
            expected = torch.as_tensor(reference[targets[mask].cpu().numpy()], device=device)
            loss = (predicted - expected).square().sum()
            ids = _nearest_ids(predicted, reference) if score else None
        return loss, int((ids == targets[mask]).sum()) if score else 0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, train_tokens = 0.0, 0
        order = torch.randperm(len(train), generator=generator).tolist()
        for memory, targets, mask in _batches(train, order, batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = run_batch(memory, targets, mask)
            count = int(mask.sum())
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            (loss / count).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite training gradient")
            optimizer.step()
            train_loss += float(loss.detach())
            train_tokens += count
            steps += 1
            examples += int(len(memory))
            tokens += count
        row = {"epoch": epoch + 1, "train_loss": train_loss / train_tokens}
        if validation:
            model.eval()
            val_loss, val_hits, val_tokens = 0.0, 0, 0
            with torch.no_grad():
                for memory, targets, mask in _batches(validation, range(len(validation)), batch_size, device):
                    loss, hits = run_batch(memory, targets, mask, score=True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite validation loss")
                    val_loss += float(loss)
                    val_hits += hits
                    val_tokens += int(mask.sum())
            accuracy, average_loss = val_hits / val_tokens, val_loss / val_tokens
            row.update({"validation_loss": average_loss, "validation_hits": val_hits,
                        "validation_tokens": val_tokens, "validation_top1": accuracy})
            if best_state is None or (accuracy, -average_loss) > (best_accuracy, -best_loss):
                best_epoch, best_accuracy, best_loss = epoch + 1, accuracy, average_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = {"format": "qwen2_ima_training_v1", "identity": identity, "identity_material": identity_material,
                     "epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "generator_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     "history": history, "best_model_state": best_state, "best_epoch": best_epoch,
                     "best_accuracy": best_accuracy, "best_loss": best_loss,
                     "steps": steps, "examples_seen": examples, "tokens_seen": tokens}
            fd, temporary = tempfile.mkstemp(prefix=checkpoint.name + ".", suffix=".tmp", dir=checkpoint.parent)
            os.close(fd)
            try:
                torch.save(state, temporary)
                os.replace(temporary, checkpoint)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    configuration = {
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "obf_dim": int(model.obf_dim), "vocab_size": int(model.vocab_size),
        "plain_embed_dim": model.plain_embed_dim,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return {"model": model, "history": history, "steps": steps, "examples_seen": examples,
            "tokens_seen": tokens, "best_epoch": best_epoch, "best_validation_top1": best_accuracy,
            "n_train_documents": len(train), "n_validation_documents": len(validation),
            "n_distinct_train_documents": len({row[1] for row in identity_material["train"]}),
            "n_distinct_validation_documents": len({row[1] for row in identity_material["validation"]}),
            "n_train_tokens": sum(len(y) for _, y in train),
            "n_validation_tokens": sum(len(y) for _, y in validation),
            "instantiated_config": configuration,
            "optimizer": "AdamW", "learning_rate": lr, "weight_decay": weight_decay, "seed": seed,
            "epochs_completed": len(history),
            "validation_selection": ("token_top1_then_loss" if best_epoch is not None else
                                     "not_run" if validation else "unused"),
            "validation_isolation": "exact token sequence overlap rejected; near-duplicate independence requires external audit",
            "identity": identity, "device": str(device), "batch_size": batch_size}


def select_qwen2_inverter(model_factory, train_pairs, validation_pairs, *, learning_rates,
                          checkpoint_dir=None, resume: bool = False, **training_options) -> dict:
    """Train the frozen learning-rate grid and select using validation only."""
    train_pairs, validation_pairs = list(train_pairs), list(validation_pairs)
    rates = [float(rate) for rate in learning_rates]
    if not validation_pairs:
        raise ValueError("learning-rate selection requires isolated validation pairs")
    if not rates or any(not np.isfinite(rate) or rate <= 0 for rate in rates) or len(set(rates)) != len(rates):
        raise ValueError("learning rates must be distinct, finite and positive")
    if "lr" in training_options or "checkpoint_path" in training_options:
        raise ValueError("learning rate and checkpoint paths are set by the grid")
    candidates, selected, best_key = [], None, None
    for rate in rates:
        path = None
        if checkpoint_dir is not None:
            name = hashlib.sha256(rate.hex().encode()).hexdigest()[:16]
            path = Path(checkpoint_dir) / f"lr-{name}.pt"
        result = fit_qwen2_inverter(
            model_factory(), train_pairs, validation_pairs=validation_pairs,
            lr=rate, checkpoint_path=path, resume=resume, **training_options,
        )
        if result["best_epoch"] is None:
            raise ValueError("learning-rate selection requires at least one training epoch")
        best_row = result["history"][result["best_epoch"] - 1]
        key = (result["best_validation_top1"], -best_row["validation_loss"])
        candidates.append({"learning_rate": rate, "best_epoch": result["best_epoch"],
                           "validation_top1": key[0], "validation_loss": best_row["validation_loss"],
                           "identity": result["identity"], "history": result["history"],
                           "optimizer_steps": result["steps"], "examples_seen": result["examples_seen"],
                           "tokens_seen": result["tokens_seen"]})
        if selected is None or key > best_key:
            selected, best_key = result, key
            selected["learning_rate"] = rate
    selected["learning_rate_grid"] = candidates
    selected["selection_rule"] = "validation token top1, then validation loss; first grid entry breaks ties"
    return selected
