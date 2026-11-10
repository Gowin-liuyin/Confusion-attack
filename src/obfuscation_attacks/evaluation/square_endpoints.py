"""Scorer-side replay of noisy embedding and output-head export weights."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import hmac
import json
import math
from pathlib import Path

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from .export_randomness import _load_tokenizer
from .square_lineage import _Weights, _export_dtype, _square_axes


def _noise_blocks(master: bytes, domain: str, seed: int, total: int):
    """Reproduce the exporter's float32 Box-Muller chunks in stream order."""
    message = json.dumps(["AloePri-random-v1", [], domain, seed],
                         ensure_ascii=True, separators=(",", ":")).encode()
    key = hmac.digest(master, message, "sha256")
    stream = Cipher(algorithms.ChaCha20(key, b"\0" * 16), mode=None).encryptor()
    for start in range(0, total, 262144):
        size = min(262144, total - start)
        pairs = (size + 1) // 2
        raw = stream.update(b"\0" * (pairs * 16))
        integers = np.frombuffer(raw, dtype="<u8") >> 11
        uniform = ((integers.astype(np.float64) + 0.5) / (2**53)).reshape(pairs, 2)
        radius = np.sqrt(-2 * np.log(uniform[:, 0]))
        angle = 2 * np.pi * uniform[:, 1]
        values = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1).reshape(-1)[:size].copy()
        yield torch.from_numpy(values).to(torch.float32)


def verify_square_endpoints(artifact: Path | str, base_model: Path | str) -> dict:
    """Rebuild every noisy endpoint row from the private export stream and base weights."""
    artifact, base_model = Path(artifact), Path(base_model)
    servers = [path for path in (artifact / "model_server", artifact / "server") if path.is_dir()]
    if len(servers) != 1:
        raise ValueError("artifact must contain exactly one server model directory")
    server = servers[0]
    config = json.loads((base_model / "config.json").read_text(encoding="utf-8"))
    secret = torch.load(artifact / "client" / "client_secret.pt", map_location="cpu", weights_only=True)
    metadata, master = secret["metadata"], secret["random_master_key"]
    width, vocab = int(config["hidden_size"]), int(config["vocab_size"])
    seed = metadata["seed"]
    if (width < 1 or 262144 % width or not isinstance(master, bytes) or len(master) != 32
            or type(seed) is not int or metadata.get("implementation_variant") != "square_monomial"):
        raise ValueError("unsupported square endpoint export")
    alpha_e, alpha_h = float(metadata["alpha_e"]), float(metadata["alpha_h"])
    if not all(math.isfinite(value) and value >= 0 for value in (alpha_e, alpha_h)):
        raise ValueError("invalid endpoint noise scales")
    vocab_inverse = secret["inv_perm_vocab"].long()
    if vocab_inverse.shape != (vocab,) or not torch.equal(torch.sort(vocab_inverse).values, torch.arange(vocab)):
        raise ValueError("invalid vocabulary inverse")
    hidden_inverse, signs = _square_axes(master, seed, width)
    tokenizer = _load_tokenizer(server)
    tokenizer_vocab = int(tokenizer.vocab_size)
    if tokenizer_vocab < 1 or tokenizer_vocab > vocab:
        raise ValueError("tokenizer vocabulary does not match model")
    special = {int(x) for x in tokenizer.all_special_ids if 0 <= int(x) < tokenizer_vocab}
    movable = torch.tensor([i for i in range(tokenizer_vocab) if i not in special], dtype=torch.long)
    fixed = torch.tensor([i for i in range(vocab) if i >= tokenizer_vocab or i in special], dtype=torch.long)
    if movable.numel() == 0:
        raise ValueError("no movable vocabulary rows")
    rows_per_block = 262144 // width
    checked = 0
    with ExitStack() as stack:
        clean, deployed = _Weights(base_model, stack), _Weights(server, stack)
        for name, alpha, domain, offset in (
            ("model.embed_tokens.weight", alpha_e, "embedding.noise", 101),
            ("lm_head.weight", alpha_h, "head.noise", 202),
        ):
            source_name = name if name in clean.keys() else "model.embed_tokens.weight"
            if name == "model.embed_tokens.weight" and source_name != name:
                raise ValueError("base embedding is missing")
            if name == "lm_head.weight" and source_name != name and not config.get("tie_word_embeddings"):
                raise ValueError("base output head is missing")
            target = deployed.tensor(name)
            source = clean.tensor(source_name)
            if source.shape != (vocab, width) or target.shape != (vocab, width):
                raise ValueError(f"endpoint tensor shape mismatch: {name}")
            if target.dtype != _export_dtype(metadata, source):
                raise ValueError(f"endpoint tensor dtype disagrees with export: {name}")
            permuted = source.index_select(0, vocab_inverse).to(target.dtype).to(torch.float32)
            sigma = float(permuted.index_select(0, movable).std(unbiased=False)) if alpha else 0.0
            if not math.isfinite(sigma):
                raise ValueError(f"endpoint noise scale is non-finite: {name}")
            blocks = _noise_blocks(master, domain, seed + offset, int(movable.numel()) * width) if alpha else None
            for start in range(0, len(movable), rows_per_block):
                rows = movable[start:start + rows_per_block]
                expected = permuted.index_select(0, rows)
                if blocks is not None:
                    noise = next(blocks).reshape(len(rows), width) * sigma
                    expected = expected + alpha * noise
                expected = (expected.index_select(1, hidden_inverse) * signs.to(torch.float32)[None, :]).to(target.dtype)
                if not torch.equal(expected, target.index_select(0, rows)):
                    raise ValueError(f"noisy endpoint does not match export replay: {name}")
                checked += int(rows.numel())
            for start in range(0, len(fixed), rows_per_block):
                rows = fixed[start:start + rows_per_block]
                expected = (permuted.index_select(0, rows).index_select(1, hidden_inverse)
                            * signs.to(torch.float32)[None, :]).to(target.dtype)
                if not torch.equal(expected, target.index_select(0, rows)):
                    raise ValueError(f"fixed endpoint rows do not match export replay: {name}")
                checked += int(rows.numel())
    return {"verified": True, "checked_rows": checked,
            "method": "exact replay of embedding and output-head permutation, noise and hidden-axis transform",
            "limitation": "bitwise noise replay depends on numerical libraries and reduction settings matching the export environment",
            "scope": "square A endpoint weights only; no privacy, utility or A+B compatibility claim"}
