"""Scorer-side checks for the random source of a square export's vocabulary map."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from pathlib import Path

import numpy as np


def _load_tokenizer(server: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(server), local_files_only=True, trust_remote_code=False)


def _replay_permutation(vocab_size: int, movable: list[int], master: bytes, seed: int) -> np.ndarray:
    """Replay the exporter HMAC/ChaCha20 Fisher-Yates stream without publishing it."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

    message = json.dumps(["AloePri-random-v1", [], "vocabulary.permutation", seed],
                         ensure_ascii=True, separators=(",", ":")).encode()
    key = hmac.digest(master, message, "sha256")
    cipher = Cipher(algorithms.ChaCha20(key, b"\0" * 16), mode=None).encryptor()
    buffer = b""
    offset = 0

    def randbelow(n: int) -> int:
        nonlocal buffer, offset
        limit = 2**64 - 2**64 % n
        while True:
            if len(buffer) - offset < 8:
                buffer = buffer[offset:] + cipher.update(b"\0" * 65536)
                offset = 0
            value = int.from_bytes(buffer[offset:offset + 8], "little")
            offset += 8
            if value < limit:
                return value % n

    shuffled = list(movable)
    for i in range(len(shuffled) - 1, 0, -1):
        j = randbelow(i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    result = np.arange(vocab_size, dtype=np.int64)
    result[np.asarray(movable, dtype=np.int64)] = shuffled
    return result


def _public_leaks(value, seed: int, master_hex: str) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            name = "".join(c for c in str(key).lower() if c.isalnum())
            if name.endswith(("permvocab", "invpermvocab", "randommasterkey")):
                return True
            if "seed" in name and not isinstance(item, bool) and str(item) == str(seed):
                return True
            if _public_leaks(item, seed, master_hex):
                return True
    elif isinstance(value, list):
        return any(_public_leaks(item, seed, master_hex) for item in value)
    elif isinstance(value, str):
        return master_hex in value.lower()
    return False


def verify_export_randomness(artifact: Path | str) -> dict:
    """Check private export evidence and public files; return no seed or permutation.

    Replay verifies consistency with the claimed cryptographic generator. It does
    not independently prove that the master key was sampled from system entropy.
    """
    import torch

    root = Path(artifact)
    try:
        secret = torch.load(root / "client" / "client_secret.pt", map_location="cpu", weights_only=True)
        metadata = secret["metadata"]
        randomness = secret["randomness"]
        master = secret["random_master_key"]
        seed = metadata["seed"]
        perm = np.asarray(secret["perm_vocab"], dtype=np.int64)
        inverse = np.asarray(secret["inv_perm_vocab"], dtype=np.int64)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("missing or unreadable private export randomness evidence") from exc
    if (not isinstance(master, bytes) or len(master) != 32 or type(seed) is not int
            or not 0 <= seed < 2**63 or not isinstance(metadata, dict) or not isinstance(randomness, dict)):
        raise ValueError("invalid private export randomness evidence")
    expected = {"random_source": "ChaCha20_HMAC_SHA256", "randomness_mode": "system_entropy"}
    if any(metadata.get(key) != value or randomness.get(key) != value for key, value in expected.items()):
        raise ValueError("export was not recorded as using the system-entropy cryptographic generator")
    if metadata.get("seed_source") != "system_random":
        raise ValueError("export seed is not recorded as system generated")
    servers = [path for path in (root / "model_server", root / "server") if path.is_dir()]
    if len(servers) != 1:
        raise ValueError("artifact must contain exactly one server model directory")
    server = servers[0]
    text_extensions = {".json", ".txt", ".md", ".yaml", ".yml", ".toml", ".py"}
    seed_pattern = re.compile(r"(?<!\d)" + re.escape(str(seed)) + r"(?!\d)")
    for path in root.rglob("*"):
        if (not path.is_file() or path.relative_to(root).parts[0] == "client"
                or path.suffix.lower() not in text_extensions):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            leaked = (_public_leaks(json.loads(text), seed, master.hex()) if path.suffix.lower() == ".json"
                      else bool(seed_pattern.search(text) or master.hex() in text.lower()))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("a public export text file is unreadable") from exc
        if leaked:
            raise ValueError("public export metadata contains secret generation material")
    if any("client_secret" in path.name.lower() for path in server.rglob("*") if path.is_file()):
        raise ValueError("server export contains a client secret file")
    from safetensors import safe_open

    for path in server.rglob("*.safetensors"):
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = handle.keys()
                weight_metadata = handle.metadata() or {}
                if (any("".join(c for c in name.lower() if c.isalnum())
                        .endswith(("permvocab", "invpermvocab", "randommasterkey")) for name in keys)
                        or _public_leaks(weight_metadata, seed, master.hex())):
                    raise ValueError("server weight header contains secret generation material")
        except (OSError, RuntimeError) as exc:
            raise ValueError("server weight header is unreadable") from exc
    try:
        tokenizer = _load_tokenizer(server)
        size = int(json.loads((server / "config.json").read_text(encoding="utf-8"))["vocab_size"])
        token_vocab = int(tokenizer.vocab_size)
        special = {int(token) for token in tokenizer.all_special_ids if 0 <= int(token) < token_vocab}
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("cannot verify the tokenizer and model vocabulary") from exc
    if size < token_vocab or token_vocab < 1 or perm.shape != (size,) or inverse.shape != (size,):
        raise ValueError("vocabulary permutation dimensions do not match the model")
    movable = [token for token in range(token_vocab) if token not in special]
    if not np.array_equal(perm, _replay_permutation(size, movable, master, seed)):
        raise ValueError("private vocabulary permutation does not match the cryptographic export stream")
    if not np.array_equal(inverse[perm], np.arange(size)):
        raise ValueError("private vocabulary permutation and inverse are inconsistent")
    return {"verified": True, "method": "private cryptographic stream replay and public text and weight-header leak scan",
            "limitation": "system entropy is recorded, not independently provable from the resulting key"}
