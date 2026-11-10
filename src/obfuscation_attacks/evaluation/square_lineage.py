"""Scorer-side exact decoder lineage verification for the square export variant."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import hmac
import json
from pathlib import Path

import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from safetensors import safe_open


def _export_dtype(metadata: dict, original: torch.Tensor) -> torch.dtype:
    value = str(metadata.get("dtype") or "").lower()
    choices = {"float32": torch.float32, "fp32": torch.float32,
               "float16": torch.float16, "fp16": torch.float16,
               "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
               "auto": original.dtype}
    if value not in choices:
        raise ValueError("export dtype is missing or unsupported")
    return choices[value]


def _square_axes(master: bytes, seed: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the exporter's square-monomial stream; axes stay in this process."""
    message = json.dumps(["AloePri-random-v1", [], "square.monomial", seed + 5000],
                         ensure_ascii=True, separators=(",", ":")).encode()
    key = hmac.digest(master, message, "sha256")
    stream = Cipher(algorithms.ChaCha20(key, b"\0" * 16), mode=None).encryptor()
    buffer = b""
    offset = 0

    def take(n):
        nonlocal buffer, offset
        if len(buffer) - offset < n:
            buffer = buffer[offset:] + stream.update(b"\0" * max(65536, n))
            offset = 0
        value = buffer[offset:offset + n]
        offset += n
        return value

    indices = list(range(width))
    for i in range(width - 1, 0, -1):
        n = i + 1
        limit = 2**64 - 2**64 % n
        while True:
            value = int.from_bytes(take(8), "little")
            if value < limit:
                j = value % n
                break
        indices[i], indices[j] = indices[j], indices[i]
    if width > 1 and indices == list(range(width)):
        indices[0], indices[1] = indices[1], indices[0]
    signs = torch.tensor([1 if byte & 1 else -1 for byte in take(width)], dtype=torch.int8)
    if bool(torch.all(signs == 1)):
        signs[0] = -1
    return torch.argsort(torch.tensor(indices, dtype=torch.long)), signs


class _Weights:
    def __init__(self, root: Path, stack: ExitStack):
        index = root / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            mapping = None
        self.handles = {}
        self.mapping = mapping
        files = sorted(set(mapping.values())) if mapping is not None else ["model.safetensors"]
        for name in files:
            if Path(name).name != name:
                raise ValueError("invalid weight shard path")
            self.handles[name] = stack.enter_context(safe_open(str(root / name), framework="pt", device="cpu"))

    def tensor(self, name: str) -> torch.Tensor:
        try:
            shard = self.mapping[name] if self.mapping is not None else "model.safetensors"
            return self.handles[shard].get_tensor(name)
        except (KeyError, RuntimeError) as exc:
            raise ValueError(f"missing or unreadable decoder tensor: {name}") from exc

    def keys(self) -> set[str]:
        return {name for handle in self.handles.values() for name in handle.keys()}


def verify_square_lineage(artifact: Path | str, base_model: Path | str) -> dict:
    """Compare every transformed decoder matrix and norm, without exposing axes."""
    artifact, base_model = Path(artifact), Path(base_model)
    servers = [path for path in (artifact / "model_server", artifact / "server") if path.is_dir()]
    if len(servers) != 1:
        raise ValueError("artifact must contain exactly one server model directory")
    server = servers[0]
    base_config = json.loads((base_model / "config.json").read_text(encoding="utf-8"))
    deployed_config = json.loads((server / "config.json").read_text(encoding="utf-8"))
    ignored = {"dtype", "torch_dtype", "transformers_version", "_name_or_path", "_commit_hash",
               "rope_parameters", "rope_theta", "rope_scaling", "tie_word_embeddings"}
    def rope(config):
        if config.get("rope_parameters") is not None:
            return config["rope_parameters"]
        return {**(config.get("rope_scaling") or {}), "rope_theta": config.get("rope_theta")}
    fields = (set(base_config) | set(deployed_config)) - ignored
    if (base_config.get("model_type") != "llama" or deployed_config.get("tie_word_embeddings") is not False
            or any(base_config.get(field) != deployed_config.get(field) for field in fields)
            or rope(base_config) != rope(deployed_config)):
        raise ValueError("base and deployment model configurations do not match")
    secret = torch.load(artifact / "client" / "client_secret.pt", map_location="cpu", weights_only=True)
    metadata, master = secret["metadata"], secret["random_master_key"]
    seed, width = metadata["seed"], int(base_config["hidden_size"])
    if (not isinstance(master, bytes) or len(master) != 32 or type(seed) is not int
            or metadata.get("implementation_variant") != "square_monomial"
            or metadata.get("num_layers") != base_config["num_hidden_layers"]
            or metadata.get("transform_dim") != width):
        raise ValueError("unsupported or incomplete square export metadata")
    inverse, signs = _square_axes(master, seed, width)
    checks: list[tuple[str, str]] = [("model.norm.weight", "norm")]
    for layer in range(int(base_config["num_hidden_layers"])):
        prefix = f"model.layers.{layer}"
        checks.extend((f"{prefix}.{name}.weight", "norm") for name in
                      ("input_layernorm", "post_attention_layernorm"))
        checks.extend((f"{prefix}.self_attn.{name}.weight", "columns") for name in ("q_proj", "k_proj", "v_proj"))
        checks.append((f"{prefix}.self_attn.o_proj.weight", "rows"))
        checks.extend((f"{prefix}.mlp.{name}.weight", "columns") for name in ("gate_proj", "up_proj"))
        checks.append((f"{prefix}.mlp.down_proj.weight", "rows"))
        if base_config.get("attention_bias") or base_config.get("mlp_bias"):
            raise ValueError("square-lineage verification does not yet support biased layers")
    with ExitStack() as stack:
        clean, deployed = _Weights(base_model, stack), _Weights(server, stack)
        decoder_names = {name for name, _ in checks}
        permitted = decoder_names | {"model.embed_tokens.weight", "lm_head.weight"}
        if (clean.keys() - permitted or deployed.keys() - permitted
                or not (decoder_names | {"model.embed_tokens.weight"}) <= clean.keys()
                or not permitted <= deployed.keys()):
            raise ValueError("base or deployment contains missing or unsupported model tensors")
        for name, operation in checks:
            original, actual = clean.tensor(name), deployed.tensor(name)
            if actual.dtype != _export_dtype(metadata, original):
                raise ValueError(f"decoder tensor dtype disagrees with export: {name}")
            original = original.to(dtype=actual.dtype)
            if operation == "norm":
                expected = original.index_select(0, inverse)
            elif operation == "columns":
                expected = original.index_select(1, inverse) * signs.to(original.dtype)[None, :]
            else:
                expected = original.index_select(0, inverse) * signs.to(original.dtype)[:, None]
            if expected.shape != actual.shape or not torch.equal(expected, actual):
                raise ValueError(f"decoder tensor does not match transformed base: {name}")
    return {"verified": True, "checked_tensors": len(checks),
            "method": "exact comparison of all decoder matrices and norms after private transform replay",
            "scope": "decoder weights and model configuration; embedding and output-head noise are checked separately"}
