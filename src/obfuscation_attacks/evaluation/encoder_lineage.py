"""Check consistency of an encoder export with its declared training inputs.

This is a content-binding check, not independent proof of training or acceptance.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commitment(perm: torch.Tensor) -> str:
    if not isinstance(perm, torch.Tensor) or perm.ndim != 1 or perm.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("invalid vocabulary permutation")
    perm = perm.cpu().long()
    if perm.numel() == 0 or not torch.equal(torch.sort(perm).values, torch.arange(perm.numel())):
        raise ValueError("invalid vocabulary permutation")
    values = perm.numpy().astype("<i8", copy=False)
    return hashlib.sha256(b"osnip-token-map-v1\0" + len(values).to_bytes(8, "little") + values.tobytes()).hexdigest()


def verify_encoder_binding(root: str | Path) -> dict:
    root = Path(root)
    binding = torch.load(root / "osnip/training_binding.pt", map_location="cpu", weights_only=True)
    if not isinstance(binding, dict) or binding.get("format") != "osnip_encoder_binding_v1" or type(binding.get("global_step")) is not int or binding["global_step"] <= 0:
        raise ValueError("invalid encoder training binding")
    server = root / "model_server"
    files = {p.relative_to(server).as_posix(): _sha(p) for p in sorted(server.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if not files or files != binding.get("model_files_sha256"):
        raise ValueError("encoder training server contents do not match")
    secret = torch.load(root / "client/client_secret.pt", map_location="cpu", weights_only=True)
    if not isinstance(secret, dict) or _commitment(secret.get("perm_vocab")) != binding.get("token_map_commitment_sha256"):
        raise ValueError("encoder training vocabulary permutation does not match")
    encoder = torch.load(root / "osnip/encryptor.pt", map_location="cpu", weights_only=True)
    saved = binding.get("encoder")
    if not isinstance(encoder, dict) or not isinstance(saved, dict) or not encoder or encoder.keys() != saved.keys():
        raise ValueError("encoder training binding state does not match")
    for name, tensor in encoder.items():
        other = saved[name]
        if not isinstance(tensor, torch.Tensor) or not isinstance(other, torch.Tensor) or tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other) or not torch.isfinite(tensor).all():
            raise ValueError("encoder training binding state does not match")
    return {"verified": True, "method": "training-output binding consistency", "limitation": "does not independently establish training history, data isolation, budgets, or security acceptance"}
