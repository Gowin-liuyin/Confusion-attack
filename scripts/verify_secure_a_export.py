"""Verify a standalone A export with a small real-model forward probe.

This does not certify downstream quality, privacy, or compatibility with B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  # Load before torch to share the installed OpenMP runtime.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.evaluation.export_randomness import verify_export_randomness  # noqa: E402
from obfuscation_attacks.evaluation import export_randomness  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(artifact: Path) -> dict:
    randomness = verify_export_randomness(artifact)
    server = artifact / "server"
    if not server.is_dir() or (artifact / "model_server").exists():
        raise ValueError("expected a standalone A export with exactly one server directory")
    tokenizer = AutoTokenizer.from_pretrained(str(server), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(server), local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
    ).eval()
    secret = torch.load(artifact / "client" / "client_secret.pt", map_location="cpu", weights_only=True)
    permutation = secret["perm_vocab"].long()
    prompt = "The capital of France is"
    clean = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    server_ids = permutation[clean]
    with torch.inference_mode():
        from_ids = model(input_ids=server_ids, use_cache=False).logits[:, -1].float()
        embeds = model.get_input_embeddings()(server_ids)
        from_embeds = model(inputs_embeds=embeds, use_cache=False).logits[:, -1].float()
    if not torch.isfinite(from_ids).all() or not torch.isfinite(from_embeds).all():
        raise ValueError("server forward produced non-finite logits")
    difference = float(torch.max(torch.abs(from_ids - from_embeds)))
    if difference > 1e-5:
        raise ValueError("token and embedding forward paths disagree")
    client_logits = from_ids[0, permutation]
    predicted = int(torch.argmax(client_logits))
    if from_ids[0, permutation[predicted]] != torch.max(from_ids):
        raise ValueError("client output mapping is inconsistent with server logits")
    files = [path for path in sorted(artifact.rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
    return {
        "kind": "standalone_a_secure_export_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact.resolve()),
        "artifact_files_sha256": {path.relative_to(artifact).as_posix(): _sha256(path) for path in files},
        "verifier_sha256": _sha256(Path(__file__)),
        "randomness_checker_sha256": _sha256(Path(export_randomness.__file__)),
        "randomness": randomness,
        "probe": {"input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "input_tokens": int(clean.numel()), "vocab_size": int(from_ids.shape[-1]),
                  "token_vs_embedding_max_logit_difference": difference,
                  "client_mapping_consistent": True, "finite_logits": True,
                  "device": "cpu", "dtype": str(next(model.parameters()).dtype)},
        "status": "passed_probe",
        "scope": "standalone A engineering variant only; not an A+B compatibility, privacy, utility, or formal experiment result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("refusing to overwrite an existing verification report")
    result = verify(args.artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Verified standalone A probe; report: {args.report}")


if __name__ == "__main__":
    main()
