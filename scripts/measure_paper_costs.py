#!/usr/bin/env python3
"""Short, explicitly-labelled resource measurements for the heavy paper-v1 cells.

This is NOT a paper result and never writes into the attack result tree. It loads
the verified deployment artifact once, measures representative per-step costs on
this machine, and reports the hardware limits. CPU-only torch is recorded as a
blocker for GPU-scale extrapolation; no budget is reduced."

Usage::

    python scripts/measure_paper_costs.py --manifest configs/paper/manifest.paper-v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from obfuscation_attacks.data.corpora import load_alpaca  # noqa: E402
from obfuscation_attacks.schemes.ab_deployment import ABDeployment, knn_topk  # noqa: E402


def _peak_rss_gb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:
        return 0.0


def _ids(tokenizer, text: str, length: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(ids) < length:
        reps = (length // max(len(ids), 1)) + 1
        ids = ids.repeat(reps)[:length]
    return ids[:length].long()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "paper" / "manifest.paper-v1.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "reports" / "privacy_rebuild_20261107" / "round4"
                        / "resource_measurements.json")
    parser.add_argument("--root-key-file", type=Path, default=ROOT / ".secrets" / "paper-v1.root")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    artifact = Path(manifest["artifacts"]["ab"]["path"])
    model_path = Path(manifest["models"]["llama-3.2-1b-base"]["path"])
    root_key = args.root_key_file.read_bytes()[:32] if args.root_key_file.is_file() else None

    t0 = time.time()
    dep = ABDeployment(
        artifact=artifact, clean_model=model_path, dtype="float32",
        layers=(1, 5, 10, 15), max_len=None, load_models="deployment",
        strict_length=True, expected_variant="base", root_key=root_key,
    )
    load_seconds = round(time.time() - t0, 1)
    model = dep.model
    tokenizer = dep.tokenizer
    corpus = load_alpaca(Path(manifest["data"]["path"]),
                         seed=int(manifest["data"]["split_seed"]), n_knn=32)
    sample = corpus["knn"].texts()[:4]
    results: dict = {
        "kind": "resource_measurement_only_not_a_paper_result",
        "hardware": {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(),
            "rss_gb_after_load": _peak_rss_gb(),
        },
        "deployment_load_seconds": load_seconds,
        "forward": {},
        "knn_full_vocab": {},
        "isa_step": {},
        "generation": {},
    }

    # 1) decoder forward cost at representative lengths
    for length in (64, 128, 256):
        ids = _ids(tokenizer, sample[0], length).unsqueeze(0)
        times = []
        with torch.no_grad():
            for _ in range(3):
                start = time.time()
                model(input_ids=ids, output_hidden_states=True)
                times.append(time.time() - start)
        results["forward"][f"len_{length}"] = {
            "seconds_median": round(float(np.median(times)), 3),
            "tokens_per_second": round(length / float(np.median(times)), 1),
        }

    # 2) VMA-H full-vocabulary replay: one position = 1 query x 128256 x 2048
    queries = dep.E_dep[np.random.default_rng(0).integers(0, dep.vocab_size, size=64)].astype(np.float32)
    table = dep.table("ab")
    start = time.time()
    table_norm = table / np.maximum(np.linalg.norm(table, axis=1, keepdims=True), 1e-12)
    norm_seconds = time.time() - start
    one = knn_topk(queries[:1], table, k=100, metric="cosine", table_norm=table_norm)
    del one
    start = time.time()
    knn_topk(queries, table, k=100, metric="cosine", table_norm=table_norm)
    batch_seconds = time.time() - start
    start = time.time()
    knn_topk(queries[:1], table, k=100, metric="l2")
    single_l2 = time.time() - start
    results["knn_full_vocab"] = {
        "table_norm_seconds_once": round(norm_seconds, 3),
        "per_position_cosine_seconds_cached_norm": round(batch_seconds / len(queries), 4),
        "per_position_l2_seconds_no_cache": round(single_l2, 4),
        "vocab": int(dep.vocab_size),
        "hidden": int(dep.hidden),
    }

    # 3) ISA-style optimization step (forward + backward through inputs_embeds)
    ids = _ids(tokenizer, sample[0], 64)
    embeddings = torch.from_numpy(dep.E_dep[ids.numpy()]).clone().float().requires_grad_(True)
    times = []
    for _ in range(3):
        start = time.time()
        out = model(inputs_embeds=embeddings.unsqueeze(0), output_hidden_states=True)
        loss = out.hidden_states[1].float().pow(2).mean()
        loss.backward()
        embeddings.grad = None
        times.append(time.time() - start)
    results["isa_step"] = {
        "sequence_length": 64,
        "forward_backward_seconds_median": round(float(np.median(times)), 3),
    }

    # 4) full-prefix client generation step (encode + encoder + forward + argmax)
    text = sample[0]
    token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    key = dep._key_for(0)
    step_times = []
    with torch.no_grad():
        prefix = token_ids
        for _ in range(4):
            start = time.time()
            permuted = torch.from_numpy(dep.perm[prefix.numpy()]).long()
            h = torch.from_numpy(dep.E_dep[permuted.numpy()]).float()
            z = dep.iso_norm_project(dep._encoder(h, torch.from_numpy(key)), h)
            out = model(inputs_embeds=z.unsqueeze(0), output_hidden_states=True)
            next_perm = int(out.logits[0, -1].argmax())
            next_id = int(np.where(dep.perm == next_perm)[0][0])
            prefix = torch.cat([prefix, torch.tensor([next_id])])
            step_times.append(time.time() - start)
    results["generation"] = {
        "prompt_tokens": int(len(token_ids)),
        "per_token_seconds_median_full_prefix": round(float(np.median(step_times)), 3),
        "tokens_per_second": round(1.0 / float(np.median(step_times)), 3),
    }
    results["rss_gb_peak"] = _peak_rss_gb()
    results["note"] = (
        "CPU-only torch; these are measured single-process costs on this machine, not "
        "GPU estimates. Short runs are for resource planning only and do not satisfy any "
        "paper-v1 budget."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
