#!/usr/bin/env python3
"""Probe the packaged A+B artifact against the clean Llama-3.2-1B weights.

Answers, with numbers, the questions the reproduction needs before any attack can
be claimed faithful:

1. Is ``model_server/`` a *permuted* version of the clean embedding table (so the
   AloePri token permutation is real and recoverable), and how large is the noise?
2. What does the OSNIP encoder actually do to a token embedding (magnitude,
   cosine to the clean embedding, per-key variance)?
3. How expensive is one forward pass / one hidden-state extraction on this CPU,
   i.e. how many texts can a real-arm attack afford?

Writes ``reports/ab_probe/probe.json``; prints a short table.
"""

from __future__ import annotations


import os

# BLAS threading must be set before numpy is imported: without it the
# 1000x2048x128256 GEMM runs at ~1 GFLOPS (>10 min on this box), with it ~350 GFLOPS (1.5 s).
# Measured with scripts/_bench_blas.py; ``setdefault`` keeps an explicit caller override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(min(24, os.cpu_count() or 8)))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = Path(r"F:\OSNIP_package_20261101\ab_runnable")
DEFAULT_CLEAN = Path(r"F:\Models\Llama-3.2-1B-Instruct")


def load_model(path: Path, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    ap.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "ab_probe" / "probe.json")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    report: dict = {"artifact": str(args.artifact), "clean_model": str(args.clean)}

    t0 = time.time()
    tok, dep = load_model(args.artifact / "model_server")
    report["load_seconds_deployment"] = round(time.time() - t0, 1)
    t0 = time.time()
    _, clean = load_model(args.clean)
    report["load_seconds_clean"] = round(time.time() - t0, 1)

    E_dep = dep.get_input_embeddings().weight.detach().float().numpy()
    E_clean = clean.get_input_embeddings().weight.detach().float().numpy()
    report["vocab"] = int(E_dep.shape[0])
    report["hidden"] = int(E_dep.shape[1])
    report["vocab_match"] = bool(E_dep.shape == E_clean.shape)
    report["n_layers"] = int(dep.config.num_hidden_layers)

    secret = torch.load(args.artifact / "client" / "client_secret.pt", map_location="cpu", weights_only=False)
    report["client_secret_keys"] = sorted(secret.keys())
    perm = secret["perm_vocab"].long().numpy()
    report["perm_invertible"] = bool(
        np.array_equal(np.asarray(secret["inv_perm_vocab"].long().numpy())[perm], np.arange(len(perm)))
    )
    report["perm_fixed_points"] = int((perm == np.arange(len(perm))).sum())

    # --- is E_dep a permuted + noised E_clean? ------------------------------
    # rows are unit-ish; compare each clean row to the row the permutation claims.
    rng = np.random.default_rng(0)
    probe_ids = rng.choice(len(perm), size=512, replace=False)
    dep_rows = E_dep[perm[probe_ids]]
    clean_rows = E_clean[probe_ids]
    cos = np.sum(dep_rows * clean_rows, axis=1) / (
        np.linalg.norm(dep_rows, axis=1) * np.linalg.norm(clean_rows, axis=1) + 1e-12
    )
    report["perm_claim_cosine"] = {
        "mean": float(cos.mean()), "p05": float(np.percentile(cos, 5)),
        "p50": float(np.percentile(cos, 50)), "min": float(cos.min()),
    }
    rel = np.linalg.norm(dep_rows - clean_rows, axis=1) / (np.linalg.norm(clean_rows, axis=1) + 1e-12)
    report["perm_claim_rel_l2"] = {"mean": float(rel.mean()), "p50": float(np.median(rel))}

    # --- nearest-row identification: does the deployment table still map 1:1?
    # For a sample of clean rows, find the closest deployment row by cosine.
    sub = rng.choice(len(perm), size=256, replace=False)
    C = E_clean[sub] / (np.linalg.norm(E_clean[sub], axis=1, keepdims=True) + 1e-12)
    D = E_dep / (np.linalg.norm(E_dep, axis=1, keepdims=True) + 1e-12)
    sims = C @ D.T
    nn = sims.argmax(axis=1)
    hit_perm = float(np.mean(nn == perm[sub]))
    report["nn_clean_to_deployment_hit_at_permuted_row"] = hit_perm
    report["nn_sim_of_hit"] = float(sims[np.arange(len(sub)), nn].mean())

    # --- OSNIP encoder behaviour -------------------------------------------
    cfg = json.loads((args.artifact / "osnip" / "config.json").read_text())
    W = int(cfg["width"])
    key_dim = int(cfg.get("key_dim", 32))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    import importlib.util

    spec = importlib.util.spec_from_file_location("ab_runtime", args.artifact / "ab_runtime.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    enc = ab.KeyConditionedEncryptor(W, key_dim=key_dim, hidden_dim=hidden_dim)
    enc.load_state_dict(torch.load(args.artifact / "osnip" / "encryptor.pt", map_location="cpu", weights_only=True))
    enc.eval()
    report["encryptor_params"] = int(sum(p.numel() for p in enc.parameters()))

    ids = torch.tensor([[1, 5000, 12345, 30000, 99999, 7]], dtype=torch.long)
    mapped = torch.from_numpy(perm[ids.numpy()])
    h = dep.get_input_embeddings()(mapped).float()
    key = torch.randn(key_dim, generator=torch.Generator("cpu").manual_seed(0))
    with torch.no_grad():
        raw = enc(h, key)
        z = ab.iso_norm_project(raw, h)
    hd = h[0].numpy()
    zd = z[0].numpy()
    report["osnip_wire"] = {
        "cos_to_h_mean": float(np.mean(np.sum(hd * zd, axis=1) / (np.linalg.norm(hd, axis=1) * np.linalg.norm(zd, axis=1)))),
        "rel_move_mean": float(np.mean(np.linalg.norm(zd - hd, axis=1) / np.linalg.norm(hd, axis=1))),
        "norm_ratio_mean": float(np.mean(np.linalg.norm(zd, axis=1) / np.linalg.norm(hd, axis=1))),
    }
    # per-key spread: same token, different keys
    keys = torch.randn(5, key_dim, generator=torch.Generator("cpu").manual_seed(1))
    with torch.no_grad():
        zs = [ab.iso_norm_project(enc(h, k), h)[0].numpy() for k in keys]
    pair_cos = []
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            a, b = zs[i], zs[j]
            pair_cos.append(float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)))))
    report["osnip_key_spread_cos"] = {"mean": float(np.mean(pair_cos)), "min": float(np.min(pair_cos))}

    # --- timing -------------------------------------------------------------
    text = ("The city of Paris is the capital of France and its largest city, "
            "with a population of about two million people in the city proper.")
    enc_ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    report["probe_seq_len"] = int(enc_ids.shape[1])
    timings = {}
    for name, model in (("deployment", dep), ("clean", clean)):
        for label, kwargs in (("forward", {}), ("hidden", {"output_hidden_states": True})):
            model.eval()
            t0 = time.time()
            with torch.no_grad():
                model(input_ids=enc_ids, **kwargs)
            timings[f"{name}_{label}"] = round(time.time() - t0, 2)
    with torch.no_grad():
        t0 = time.time()
        out = dep(inputs_embeds=z, output_hidden_states=True)
        timings["deployment_inputs_embeds_hidden"] = round(time.time() - t0, 2)
    report["timings_seconds"] = timings
    report["n_hidden_layers_returned"] = int(len(out.hidden_states))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
