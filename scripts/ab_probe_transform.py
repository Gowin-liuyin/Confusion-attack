#!/usr/bin/env python3
"""Second probe: identify the AloePri ``square_monomial`` transform in the packaged
A+B artifact, and measure what a weight-only attacker (KW) can recover.

Hypothesis (from ``client/client_secret.pt`` metadata:
``transform_family = square_monomial_standard_shape``, ``transform_dim = 2048``):
the server weights are the clean weights transformed by a **monomial** operator
``Q = (Pi D)^2`` -- a permutation of the 2048 hidden coordinates composed with a
diagonal scaling, applied twice -- plus relative noise ``alpha_e`` on embeddings
and ``alpha_h`` on hidden-side weights, plus a token permutation on the vocab
axis.  Concretely, for the embedding table::

    E_dep[perm[i], :] = ((E_clean[i, :] * d) [:, pi] ...) + noise

i.e. the *column* map is a permutation times a scale, not a dense rotation.

If that is true then

* a row-wise cosine between ``E_dep[perm]`` and ``E_clean`` is ~0 (coordinate
  mismatch) -- what probe 1 measured; but
* the **column** correlation matrix between them is, up to noise, a scaled
  permutation matrix, i.e. the transform is recoverable from weights alone;
* per-row *sorted magnitude profiles* (the VMA invariant) survive the transform.

This probe reports all three, plus the vocab-permutation structure (fixed points,
special tokens, tail rows) and the per-tensor transform signature for the other
weight matrices.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ART = Path(r"F:\OSNIP_package_20261101\ab_runnable")
CLEAN = Path(r"F:\Models\Llama-3.2-1B-Instruct")


def load_embed(path: Path) -> np.ndarray:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32)
    m.eval()
    w = m.get_input_embeddings().weight.detach().float().numpy().copy()
    del m
    return w


def column_fit(clean: np.ndarray, deployed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit signed column scales after matching absolute correlations."""
    clean = clean - clean.mean(axis=0, keepdims=True)
    deployed = deployed - deployed.mean(axis=0, keepdims=True)
    clean_norm = np.linalg.norm(clean, axis=0)
    deployed_norm = np.linalg.norm(deployed, axis=0)
    corr = (clean / (clean_norm + 1e-12)).T @ (deployed / (deployed_norm + 1e-12))
    best = np.abs(corr).argmax(axis=1)
    scales = np.sum(clean * deployed[:, best], axis=0) / (np.sum(clean * clean, axis=0) + 1e-12)
    residual = np.linalg.norm(deployed[:, best] - clean * scales, axis=1) / (
        np.linalg.norm(deployed[:, best], axis=1) + 1e-12
    )
    return corr, best, residual


def main() -> None:
    out_path = ROOT / "reports" / "ab_probe" / "oracle_transform_probe.json"
    report: dict = {"classification": "oracle_diagnostic", "uses_secret_row_permutation": True,
                    "applicability": "diagnostic only; not a public attack or privacy result"}
    t0 = time.time()
    E_dep = load_embed(ART / "model_server")
    E_clean = load_embed(CLEAN)
    report["load_seconds"] = round(time.time() - t0, 1)

    secret = torch.load(ART / "client" / "client_secret.pt", map_location="cpu", weights_only=False)
    perm = secret["perm_vocab"].long().numpy()
    meta = secret["metadata"]
    report["metadata"] = {k: meta[k] for k in
                          ("transform_family", "implementation_variant", "alpha_e", "alpha_h",
                           "transform_dim", "tail_row_count", "special_ids_fixed", "tail_rows_fixed",
                           "paper_algorithm1_reproduced")}

    # ---- token-permutation structure --------------------------------------
    fixed = np.flatnonzero(perm == np.arange(len(perm)))
    report["perm"] = {
        "fixed_points": int(len(fixed)),
        "fixed_are_special_or_tail": bool(np.all((fixed >= 128000) | (fixed < 3))),
        "max_displacement": int(np.abs(perm - np.arange(len(perm))).max()),
    }
    # tail rows fixed?
    report["tail_rows_identical"] = bool(np.allclose(E_dep[128000:], E_clean[128000:], atol=0))

    # ---- monomial hypothesis on the embedding table ------------------------
    # Align rows by the secret permutation, then look at column correlation.
    A = E_dep[perm]                       # [V, 2048] deployment rows in clean order
    rng = np.random.default_rng(0)
    rows = rng.choice(len(perm), size=8192, replace=False)
    As = A[rows]
    Cs = E_clean[rows]
    As = As - As.mean(0, keepdims=True)
    Cs = Cs - Cs.mean(0, keepdims=True)
    t0 = time.time()
    corr, best, resid = column_fit(Cs, As)
    report["corr_seconds"] = round(time.time() - t0, 1)
    best_val = corr[np.arange(corr.shape[0]), best]
    report["column_map"] = {
        "unique_targets": int(len(set(best.tolist()))),
        "is_bijection": bool(len(set(best.tolist())) == corr.shape[0]),
        "best_abs_corr_mean": float(np.mean(np.abs(best_val))),
        "best_abs_corr_p05": float(np.percentile(np.abs(best_val), 5)),
        "second_best_abs_corr_mean": float(
            np.mean(np.partition(np.abs(corr), -2, axis=1)[:, -2])),
    }
    # implied scale factors if the map is a scaled permutation
    scale = np.array([As[:, best[j]].std() / (Cs[:, j].std() + 1e-12) for j in range(As.shape[1])])
    report["column_scale"] = {"mean": float(scale.mean()), "min": float(scale.min()),
                              "max": float(scale.max())}

    # ---- does the monomial map explain the deployment rows? ----------------
    # Residual compares matched columns, including their fitted signs.
    report["monomial_fit_rel_error"] = {"mean": float(resid.mean()), "p50": float(np.median(resid)),
                                        "p95": float(np.percentile(resid, 95))}

    # ---- VMA invariant: per-row sorted magnitude profile -------------------
    prof_dep = np.sort(np.abs(As), axis=1)[:, ::-1]
    prof_clean = np.sort(np.abs(Cs), axis=1)[:, ::-1]
    # scale-free compare: correlation of profiles
    pd = prof_dep / (np.linalg.norm(prof_dep, axis=1, keepdims=True) + 1e-12)
    pc = prof_clean / (np.linalg.norm(prof_clean, axis=1, keepdims=True) + 1e-12)
    prof_corr = np.sum(pd * pc, axis=1)
    report["rowsort_profile_corr"] = {"mean": float(prof_corr.mean()),
                                      "p05": float(np.percentile(prof_corr, 5))}

    # ---- transform signature of the other tensors --------------------------
    from safetensors.torch import load_file

    idx = json.loads((ART / "model_server" / "model.safetensors.index.json").read_text())
    dep_state = {}
    for shard in sorted(set(idx["weight_map"].values())):
        dep_state.update(load_file(str(ART / "model_server" / shard), device="cpu"))
    clean_state = {}
    cidx_path = CLEAN / "model.safetensors.index.json"
    if cidx_path.exists():
        cidx = json.loads(cidx_path.read_text())
        cshards = sorted(set(cidx["weight_map"].values()))
        for shard in cshards:
            clean_state.update(load_file(str(CLEAN / shard), device="cpu"))
    else:
        clean_state = load_file(str(CLEAN / "model.safetensors"), device="cpu")

    sig = {}
    for name, tensor in dep_state.items():
        if name not in clean_state:
            sig[name] = {"status": "missing_in_clean"}
            continue
        c = clean_state[name].float()
        d = tensor.float()
        if c.shape != d.shape:
            sig[name] = {"status": "shape_mismatch", "dep": list(d.shape), "clean": list(c.shape)}
            continue
        cf, df = c.reshape(-1), d.reshape(-1)
        same = bool(torch.equal(c, d))
        cos = float(torch.dot(cf, df) / (cf.norm() * df.norm() + 1e-12))
        sig[name] = {"same": same, "flat_cos": round(cos, 6),
                     "rel_l2": round(float((d - c).norm() / (c.norm() + 1e-12)), 4)}
    report["tensor_signature"] = sig
    report["tensor_summary"] = {
        "identical": sorted(n for n, s in sig.items() if s.get("same")),
        "changed": sorted(n for n, s in sig.items() if s.get("same") is False),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tensor_signature"},
                     indent=2, ensure_ascii=False))
    print("\nidentical tensors:", len(report["tensor_summary"]["identical"]),
          "| changed tensors:", len(report["tensor_summary"]["changed"]))
    print("changed sample:", report["tensor_summary"]["changed"][:12])
    print("identical sample:", report["tensor_summary"]["identical"][:12])


if __name__ == "__main__":
    main()
