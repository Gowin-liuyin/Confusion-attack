#!/usr/bin/env python3
"""Probe: dense-mixing inversion on the packaged A+B artifact.

1. solve the hidden-axis mixing M from q/k/v across layers (residual must be ~alpha_h);
2. un-mix the deployment embedding table with M^-1;
3. nearest-neighbour the token permutation and score it against the secret permutation.

Writes ``reports/ab_probe/dense_mixing.json`` and prints the chain.
"""

from __future__ import annotations


import os

# BLAS threading must be set before numpy is imported: without it the
# 1000x2048x128256 GEMM runs at ~1 GFLOPS (>10 min on this box), with it ~350 GFLOPS (1.5 s).
# Measured with scripts/_bench_blas.py; ``setdefault`` keeps an explicit caller override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(min(24, os.cpu_count() or 8)))

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.attacks.vma.dense_mixing import recover_mixing, unmix_table  # noqa: E402
from obfuscation_attacks.attacks.vma.table9 import token_permutation_from_embedding  # noqa: E402
from obfuscation_attacks.schemes.ab_deployment import ABDeployment  # noqa: E402
from obfuscation_attacks.evaluation.real_battery import (  # noqa: E402
    CLEAN_ACCESS_SCENARIOS,
)


def require_clean_scenario(scenario: str) -> str:
    """Fail closed: these probes read the clean weights and embeddings."""
    if scenario not in CLEAN_ACCESS_SCENARIOS:
        raise SystemExit(
            f"refusing to read clean reference weights/embeddings in "
            f"scenario {scenario!r}; this probe requires one of "
            f"{sorted(CLEAN_ACCESS_SCENARIOS)}"
        )
    return scenario


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S-public")
    args = ap.parse_args()
    require_clean_scenario(args.scenario)
    t0 = time.time()
    dep = ABDeployment(layers=(1,), max_len=16)
    report: dict = {"artifact": str(dep.artifact), "alpha_h_metadata": dep.secret_metadata.get("alpha_h"),
                    "alpha_e_metadata": dep.secret_metadata.get("alpha_e")}

    mix = recover_mixing(dep)
    report["mixing"] = {k: v for k, v in mix.items() if k != "M"}
    report["mixing"]["per_matrix_sample"] = dict(list(mix["per_matrix"].items())[:4])
    print(f"[dense] M recovered from {mix['n_proposals']} matrices: "
          f"residual mean={mix['residual_mean']:.4f} max={mix['residual_max']:.4f} "
          f"cond={mix['conditional_number']:.1f} spread={mix['proposal_spread_mean']:.2e}", flush=True)

    unmixed = unmix_table(dep.E_dep, mix["M"])
    unmix_residual = float(
        np.linalg.norm(unmixed - dep.E_clean) / (np.linalg.norm(dep.E_clean) + 1e-12)
    )
    report["unmix"] = {"relative_residual": unmix_residual}
    print(f"[dense] un-mixed embedding residual={unmix_residual:.4f}", flush=True)

    rng = np.random.default_rng(20261106)
    for n_types in (1000, 5000, 20000):
        ref = np.sort(rng.choice(len(dep.perm), size=n_types, replace=False))
        res = token_permutation_from_embedding(dep, unmixed, ref)
        hits = sum(
            1 for clean_id in ref
            if res["mapping"].get(int(dep.perm[int(clean_id)])) == int(clean_id)
        )
        accuracy = hits / max(1, len(ref))
        report.setdefault("token_recovery", {})[f"n{n_types}"] = {
            "accuracy": accuracy, "chance": 1 / len(dep.perm)}
        print(f"[dense] token permutation over {n_types} types: accuracy={accuracy:.4f}",
              flush=True)

    # function check: the same M^-1 on the output head should keep logits close
    w_clean, w_dep = dep.weights("clean"), dep.weights("deployment")
    head_c = w_clean["lm_head.weight"]
    head_d = w_dep["lm_head.weight"]
    report["lm_head_relation"] = {
        "clean_shape": list(head_c.shape),
        "dense_fit_residual_rows": float(
            np.linalg.norm(head_d - head_c @ mix["M"].astype(np.float32))
            / (np.linalg.norm(head_d) + 1e-12)),
    }
    report["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "reports" / "ab_probe" / "dense_mixing.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[dense] wrote {out} in {report['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
