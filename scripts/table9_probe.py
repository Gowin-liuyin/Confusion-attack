#!/usr/bin/env python3
"""Probe: does the Table-9 hidden-axis route recover the token permutation?

Runs the geometry only (no payload writing) so the hypothesis can be checked in a
minute before it is wired into the battery:

1. recover the hidden-axis column map from q/k/v/o across all 16 layers (voting);
2. un-mix the deployment embedding table with that map + least-squares scale;
3. nearest-neighbour a reference type set against the un-mixed rows and score it
   against the secret permutation.

Writes ``reports/ab_probe/table9_geometry.json``.
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

from obfuscation_attacks.attacks.vma.table9 import (  # noqa: E402
    recover_hidden_axis,
    token_permutation_from_embedding,
    unmix_embedding_table,
)
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
    out_path = ROOT / "reports" / "ab_probe" / "table9_geometry.json"
    t0 = time.time()
    dep = ABDeployment(layers=(1, 5, 10, 15), max_len=32)
    report: dict = {"artifact": str(dep.artifact), "hidden": dep.hidden, "layers": dep.n_layers}

    axis = recover_hidden_axis(dep)
    report["hidden_axis"] = {
        "voted": axis["voted"].as_dict(),
        "agreement": axis["agreement"],
        "n_proposals": axis["n_proposals"],
        "per_layer_sample": {k: v for k, v in list(axis["per_layer"].items())[:4]},
    }
    print(f"[table9] hidden axis: corr={axis['voted'].corr_best:.4f} "
          f"second={axis['voted'].corr_second:.4f} bijection={axis['voted'].bijection:.3f} "
          f"agreement={axis['agreement']:.3f}", flush=True)

    mix = unmix_embedding_table(dep, axis["voted"].col_map)
    report["unmix"] = {"relative_residual": mix["relative_residual"],
                       "scale": mix["scale_summary"]}
    print(f"[table9] un-mix residual={mix['relative_residual']:.4f} "
          f"scale mean={mix['scale_summary']['mean']:.3f}", flush=True)

    rng = np.random.default_rng(20261106)
    for n_types in (1000, 5000):
        ref = np.sort(rng.choice(len(dep.perm), size=n_types, replace=False))
        res = token_permutation_from_embedding(dep, mix["unmixed"], ref)
        hits = sum(
            1 for clean_id in ref
            if res["mapping"].get(int(dep.perm[int(clean_id)])) == int(clean_id)
        )
        accuracy = hits / max(1, len(ref))
        report.setdefault("token_recovery", {})[f"n{n_types}"] = {
            "accuracy": accuracy, "chance": 1 / len(dep.perm)}
        print(f"[table9] token permutation over {n_types} types: "
              f"accuracy={accuracy:.4f} (chance {1/len(dep.perm):.2e})", flush=True)

    report["seconds"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[table9] wrote {out_path} in {report['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
