#!/usr/bin/env python3
"""Oracle-only timing diagnostic; does not measure a public VMA attack."""

from __future__ import annotations


import os

# BLAS threading must be set before numpy is imported: without it the
# 1000x2048x128256 GEMM runs at ~1 GFLOPS (>10 min on this box), with it ~350 GFLOPS (1.5 s).
# Measured with scripts/_bench_blas.py; ``setdefault`` keeps an explicit caller override.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(min(24, os.cpu_count() or 8)))

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from obfuscation_attacks.attacks.vma.dense_mixing import recover_mixing, unmix_rows  # noqa: E402
from obfuscation_attacks.attacks.vma.weight_rowsort import recover_permutations  # noqa: E402
from obfuscation_attacks.schemes.ab_deployment import ABDeployment  # noqa: E402
from obfuscation_attacks.evaluation.real_battery import (  # noqa: E402
    CLEAN_ACCESS_SCENARIOS,
)


def require_clean_scenario(scenario: str) -> str:
    """Fail closed: this probe reads the clean weights and embeddings."""
    if scenario not in CLEAN_ACCESS_SCENARIOS:
        raise SystemExit(
            f"refusing to read clean reference weights/embeddings in "
            f"scenario {scenario!r}; this probe requires one of "
            f"{sorted(CLEAN_ACCESS_SCENARIOS)}"
        )
    return scenario


def timed(label, fn):
    t0 = time.time()
    out = fn()
    print(f"  {label:38s} {time.time() - t0:7.1f}s", flush=True)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S-public")
    ap.add_argument("--oracle-diagnostic", action="store_true",
                    help="acknowledge that row selection uses the secret permutation")
    args = ap.parse_args()
    require_clean_scenario(args.scenario)
    if not args.oracle_diagnostic:
        ap.error("this profiling diagnostic uses the secret permutation; pass --oracle-diagnostic")
    print("ORACLE DIAGNOSTIC: secret permutation used; accuracy is not a public attack result", flush=True)
    t0 = time.time()
    dep = ABDeployment(layers=(1,), max_len=16)
    print(f"  {'load models + tables':38s} {time.time() - t0:7.1f}s", flush=True)

    rng = np.random.default_rng(0)
    ref = np.sort(rng.choice(len(dep.perm), size=1000, replace=False))
    X = dep.E_clean[ref]
    Y = dep.E_dep[dep.perm[ref]]

    rec = timed("recover_mixing (4 x q_proj lstsq)", lambda: recover_mixing(
        dep, layers=(1, 5, 10, 15), groups=("self_attn.q_proj",), getter=dep.weight_tensor))
    unmixed_rows = timed(
        "unmix_rows (E'[rows] @ M^-1)",
        lambda: unmix_rows(dep.E_dep, rec["M"], dep.perm[ref]),
    )

    def _rowsort():
        return recover_permutations(X, Y)

    timed("row_sort recover_permutations (1000)", _rowsort)

    def _nn():
        table = unmixed_rows / (
            np.linalg.norm(unmixed_rows, axis=1, keepdims=True) + 1e-12
        )
        q = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        hits = 0
        for s in range(0, len(ref), 256):
            nn = (q[s:s + 256] @ table.T).argmax(axis=1)
            hits += int(np.sum(nn == np.arange(s, min(s + 256, len(ref)))))
        return hits / len(ref)

    acc = timed("dense-mixing token NN (1000)", _nn)
    print(f"  accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
