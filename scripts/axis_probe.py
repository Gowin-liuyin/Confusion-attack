#!/usr/bin/env python3
"""Which axis does the packaged transform touch? Row vs column vs dense.

The Table-9 route depends on the answer: a monomial on the *hidden* axis makes the
deployment columns scaled copies of clean columns; a left-multiplied monomial makes the
*rows* copies; a dense M^2 makes neither and instead admits a dense least-squares fit.

Measures, for one attention and one MLP matrix:
  * row matching: each deployment row's best clean row by cosine (+ bijection rate)
  * column matching: same along the other axis
  * dense fit: residual of ``dep ~ clean @ M`` (lstsq) -- ~0.01 would mean dense-linear
    with the artifact's alpha_h = 0.01 weight noise, ~1.0 means no linear relation.
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

from obfuscation_attacks.schemes.ab_deployment import ABDeployment  # noqa: E402


def _axis_stats(clean: np.ndarray, dep: np.ndarray) -> dict:
    c = np.asarray(clean, dtype=np.float32)
    d = np.asarray(dep, dtype=np.float32)
    out = {}
    for axis, name in ((0, "rows"), (1, "cols")):
        cc = c if axis == 1 else c
        dd = d
        if axis == 0:            # rows: vectors live along axis 1
            cc = cc
            dd = dd
        else:                    # columns: vectors live along axis 0
            cc = cc.T
            dd = dd.T
        cn = cc - cc.mean(axis=1, keepdims=True)
        dn = dd - dd.mean(axis=1, keepdims=True)
        cn /= np.linalg.norm(cn, axis=1, keepdims=True) + 1e-12
        dn /= np.linalg.norm(dn, axis=1, keepdims=True) + 1e-12
        corr = dn @ cn.T
        best = corr.max(axis=1)
        second = np.partition(corr, -2, axis=1)[:, -2]
        am = corr.argmax(axis=1)
        out[name] = {
            "best_corr_mean": float(best.mean()),
            "second_corr_mean": float(second.mean()),
            "bijection_rate": float(len(set(am.tolist())) / corr.shape[0]),
        }
    # dense fit dep ~ clean @ M  (clean is [n, m], dep is [n, m])
    t0 = time.time()
    M, *_ = np.linalg.lstsq(c.astype(np.float64), d.astype(np.float64), rcond=None)
    resid = float(np.linalg.norm(d - c @ M) / (np.linalg.norm(d) + 1e-12))
    out["dense_fit"] = {"relative_residual": resid, "seconds": round(time.time() - t0, 1)}
    return out


def main() -> None:
    dep = ABDeployment(layers=(1,), max_len=16)
    wc = dep.weights("clean")
    wd = dep.weights("deployment")
    report = {}
    for name in (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ):
        if name not in wc:
            report[name] = {"missing": True}
            continue
        if wc[name].shape[0] > 20000 or wd[name].shape[0] > 20000:
            # the pairwise row/column match is O(n^2) and the dense fit is O(n*m^2); the
            # 128,256-row embedding/head matrices are out of scope here (they are covered
            # by the un-mix diagnostics in dense_mixing_probe.py instead)
            report[name] = {"skipped_large": True, "shape": list(wc[name].shape)}
            print(f"{name}\n  skipped (shape {list(wc[name].shape)} too large for O(n^2) probe)",
                  flush=True)
            out_partial = ROOT / "reports" / "ab_probe" / "axis_probe.json"
            out_partial.parent.mkdir(parents=True, exist_ok=True)
            out_partial.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            continue
        t0 = time.time()
        stats = _axis_stats(wc[name], wd[name])
        stats["seconds"] = round(time.time() - t0, 1)
        stats["shape"] = list(wc[name].shape)
        report[name] = stats
        (ROOT / "reports" / "ab_probe" / "axis_probe.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name}\n  rows best={stats['rows']['best_corr_mean']:.3f} "
              f"bij={stats['rows']['bijection_rate']:.2f} | "
              f"cols best={stats['cols']['best_corr_mean']:.3f} "
              f"bij={stats['cols']['bijection_rate']:.2f} | "
              f"dense resid={stats['dense_fit']['relative_residual']:.3f}", flush=True)
    out = ROOT / "reports" / "ab_probe" / "axis_probe.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
