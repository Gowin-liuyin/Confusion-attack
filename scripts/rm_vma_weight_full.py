"""Closed-loop AloePri weight-structure VMA on real Llama weights.

Recovers ``Y = Z1 X Z2`` (shared row permutation Z1, per-matrix column
permutation Z2) with the actual ``WeightRowSortVMA`` attack, using real
2048 x 2048 weight matrices of the 1B model (q_proj / o_proj of two decoder
layers, so every matrix shares the row dimension and the same secret Z1).

Two noise settings per seed, mirroring the paper: ``alpha_e = 0`` (exact
relation) and ``alpha_e = 1.0`` (AloePri's recommended noisy weights,
``W* = W_e + alpha_e E``). The paper does not pin the scale of ``E``; this probe
adds noise with std ``alpha_e * std(X)`` (i.e. relative to the matrix's own
std), and every record states that convention in ``noise_scale`` so the probe
numbers are never mistaken for the paper's absolute-noise calibre.

Scope note: the row-sort matcher materialises an ``n x n x m`` distance tensor,
so the 128,256-row embedding table is out of reach on this machine. It runs on
complete real weight matrices (the GQA key/value projections, 512 x 2048, of two
decoder layers), which share one row dimension and therefore one secret Z1 - the
Table 9 setting, with no row subsampling.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from rm_common import OUTDIR, append, load_done, log, status_value

RESULTS = OUTDIR / "vma_weight_full.jsonl"
MATRIX_SIZES = "512x2048"
#: Noise-scale convention recorded with every row: the probe's additive noise has
#: std ``alpha_e * std(X)`` of the obfuscated matrix itself. The paper leaves the
#: scale of E unspecified, so probe numbers are NOT the paper's absolute calibre.
NOISE_SCALE = "alpha_e * std(X) per matrix (relative)"


def failure_record(seed: int, alpha_e: float, status, reason, layers=None,
                   seconds: float | None = None) -> dict:
    """A failed unit that keeps its (variant, alpha_e, seed) identity."""
    rec = {
        "job_id": f"vma_weight:{alpha_e}:{seed}",
        "family": "vma",
        "variant": "weight_rowsort",
        "seed": seed,
        "alpha_e": alpha_e,
        "noise_scale": NOISE_SCALE,
        "matrix_shape": MATRIX_SIZES,
        "status": status_value(status),
        "status_reason": reason,
        "probe_only": True,
    }
    if layers is not None:
        rec["matrices"] = 2 * len(layers)
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 1)
    return rec


def load_cpu(model_dir: str):
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded {model_dir} fp32/cpu in {time.time()-t0:.1f}s", tag="model")
    return model


def matrices(model, layers):
    out = []
    # GQA key/value projections are real 512 x 2048 matrices: they share the same
    # row count (so one Z1 applies to all of them) and 512 rows keeps the
    # attack's n x n x m distance tensor inside RAM.
    for l in layers:
        for kind in ("k_proj", "v_proj"):
            w = getattr(model.model.layers[l].self_attn, kind).weight
            out.append(w.detach().cpu().numpy().astype(np.float64))
    return out


def run_case(model, seed: int, layers, alpha_e: float) -> dict:
    from obfuscation_attacks.attacks.vma.weight_rowsort import WeightRowSortVMA
    from obfuscation_attacks.data.contracts import AttackSample, Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    X = matrices(model, layers)
    n = X[0].shape[0]
    rng = np.random.default_rng(seed)
    z1 = rng.permutation(n)                      # shared secret row permutation
    z2 = [rng.permutation(x.shape[1]) for x in X]
    Y = []
    for l, x in enumerate(X):
        y = x[np.ix_(z1, z2[l])]
        if alpha_e:
            y = y + alpha_e * x.std() * rng.standard_normal(y.shape)
        Y.append(y)

    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=n, vocab_size=n,
        known_weights={"X": np.stack(X, 0), "Y": np.stack(Y, 0)},
        meta={"true_mapping": {int(i): int(z1[i]) for i in range(n)}},
        text_source="real-weights", data_split="eval")
    ctx = build_context(
        GenericScheme("vma", frozenset({Capability.WHITE_BOX_WEIGHTS})), sample)
    t0 = time.time()
    res = WeightRowSortVMA().run(ctx)
    dt = time.time() - t0
    if res.recovered_permutations is None:
        return failure_record(seed, alpha_e, res.status, res.status_reason,
                              layers, dt)
    met = compute_metrics(res, sample, ks=(1, 5, 10)).get("mapping_recovery", {})
    p1 = np.asarray(res.recovered_permutations["Z1"])
    return {
        "job_id": f"vma_weight:{alpha_e}:{seed}", "family": "vma",
        "variant": "weight_rowsort", "seed": seed, "alpha_e": alpha_e,
        "noise_scale": NOISE_SCALE,
        "matrices": len(X), "matrix_shape": MATRIX_SIZES,
        "rows": int(n), "z1_top1": met.get("mapping_top1_accuracy"),
        "z1_top10": met.get("mapping_asr@10"),
        "verified": bool(res.diagnostics.get("verified")),
        "partial": bool(res.diagnostics.get("partial")),
        "collisions": bool(res.diagnostics.get("collisions")),
        "reconstruction_residual": res.diagnostics.get("reconstruction_residual"),
        "min_margin": res.diagnostics.get("min_margin"),
        "z1_exact": bool(np.array_equal(p1, z1)),
        "seconds": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="4,8")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    done = load_done(RESULTS, required_fields=(
        "variant", "seed", "alpha_e", "z1_top1"))
    model = load_cpu(args.model)
    for s in seeds:
        for alpha in (0.0, 1.0):
            jid = f"vma_weight:{alpha}:{s}"
            if jid in done:
                continue
            rec = run_case(model, s, layers, alpha)
            append(RESULTS, rec)
            log(f"{jid}: z1_top1={rec.get('z1_top1')} verified={rec.get('verified')} "
                f"residual={rec.get('reconstruction_residual')} ({rec.get('seconds')}s)")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
