"""Closed-loop invariant attacks (IA) on the plaintext Llama-3.2-1B.

Three sub-experiments, all on the real model and the full 128,256-token vocab:

* ``ednn`` - the EDNN glide-reflection invariant on real prompt embeddings,
  against the full vocabulary reference table (24 prompts, all layers 4/8/12).
* ``gate`` - Gate-IA mapping recovery over ALL 128,256 tokens: the secret is a
  row permutation of the real embedding table plus an independent column
  permutation of each layer's real ``gate_proj``. Invariants are computed with
  the actual ``gate_invariant`` (chunked over tokens; identical arithmetic),
  matching with the actual ``match_by_invariant``.
* ``attn`` - Attn-IA mapping recovery over ALL 128,256 tokens: Q = E Wq^T for
  the first 8 query heads of one layer, obfuscated by a per-RoPE-block invertible 2x2
  transform (``half_split`` pairing) and a row permutation of the vocab. The
  invariant uses the actual ``stacked_invariant`` / ``whitening_invariant``;
  matching is chunked over observed rows because the 128k x 128k x 2048
  distance matrix does not fit in memory (same nearest-neighbour rule).

Faithfulness notes recorded with every result:
* these runs use the invariant-preserving construction (no additive embedding
  noise) - AloePri's alpha_e noise is what breaks the invariant, so this is the
  attack's ceiling, not a claim about the defended system;
* no low-rank / vocabulary subsampling is applied to the embedding table.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from rm_common import OUTDIR, PROMPTS, append, load_done, log

RESULTS = OUTDIR / "ia_full.jsonl"
LAYERS = (4, 8, 12)
GATE_LAYERS = 4  # number of layers stacked in the gate invariant
ATTN_LAYER = 8
CHUNK = 512
ATTN_HEADS = 8  # distinguishes heads; the paper's invariant analysis uses one


def nn_match_topk(P: np.ndarray, O: np.ndarray, top_k: int, chunk: int = 1024):
    """Nearest-neighbour matching by chunked BLAS (same rule as match_by_invariant).

    ``match_by_invariant`` builds a full ``V x D`` difference per observed row,
    which is O(V*D) memory *per row* - unusable at V=128k, D=2048. With the
    identity ``||p-o||^2 = ||p||^2 + ||o||^2 - 2 p.o`` the whole distance matrix
    is a matrix product, computed here in row chunks. The ranking is identical.
    """
    P = np.ascontiguousarray(P, dtype=np.float32)
    O = np.ascontiguousarray(O, dtype=np.float32)
    Pn = np.einsum("ij,ij->i", P, P)
    On = np.einsum("ij,ij->i", O, O)
    k = min(int(top_k), P.shape[0])
    mapping: dict[int, int] = {}
    candidates: dict[int, list[int]] = {}
    for s in range(0, O.shape[0], chunk):
        stop = min(s + chunk, O.shape[0])
        D = On[s:stop, None] + Pn[None, :] - 2.0 * (O[s:stop] @ P.T)
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        dsort = np.take_along_axis(D, part, axis=1)
        order = np.argsort(dsort, axis=1, kind="stable")
        ranked = np.take_along_axis(part, order, axis=1)
        for j in range(stop - s):
            candidates[s + j] = [int(x) for x in ranked[j]]
            mapping[s + j] = int(ranked[j, 0])
    return mapping, candidates


def load_cpu(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded {model_dir} fp32/cpu in {time.time()-t0:.1f}s", tag="model")
    return model, tok


def embedding_table(model) -> np.ndarray:
    return model.get_input_embeddings().weight.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# EDNN
# ---------------------------------------------------------------------------
def run_ednn(model, tok, E, prompt_idx: int, layer: int, seed: int) -> dict:
    from obfuscation_attacks.attacks.ia.ednn import EDNN
    from obfuscation_attacks.data.contracts import AttackSample, Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    ids = tok(PROMPTS[prompt_idx], return_tensors="pt")["input_ids"][0]
    ids = ids[ids < E.shape[0]]
    content = ids[1:]
    # EDNN observes the *embedding* layer, not a hidden state: take the clean
    # token embeddings and apply the glide-reflection as the obfuscation.
    with torch.inference_mode():
        e = model.get_input_embeddings()(content).detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    d = e.shape[1]
    ones = np.ones(d)
    obf = np.empty_like(e)
    for i in range(e.shape[0]):
        a = float(rng.uniform(0.1, 1.0)); b = float(rng.uniform(0.0, 1.0))
        l = ones * a
        obf[i] = e[i] - 2.0 * (e[i] @ l) / (l @ l) * l + ones * b
    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=int(content.numel()),
        vocab_size=int(E.shape[0]), layer=layer,
        obfuscated_embeddings=obf, reference_embeddings=E,
        plaintext_token_ids=content.numpy(), text_source="real", data_split="eval",
    )
    ctx = build_context(GenericScheme("ia", frozenset({Capability.EMBEDDINGS})), sample)
    t0 = time.time()
    res = EDNN(top_k=10).run(ctx)
    met = compute_metrics(res, sample)["token_recovery"]
    rec = [int(x) for x in res.recovered_token_ids.tolist()]
    return {
        "job_id": f"ia_ednn:{prompt_idx}:L{layer}", "family": "ia", "variant": "ednn",
        "prompt_idx": prompt_idx, "layer": layer, "seed": seed,
        "vocab": int(E.shape[0]), "n_positions": int(content.numel()),
        "asr1": met["asr@1"], "asr5": met["asr@5"], "ttrsr": met["ttrsr"],
        "original": tok.decode(content.tolist()), "recovered": tok.decode(rec),
        "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Gate-IA over the full vocabulary
# ---------------------------------------------------------------------------
def gate_full(E, model, seed: int) -> dict:
    from obfuscation_attacks.attacks.ia.core import match_by_invariant
    from obfuscation_attacks.attacks.ia.gate import gate_invariant

    V, d = E.shape
    rng = np.random.default_rng(seed)
    W = np.stack([model.model.layers[l].mlp.gate_proj.weight.detach().cpu().numpy().T
                  for l in range(GATE_LAYERS)], 0)  # (L, d, hf)
    perm = rng.permutation(V)
    col_perms = [rng.permutation(W.shape[2]) for _ in range(GATE_LAYERS)]
    Wo = np.stack([W[l][:, col_perms[l]] for l in range(GATE_LAYERS)], 0)

    def plain_inv_chunk(rows):
        return gate_invariant(E[rows], W)

    def obf_inv_chunk(rows):
        return gate_invariant(E[perm[rows]], Wo)

    t0 = time.time()
    plain_inv = np.empty((V, GATE_LAYERS), dtype=np.float64)
    obf_inv = np.empty((V, GATE_LAYERS), dtype=np.float64)
    for s in range(0, V, 8192):
        rows = np.arange(s, min(s + 8192, V))
        plain_inv[rows] = plain_inv_chunk(rows)
        obf_inv[rows] = obf_inv_chunk(rows)

    mapping: dict[int, int] = {}
    candidates: dict[int, list[int]] = {}
    for s in range(0, V, 8192):
        rows = np.arange(s, min(s + 8192, V))
        m, c = match_by_invariant(plain_inv, obf_inv[rows], top_k=10)
        for j, r in enumerate(rows):
            mapping[int(r)] = int(m[j])
            candidates[int(r)] = [int(x) for x in c[j]]
    correct = sum(1 for i in range(V) if mapping[i] == perm[i])
    top10 = sum(1 for i in range(V) if perm[i] in candidates[i][:10])
    return {
        "job_id": f"ia_gate:full:{seed}", "family": "ia", "variant": "gate_full_vocab",
        "seed": seed, "vocab": V, "layers": GATE_LAYERS, "invariant_dim": GATE_LAYERS,
        "mapping_top1": correct / V, "mapping_top10": top10 / V,
        "num_observed_symbols": V, "chance": 1.0 / V,
        "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Attn-IA over the full vocabulary
# ---------------------------------------------------------------------------
def attn_full(E, model, seed: int, layer: int) -> dict:
    from obfuscation_attacks.attacks.ia.attn import stacked_invariant

    V, _ = E.shape
    rng = np.random.default_rng(seed)
    Wq = model.model.layers[layer].self_attn.q_proj.weight.detach().cpu().numpy()  # (d, d)
    H = min(int(model.config.num_attention_heads), ATTN_HEADS)
    hd = model.config.head_dim
    t0 = time.time()
    Q = (E @ Wq.T[:, : H * hd]).astype(np.float32)  # first H heads; (V, H*hd)
    del Wq
    perm = rng.permutation(V)
    Q_obf = Q.copy()
    half = hd // 2
    for h in range(H):
        base = h * hd
        for j in range(half):
            A = rng.standard_normal((2, 2))
            while abs(np.linalg.det(A)) < 0.3:
                A = rng.standard_normal((2, 2))
            cols = [base + j, base + j + half]
            Q_obf[:, cols] = Q[:, cols] @ A
    Q_obf = Q_obf[perm]  # secret row permutation (obf row i <- plain perm[i])
    Qs = Q.reshape(V, H, hd).transpose(1, 0, 2)
    Qo = Q_obf.reshape(V, H, hd).transpose(1, 0, 2)

    plain_inv, plain_keys = stacked_invariant({"Q": Qs}, block_size=2, pairing="half_split")
    obf_inv, obf_keys = stacked_invariant({"Q": Qo}, block_size=2, pairing="half_split")
    common = [k for k in plain_keys if k in set(obf_keys)]
    p_idx = [plain_keys.index(k) for k in common]
    o_idx = [obf_keys.index(k) for k in common]
    P = plain_inv[:, p_idx]
    O = obf_inv[:, o_idx]

    mapping, candidates = nn_match_topk(P, O, top_k=10)
    correct = sum(1 for i in range(V) if mapping[i] == perm[i])
    top10 = sum(1 for i in range(V) if perm[i] in candidates[i][:10])
    return {
        "job_id": f"ia_attn:full:{layer}:{seed}", "family": "ia",
        "variant": "attn_full_vocab", "seed": seed, "layer": layer,
        "vocab": V, "heads": H, "invariant_dim": int(P.shape[1]),
        "mapping_top1": correct / V, "mapping_top10": top10 / V,
        "num_observed_symbols": V, "chance": 1.0 / V,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--which", default="ednn,gate,attn")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--edge-seeds", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()
    which = [w.strip() for w in args.which.split(",") if w.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    done = load_done(RESULTS)
    model, tok = load_cpu(args.model)
    E = embedding_table(model)
    log(f"IA full: E={E.shape}, which={which}, seeds={seeds}")

    if "ednn" in which:
        for pi in range(len(PROMPTS)):
            for l in LAYERS:
                jid = f"ia_ednn:{pi}:L{l}"
                if jid in done:
                    continue
                rec = run_ednn(model, tok, E, pi, l, seed=0)
                append(RESULTS, rec)
                log(f"{jid}: asr1={rec['asr1']:.3f} asr5={rec['asr5']:.3f} "
                    f"ttrsr={rec['ttrsr']:.3f} ({rec['seconds']}s)")

    if "gate" in which:
        for s in seeds:
            jid = f"ia_gate:full:{s}"
            if jid in done:
                continue
            rec = gate_full(E, model, s)
            append(RESULTS, rec)
            log(f"{jid}: top1={rec['mapping_top1']:.4f} top10={rec['mapping_top10']:.4f} "
                f"({rec['seconds']}s)")

    if "attn" in which:
        for s in seeds:
            jid = f"ia_attn:full:{ATTN_LAYER}:{s}"
            if jid in done:
                continue
            rec = attn_full(E, model, s, ATTN_LAYER)
            append(RESULTS, rec)
            log(f"{jid}: top1={rec['mapping_top1']:.4f} top10={rec['mapping_top10']:.4f} "
                f"dim={rec['invariant_dim']} ({rec['seconds']}s)")

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
