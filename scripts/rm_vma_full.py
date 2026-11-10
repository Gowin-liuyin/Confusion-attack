"""Full-vocabulary, autoregressive VMA replay on the plaintext Llama-3.2-1B.

This is the faithful B1/OSNIP formula:

    xhat_i = argmin_{v in V} || M_{<=l}([xhat_{<i}, v])_i - h_i ||_1

with the FULL 128,256-token vocabulary, the RECOVERED prefix ``xhat_{<i}``
(not the ground-truth prefix), and every layer decoded independently. The only
batching is over candidate tokens; the L1 rule, prefix and argmin are unchanged.

A self-check compares cached one-token replay against a full-sequence forward
before any prompt is decoded.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from rm_common import OUTDIR, PROMPTS, append, load_done, load_model, log

VMA_LAYERS = [1, 2, 4, 6, 8, 10, 12, 16]
RESULTS_DEFAULT = OUTDIR / "vma_full.jsonl"


def cache_for_prefix(model, prefix_ids: list[int]):
    ids = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        out = model.model(input_ids=ids, use_cache=True, return_dict=True)
    return out.past_key_values


def candidate_dists(model, base_cache, prefix_len: int, victim: dict[int, torch.Tensor],
                    vocab: int, batch: int, sorted_l1: bool = False) -> dict[int, torch.Tensor]:
    """Distances from every vocabulary candidate's layer-l state to the victim.

    ``sorted_l1`` is the OSNIP variant's rule (Algorithms 5/6, hidden-dimension
    permutation): ``sum |sort(a) - sort(b)|``, i.e. the same L1 after sorting each
    vector, which is invariant to a permutation of the hidden dimensions.
    """
    out_d = {l: torch.empty(vocab, dtype=torch.float32, device="cuda") for l in victim}
    for start in range(0, vocab, batch):
        stop = min(start + batch, vocab)
        cand = torch.arange(start, stop, dtype=torch.long, device="cuda").unsqueeze(1)
        n = stop - start
        cache = copy.deepcopy(base_cache)
        cache.batch_repeat_interleave(n)
        pos = torch.full((n, 1), prefix_len, dtype=torch.long, device="cuda")
        mask = torch.ones((n, prefix_len + 1), dtype=torch.long, device="cuda")
        with torch.inference_mode():
            out = model.model(
                input_ids=cand, past_key_values=cache, position_ids=pos,
                attention_mask=mask, use_cache=True, output_hidden_states=True,
                return_dict=True,
            )
        for l in victim:
            g = out.hidden_states[l][:, -1, :].float()
            if sorted_l1:
                g = torch.sort(g, dim=-1).values
                v = torch.sort(victim[l], dim=-1).values
            else:
                v = victim[l]
            out_d[l][start:stop] = (g - v).abs().sum(dim=-1)
        del out, cache, cand, pos, mask
    return out_d


def self_check(model, tok, prompt: str, layer: int = 8, i: int = 3,
               batch: int = 4096) -> dict:
    """Decision-level check: cached replay must rank the same candidates as a
    full-sequence forward.

    A max-abs hidden-state threshold is not usable here: fp16 kernels differ by
    ~6e-2 at layer 16 (an fp32 run of the same check matches exactly), while the
    VMA decision depends on the L1 argmin. So compare the top-1/top-5 candidates
    and the true-token rank of cached replay against a full forward over the
    full vocabulary, which is exactly the operation the attack performs.
    """
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    prefix, true = ids[:i], ids[i]
    V = int(model.config.vocab_size)
    base = cache_for_prefix(model, prefix)
    with torch.inference_mode():
        hs = model.model(input_ids=torch.tensor([ids[:i + 1]], device="cuda"),
                         use_cache=False, output_hidden_states=True,
                         return_dict=True).hidden_states
        victim = hs[layer][0, i].float()
        cached_d = candidate_dists(model, base, len(prefix), {layer: victim}, V, batch)[layer]
        full_d = torch.empty(V, device="cuda")
        for start in range(0, V, 512):
            c = torch.arange(start, min(start + 512, V), device="cuda").tolist()
            out = model.model(
                input_ids=torch.tensor([prefix + [x] for x in c], device="cuda"),
                use_cache=False, output_hidden_states=True, return_dict=True)
            full_d[start:start + len(c)] = (out.hidden_states[layer][:, -1, :].float()
                                            - victim).abs().sum(dim=-1)
    a_top = torch.argsort(cached_d)[:5].tolist()
    b_top = torch.argsort(full_d)[:5].tolist()
    res = {
        "layer": layer,
        "position": i,
        "cached_top1": int(a_top[0]),
        "full_top1": int(b_top[0]),
        "top1_match": a_top[0] == b_top[0],
        "top5_overlap": len(set(a_top) & set(b_top)),
        "true_rank_cached": int((cached_d < cached_d[true]).sum()),
        "true_rank_full": int((full_d < full_d[true]).sum()),
        "max_abs_diff_true_token": float((cached_d[true] - full_d[true]).abs()),
    }
    return res


def run_prompt(model, tok, pi: int, layers: list[int], batch: int,
               sorted_l1: bool = False, tag: str = "vma_full") -> list[dict]:
    ids = tok(PROMPTS[pi], return_tensors="pt")["input_ids"][0]
    full = ids.tolist()
    n = len(full)
    if n < 2:
        return []
    V = int(model.config.vocab_size)
    content = full[1:]
    ids = ids.to("cuda")
    with torch.inference_mode():
        hs = model.model(input_ids=ids.unsqueeze(0), use_cache=False,
                         output_hidden_states=True, return_dict=True).hidden_states
    victim = {l: hs[l][0].float() for l in layers}

    # independent autoregressive decode per layer: prefix is that layer's xhat
    recovered = {l: [] for l in layers}
    ranks = {l: [] for l in layers}
    t0 = time.time()
    for i in range(1, n):
        for l in layers:
            prefix = [full[0]] + recovered[l]
            base = cache_for_prefix(model, prefix)
            d = candidate_dists(model, base, len(prefix), {l: victim[l][i]}, V, batch,
                                sorted_l1=sorted_l1)[l]
            best = int(torch.argmin(d))
            true = int(full[i])
            ranks[l].append(int((d < d[true]).sum()))
            recovered[l].append(best)
            del d, base
    dt = time.time() - t0

    from obfuscation_attacks.metrics import ttrsr

    recs = []
    for l in layers:
        rec = recovered[l]
        rank = torch.tensor(ranks[l], dtype=torch.float32)
        recs.append({
            "job_id": f"{tag}:{pi}:L{l}",
            "family": "vma",
            "variant": ("osnip_sorted_l1_full_vocab" if sorted_l1
                        else "original_autoregressive_full_vocab"),
            "prompt_idx": pi,
            "layer": l,
            "vocab": V,
            "positions": n - 1,
            "autoregressive_prefix": True,
            "asr1": float((rank == 0).float().mean()),
            "asr5": float((rank < 5).float().mean()),
            "mean_true_rank": float(rank.mean()),
            "ttrsr": float(ttrsr(rec, content)),
            "exact": bool(rec == content),
            "original": tok.decode(content),
            "recovered": tok.decode(rec),
            "seconds": round(dt, 1),
        })
    log(f"{tag} prompt {pi}: {n-1} positions x {len(layers)} layers x {V} candidates in {dt:.0f}s")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--layers", default=",".join(map(str, VMA_LAYERS)))
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--distance", default="l1", choices=["l1", "sorted_l1"],
                    help="sorted_l1 runs the OSNIP projection variant")
    ap.add_argument("--out", default="vma_full.jsonl")
    ap.add_argument("--tag", default="vma_full")
    args = ap.parse_args()
    RESULTS = OUTDIR / args.out
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    prompts = [int(x) for x in args.prompts.split(",") if x.strip()] if args.prompts else list(range(len(PROMPTS)))
    done = load_done(RESULTS)
    pending = [p for p in prompts if f"{args.tag}:{p}:L{layers[0]}" not in done]
    log(f"VMA full-vocab: {len(pending)}/{len(prompts)} prompts pending (resume file {RESULTS.name})")
    if not pending:
        log("ALL DONE")
        return 0

    model, tok = load_model(args.model, dtype="float16", device="cuda")
    chk = self_check(model, tok, PROMPTS[0], layer=8, i=3, batch=args.batch)
    log(f"cached-vs-full decision self-check: {json.dumps(chk)}")
    if not chk["top1_match"] or chk["top5_overlap"] < 4:
        raise SystemExit("cached replay disagrees with full forward; refusing to run")

    for pi in pending:
        recs = run_prompt(model, tok, pi, layers, args.batch,
                          sorted_l1=(args.distance == "sorted_l1"), tag=args.tag)
        for r in recs:
            append(RESULTS, r)
        torch.cuda.empty_cache()
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
