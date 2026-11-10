"""Closed-loop NN/KNN baselines on the plaintext Llama-3.2-1B (full vocabulary).

Two observation levels, both against the real 128,256 x 2048 embedding table:

* ``embeddings`` - the attacker sees the token embeddings themselves. On a
  plaintext model this is the identity map, so it measures the harness, not an
  attack.
* ``hidden_states`` - the attacker sees the layer-l hidden state and matches it
  to the embedding table by cosine / euclidean distance (the NN/KNN baseline the
  ISA and AloePri papers compare against). This is the informative number.

24 prompts x layers {1,2,4,6,8,10,12,16} x {cosine, euclidean}.
"""

from __future__ import annotations

import argparse
import time

import torch

from rm_common import OUTDIR, PROMPTS, append, load_done, load_model, log

RESULTS = OUTDIR / "nn_full.jsonl"
LAYERS = (1, 2, 4, 6, 8, 10, 12, 16)


def run(model, tok, E_np, pi, layer, level, metric, device="cuda"):
    from obfuscation_attacks.attacks.nn_knn.core import run_knn
    from obfuscation_attacks.data.contracts import AttackSample, Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    ids = tok(PROMPTS[pi], return_tensors="pt")["input_ids"][0]
    content = ids[1:]
    truth = content.numpy()
    if level == "hidden_states":
        with torch.inference_mode():
            hs = model.model(input_ids=ids.to(device).unsqueeze(0), use_cache=False,
                             output_hidden_states=True, return_dict=True).hidden_states
        # run_knn scores whatever vector the scheme exposes; the layer state is
        # handed over through the same slot as embeddings (state -> table match).
        obs = hs[layer][0, 1:].float().cpu().numpy()
        caps = frozenset({Capability.EMBEDDINGS})
        sample = AttackSample(
            tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=len(truth),
            vocab_size=int(E_np.shape[0]), layer=layer,
            obfuscated_embeddings=obs, reference_embeddings=E_np,
            plaintext_token_ids=truth, text_source="real", data_split="eval")
    else:
        obs = E_np[truth]
        caps = frozenset({Capability.EMBEDDINGS})
        sample = AttackSample(
            tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=len(truth),
            vocab_size=int(E_np.shape[0]), layer=layer,
            obfuscated_embeddings=obs, reference_embeddings=E_np,
            plaintext_token_ids=truth, text_source="real", data_split="eval")
    ctx = build_context(GenericScheme("nn", caps), sample)
    res = run_knn(ctx, metric, 10)
    met = compute_metrics(res, sample)["token_recovery"]
    rec = [int(x) for x in res.recovered_token_ids.tolist()]
    return {
        "job_id": f"nn:{level}:{metric}:{pi}:L{layer}", "family": "nn_knn",
        "variant": f"{level}_{metric}", "level": level, "metric": metric,
        "prompt_idx": pi, "layer": layer, "vocab": int(E_np.shape[0]),
        "positions": len(truth),
        "asr1": met["asr@1"], "asr5": met["asr@5"], "ttrsr": met["ttrsr"],
        "original": tok.decode(truth.tolist()), "recovered": tok.decode(rec),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--levels", default="embeddings,hidden_states")
    ap.add_argument("--metrics", default="cosine,euclidean")
    ap.add_argument("--device", default="cuda",
                    help="cpu keeps the baseline off the GPU when other runs need it")
    args = ap.parse_args()
    levels = args.levels.split(",")
    metrics = args.metrics.split(",")
    done = load_done(RESULTS)
    jobs = [(pi, l, lev, m) for lev in levels for m in metrics for pi in range(len(PROMPTS))
            for l in LAYERS]
    pending = [j for j in jobs
               if f"nn:{j[2]}:{j[3]}:{j[0]}:L{j[1]}" not in done]
    log(f"NN/KNN: {len(pending)}/{len(jobs)} jobs pending")
    dtype = "float16" if args.device == "cuda" else "float32"
    model, tok = load_model(args.model, dtype=dtype, device=args.device)
    E_np = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    for pi, l, lev, m in pending:
        t0 = time.time()
        rec = run(model, tok, E_np, pi, l, lev, m, args.device)
        append(RESULTS, rec)
        if lev == "hidden_states" and m == "cosine" and pi % 6 == 0:
            log(f"{rec['job_id']}: asr1={rec['asr1']:.3f} asr5={rec['asr5']:.3f} "
                f"ttrsr={rec['ttrsr']:.3f} ({time.time()-t0:.1f}s)")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
