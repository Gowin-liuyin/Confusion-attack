"""Extended real-model validation (follow-ups a/b/c) on Llama-3.2-1B-Instruct.

Experiments (flag-driven via --exp):
  imakeyed  (c) IMA with a KEYED obfuscation on REAL embeddings: train on
            attacker-sampled keys, eval on a held-out client key (faithful,
            expect ~chance) vs a known-key run (capacity, expect high).
  vmaksweep (a) VMA autoregressive replay with larger candidate pools K.
  vmafull   (a) VMA replay with the FULL 128,256 vocab for one position
            (definitive: is the true token the global L1-argmin?).
  isaladder (b) ISA-ER recovery vs optimization-step budget (300/1000/3000).

Results are merged into reports/real_model/results_ext.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from obfuscation_attacks.data.contracts import AttackSample, Capability
from obfuscation_attacks.schemes.generic import GenericScheme
from obfuscation_attacks.schemes.base import build_context
from obfuscation_attacks.schemes.mechanism import AloePriToyMechanism, attacker_oracle
from obfuscation_attacks.evaluation.scoring import compute_metrics
from obfuscation_attacks.attacks.ima.classify import ClassifyInverter
from obfuscation_attacks.attacks.isa.er import ERAttack
from obfuscation_attacks.attacks.vma.autoregressive import _l1
from obfuscation_attacks.metrics import ttrsr

PROMPT = "The quick brown fox jumps over the lazy dog near the river bank at dawn."
OUT = ROOT / "reports" / "real_model" / "results_ext.json"
RESULTS: dict = json.loads(OUT.read_text()) if OUT.exists() else {}


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")


def load_model(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load] {time.time()-t0:.1f}s vocab={model.config.vocab_size} layers={model.config.num_hidden_layers} (cpu fp32)")
    return model, tok


# ---------------------------------------------------------------------------
# (c) IMA with a keyed obfuscation on REAL Llama embeddings
# ---------------------------------------------------------------------------
def exp_imakeyed(model, tok, E_np, *, M=128, N=6, seed=0):
    print(f"\n=== (c) IMA keyed real-embedding scenario (M={M} working vocab) ===")
    rng = np.random.default_rng(seed)
    vocab_ids = rng.choice(E_np.shape[0], size=M, replace=False)
    E_sub = E_np[vocab_ids].astype(np.float64)  # REAL Llama embedding rows, relabelled 0..M-1
    d = E_sub.shape[1]
    A = rng.standard_normal((d, d))
    mech = AloePriToyMechanism(E_sub, alpha_e=0.0, A=A, seed=seed)
    client_key = mech.sample_key(np.random.default_rng(seed + 777))

    corpus = [rng.integers(0, M, size=N) for _ in range(60)]
    corpus.append(np.arange(M))  # full label coverage
    held = np.resize(np.arange(10), N)
    held_t = tuple(int(x) for x in held)
    corpus = [s for s in corpus if tuple(int(x) for x in s) != held_t]
    eval_obf = mech.obfuscate_embeddings(held, client_key)

    def run(fn, tag):
        sample = AttackSample(
            tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=N, vocab_size=M,
            obfuscated_embeddings=eval_obf, reference_embeddings=E_sub, reference_corpus=corpus,
            plaintext_token_ids=held, text_source="real_embeddings", data_split="eval",
        )
        scheme = GenericScheme("imakeyed", frozenset({Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}), obfuscation_fn=fn)
        ctx = build_context(scheme, sample)
        res = ClassifyInverter(knowledge="GB", epochs=150, lr=2e-2, hidden_dims=(128,), seed=0).run(ctx)
        acc = float((res.recovered_token_ids == held).mean())
        print(f"  {tag:22s} asr@1={acc:.3f}  (chance=1/M={1.0/M:.4f})")
        return acc

    t0 = time.time()
    unk = run(attacker_oracle(mech, attacker_known_key=False, seed=seed), "unknown-key (faithful)")
    kn = run(attacker_oracle(mech, client_key=client_key, attacker_known_key=True, seed=seed), "known-key (capacity)")
    print(f"  [{time.time()-t0:.1f}s]  unknown-key ~ chance (AloePri defends); known-key proves inverter capacity")
    RESULTS["ima_keyed_real"] = {
        "working_vocab_M": M, "chance": 1.0 / M, "unknown_key_asr1": unk, "known_key_asr1": kn,
        "obfuscation": "permutation(secret key) + shared random linear map on REAL Llama embeddings, alpha_e=0",
        "note": "faithful threat model: attacker knows the algorithm, not the client key",
    }
    save()


# ---------------------------------------------------------------------------
# VMA replay helpers
# ---------------------------------------------------------------------------
def _victim_states(model, ids, layers):
    with torch.inference_mode():
        hs = model(ids.unsqueeze(0), output_hidden_states=True, use_cache=False).hidden_states
    return {l: hs[l][0].numpy() for l in layers}


def exp_vmaksweep(model, tok, *, Ks=(256, 1024), max_content=6, main_layer=8, seed=0):
    print(f"\n=== (a) VMA replay candidate-pool sweep K in {Ks} (main layer {main_layer}) ===")
    rng = np.random.default_rng(seed)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0][: 1 + max_content]
    full = ids.tolist()
    n = len(full)
    V = model.config.vocab_size
    layers = [main_layer]
    victim = _victim_states(model, ids, layers)
    content = full[1:]
    out = {}
    for K in Ks:
        hits = 0
        recovered = []
        t0 = time.time()
        for i in range(1, n):
            true_tok = full[i]
            cand = {true_tok, *content}
            while len(cand) < K:
                cand.add(int(rng.integers(0, V)))
            cand = list(cand)[:K]
            if true_tok not in cand:
                cand[0] = true_tok
            batch = torch.tensor([full[:i] + [c] for c in cand], dtype=torch.long)
            with torch.inference_mode():
                g = model(batch, output_hidden_states=True, use_cache=False).hidden_states[main_layer][:, -1, :].numpy()
            h_i = victim[main_layer][i]
            pred = cand[int(np.argmin([_l1(g[k], h_i) for k in range(len(cand))]))]
            hits += int(pred == true_tok)
            recovered.append(pred)
        acc = hits / (n - 1)
        tt = ttrsr(recovered, content)
        print(f"  K={K:5d}: asr@1={acc:.3f}  ttrsr={tt:.3f}  ({time.time()-t0:.1f}s)")
        out[K] = {"asr@1": acc, "ttrsr": tt}
    RESULTS["vma_ksweep"] = {"main_layer": main_layer, "positions": n - 1, "by_K": out}
    save()


def exp_vmafull(model, tok, *, layers=(1, 8, 16), position=1, seed=0, batch=256):
    print(f"\n=== (a) VMA FULL-VOCAB replay at position {position} (all {model.config.vocab_size} tokens) ===")
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0][: position + 1]
    full = ids.tolist()
    true_tok = full[position]
    prefix = full[:position]
    V = model.config.vocab_size
    victim = _victim_states(model, ids, list(layers))
    dist = {l: np.empty(V, dtype=np.float64) for l in layers}
    t0 = time.time()
    for start in range(0, V, batch):
        cand = list(range(start, min(start + batch, V)))
        b = torch.tensor([prefix + [c] for c in cand], dtype=torch.long)
        with torch.inference_mode():
            hs = model(b, output_hidden_states=True, use_cache=False).hidden_states
        for l in layers:
            g = hs[l][:, -1, :].numpy()
            h_i = victim[l][position]
            dist[l][start:start + len(cand)] = np.sum(np.abs(g - h_i), axis=1)
        if (start // batch) % 100 == 0:
            print(f"    ... {start}/{V}  ({time.time()-t0:.0f}s)")
    res = {}
    for l in layers:
        argmin = int(np.argmin(dist[l]))
        rank = int((dist[l] < dist[l][true_tok]).sum())  # 0 == true token is the global argmin
        res[l] = {"argmin_is_true": argmin == true_tok, "true_token_rank": rank,
                  "true_dist": float(dist[l][true_tok]), "min_dist": float(dist[l][argmin])}
        print(f"  layer {l:2d}: true_token_rank={rank} (0=global argmin)  argmin_is_true={argmin==true_tok}  "
              f"true_dist={dist[l][true_tok]:.3f} min_dist={dist[l][argmin]:.3f}")
    print(f"  [{time.time()-t0:.0f}s total]")
    RESULTS["vma_full_vocab"] = {"position": position, "true_token": true_tok, "by_layer": res,
                                 "note": "full 128,256-candidate L1 replay; rank 0 == true token beats the entire vocab"}
    save()


# ---------------------------------------------------------------------------
# (b) ISA-ER recovery vs step budget
# ---------------------------------------------------------------------------
class _LlamaHandle:
    def __init__(self, m):
        self.model = m
        self.vocab_size = m.config.vocab_size
        self.hidden_size = m.config.hidden_size
        self.num_layers = m.config.num_hidden_layers
        self._E = m.get_input_embeddings().weight.detach()

    def embedding_matrix(self):
        return self._E

    def forward_from_embeddings(self, inputs_embeds, layer=None):
        out = self.model.model(inputs_embeds=inputs_embeds.unsqueeze(0), output_hidden_states=True, use_cache=False)
        l = self.num_layers if layer is None else int(layer)
        return out.hidden_states[l][0]

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self


def exp_isaladder(model, tok, E_np, *, layer=2, n_content=5, budgets=(300, 1000, 3000), seed=0):
    print(f"\n=== (b) ISA-ER recovery vs step budget (layer {layer}, budgets {budgets}) ===")
    handle = _LlamaHandle(model)
    content = tok(PROMPT, return_tensors="pt")["input_ids"][0][1: 1 + n_content]
    with torch.inference_mode():
        target = handle.forward_from_embeddings(handle.embedding_matrix()[content], layer).numpy()
    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=int(content.numel()), vocab_size=E_np.shape[0],
        layer=layer, obfuscated_hidden_states=target, reference_embeddings=E_np,
        plaintext_token_ids=content.numpy(), text_source="real", data_split="eval",
    )
    ladder = {}
    for steps in budgets:
        ctx = build_context(GenericScheme("isa", frozenset({Capability.HIDDEN_STATES})), sample, attacker_model=handle)
        t0 = time.time()
        res = ERAttack(distance="l2", lr=0.1, steps=steps, top_k=5, seed=seed).run(ctx)
        m = compute_metrics(res, sample)["token_recovery"]
        rec = tok.decode(res.recovered_token_ids.tolist())
        print(f"  steps={steps:5d}: ttrsr={m['ttrsr']:.3f} asr@5={m['asr@5']:.3f} "
              f"state_dist={res.diagnostics['final_state_distance']:.4f} ({time.time()-t0:.0f}s)  rec={rec!r}")
        ladder[steps] = {"ttrsr": m["ttrsr"], "asr@5": m["asr@5"],
                         "state_distance": res.diagnostics["final_state_distance"], "recovered": rec}
    RESULTS["isa_ladder"] = {"layer": layer, "n_content": n_content, "original": tok.decode(content),
                             "by_steps": ladder, "note": "paper uses ~5e4 steps; this is a CPU-budget ladder"}
    save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--exp", required=True, help="comma: imakeyed,vmaksweep,vmafull,isaladder")
    args = ap.parse_args()
    exps = [e.strip() for e in args.exp.split(",") if e.strip()]
    model, tok = load_model(args.model)
    E_np = model.get_input_embeddings().weight.detach().numpy()
    table = {
        "imakeyed": lambda: exp_imakeyed(model, tok, E_np),
        "vmaksweep": lambda: exp_vmaksweep(model, tok),
        "vmafull": lambda: exp_vmafull(model, tok),
        "isaladder": lambda: exp_isaladder(model, tok, E_np),
    }
    for e in exps:
        table[e]()
    print(f"\n[done] {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
