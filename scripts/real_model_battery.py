"""Large-scale, resumable attack-validation battery on a real plaintext LLM.

Runs the repository's own attack classes against Llama-3.2-1B-Instruct over MANY
prompts / seeds / layers to produce statistically meaningful numbers. Every
completed job is appended to a JSONL so the battery is RESUMABLE across the
per-call time limit: re-run until it prints "ALL DONE", then aggregate.

Usage:
  python scripts/real_model_battery.py --only nn,ednn,gate,attn,tfma,ima      # cheap
  python scripts/real_model_battery.py --only vma,isa --budget 0              # heavy (background)
  python scripts/real_model_battery.py --aggregate                            # build summary + print
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
from obfuscation_attacks.attacks.nn_knn.core import run_knn
from obfuscation_attacks.attacks.ia.ednn import EDNN
from obfuscation_attacks.attacks.ia.gate import GateIA
from obfuscation_attacks.attacks.ia.attn import AttnIA
from obfuscation_attacks.attacks.tfma.frequency_matching import IndependentTFMA
from obfuscation_attacks.attacks.ima.classify import ClassifyInverter
from obfuscation_attacks.attacks.isa.er import ERAttack
from obfuscation_attacks.attacks.vma.autoregressive import _l1
from obfuscation_attacks.metrics import ttrsr

OUTDIR = ROOT / "reports" / "real_model"
JSONL = OUTDIR / "battery.jsonl"
LOG = OUTDIR / "battery.log"
LAYERS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
VMA_LAYERS = [1, 2, 4, 6, 8, 10, 12, 16]

# 24 diverse prompts (natural language, QA, code, numbers, names, dates, multi-domain)
PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
    "Artificial intelligence is transforming how scientists analyze large biological datasets.",
    "She carefully placed the antique porcelain vase on the oak shelf by the window.",
    "Quantum computers exploit superposition and entanglement to solve certain problems faster.",
    "The committee will meet on March 14th, 2026 to finalize the annual budget report.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
    "He whispered that the treasure was buried beneath the third stone from the gate.",
    "Machine learning models require careful validation to avoid overfitting the training data.",
    "The chef added a pinch of saffron, two cloves of garlic, and fresh basil leaves.",
    "Despite the heavy rain, the marathon runners pressed on toward the finish line.",
    "Please transfer 2,450 dollars to account number 8837-2291 before Friday noon.",
    "The ancient library in Alexandria once held hundreds of thousands of scrolls.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Climate scientists warn that rising ocean temperatures threaten coral reef ecosystems.",
    "Why did the stock market react so sharply to the central bank interest rate decision?",
    "Her thesis explored the economic consequences of migration in nineteenth century Europe.",
    "The spacecraft entered orbit around Jupiter after a seven year interplanetary journey.",
    "Mix the flour, sugar, and baking powder, then fold in the melted butter gently.",
    "Linguists study how children acquire grammar without explicit formal instruction.",
    "The detective examined the fingerprints, the torn letter, and the muddy footprints.",
    "Renewable energy capacity grew rapidly as solar panel costs fell over the last decade.",
    "A balanced diet includes proteins, carbohydrates, healthy fats, vitamins, and minerals.",
    "The orchestra tuned their instruments as the conductor raised his baton for silence.",
    "Blockchain networks record transactions in immutable cryptographically linked blocks.",
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done_ids():
    if not JSONL.exists():
        return set()
    ids = set()
    for ln in JSONL.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            ids.add(json.loads(ln)["job_id"])
    return ids


def append(rec):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
def gen_jobs():
    jobs = []
    # cheap
    for pi in range(len(PROMPTS)):
        jobs.append((f"nn:{pi}", "nn", {"pi": pi}))
    for s in range(10):
        jobs.append((f"ednn:{s}", "ednn", {"seed": s, "M": 64}))
        jobs.append((f"gate:{s}", "gate", {"seed": s, "M": 64}))
        jobs.append((f"attn:{s}", "attn", {"seed": s, "M": 64}))
    for s in range(8):
        jobs.append((f"tfma:{s}", "tfma", {"seed": s}))
    for s in range(5):
        for M in (128, 256):
            jobs.append((f"ima:{s}:{M}", "ima", {"seed": s, "M": M}))
    # heavy: VMA candidate-restricted - K=256 seed0 across ALL prompts (breadth, ~375 positions)
    for pi in range(len(PROMPTS)):
        jobs.append((f"vma:{pi}:256:0", "vma", {"pi": pi, "K": 256, "seed": 0, "cap": 20}))
    # heavy: ISA ladder - 500-step across 6 prompts (breadth), 3000-step on 3 (climb), layer sweep
    for pi in range(6):
        jobs.append((f"isa:{pi}:2:500:0", "isa", {"pi": pi, "layer": 2, "steps": 500, "seed": 0}))
    for pi in range(3):
        jobs.append((f"isa:{pi}:2:3000:0", "isa", {"pi": pi, "layer": 2, "steps": 3000, "seed": 0}))
    for layer in (1, 4, 8):
        jobs.append((f"isa:0:{layer}:3000:0", "isa", {"pi": 0, "layer": layer, "steps": 3000, "seed": 0}))
    # heavy: VMA hardness (K=1024 on 5 prompts)
    for pi in range(5):
        jobs.append((f"vma:{pi}:1024:0", "vma", {"pi": pi, "K": 1024, "seed": 0, "cap": 16}))
    return jobs


# ---------------------------------------------------------------------------
def load_model(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded model in {time.time()-t0:.1f}s (cpu fp32, vocab={model.config.vocab_size})")
    return model, tok


class LlamaHandle:
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


# ---- per-job runners ------------------------------------------------------
def job_nn(model, tok, E_np, p):
    ids = tok(PROMPTS[p["pi"]], return_tensors="pt")["input_ids"][0]
    content = ids[1:]
    truth = content.numpy().tolist()
    with torch.inference_mode():
        hs = model(ids.unsqueeze(0), output_hidden_states=True, use_cache=False).hidden_states
    V = E_np.shape[0]
    per_layer = {}
    for l in LAYERS:
        H = hs[l][0, 1:].numpy()
        res = run_knn(
            build_context(GenericScheme("s", frozenset({Capability.EMBEDDINGS})),
                          AttackSample(tokenizer_id="l", model_id="l", seq_len=len(truth), vocab_size=V,
                                       obfuscated_embeddings=H, reference_embeddings=E_np,
                                       plaintext_token_ids=content.numpy())),
            "cosine", 10)
        per_layer[l] = {"hits1": int(sum(truth[i] in res.candidates[i][:1] for i in range(len(truth)))),
                        "hits5": int(sum(truth[i] in res.candidates[i][:5] for i in range(len(truth)))),
                        "n": len(truth)}
    return {"n_positions": len(truth), "per_layer": per_layer}


def job_ednn(model, tok, E_np, p):
    rng = np.random.default_rng(p["seed"]); V, d = E_np.shape; M = p["M"]
    sids = rng.choice(V, size=M, replace=False); Es = E_np[sids]; ones = np.ones(d)
    obf = np.empty_like(Es)
    for i in range(M):
        a = float(rng.uniform(0.1, 1.0)); b = float(rng.uniform(0.0, 1.0)); e = Es[i]
        obf[i] = e - 2.0 * (e @ (ones * a)) / ((ones * a) @ (ones * a)) * (ones * a) + ones * b
    sample = AttackSample(tokenizer_id="l", model_id="l", seq_len=M, vocab_size=V,
                          obfuscated_embeddings=obf, reference_embeddings=E_np, plaintext_token_ids=sids.astype(int))
    m = compute_metrics(EDNN(top_k=10).run(build_context(GenericScheme("e", frozenset({Capability.EMBEDDINGS})), sample)), sample)["token_recovery"]
    return {"asr1": m["asr@1"], "ttrsr": m["ttrsr"], "M": M}


def job_gate(model, tok, E_np, p):
    rng = np.random.default_rng(p["seed"]); V, d = E_np.shape; M = p["M"]; L = 4
    sids = rng.choice(V, size=M, replace=False); Es = E_np[sids]
    W = np.stack([model.model.layers[l].mlp.gate_proj.weight.detach().numpy().T for l in range(L)], 0)
    perm = rng.permutation(M); E_obf = Es[perm]; W_obf = np.empty_like(W)
    for l in range(L):
        W_obf[l] = W[l][:, rng.permutation(W.shape[2])]
    sample = AttackSample(tokenizer_id="l", model_id="l", seq_len=M, vocab_size=M,
                          known_weights={"E": Es, "E_obf": E_obf, "W_gate": W, "W_gate_obf": W_obf},
                          meta={"true_mapping": {int(i): int(perm[i]) for i in range(M)}})
    m = compute_metrics(GateIA(top_k=10).run(build_context(GenericScheme("g", frozenset({Capability.WHITE_BOX_WEIGHTS})), sample)), sample).get("mapping_recovery", {})
    return {"mapping_top1": m.get("mapping_top1_accuracy"), "M": M}


def job_attn(model, tok, E_np, p):
    rng = np.random.default_rng(p["seed"]); V, d = E_np.shape; M = p["M"]
    hd = model.config.head_dim; H = 8
    sids = rng.choice(V, size=M, replace=False); Es = E_np[sids]
    qw = model.model.layers[0].self_attn.q_proj.weight.detach().numpy()
    Q = (Es @ qw.T)[:, : H * hd].reshape(M, H, hd).transpose(1, 0, 2)
    half = hd // 2; perm = rng.permutation(M); Q_obf = np.empty_like(Q)
    for hh in range(H):
        Qh = Q[hh].copy()
        for j in range(half):
            A = rng.standard_normal((2, 2))
            while abs(np.linalg.det(A)) < 0.3:
                A = rng.standard_normal((2, 2))
            Qh[:, [j, j + half]] = Qh[:, [j, j + half]] @ A
        Q_obf[hh] = Qh[perm]
    sample = AttackSample(tokenizer_id="l", model_id="l", seq_len=M, vocab_size=M,
                          known_weights={"Q": Q, "Q_obf": Q_obf},
                          meta={"true_mapping": {int(i): int(perm[i]) for i in range(M)}})
    m = compute_metrics(AttnIA(block_size=2, pairing="half_split").run(build_context(GenericScheme("a", frozenset({Capability.WHITE_BOX_WEIGHTS})), sample)), sample).get("mapping_recovery", {})
    return {"mapping_top1": m.get("mapping_top1_accuracy"), "M": M}


def job_tfma(model, tok, E_np, p, model_dir):
    rng = np.random.default_rng(p["seed"])
    txt = (Path(model_dir) / "README.md")
    text = txt.read_text(encoding="utf-8", errors="ignore") if txt.exists() else " ".join(PROMPTS) * 50
    ids = [t for t in tok(text, return_tensors="pt")["input_ids"][0].tolist() if t < model.config.vocab_size]
    half = len(ids) // 2; prior, plain = ids[:half], ids[half:]
    V = model.config.vocab_size
    perm = rng.permutation(V); cipher = perm[np.asarray(plain)]; inv = np.empty(V, dtype=int); inv[perm] = np.arange(V)
    observed = sorted(set(int(c) for c in cipher))
    sample = AttackSample(tokenizer_id="l", model_id="l", seq_len=len(cipher), vocab_size=V,
                          ciphertext_token_ids=cipher, reference_token_counts={int(t): int(c) for t, c in Counter(prior).items()},
                          meta={"true_mapping": {int(c): int(inv[c]) for c in observed}})
    m = compute_metrics(IndependentTFMA(top_k=100).run(build_context(GenericScheme("t", frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION})), sample)), sample, ks=(1, 10, 100))["mapping_recovery"]
    return {"top1": m.get("mapping_top1_accuracy"), "asr10": m.get("mapping_asr@10"), "asr100": m.get("mapping_asr@100"),
            "n_symbols": m.get("num_observed_symbols"), "chance": 1.0 / max(m.get("num_observed_symbols", 1), 1)}


def job_ima(model, tok, E_np, p):
    rng = np.random.default_rng(p["seed"]); M = p["M"]; N = 6
    vids = rng.choice(E_np.shape[0], size=M, replace=False); E_sub = E_np[vids].astype(np.float64)
    A = rng.standard_normal((E_sub.shape[1], E_sub.shape[1]))
    mech = AloePriToyMechanism(E_sub, alpha_e=0.0, A=A, seed=p["seed"])
    ck = mech.sample_key(np.random.default_rng(p["seed"] + 777))
    corpus = [rng.integers(0, M, size=N) for _ in range(60)] + [np.arange(M)]
    held = np.resize(np.arange(10), N); ht = tuple(int(x) for x in held)
    corpus = [s for s in corpus if tuple(int(x) for x in s) != ht]
    obf = mech.obfuscate_embeddings(held, ck)

    def run(fn):
        sample = AttackSample(tokenizer_id="l", model_id="l", seq_len=N, vocab_size=M,
                              obfuscated_embeddings=obf, reference_embeddings=E_sub, reference_corpus=corpus,
                              plaintext_token_ids=held)
        r = ClassifyInverter(knowledge="GB", epochs=150, lr=2e-2, hidden_dims=(128,), seed=0).run(
            build_context(GenericScheme("i", frozenset({Capability.EMBEDDINGS, Capability.KNOWN_OBFUSCATION_ALGORITHM}), obfuscation_fn=fn), sample))
        return float((r.recovered_token_ids == held).mean())

    unk = run(attacker_oracle(mech, attacker_known_key=False, seed=p["seed"]))
    kn = run(attacker_oracle(mech, client_key=ck, attacker_known_key=True, seed=p["seed"]))
    return {"M": M, "unknown_key_asr1": unk, "known_key_asr1": kn, "chance": 1.0 / M}


def job_vma(model, tok, E_np, p):
    rng = np.random.default_rng(100 + p["seed"]); K = p["K"]
    ids = tok(PROMPTS[p["pi"]], return_tensors="pt")["input_ids"][0][: 1 + p["cap"]]
    full = ids.tolist(); n = len(full); V = model.config.vocab_size
    with torch.inference_mode():
        hs = model(ids.unsqueeze(0), output_hidden_states=True, use_cache=False).hidden_states
    victim = {l: hs[l][0].numpy() for l in VMA_LAYERS}
    content = full[1:]
    per_layer = {l: 0 for l in VMA_LAYERS}
    rec_main = []
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
            out = model(batch, output_hidden_states=True, use_cache=False).hidden_states
        for l in VMA_LAYERS:
            g = out[l][:, -1, :].numpy(); h_i = victim[l][i]
            pred = cand[int(np.argmin([_l1(g[k], h_i) for k in range(len(cand))]))]
            per_layer[l] += int(pred == true_tok)
            if l == 8:
                rec_main.append(pred)
    return {"pi": p["pi"], "K": K, "seed": p["seed"], "n_positions": n - 1,
            "per_layer_hits": {str(l): per_layer[l] for l in VMA_LAYERS},
            "ttrsr_main_l8": ttrsr(rec_main, content),
            "exact_main_l8": bool(rec_main == content)}


def job_isa(model, tok, E_np, p):
    handle = LlamaHandle(model)
    content = tok(PROMPTS[p["pi"]], return_tensors="pt")["input_ids"][0][1: 1 + 5]
    with torch.inference_mode():
        target = handle.forward_from_embeddings(handle.embedding_matrix()[content], p["layer"]).numpy()
    sample = AttackSample(tokenizer_id="l", model_id="l", seq_len=int(content.numel()), vocab_size=E_np.shape[0],
                          layer=p["layer"], obfuscated_hidden_states=target, reference_embeddings=E_np,
                          plaintext_token_ids=content.numpy())
    r = ERAttack(distance="l2", lr=0.1, steps=p["steps"], top_k=5, seed=p["seed"]).run(
        build_context(GenericScheme("isa", frozenset({Capability.HIDDEN_STATES})), sample, attacker_model=handle))
    m = compute_metrics(r, sample)["token_recovery"]
    return {"pi": p["pi"], "layer": p["layer"], "steps": p["steps"],
            "ttrsr": m["ttrsr"], "asr5": m["asr@5"], "state_distance": r.diagnostics["final_state_distance"]}


RUNNERS = {"nn": job_nn, "ednn": job_ednn, "gate": job_gate, "attn": job_attn,
           "ima": job_ima, "vma": job_vma, "isa": job_isa}


def run_battery(model_dir, only, budget):
    jobs = gen_jobs()
    if only:
        jobs = [j for j in jobs if j[1] in only]
    have = done_ids()
    pending = [j for j in jobs if j[0] not in have]
    log(f"battery: {len(jobs)} jobs in scope, {len(have & {j[0] for j in jobs})} done, {len(pending)} pending (budget={budget}s)")
    if not pending:
        log("ALL DONE")
        return
    model, tok = load_model(model_dir)
    E_np = model.get_input_embeddings().weight.detach().numpy()
    t_start = time.time()

    def est(jt, pr):
        if jt == "vma":
            return 820 if pr["K"] > 256 else 280
        if jt == "isa":
            return 1150 if pr["steps"] > 500 else 230
        return 80

    for jid, jtype, params in pending:
        if budget > 0 and (time.time() - t_start) + est(jtype, params) > budget:
            log(f"budget guard: next job {jid} (est {est(jtype, params)}s) would exceed {budget}s; deferring (re-run to continue)")
            return
        t0 = time.time()
        try:
            if jtype == "tfma":
                rec = job_tfma(model, tok, E_np, params, model_dir)
            else:
                rec = RUNNERS[jtype](model, tok, E_np, params)
            rec.update({"job_id": jid, "type": jtype, "seconds": round(time.time() - t0, 1)})
            append(rec)
            log(f"done {jid} ({time.time()-t0:.0f}s): " + json.dumps({k: v for k, v in rec.items() if k not in ('per_layer', 'per_layer_hits', 'job_id', 'type')})[:160])
        except Exception as exc:
            import traceback
            log(f"FAILED {jid}: {exc}")
            traceback.print_exc()
    log("ALL DONE")


# ---------------------------------------------------------------------------
def aggregate():
    recs = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()] if JSONL.exists() else []
    by = defaultdict(list)
    for r in recs:
        by[r["type"]].append(r)
    S = {}

    def mstd(xs):
        xs = [x for x in xs if x is not None]
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "min": float(np.min(xs)), "max": float(np.max(xs)), "n": len(xs)} if xs else None

    # NN layer sweep (micro-avg over positions)
    if by["nn"]:
        agg = {}
        for l in LAYERS:
            h1 = sum(r["per_layer"][str(l)]["hits1"] if str(l) in r["per_layer"] else r["per_layer"][l]["hits1"] for r in by["nn"])
            h5 = sum(r["per_layer"][str(l)]["hits5"] if str(l) in r["per_layer"] else r["per_layer"][l]["hits5"] for r in by["nn"])
            npos = sum(r["per_layer"][str(l)]["n"] if str(l) in r["per_layer"] else r["per_layer"][l]["n"] for r in by["nn"])
            agg[l] = {"asr1": h1 / npos, "asr5": h5 / npos, "n_positions": npos}
        S["nn_layer_sweep"] = {"prompts": len(by["nn"]), "total_positions": agg[LAYERS[0]]["n_positions"], "by_layer": agg}
    if by["ednn"]:
        S["ednn"] = {"asr1": mstd([r["asr1"] for r in by["ednn"]]), "trials": len(by["ednn"]), "M": by["ednn"][0]["M"]}
    if by["gate"]:
        S["gate"] = {"mapping_top1": mstd([r["mapping_top1"] for r in by["gate"]]), "trials": len(by["gate"])}
    if by["attn"]:
        S["attn"] = {"mapping_top1": mstd([r["mapping_top1"] for r in by["attn"]]), "trials": len(by["attn"])}
    if by["tfma"]:
        S["tfma"] = {"top1": mstd([r["top1"] for r in by["tfma"]]), "asr10": mstd([r["asr10"] for r in by["tfma"]]),
                     "asr100": mstd([r["asr100"] for r in by["tfma"]]), "chance": by["tfma"][0]["chance"],
                     "n_symbols": by["tfma"][0]["n_symbols"], "trials": len(by["tfma"])}
    if by["ima"]:
        for M in sorted(set(r["M"] for r in by["ima"])):
            sub = [r for r in by["ima"] if r["M"] == M]
            S[f"ima_M{M}"] = {"unknown_key_asr1": mstd([r["unknown_key_asr1"] for r in sub]),
                              "known_key_asr1": mstd([r["known_key_asr1"] for r in sub]),
                              "chance": 1.0 / M, "trials": len(sub)}
    if by["vma"]:
        for K in sorted(set(r["K"] for r in by["vma"])):
            sub = [r for r in by["vma"] if r["K"] == K]
            layer_asr = {}
            for l in VMA_LAYERS:
                hits = sum(r["per_layer_hits"][str(l)] for r in sub)
                npos = sum(r["n_positions"] for r in sub)
                layer_asr[l] = hits / npos
            S[f"vma_K{K}"] = {"jobs": len(sub), "total_positions": sum(r["n_positions"] for r in sub),
                              "layer_asr1": layer_asr,
                              "ttrsr_l8": mstd([r["ttrsr_main_l8"] for r in sub]),
                              "exact_sentence_rate_l8": float(np.mean([r["exact_main_l8"] for r in sub]))}
    if by["isa"]:
        for key in sorted(set((r["layer"], r["steps"]) for r in by["isa"])):
            sub = [r for r in by["isa"] if (r["layer"], r["steps"]) == key]
            S[f"isa_L{key[0]}_s{key[1]}"] = {"ttrsr": mstd([r["ttrsr"] for r in sub]), "asr5": mstd([r["asr5"] for r in sub]),
                                             "state_distance": mstd([r["state_distance"] for r in sub]), "trials": len(sub)}
    (OUTDIR / "battery_summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    print(json.dumps(S, indent=2))
    log(f"aggregated {len(recs)} job records -> battery_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--only", default="")
    ap.add_argument("--budget", type=int, default=1500, help="seconds; 0 = run until all pending done")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    only = set(s.strip() for s in args.only.split(",") if s.strip())
    run_battery(args.model, only, args.budget)


if __name__ == "__main__":
    raise SystemExit(main())
