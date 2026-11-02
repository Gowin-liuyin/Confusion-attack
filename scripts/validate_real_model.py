"""Validate the attack implementations against a REAL (plaintext) LLM.

Runs the repository's own attack classes against Llama-3.2-1B-Instruct loaded
from disk, so the numbers come from the real implementation on real weights /
hidden states / tokenizer frequencies - not synthetic toys.

Honesty notes baked in:
* CPU-only (torch+cpu, no CUDA here) and vocab = 128,256, so full-vocab VMA
  replay is infeasible; the VMA/OSNIP section is CANDIDATE-RESTRICTED (K per
  position) and labelled as such - it measures the attack's L1 matching rule on
  real hidden states, not a full-vocab paper run.
* ISA-ER uses a small step budget (paper uses ~5e4); its number is a
  budget-limited lower bound, labelled.
* NN full-vocab, IA on real weights, and TFMA on real token frequencies are
  exact / unrestricted.

Usage:  python scripts/validate_real_model.py [--model F:\\Models\\Llama-3.2-1B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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
from obfuscation_attacks.evaluation.scoring import compute_metrics
from obfuscation_attacks.attacks.nn_knn.euclidean_knn import EuclideanKNN
from obfuscation_attacks.attacks.nn_knn.cosine_knn import CosineKNN
from obfuscation_attacks.attacks.nn_knn.core import run_knn
from obfuscation_attacks.attacks.ia.ednn import EDNN
from obfuscation_attacks.attacks.ia.gate import GateIA
from obfuscation_attacks.attacks.ia.attn import AttnIA
from obfuscation_attacks.attacks.tfma.frequency_matching import IndependentTFMA
from obfuscation_attacks.attacks.vma.autoregressive import _l1
from obfuscation_attacks.attacks.isa.er import ERAttack
from obfuscation_attacks.metrics import asr_at_k, ttrsr

RESULTS: dict = {}
PROMPT = "The quick brown fox jumps over the lazy dog near the river bank at dawn."


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
def load_model(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load] {time.time()-t0:.1f}s  layers={model.config.num_hidden_layers} "
          f"hidden={model.config.hidden_size} vocab={model.config.vocab_size} device=cpu dtype=fp32")
    return model, tok


# ---------------------------------------------------------------------------
# 1 + 2.  NN: embedding inversion (full vocab) and hidden-state layer sweep
# ---------------------------------------------------------------------------
def section_nn(model, tok, E_np):
    print("\n=== NN / KNN (real embeddings, FULL 128,256 vocab) ===")
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0]
    content = ids[1:]  # drop BOS
    V = E_np.shape[0]

    # (1) embedding inversion: observe h = E[token] exactly, recover by NN.
    h = E_np[content.numpy()]
    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=int(content.numel()),
        vocab_size=V, obfuscated_embeddings=h, reference_embeddings=E_np,
        plaintext_token_ids=content.numpy(), text_source="real", data_split="eval",
    )
    scheme = GenericScheme("real_nn", frozenset({Capability.EMBEDDINGS}))
    ctx = build_context(scheme, sample)
    for atk, label in ((EuclideanKNN(top_k=10), "euclidean"), (CosineKNN(top_k=10), "cosine")):
        m = compute_metrics(atk.run(ctx), sample)["token_recovery"]
        print(f"  embedding->token [{label:9s}]  asr@1={m['asr@1']:.3f}  asr@5={m['asr@5']:.3f}  ttrsr={m['ttrsr']:.3f}")
        RESULTS[f"nn_embedding_{label}"] = {k: m[k] for k in ("asr@1", "asr@5", "ttrsr")}

    # (2) hidden-state -> token NN per layer (how recoverable is the token id?)
    with torch.inference_mode():
        hs = model(ids.unsqueeze(0), output_hidden_states=True, use_cache=False).hidden_states
    truth = content.numpy().tolist()
    layers = [0, 1, 2, 4, 8, 12, 16]
    print("  hidden-state -> token (cosine NN vs embedding table), per layer:")
    sweep = {}
    for l in layers:
        H = hs[l][0, 1:].numpy()  # content positions at layer l
        res = run_knn(
            build_context(
                GenericScheme("s", frozenset({Capability.EMBEDDINGS})),
                AttackSample(tokenizer_id="l", model_id="l", seq_len=len(truth), vocab_size=V,
                             obfuscated_embeddings=H, reference_embeddings=E_np,
                             plaintext_token_ids=content.numpy()),
            ),
            "cosine", 10,
        )
        cand = res.candidates
        a1 = asr_at_k(cand, truth, 1)
        a5 = asr_at_k(cand, truth, 5)
        print(f"    layer {l:2d}: asr@1={a1:.3f}  asr@5={a5:.3f}")
        sweep[l] = {"asr@1": a1, "asr@5": a5}
    RESULTS["nn_hidden_layer_sweep"] = sweep


# ---------------------------------------------------------------------------
# 3.  VMA autoregressive replay (candidate-restricted) + OSNIP layer sweep
# ---------------------------------------------------------------------------
def section_vma(model, tok, *, K=96, max_content=8, main_layer=8, seed=0):
    print(f"\n=== VMA autoregressive replay (real model; CANDIDATE-RESTRICTED K={K}, CPU) ===")
    rng = _rng(seed)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0]
    ids = ids[: 1 + max_content]  # BOS + up to max_content content tokens
    full = ids.tolist()
    n = len(full)
    V = model.config.vocab_size
    layers = [1, 2, 4, 8, 12, 16]

    with torch.inference_mode():
        hs_true = model(ids.unsqueeze(0), output_hidden_states=True, use_cache=False).hidden_states
    # victim states per layer: (n, d)
    victim = {l: hs_true[l][0].numpy() for l in layers}

    content_tokens = full[1:]
    per_layer_hits = {l: 0 for l in layers}
    recovered_main = []
    total_positions = 0
    t0 = time.time()
    for i in range(1, n):  # recover content positions 1..n-1
        true_tok = full[i]
        # candidate pool: true token + other content tokens + random distractors
        cand = {true_tok, *content_tokens}
        while len(cand) < K:
            cand.add(int(rng.integers(0, V)))
        cand = list(cand)[:K]
        if true_tok not in cand:
            cand[0] = true_tok
        prefix = full[:i]  # teacher-forced true prefix
        batch = torch.tensor([prefix + [c] for c in cand], dtype=torch.long)  # (K, i+1)
        with torch.inference_mode():
            out = model(batch, output_hidden_states=True, use_cache=False).hidden_states
        total_positions += 1
        for l in layers:
            g = out[l][:, -1, :].numpy()  # (K, d) replayed state at position i
            h_i = victim[l][i]
            dists = [_l1(g[k], h_i) for k in range(len(cand))]  # the attack's L1 metric
            pred = cand[int(np.argmin(dists))]
            if pred == true_tok:
                per_layer_hits[l] += 1
            if l == main_layer:
                recovered_main.append(pred)
    dt = time.time() - t0
    print(f"  replayed {total_positions} positions x {K} candidates at layers {layers} in {dt:.1f}s")
    sweep = {}
    for l in layers:
        acc = per_layer_hits[l] / max(total_positions, 1)
        print(f"    layer {l:2d}: token L1-recovery asr@1 = {acc:.3f}")
        sweep[l] = acc
    tt = ttrsr(recovered_main, content_tokens)
    rec_text = tok.decode(recovered_main)
    orig_text = tok.decode(content_tokens)
    print(f"  [layer {main_layer}] TTRSR={tt:.3f}")
    print(f"    original : {orig_text!r}")
    print(f"    recovered: {rec_text!r}")
    RESULTS["vma_replay"] = {
        "candidate_pool_K": K, "positions": total_positions, "layer_sweep_asr1": sweep,
        "main_layer": main_layer, "ttrsr": tt, "original": orig_text, "recovered": rec_text,
        "note": "candidate-restricted (CPU, 128k vocab); L1 matching rule from the real attack",
    }


# ---------------------------------------------------------------------------
# 4.  IA on REAL Llama weights (EDNN / Gate-IA / Attn-IA)
# ---------------------------------------------------------------------------
def section_ia(model, tok, E_np, *, M=64, seed=0):
    print("\n=== IA on REAL Llama weights (EDNN / Gate-IA / Attn-IA) ===")
    rng = _rng(seed)
    V, d = E_np.shape
    sample_ids = rng.choice(V, size=M, replace=False)
    Es = E_np[sample_ids]  # (M, d) real embeddings

    # ---- EDNN: glide reflection on real embeddings, recover vs FULL vocab -----
    ones = np.ones(d)
    obf = np.empty_like(Es)
    for i in range(M):
        a = float(rng.uniform(0.1, 1.0)); b = float(rng.uniform(0.0, 1.0))
        e = Es[i]
        obf[i] = e - 2.0 * (e @ (ones * a)) / ((ones * a) @ (ones * a)) * (ones * a) + ones * b
    sample = AttackSample(
        tokenizer_id="l", model_id="l", seq_len=M, vocab_size=V,
        obfuscated_embeddings=obf, reference_embeddings=E_np,
        plaintext_token_ids=sample_ids.astype(int), text_source="real", data_split="eval",
    )
    ctx = build_context(GenericScheme("ednn", frozenset({Capability.EMBEDDINGS})), sample)
    m = compute_metrics(EDNN(top_k=10).run(ctx), sample)["token_recovery"]
    print(f"  EDNN  (glide on real E, FULL-vocab reference)   asr@1={m['asr@1']:.3f}  ttrsr={m['ttrsr']:.3f}")
    RESULTS["ia_ednn_real"] = {k: m[k] for k in ("asr@1", "ttrsr")}

    # ---- Gate-IA: real gate_proj weights, permutation + Zffn (alpha_e=0) ------
    n_gate_layers = 4
    W_gate = np.stack([
        model.model.layers[l].mlp.gate_proj.weight.detach().numpy().T  # (d, hf)
        for l in range(n_gate_layers)
    ], axis=0)  # (L, d, hf)
    perm = rng.permutation(M)
    E_obf = Es[perm]
    W_gate_obf = np.empty_like(W_gate)
    for l in range(n_gate_layers):
        zffn = rng.permutation(W_gate.shape[2])
        W_gate_obf[l] = W_gate[l][:, zffn]
    gsample = AttackSample(
        tokenizer_id="l", model_id="l", seq_len=M, vocab_size=M,
        known_weights={"E": Es, "E_obf": E_obf, "W_gate": W_gate, "W_gate_obf": W_gate_obf},
        secret_permutation=perm, text_source="real", data_split="eval",
        meta={"true_mapping": {int(i): int(perm[i]) for i in range(M)}},
    )
    gctx = build_context(GenericScheme("gate", frozenset({Capability.WHITE_BOX_WEIGHTS})), gsample)
    gm = compute_metrics(GateIA(top_k=10).run(gctx), gsample).get("mapping_recovery", {})
    print(f"  Gate-IA (real gate_proj, {n_gate_layers} layers, alpha_e=0)   "
          f"mapping_top1={gm.get('mapping_top1_accuracy')}")
    RESULTS["ia_gate_real"] = gm

    # ---- Attn-IA: real q_proj, per head, RoPE half_split blocks ---------------
    head_dim = model.config.head_dim
    n_heads = model.config.num_attention_heads
    qw = model.model.layers[0].self_attn.q_proj.weight.detach().numpy()  # (n_heads*head_dim, d)
    Q_full = Es @ qw.T  # (M, n_heads*head_dim)
    use_heads = 8
    Q = Q_full[:, : use_heads * head_dim].reshape(M, use_heads, head_dim).transpose(1, 0, 2)  # (H, M, hd)
    half = head_dim // 2
    perm_a = rng.permutation(M)
    Q_obf = np.empty_like(Q)
    for hh in range(use_heads):
        Qh = Q[hh].copy()
        for j in range(half):  # RoPE rotate_half pairs (j, j+half)
            A = rng.standard_normal((2, 2))
            while abs(np.linalg.det(A)) < 0.3:
                A = rng.standard_normal((2, 2))
            Qh[:, [j, j + half]] = Qh[:, [j, j + half]] @ A
        Q_obf[hh] = Qh[perm_a]
    asample = AttackSample(
        tokenizer_id="l", model_id="l", seq_len=M, vocab_size=M,
        known_weights={"Q": Q, "Q_obf": Q_obf}, secret_permutation=perm_a,
        text_source="real", data_split="eval",
        meta={"true_mapping": {int(i): int(perm_a[i]) for i in range(M)}},
    )
    actx = build_context(GenericScheme("attn", frozenset({Capability.WHITE_BOX_WEIGHTS})), asample)
    am = compute_metrics(AttnIA(block_size=2, pairing="half_split").run(actx), asample).get("mapping_recovery", {})
    print(f"  Attn-IA (real q_proj, {use_heads} heads, half_split, alpha_e=0)   "
          f"mapping_top1={am.get('mapping_top1_accuracy')}")
    RESULTS["ia_attn_real"] = am


# ---------------------------------------------------------------------------
# 5.  TFMA on REAL tokenized text (frequency prior vs a permutation cipher)
# ---------------------------------------------------------------------------
def section_tfma(model, tok, model_dir, *, seed=0):
    print("\n=== TFMA on REAL tokenized text (Llama tokenizer frequencies) ===")
    rng = _rng(seed)
    readme = Path(model_dir) / "README.md"
    text = readme.read_text(encoding="utf-8", errors="ignore") if readme.exists() else (PROMPT * 200)
    ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()
    ids = [t for t in ids if t < model.config.vocab_size]
    half = len(ids) // 2
    prior, plain = ids[:half], ids[half:]
    V = model.config.vocab_size
    prior_counts = {int(t): int(c) for t, c in Counter(prior).items()}
    perm = rng.permutation(V)
    cipher = perm[np.asarray(plain)]
    inv = np.empty(V, dtype=int); inv[perm] = np.arange(V)
    observed = sorted(set(int(c) for c in cipher))
    true_map = {int(c): int(inv[c]) for c in observed}
    sample = AttackSample(
        tokenizer_id="llama", model_id="l", seq_len=len(cipher), vocab_size=V,
        ciphertext_token_ids=cipher, reference_token_counts=prior_counts,
        secret_permutation=perm, text_source="real", data_split="eval",
        meta={"true_mapping": true_map},
    )
    scheme = GenericScheme("tfma", frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION}))
    ctx = build_context(scheme, sample)
    m = compute_metrics(IndependentTFMA(top_k=100).run(ctx), sample, ks=(1, 10, 100))["mapping_recovery"]
    n_sym = m.get("num_observed_symbols")
    print(f"  tokens={len(ids)}  distinct cipher symbols={n_sym}")
    print(f"  mapping_top1={m.get('mapping_top1_accuracy'):.3f}  asr@10={m.get('mapping_asr@10'):.3f}  "
          f"asr@100={m.get('mapping_asr@100'):.3f}  (chance@1 ~ {1.0/max(n_sym,1):.4f})")
    RESULTS["tfma_real_text"] = m


# ---------------------------------------------------------------------------
# 6.  ISA-ER: gradient inversion of a real hidden state (budget-limited)
# ---------------------------------------------------------------------------
def section_isa(model, tok, E_np, *, layer=2, n_content=5, steps=300, seed=0):
    print(f"\n=== ISA-ER gradient inversion (real model, layer {layer}, {steps} steps - BUDGET-LIMITED) ===")

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
            x = inputs_embeds.unsqueeze(0)
            out = self.model.model(inputs_embeds=x, output_hidden_states=True, use_cache=False)
            l = self.num_layers if layer is None else int(layer)
            return out.hidden_states[l][0]

        def freeze(self):
            for p in self.model.parameters():
                p.requires_grad_(False)
            return self

    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0][1 : 1 + n_content]  # content tokens, NO BOS
    content = ids
    handle = LlamaHandle(model)
    # Fair ER target: the layer-l state of the TRUE token embeddings run through
    # the SAME inputs_embeds path the attack optimizes (positions 0..n-1, no BOS),
    # so the target is reachable and ER is tested on equal footing.
    with torch.inference_mode():
        E_true = handle.embedding_matrix()[content]
        target = handle.forward_from_embeddings(E_true, layer).numpy()

    sample = AttackSample(
        tokenizer_id="l", model_id="l", seq_len=int(content.numel()), vocab_size=E_np.shape[0],
        layer=layer, obfuscated_hidden_states=target, reference_embeddings=E_np,
        plaintext_token_ids=content.numpy(), text_source="real", data_split="eval",
    )
    # Build the context with position offset: forward_from_embeddings runs the
    # dummy as positions 0..n-1 (no BOS), matched to the victim's content states.
    ctx = build_context(
        GenericScheme("isa", frozenset({Capability.HIDDEN_STATES})), sample, attacker_model=handle,
    )
    t0 = time.time()
    res = ERAttack(distance="l2", lr=0.1, steps=steps, top_k=5, seed=seed).run(ctx)
    m = compute_metrics(res, sample)["token_recovery"]
    print(f"  {time.time()-t0:.1f}s  ttrsr={m['ttrsr']:.3f}  asr@5={m['asr@5']:.3f}  "
          f"final_state_distance={res.diagnostics.get('final_state_distance'):.4f}")
    print(f"    original : {tok.decode(content)!r}")
    print(f"    recovered: {tok.decode(res.recovered_token_ids.tolist())!r}")
    RESULTS["isa_er"] = {"layer": layer, "steps": steps, "ttrsr": m["ttrsr"], "asr@5": m["asr@5"],
                          "final_state_distance": res.diagnostics.get("final_state_distance"),
                          "note": "budget-limited (paper uses ~5e4 steps)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--skip", default="", help="comma-separated: nn,vma,ia,tfma,isa")
    args = ap.parse_args()
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    model, tok = load_model(args.model)
    E_np = model.get_input_embeddings().weight.detach().numpy()

    for name, fn in (
        ("nn", lambda: section_nn(model, tok, E_np)),
        ("ia", lambda: section_ia(model, tok, E_np)),
        ("tfma", lambda: section_tfma(model, tok, args.model)),
        ("vma", lambda: section_vma(model, tok)),
        ("isa", lambda: section_isa(model, tok, E_np)),
    ):
        if name in skip:
            continue
        try:
            fn()
        except Exception as exc:  # keep going; report the failure
            import traceback
            print(f"\n[!] section {name} failed: {exc}")
            traceback.print_exc()
            RESULTS[f"{name}_error"] = str(exc)

    out_dir = ROOT / "reports" / "real_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
