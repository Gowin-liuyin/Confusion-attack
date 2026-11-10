"""Closed-loop IMA variants (InvBERT Seq2Seq / AloePri Qwen2 inverter) on real embeddings.

Same keyed threat model as the Classify run: the attacker knows the algorithm and
trains with keys it samples itself; the client's permutation is held out. These
two inverters are sequence models (BB decoder / Qwen2-2-layer-8-head backbone),
so they are run on a smaller working vocabulary and pair budget than Classify -
they train on CPU here - and both the unknown-key and known-key settings are
reported so the key, not capacity, is the variable under test.

Paths:
* ``ima.seq2seq``        - InvBERT Seq2Seq (BB), greedy decode until EOS.
* ``ima.qwen2_aloepri``  - AloePri Qwen2 inversion model, both the
  embedding-regression (paper-experiment) and token-index output paths.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import torch

from rm_common import OUTDIR, PROMPTS, ROOT, append, load_done, log, status_value

RESULTS = OUTDIR / "ima_variants.jsonl"
SEQ_LEN = 16


def failure_record(jid: str, attack_name: str, qwen_path: str, known_key: bool,
                   M: int, seed: int, epochs: int, status, reason,
                   seconds: float | None = None) -> dict:
    """A failed unit that keeps its variant/output-path/key/seed identity."""
    rec = {
        "job_id": jid,
        "family": "ima",
        "variant": attack_name,
        "output_path": qwen_path if attack_name == "qwen2" else "greedy_bb",
        "seed": seed,
        "working_vocab": M,
        "attacker_known_key": known_key,
        "epochs": epochs,
        "status": status_value(status),
        "status_reason": reason,
        "probe_only": True,
    }
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 1)
    return rec


def read_corpus_text() -> str:
    parts = []
    for root in ("docs", "src", "scripts"):
        for p in (ROOT / root).rglob("*"):
            if p.suffix in {".md", ".py"} and p.is_file():
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    parts.extend(PROMPTS)
    return "\n".join(parts)


def setup(model_dir: str, M: int, seed: int):
    from transformers import AutoTokenizer

    from obfuscation_attacks.schemes.mechanism import AloePriToyMechanism

    tok = AutoTokenizer.from_pretrained(model_dir)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    E = model.get_input_embeddings().weight.detach().numpy().copy()
    V = int(E.shape[0])
    del model

    text = read_corpus_text()
    ids = [int(t) for t in tok(text, return_tensors="pt")["input_ids"][0].tolist() if t < V]
    vocab_ids = [t for t, _ in Counter(ids).most_common(M)]
    local = {t: i for i, t in enumerate(vocab_ids)}
    E_sub = E[vocab_ids].astype(np.float64)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((E_sub.shape[1], E_sub.shape[1]))
    mech = AloePriToyMechanism(E_sub, alpha_e=0.0, A=A, seed=seed)
    client_key = mech.sample_key(np.random.default_rng(seed + 777))
    stream = [local[t] for t in ids if t in local]
    return tok, mech, client_key, E_sub, vocab_ids, stream


def run_case(model_dir: str, attack_name: str, M: int, n_seqs: int, epochs: int,
             known_key: bool, seed: int, qwen_path: str = "embedding") -> dict:
    from obfuscation_attacks.attacks.ima.qwen2_aloepri import Qwen2AloePriInverter
    from obfuscation_attacks.attacks.ima.seq2seq import Seq2SeqInverter
    from obfuscation_attacks.data.contracts import AttackSample, Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme
    from obfuscation_attacks.schemes.mechanism import attacker_oracle

    tok, mech, client_key, E_sub, vocab_ids, stream = setup(model_dir, M, seed)
    jid = f"ima_{attack_name}:{qwen_path if attack_name == 'qwen2' else 'bb'}:" \
          f"{'known' if known_key else 'unknown'}:{seed}"
    if n_seqs < 1 or (n_seqs + 1) * SEQ_LEN > len(stream):
        return failure_record(jid, attack_name, qwen_path, known_key, M, seed,
                              epochs, "failed", "insufficient_data: training budget leaves no independent held-out sequence")
    n_seq = n_seqs
    corpus = [np.asarray(stream[i * SEQ_LEN:(i + 1) * SEQ_LEN], dtype=int)
              for i in range(n_seq)]
    held = np.asarray(stream[-SEQ_LEN:], dtype=int)
    obf = mech.obfuscate_embeddings(held, client_key)
    oracle = attacker_oracle(mech, client_key=client_key,
                             attacker_known_key=known_key, seed=seed)
    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=int(held.size),
        vocab_size=M, obfuscated_embeddings=obf, reference_embeddings=E_sub,
        reference_corpus=corpus, plaintext_token_ids=held)
    common = dict(seed=seed, epochs=epochs, lr=1e-3, batch_size=128, top_k=5)
    if attack_name == "seq2seq":
        attack = Seq2SeqInverter(**common)
    else:
        attack = Qwen2AloePriInverter(objective=qwen_path, **common)
    ctx = build_context(
        GenericScheme("ima", frozenset({Capability.EMBEDDINGS,
                                       Capability.KNOWN_OBFUSCATION_ALGORITHM}),
                      obfuscation_fn=oracle), sample)
    t0 = time.time()
    res = attack.run(ctx)
    dt = time.time() - t0
    if res.recovered_token_ids is None:
        return failure_record(jid, attack_name, qwen_path, known_key, M, seed,
                              epochs, res.status, res.status_reason, dt)
    rec = [int(x) for x in res.recovered_token_ids.tolist()]
    from obfuscation_attacks.metrics import symbol_error_rate
    truth = [int(x) for x in held.tolist()]
    # The BB decoder emits its own length, so per-position ASR only applies when
    # the output aligns with the truth; SER is always defined.
    ser = symbol_error_rate(rec, truth)
    met = (compute_metrics(res, sample)["token_recovery"] if len(rec) == len(truth)
           else {"asr@1": None, "asr@5": None, "ttrsr": None})
    return {
        "job_id": jid, "family": "ima", "variant": attack_name,
        "output_path": qwen_path if attack_name == "qwen2" else "greedy_bb",
        "seed": seed, "working_vocab": M, "n_train_seqs": n_seq,
        "attacker_known_key": known_key, "epochs": epochs,
        "asr1": met["asr@1"], "asr5": met["asr@5"], "ttrsr": met["ttrsr"],
        "ser": ser, "length_mismatch": len(rec) != len(truth),
        "chance": 1.0 / M,
        "recovered_length": res.diagnostics.get("recovered_length"),
        "train_final_loss": res.diagnostics.get("train_final_loss"),
        "original": tok.decode([vocab_ids[i] for i in truth]),
        "recovered": tok.decode([vocab_ids[i] for i in rec if 0 <= i < len(vocab_ids)]),
        "seconds": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--train-seqs", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    done = load_done(RESULTS, required_fields=(
        "variant", "output_path", "seed", "working_vocab",
        "attacker_known_key", "ttrsr"))
    plan = [("seq2seq", "bb"), ("qwen2", "embedding"), ("qwen2", "token")]
    for s in seeds:
        for name, path in plan:
            for known in (False, True):
                jid = f"ima_{name}:{path if name == 'qwen2' else 'bb'}:" \
                      f"{'known' if known else 'unknown'}:{s}"
                if jid in done:
                    continue
                rec = run_case(args.model, name, args.vocab, args.train_seqs,
                               args.epochs, known, s, qwen_path=path)
                append(RESULTS, rec)
                log(f"{jid}: asr1={rec.get('asr1')} ttrsr={rec.get('ttrsr')} "
                    f"({rec.get('seconds')}s) {rec.get('reason', '')}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
