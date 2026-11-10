"""Closed-loop IMA (InvBERT Classify) on the plaintext Llama-3.2-1B.

Faithful keyed threat model: the obfuscation algorithm is public, the client's
token permutation is secret. The attacker synthesises training pairs with keys
it samples itself (``attacker_known_key=False``) and evaluates on the held-out
client key; a white-box control (``attacker_known_key=True``) shows the same
classifier can learn when the key is known.

Scale: working vocabularies of 4,096 and 16,384 REAL Llama tokens (the most
frequent tokens of the public corpus, embeddings taken from the real 2048-d
table). A 128,256-way classifier would need millions of training pairs; that is
a corpus/compute limit of this machine, not an algorithmic shortcut, and it is
recorded in every result. Training pairs come from real text sequences.

Artifacts: ``reports/closed_loop/ima_full.jsonl``.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import torch

from rm_common import OUTDIR, PROMPTS, ROOT, append, load_done, log, status_value

RESULTS = OUTDIR / "ima_full.jsonl"
SEQ_LEN = 16


def failure_record(M: int, known_key: bool, seed: int, status, reason,
                   seconds: float | None = None) -> dict:
    """A failed unit that keeps its (variant, vocab, key-setting, seed) identity."""
    rec = {
        "job_id": f"ima:{M}:{'known' if known_key else 'unknown'}:{seed}",
        "family": "ima",
        "variant": "invbert_classify_keyed",
        "working_vocab": M,
        "seq_len": SEQ_LEN,
        "seed": seed,
        "attacker_known_key": known_key,
        "knowledge": "GB",
        "status": status_value(status),
        "status_reason": reason,
        "probe_only": True,
    }
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 1)
    return rec


def read_corpus_text() -> str:
    """All public English/code text in the repo (~0.5 MB) plus the prompts."""
    parts = []
    for root in ("src", "docs", "scripts", "configs"):
        for p in (ROOT / root).rglob("*"):
            if p.suffix in {".py", ".md", ".yaml", ".yml"} and p.is_file():
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    parts.extend(PROMPTS)
    return "\n".join(parts)


def run_case(model_dir: str, M: int, n_train: int, known_key: bool, seed: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from obfuscation_attacks.attacks.ima.classify import ClassifyInverter
    from obfuscation_attacks.data.contracts import (
        AttackSample, Capability,
    )
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme
    from obfuscation_attacks.schemes.mechanism import AloePriToyMechanism, attacker_oracle

    t0 = time.time()
    # Extract the embedding table on CPU: the GPU is busy with the ISA 5e4-step
    # runs, and the IMA experiment only needs the table plus its own classifier.
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    E_full = model.get_input_embeddings().weight.detach().numpy().copy()
    del model

    text = read_corpus_text()
    ids = [int(t) for t in tok(text, return_tensors="pt")["input_ids"][0].tolist()
           if t < E_full.shape[0]]
    freq = Counter(ids)
    vocab_ids = [t for t, _ in freq.most_common(M)]
    if len(vocab_ids) < M:
        return failure_record(
            M, known_key, seed, "failed",
            f"corpus has only {len(vocab_ids)} distinct tokens < M={M}",
            seconds=time.time() - t0,
        )
    local = {t: i for i, t in enumerate(vocab_ids)}
    E_sub = E_full[vocab_ids].astype(np.float64)
    del E_full

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((E_sub.shape[1], E_sub.shape[1]))
    mech = AloePriToyMechanism(E_sub, alpha_e=0.0, A=A, seed=seed)
    client_key = mech.sample_key(np.random.default_rng(seed + 777))

    stream = [local[t] for t in ids if t in local]
    if n_train < 1 or (n_train + 1) * SEQ_LEN > len(stream):
        return failure_record(M, known_key, seed, "failed", "insufficient_data: training budget leaves no independent held-out sequence", seconds=time.time() - t0)
    n_seq = n_train
    corpus = [np.asarray(stream[i * SEQ_LEN:(i + 1) * SEQ_LEN], dtype=int)
              for i in range(n_seq)]
    # Held-out evaluation sequence: real text after the training slice.
    held = np.asarray(stream[-SEQ_LEN:], dtype=int)
    obf = mech.obfuscate_embeddings(held, client_key)

    oracle = attacker_oracle(mech, client_key=client_key,
                             attacker_known_key=known_key, seed=seed)
    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=int(held.size),
        vocab_size=M, obfuscated_embeddings=obf, reference_embeddings=E_sub,
        reference_corpus=corpus, plaintext_token_ids=held,
    )
    attack = ClassifyInverter(
        knowledge="GB", epochs=120, lr=2e-2, hidden_dims=(256,),
        batch_size=512, seed=seed, device="cuda", top_k=5,
    )
    ctx = build_context(
        GenericScheme("ima", frozenset({Capability.EMBEDDINGS,
                                        Capability.KNOWN_OBFUSCATION_ALGORITHM}),
                      obfuscation_fn=oracle), sample)
    t0 = time.time()
    res = attack.run(ctx)
    dt = time.time() - t0
    met = compute_metrics(res, sample)["token_recovery"]
    rec_ids = [int(x) for x in res.recovered_token_ids.tolist()]
    return {
        "job_id": f"ima:{M}:{'known' if known_key else 'unknown'}:{seed}",
        "seed": seed,
        "family": "ima", "variant": "invbert_classify_keyed",
        "working_vocab": M, "seq_len": SEQ_LEN, "n_train_seqs": n_seq,
        "n_train_pairs": int(n_seq * SEQ_LEN),
        "attacker_known_key": known_key,
        "knowledge": "GB",
        "asr1": met["asr@1"], "asr5": met["asr@5"], "ttrsr": met["ttrsr"],
        "chance": 1.0 / M,
        "train_accuracy": res.diagnostics.get("train_accuracy"),
        "train_final_loss": res.diagnostics.get("train_final_loss"),
        "original": tok.decode([vocab_ids[i] for i in held]),
        "recovered": tok.decode([vocab_ids[i] for i in rec_ids]),
        "seconds": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--sizes", default="4096,16384")
    ap.add_argument("--train-seqs", type=int, default=12000)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    # Existing successful rows predate the ``seed`` field (it is already part of
    # ``job_id``); require the grouping identity and a result instead.
    done = load_done(RESULTS, required_fields=(
        "variant", "working_vocab", "attacker_known_key", "ttrsr"))
    for M in sizes:
        for known in (False, True):
            for s in seeds:
                jid = f"ima:{M}:{'known' if known else 'unknown'}:{s}"
                if jid in done:
                    continue
                rec = run_case(args.model, M, args.train_seqs, known, s)
                append(RESULTS, rec)
                log(f"{jid}: asr1={rec.get('asr1')} asr5={rec.get('asr5')} "
                    f"chance={rec.get('chance'):.2e} train_acc={rec.get('train_accuracy')} "
                    f"({rec.get('seconds')}s)")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
