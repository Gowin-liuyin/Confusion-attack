"""Closed-loop SDA-2021 on real English text (character-level).

The paper's task: public plaintext enciphered with a fresh random 1:1
substitution key; the attacker learns freq-rank(cipher) -> plaintext on its own
public corpus and deciphers a held-out cipher of the same language. This script
uses real English/code text from the repository, a character alphabet built from
that corpus, and a 256-character held-out cipher (the paper's length where
decipherment is strongest).

Resource note recorded in every result: the paper trains on 2M synthesised pairs
for 20 epochs on a V100 (~110 min/epoch). On this 8 GB RTX 4060 the run uses
100k pairs and 8 epochs with a smaller transformer; the objective, injective
frequency-rank encoding, length-matched beam search and 1:1 key are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np

from rm_common import OUTDIR, PROMPTS, ROOT, append, load_done, log, status_value, sda_probe_identity

RESULTS = OUTDIR / "sda_full.jsonl"
CIPHER_LEN = 256
METHOD = "sda2021"
#: Per-job training checkpoints. At num_train >= 50,000 ``fit_sda`` switches to
#: disk pair storage and *requires* a checkpoint/window-manifest path for
#: provenance (it refuses otherwise), and a crashed multi-hour job must resume
#: instead of restarting from epoch 0 on every queue retry.
def checkpoint_path(jid: str):
    """Checkpoint file (and implicit window manifest) for one job id.

    ``OUTDIR`` is resolved at call time so tests (and callers) can redirect the
    artifact root after import.
    """
    directory = OUTDIR / "sda_checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{jid.replace(':', '__')}.pt"

#: Stamped on every record so a probe payload cannot pass as paper-v1 evidence.
PROBE_FIELDS = {"profile": "probe", "probe_only": True}
#: The docstring's resource caveat, recorded per result instead of only in prose.
RESOURCE_NOTE = (
    "8 GB RTX 4060 probe: the paper trains 2M synthesised pairs for 20 epochs on "
    "a V100; the actual budget is in num_train_pairs/epochs. Objective, injective "
    "frequency-rank encoding, length-matched beam search and 1:1 key unchanged."
)


def corpus_hash(text: str) -> str:
    """Content hash of the live corpus (repo text + prompts)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def implementation_identity(args, chash: str, seed: int, homophonic: bool = False) -> dict:
    config = {"method": METHOD, "seed": seed, "homophonic": homophonic,
              "num_train": args.num_train, "epochs": args.epochs, "beam": args.beam,
              "cipher_len": CIPHER_LEN, "model": str(getattr(args, "model", ""))}
    return sda_probe_identity(__file__, config, chash)


def job_id(args, chash: str, seed: int) -> str:
    """Job identity carries every config field that changes the result.

    The corpus is live repo text, so its content hash is part of the identity:
    a rerun after the corpus changed is a new job, not a stale success.
    """
    return (f"{METHOD}:{args.num_train}:{args.epochs}:{args.beam}:"
            f"{chash[:16]}:{seed}:" + ":".join(implementation_identity(args, chash, seed)[k][:16]
            for k in ("code_sha256", "config_sha256", "protocol_sha256", "environment_sha256")))


def failure_record(seed: int, chash: str, status, reason, args,
                   seconds: float | None = None) -> dict:
    """A failed unit that keeps its config/corpus identity for the denominator."""
    rec = {
        "job_id": job_id(args, chash, seed),
        "family": "sda",
        "method": METHOD,
        "variant": "frequency_rank_lookup_probe",
        "seed": seed,
        "num_train_pairs": args.num_train,
        "epochs": args.epochs,
        "beam_width": args.beam,
        "corpus_sha256": chash,
        "implementation_identity": implementation_identity(args, chash, seed),
        "resource_note": RESOURCE_NOTE,
        "status": status_value(status),
        "status_reason": reason,
        **PROBE_FIELDS,
    }
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 1)
    return rec


def success_record(args, chash: str, seed: int, *, alphabet_size: int,
                   cipher_len: int, meter: dict, original: str,
                   recovered: str, diagnostics: dict, seconds: float) -> dict:
    """A completed unit; carries the probe marker and its resource note."""
    return {
        "job_id": job_id(args, chash, seed), "family": "sda",
        "method": METHOD, "variant": "frequency_rank_lookup_probe",
        "seed": seed, "alphabet": alphabet_size, "cipher_len": int(cipher_len),
        "num_train_pairs": args.num_train, "epochs": args.epochs,
        "beam_width": args.beam, "architecture": "d128/2+2/ff256",
        "corpus_sha256": chash,
        "implementation_identity": implementation_identity(args, chash, seed),
        "asr1": meter["asr@1"], "ttrsr": meter["ttrsr"],
        "ser": meter.get("symbol_error_rate"), "bleu4": meter.get("bleu4"),
        "original": original, "recovered": recovered,
        "final_train_loss": diagnostics.get("final_train_loss"),
        "resource_note": RESOURCE_NOTE,
        "seconds": round(float(seconds), 1),
        **PROBE_FIELDS,
    }


def read_corpus() -> str:
    parts = []
    for root in ("README.md", "docs", "src"):
        p = ROOT / root
        if p.is_file():
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            continue
        for q in p.rglob("*"):
            if q.suffix in {".md", ".py"} and q.is_file():
                parts.append(q.read_text(encoding="utf-8", errors="ignore"))
    parts.extend(PROMPTS)
    return "\n".join(parts)


def run(model_dir: str, seed: int, args, text: str, chash: str) -> dict:
    from obfuscation_attacks.attacks.sda import encodings as enc
    from obfuscation_attacks.attacks.sda.sda2021_seq2seq import SDA2021
    from obfuscation_attacks.data.contracts import AttackSample, Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    alphabet = sorted(set(text))
    c2i = {c: i for i, c in enumerate(alphabet)}
    A = len(alphabet)
    split = int(len(text) * 0.8)
    train_text, test_text = text[:split], text[split:]
    train_ids = np.asarray([c2i[c] for c in train_text], dtype=int)
    # reference corpus: overlapping chunks give the pair sampler real contexts
    chunk = 512
    corpus = [train_ids[i:i + chunk] for i in range(0, len(train_ids) - 16, chunk // 2)]
    plain_chars = [c for c in test_text if c in c2i][:CIPHER_LEN]
    plain = np.asarray([c2i[c] for c in plain_chars], dtype=int)
    rng = np.random.default_rng(seed)
    key = enc.make_substitution_key(list(range(A)), rng=rng, homophonic=False,
                                    cipher_alphabet_size=A)
    cipher = enc.apply_substitution(plain, key)

    sample = AttackSample(
        tokenizer_id="char", model_id="english-char", seq_len=int(plain.size),
        vocab_size=A, ciphertext_token_ids=cipher,
        reference_corpus=corpus, plaintext_token_ids=plain,
        plaintext="".join(plain_chars), text_source="real", data_split="eval",
    )
    ctx = build_context(
        GenericScheme("sda", frozenset({Capability.TOKEN_CIPHERTEXT,
                                        Capability.DETERMINISTIC_PERMUTATION})), sample)
    attack = SDA2021(
        epochs=args.epochs, num_train=args.num_train, min_len=16, max_len=256,
        d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1,
        lr=1e-3, batch_size=32, beam_width=args.beam, seed=seed, device="cuda",
        checkpoint_path=str(checkpoint_path(job_id(args, chash, seed))),
        resume=checkpoint_path(job_id(args, chash, seed)).exists(),
    )
    t0 = time.time()
    res = attack.run(ctx)
    dt = time.time() - t0
    if res.recovered_token_ids is None:
        return failure_record(seed, chash, res.status, res.status_reason, args, dt)
    met = compute_metrics(res, sample)["token_recovery"]
    rec_chars = "".join(alphabet[int(i)] for i in res.recovered_token_ids.tolist())
    return success_record(
        args, chash, seed, alphabet_size=A, cipher_len=int(plain.size), meter=met,
        original="".join(plain_chars), recovered=rec_chars,
        diagnostics=res.diagnostics, seconds=dt,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--num-train", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--beam", type=int, default=10)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    text = read_corpus()
    chash = corpus_hash(text)
    done = load_done(RESULTS, required_fields=(
        "variant", "seed", "num_train_pairs", "epochs", "beam_width",
        "corpus_sha256", "ttrsr"))
    for s in seeds:
        jid = job_id(args, chash, s)
        if jid in done:
            continue
        rec = run(args.model, s, args, text, chash)
        append(RESULTS, rec)
        log(f"{jid}: ser={rec.get('ser')} asr1={rec.get('asr1')} "
            f"ttrsr={rec.get('ttrsr')} ({rec.get('seconds')}s)")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
