"""Closed-loop SDA-2023 (recurrence-encoded causal LM) on real English text.

Separate lineage from SDA-2021: the 2023 method encodes the cipher with
recurrence codes and trains a causal LM with ``L_CLM = -log P(X) - log P(Y|X)``
(source recurrence AND target), decoding with length-matched beam search. Both
the 1:1 and the homophonic cipher condition are run on the same real text.

Resource note recorded in every result: 8 GB GPU probe with a d128/2-layer
recurrence CLM (lr 5e-4, batch 16); the actual training budget is in each
record, not the paper scale. Records are marked ``probe_only``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time

import numpy as np

from rm_common import OUTDIR, PROMPTS, ROOT, append, load_done, log, status_value, sda_probe_identity

RESULTS = OUTDIR / "sda2023_full.jsonl"
CIPHER_LEN = 256
METHOD = "sda2023"
#: Per-job training checkpoints. At num_train >= 50,000 ``fit_sda`` switches to
#: disk pair storage and *requires* a checkpoint/window-manifest path for
#: provenance (it refuses otherwise); a crashed multi-hour job must also resume
#: instead of restarting from epoch 0 on every queue retry.
def checkpoint_path(jid: str):
    """Checkpoint file (and implicit window manifest) for one job id.

    ``OUTDIR`` is resolved at call time so tests (and callers) can redirect the
    artifact root after import.
    """
    directory = OUTDIR / "sda2023_checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{jid.replace(':', '__')}.pt"

#: Stamped on every record so a probe payload cannot pass as paper-v1 evidence.
PROBE_FIELDS = {"profile": "probe", "probe_only": True}
#: Resource caveat recorded per result instead of only in prose.
RESOURCE_NOTE = (
    "8 GB GPU probe: d128/2-layer recurrence CLM, lr 5e-4, batch 16; the actual "
    "training budget is in num_train_pairs/epochs, not the paper-scale run."
)

#: SDA2023 emits ``final_train_loss_{total,src,tgt}`` (sda2023_recurrence_clm.py);
#: older revisions used the shorter names, kept here as a fallback.
_LOSS_KEYS = (
    ("total", ("final_train_loss_total", "final_total_loss")),
    ("src", ("final_train_loss_src", "final_src_loss")),
    ("tgt", ("final_train_loss_tgt", "final_tgt_loss")),
)


def loss_diagnostics(res) -> dict:
    """Training losses from an SDA2023 result, under stable short names.

    The 2023 attack's ``-log P(X) - log P(Y|X)`` evidence lives in its
    diagnostics; reading the wrong keys silently stored nulls. A successful run
    must yield a finite total loss, otherwise ``missing_loss_reason`` records why
    the evidence is absent instead of writing an unexplained null.
    """
    diag = getattr(res, "diagnostics", None)
    if diag is None:
        diag = res if isinstance(res, dict) else {}
    losses = {}
    for name, keys in _LOSS_KEYS:
        value = next((diag[k] for k in keys if k in diag), None)
        try:
            losses[name] = None if value is None else float(value)
        except (TypeError, ValueError):
            losses[name] = None
    total = losses["total"]
    if total is None:
        reason = ("SDA2023 result carries no final_train_loss_total "
                  "(and no legacy final_total_loss)")
    elif not math.isfinite(total):
        reason = f"SDA2023 final_train_loss_total is not finite: {total!r}"
    else:
        reason = None
    return {"losses": losses, "missing_loss_reason": reason}


def has_loss_evidence(rec: dict) -> bool:
    """A completed 2023 record must carry finite total loss or an explicit reason.

    ``required_fields`` can only check that ``losses`` exists; an old row whose
    diagnostics carried no ``final_train_loss_total`` stored ``total=None`` and
    would otherwise be resumed as done without regenerating the loss evidence.
    """
    losses = rec.get("losses")
    total = losses.get("total") if isinstance(losses, dict) else None
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        try:
            if math.isfinite(float(total)):
                return True
        except (TypeError, ValueError):
            pass
    return bool(rec.get("missing_loss_reason"))


def corpus_hash(text: str) -> str:
    """Content hash of the live corpus (repo text + prompts)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def implementation_identity(args, chash: str, seed: int, homophonic: bool = False) -> dict:
    config = {"method": METHOD, "seed": seed, "homophonic": homophonic,
              "num_train": args.num_train, "epochs": args.epochs, "beam": args.beam,
              "cipher_len": CIPHER_LEN, "model": str(getattr(args, "model", ""))}
    return sda_probe_identity(__file__, config, chash)


def job_id(args, chash: str, seed: int, homophonic: bool) -> str:
    """Job identity carries every config field that changes the result.

    The corpus is live repo text, so its content hash is part of the identity:
    a rerun after the corpus changed is a new job, not a stale success.
    """
    cond = "homo" if homophonic else "oneToOne"
    return (f"{METHOD}:{cond}:{args.num_train}:{args.epochs}:{args.beam}:"
            f"{chash[:16]}:{seed}:" + ":".join(implementation_identity(args, chash, seed, homophonic)[k][:16]
            for k in ("code_sha256", "config_sha256", "protocol_sha256", "environment_sha256")))


def build_cipher(plain: np.ndarray, A: int, seed: int, homophonic: bool):
    """Substitution key and ciphertext for a held-out plaintext.

    A homophonic key needs the plaintext type/frequency counts for Algorithm 1
    (``make_substitution_key`` raises without them) and a cipher alphabet large
    enough for all allocated homophones; the 1:1 condition stays an
    ``A``-symbol permutation.
    """
    from obfuscation_attacks.attacks.sda import encodings as enc

    counts = {int(t): int(c) for t, c in zip(*np.unique(plain, return_counts=True))}
    key = enc.make_substitution_key(
        counts if homophonic else range(A),
        rng=np.random.default_rng(seed),
        homophonic=homophonic,
        cipher_alphabet_size=4 * A if homophonic else A,
        plain_counts=counts if homophonic else None,
    )
    cipher = enc.apply_substitution(plain, key, rng=np.random.default_rng(seed + 1))
    return key, cipher


def failure_record(seed: int, homophonic: bool, chash: str, status, reason,
                   args, seconds: float | None = None) -> dict:
    """A failed unit that keeps its config/corpus identity for the denominator."""
    rec = {
        "job_id": job_id(args, chash, seed, homophonic),
        "family": "sda",
        "method": METHOD,
        "variant": "recurrence_lookup_probe",
        "seed": seed,
        "cipher_type": "homophonic" if homophonic else "1:1",
        "num_train_pairs": args.num_train,
        "epochs": args.epochs,
        "beam_width": args.beam,
        "corpus_sha256": chash,
        "implementation_identity": implementation_identity(args, chash, seed, homophonic),
        "resource_note": RESOURCE_NOTE,
        "status": status_value(status),
        "status_reason": reason,
        **PROBE_FIELDS,
    }
    if seconds is not None:
        rec["seconds"] = round(float(seconds), 1)
    return rec


def success_record(args, chash: str, seed: int, homophonic: bool, *,
                   alphabet_size: int, cipher_len: int, meter: dict,
                   original: str, recovered: str, loss_info: dict,
                   seconds: float) -> dict:
    """A completed unit; carries the probe marker and its resource note."""
    return {
        "job_id": job_id(args, chash, seed, homophonic),
        "family": "sda", "method": METHOD, "variant": "recurrence_lookup_probe",
        "seed": seed, "cipher_type": "homophonic" if homophonic else "1:1",
        "alphabet": alphabet_size, "cipher_len": int(cipher_len),
        "num_train_pairs": args.num_train, "epochs": args.epochs,
        "beam_width": args.beam, "architecture": "d128/2/ff256",
        "corpus_sha256": chash,
        "implementation_identity": implementation_identity(args, chash, seed, homophonic),
        "asr1": meter["asr@1"], "ttrsr": meter["ttrsr"],
        "ser": meter.get("symbol_error_rate"),
        "original": original, "recovered": recovered,
        "losses": loss_info["losses"],
        "missing_loss": loss_info["missing_loss_reason"] is not None,
        "missing_loss_reason": loss_info["missing_loss_reason"],
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


def run(seed: int, homophonic: bool, args, text: str, chash: str) -> dict:
    from obfuscation_attacks.attacks.sda.sda2023_recurrence_clm import SDA2023
    from obfuscation_attacks.data.contracts import AttackSample, Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    alphabet = sorted(set(text))
    c2i = {c: i for i, c in enumerate(alphabet)}
    A = len(alphabet)
    split = int(len(text) * 0.8)
    train_ids = np.asarray([c2i[c] for c in text[:split]], dtype=int)
    chunk = 512
    corpus = [train_ids[i:i + chunk] for i in range(0, len(train_ids) - 16, chunk // 2)]
    plain_chars = [c for c in text[split:] if c in c2i][:CIPHER_LEN]
    plain = np.asarray([c2i[c] for c in plain_chars], dtype=int)

    _, cipher = build_cipher(plain, A, seed, homophonic)

    sample = AttackSample(
        tokenizer_id="char", model_id="english-char", seq_len=int(plain.size),
        vocab_size=A, ciphertext_token_ids=cipher, reference_corpus=corpus,
        plaintext_token_ids=plain, plaintext="".join(plain_chars),
        text_source="real", data_split="eval")
    ctx = build_context(
        GenericScheme("sda", frozenset({Capability.TOKEN_CIPHERTEXT,
                                       Capability.DETERMINISTIC_PERMUTATION})), sample)
    attack = SDA2023(
        epochs=args.epochs, num_train=args.num_train, min_len=16, max_len=256,
        homophonic=homophonic, d_model=128, nhead=4, num_layers=2, dim_ff=256,
        dropout=0.1, lr=5e-4, batch_size=16, beam_width=args.beam, seed=seed,
        device="cuda",
        checkpoint_path=str(checkpoint_path(job_id(args, chash, seed, homophonic))),
        resume=checkpoint_path(job_id(args, chash, seed, homophonic)).exists())
    t0 = time.time()
    res = attack.run(ctx)
    dt = time.time() - t0
    if res.recovered_token_ids is None:
        return failure_record(seed, homophonic, chash, res.status,
                              res.status_reason, args, dt)
    met = compute_metrics(res, sample)["token_recovery"]
    rec = "".join(alphabet[int(i)] for i in res.recovered_token_ids.tolist())
    return success_record(
        args, chash, seed, homophonic, alphabet_size=A,
        cipher_len=int(plain.size), meter=met, original="".join(plain_chars),
        recovered=rec, loss_info=loss_diagnostics(res), seconds=dt,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--num-train", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--beam", type=int, default=10)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    text = read_corpus()
    chash = corpus_hash(text)
    done = load_done(RESULTS, required_fields=(
        "variant", "seed", "cipher_type", "num_train_pairs", "epochs",
        "beam_width", "corpus_sha256", "ttrsr"), predicate=has_loss_evidence)
    for s in seeds:
        for homo in (False, True):
            jid = job_id(args, chash, s, homo)
            if jid in done:
                continue
            rec = run(s, homo, args, text, chash)
            append(RESULTS, rec)
            log(f"{jid}: ser={rec.get('ser')} asr1={rec.get('asr1')} "
                f"ttrsr={rec.get('ttrsr')} ({rec.get('seconds')}s)")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
