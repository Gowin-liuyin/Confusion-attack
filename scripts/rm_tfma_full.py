"""Closed-loop TFMA on real English text with the real Llama tokenizer.

Cipher: a random 1:1 substitution over the model's full 128,256-token
vocabulary applied to held-out public text (deterministic permutation, the
method's precondition). The attacker knows a reference corpus of plaintext
token counts (the other half of the corpus) and the ciphertext frequencies.

Runs both the paper's independent nearest-frequency matching and the separately
named global one-to-one assignment extension (Hungarian). The Hungarian cost
matrix is float64. A conservative working-set estimate gates the full observed
symbol set; if it exceeds the budget the global variant is resource-blocked,
without truncating the evaluation stream.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rm_common import OUTDIR, PROMPTS, ROOT, append, load_done, log

RESULTS = OUTDIR / "tfma_full.jsonl"
TOP_K = 100
#: Budget covers an estimated matrix, temporary arrays and solver overhead.
DEFAULT_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024
MIN_GLOBAL_SYMBOLS = 2
#: Weight-lineage fingerprinting: small model dirs are hashed in full; large
#: ones use each safetensors file's size plus its header/metadata.
SMALL_MODEL_FULL_HASH_BYTES = 64 * 1024 * 1024
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


class ResourceBlockedError(RuntimeError):
    """A run that cannot be executed within the fixed memory budget."""


def bounded_symbol_cap(requested: int, vocab_size: int,
                       budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES) -> dict:
    """Return the maximum full symbol set allowed by an estimated working set."""
    requested = int(requested)
    if requested < 1:
        raise ValueError(f"--max-global-symbols must be >= 1, got {requested}")
    vocab_size = int(vocab_size)
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    budget_cap = max(0, (int(budget_bytes) - 64 * vocab_size) // (vocab_size * 32 + 64))
    used = min(requested, budget_cap)
    reduced = used < requested
    if reduced:
        reason = (
            f"estimated float64 matrix and solver working set limits the Hungarian cap to {budget_cap}"
        )
    else:
        reason = (
            f"requested cap {requested} fits the {int(budget_bytes)}-byte memory "
            f"budget at V={vocab_size}"
        )
    if used < MIN_GLOBAL_SYMBOLS:
        raise ResourceBlockedError(
            "global_assignment resource-blocked: the "
            f"{int(budget_bytes)}-byte memory budget allows only {used} "
            f"observed symbol(s) at V={vocab_size} (requested {requested}, "
            f"minimum {MIN_GLOBAL_SYMBOLS}); raise the budget or reduce V"
        )
    return {
        "cap_requested": requested,
        "cap_used": int(used),
        "cap_reduced": bool(reduced),
        "cap_reason": reason,
        "memory_budget_bytes": int(budget_bytes),
    }


_NO_GLOBAL_CAP = {
    "cap_requested": None,
    "cap_used": None,
    "cap_reduced": False,
    "cap_reason": "not applicable: independent matching uses every observed symbol",
    "memory_budget_bytes": None,
}


def corpus_hash(text: str) -> str:
    """Content hash of the live corpus (repo text + prompts)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _weight_lineage(resolved: Path) -> bytes:
    """Fingerprint the weight files without hashing multi-GB shards.

    Each weight file contributes its name and size plus, for safetensors, the
    8-byte header length and the JSON header/metadata (which records tensor
    names/shapes/dtype). Directories whose total weight size is small are hashed
    in full. This distinguishes different weight files behind the same path.
    """
    if not resolved.is_dir():
        return b""
    files = sorted(
        path for path in resolved.iterdir()
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )
    total_bytes = sum(path.stat().st_size for path in files)
    full_hash = total_bytes <= SMALL_MODEL_FULL_HASH_BYTES
    digest = hashlib.sha256()
    for path in files:
        size = path.stat().st_size
        digest.update(path.name.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        if full_hash:
            digest.update(path.read_bytes())
        elif path.suffix.lower() == ".safetensors":
            with path.open("rb") as handle:
                raw_length = handle.read(8)
                digest.update(raw_length)
                if len(raw_length) == 8:
                    header_length = int.from_bytes(raw_length, "little")
                    if 0 <= header_length <= 100 * 1024 * 1024:
                        digest.update(handle.read(header_length))
    return digest.digest()


def model_identity(model_dir: str, vocab_size: int) -> str:
    """Stable model/tokenizer identity: path, config, weight lineage and vocab.

    A job completed with the 1B tokenizer must never satisfy a 3B rerun on the
    same corpus, and different weight files at the same path must not share an
    identity, so the path, ``config.json``, the weight-file fingerprint and the
    vocabulary size are all hashed into the id.
    """
    path = Path(model_dir).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    digest = hashlib.sha256(str(resolved).casefold().encode("utf-8"))
    config = resolved / "config.json"
    if config.is_file():
        digest.update(config.read_bytes())
    digest.update(_weight_lineage(resolved))
    digest.update(f"V{int(vocab_size)}".encode("ascii"))
    return f"{digest.hexdigest()[:12]}V{int(vocab_size)}"


def job_id(name: str, seed: int, max_global_symbols: int, top_k: int,
           chash: str, model_key: str, cap_used: int | None = None,
           budget_bytes: int | None = None) -> str:
    """Job identity carries model, corpus hash and every config field it depends on.

    For ``global_assignment`` the budget-derived ``cap_used`` and the memory
    budget itself are part of the id: changing the budget changes the truncated
    sample, so old rows must not be resumed as done.
    """
    cap_segment = ("na" if cap_used is None
                   else f"c{int(cap_used)}b{int(budget_bytes)}")
    return (f"tfma_full:{name}:m{model_key}:{int(max_global_symbols)}:"
            f"{int(top_k)}:{chash[:16]}:{cap_segment}:{int(seed)}")


def chance_top1(vocab_size: int) -> float:
    """Random top-1 accuracy over the full reference vocabulary.

    Candidates are ranked against all ``V`` reference symbols under a random
    permutation, so the chance baseline is ``1/V``. The attacked/query symbol
    count (possibly truncated for the Hungarian solve) is not the denominator.
    """
    return 1.0 / int(vocab_size)


def read_corpus() -> str:
    parts = []
    for root in ("docs", "src", "scripts"):
        for p in (ROOT / root).rglob("*"):
            if p.suffix in {".md", ".py"} and p.is_file():
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    parts.extend(PROMPTS)
    return "\n".join(parts)


def _truncate_to_symbol_cap(perm, plain, cap: int, max_iterations: int = 10000):
    """Bounded prefix whose ciphertext symbol set fits the Hungarian cap.

    Returns ``(cut_plain, cipher_g, obs_g)``. ``obs_g`` is the attacked/query
    symbol set (recorded as ``attacked_symbols``); it does not set the top-1
    chance baseline, which is ``1/V`` over the full reference vocabulary.
    """
    cap = int(cap)
    if cap < 1:
        # A non-positive cap can never be reached (one symbol always remains),
        # so the shrink loop below would never terminate.
        raise ValueError(f"symbol cap must be >= 1, got {cap}")
    cut = np.asarray(plain)
    iterations = 0
    while len({int(c) for c in perm[np.asarray(cut)]}) > cap:
        iterations += 1
        if iterations > int(max_iterations):
            raise RuntimeError(
                f"symbol truncation did not fit cap={cap} within "
                f"{int(max_iterations)} iterations"
            )
        cut = cut[: max(1, int(len(cut) * 0.9))]
    cipher_g = perm[np.asarray(cut)]
    obs_g = sorted({int(c) for c in cipher_g})
    return cut, cipher_g, obs_g


def run_seed(tok, seed: int, text: str, chash: str, model_key: str,
             model_path: str, cap_info: dict,
             max_global_symbols: int = 2500,
             done: set[str] | None = None) -> list[dict]:
    from obfuscation_attacks.attacks.tfma.frequency_matching import (
        GlobalAssignmentTFMA, IndependentTFMA,
    )
    from obfuscation_attacks.data.contracts import AttackSample, Capability
    from obfuscation_attacks.evaluation.scoring import compute_metrics
    from obfuscation_attacks.schemes.base import build_context
    from obfuscation_attacks.schemes.generic import GenericScheme

    V = int(tok.vocab_size)
    ids = [int(t) for t in tok(text, return_tensors="pt")["input_ids"][0].tolist()
           if t < V]
    half = len(ids) // 2
    prior, plain = ids[:half], ids[half:]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(V)
    cipher = perm[np.asarray(plain)]
    inv = np.empty(V, dtype=int)
    inv[perm] = np.arange(V)
    observed = sorted({int(c) for c in cipher})
    counts = Counter(prior)
    sample = AttackSample(
        tokenizer_id="llama", model_id="llama-3.2-1b", seq_len=len(cipher),
        vocab_size=V, ciphertext_token_ids=cipher,
        reference_token_counts={int(t): int(c) for t, c in counts.items()},
        meta={"true_mapping": {int(c): int(inv[c]) for c in observed}},
    )
    caps = frozenset({Capability.TOKEN_CIPHERTEXT, Capability.DETERMINISTIC_PERMUTATION})
    out = []
    variants = (("independent", IndependentTFMA(top_k=TOP_K)),
                ("global_assignment", GlobalAssignmentTFMA(top_k=TOP_K,
                    memory_budget_bytes=cap_info["memory_budget_bytes"])))
    for name, attack in variants:
        info = cap_info if name == "global_assignment" else dict(_NO_GLOBAL_CAP)
        jid = job_id(name, seed, max_global_symbols, TOP_K, chash, model_key,
                     cap_used=info["cap_used"],
                     budget_bytes=info["memory_budget_bytes"])
        if done and jid in done:
            continue
        attacked_symbols = observed
        if name == "global_assignment" and len(observed) > info["cap_used"]:
            out.append({"job_id": jid, "family": "tfma", "variant": name,
                        "status": "resource_blocked", "reason": "full observed symbol set exceeds the configured cap or estimated memory budget",
                        "n_symbols": len(observed), "text_tokens": len(cipher), **info})
            continue
        t0 = time.time()
        ctx = build_context(GenericScheme("tfma", caps), sample)
        res = attack.run(ctx)
        met = compute_metrics(res, sample, ks=(1, 10, 100)).get("mapping_recovery", {})
        out.append({
            "job_id": jid, "family": "tfma", "variant": name, "method": name,
            "seed": seed, "status": res.status.value, "status_reason": res.status_reason,
            "model": model_key, "model_path": model_path,
            "n_symbols": met.get("num_observed_symbols"),
            "cipher_tokens_used": len(cipher),
            "vocab": V,
            "corpus_sha256": chash,
            "max_global_symbols": int(max_global_symbols),
            "top_k": TOP_K,
            **info,
            "mapping_top1": met.get("mapping_top1_accuracy"),
            "mapping_asr10": met.get("mapping_asr@10"),
            "mapping_asr100": met.get("mapping_asr@100"),
            "chance": chance_top1(V),
            "attacked_symbols": int(len(attacked_symbols)),
            "text_tokens": len(cipher),
            "seconds": round(time.time() - t0, 1),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"F:\Models\Llama-3.2-1B-Instruct")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-global-symbols", type=int, default=2500,
                    help="requested maximum observed symbols for the Hungarian "
                         "solve; reduced to fit the fixed memory budget (>=1)")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    text = read_corpus()
    chash = corpus_hash(text)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    V = int(tok.vocab_size)
    mkey = model_identity(args.model, V)
    cap_info = bounded_symbol_cap(args.max_global_symbols, V)
    done = load_done(RESULTS, required_fields=(
        "variant", "seed", "model", "corpus_sha256", "max_global_symbols",
        "top_k", "cap_used", "memory_budget_bytes", "mapping_top1"))
    for s in seeds:
        jids = [
            job_id("independent", s, args.max_global_symbols, TOP_K, chash, mkey),
            job_id("global_assignment", s, args.max_global_symbols, TOP_K, chash,
                   mkey, cap_used=cap_info["cap_used"],
                   budget_bytes=cap_info["memory_budget_bytes"]),
        ]
        if all(jid in done for jid in jids):
            continue
        for rec in run_seed(tok, s, text, chash, mkey, str(args.model), cap_info,
                            args.max_global_symbols, done=done):
            append(RESULTS, rec)
            log(f"{rec['job_id']}: {rec.get('status')} top1={rec.get('mapping_top1')} "
                f"asr10={rec.get('mapping_asr10')} n={rec.get('n_symbols')} "
                f"chance={rec.get('chance')} ({rec.get('seconds')}s) "
                f"{rec.get('reason', '')}")
    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
