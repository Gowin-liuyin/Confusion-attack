"""Shared plumbing for the real-model closed-loop experiments.

All runs are on the plaintext Llama-3.2-1B-Instruct model. Artifacts are
resumable JSONL files under ``reports/closed_loop``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:  # Windows consoles default to GBK and crash on recovered non-Latin text
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUTDIR = ROOT / "reports" / "closed_loop"
LOG = OUTDIR / "run.log"

# single source of truth for the 24 prompts used by the existing battery
from real_model_battery import PROMPTS  # noqa: E402,F401

__all__ = ["OUTDIR", "PROMPTS", "append", "is_success_record", "load_done",
           "load_records", "log", "mean_std", "status_value"]


def sda_probe_identity(script: str, config: dict, corpus_sha256: str) -> dict:
    """Separate code, configuration, data, protocol and environment identities."""
    import hashlib
    import platform
    import torch

    code_paths = [Path(script), Path(__file__),
                  ROOT / "src/obfuscation_attacks/evaluation/sda_stream.py",
                  ROOT / "src/obfuscation_attacks/evaluation/sda_pair_store.py",
                  ROOT / "src/obfuscation_attacks/attacks/sda/encodings.py",
                  ROOT / "src/obfuscation_attacks/attacks/sda/_neural.py",
                  ROOT / "src/obfuscation_attacks/attacks/sda/sda2021_seq2seq.py",
                  ROOT / "src/obfuscation_attacks/attacks/sda/sda2023_recurrence_clm.py",
                  ROOT / "src/obfuscation_attacks/data/contracts.py",
                  ROOT / "src/obfuscation_attacks/schemes/base.py",
                  ROOT / "src/obfuscation_attacks/schemes/generic.py",
                  ROOT / "src/obfuscation_attacks/evaluation/scoring.py",
                  ROOT / "src/obfuscation_attacks/metrics/recovery.py",
                  ROOT / "src/obfuscation_attacks/utils/seeding.py"]
    code = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_paths}
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
    env = {"python": platform.python_version(), "torch": str(torch.__version__),
           "numpy": __import__("numpy").__version__, "platform": platform.platform(),
           "cuda": str(torch.version.cuda)}
    protocol = ROOT / "docs/PAPER_EVALUATION_PROTOCOL.md"
    return {"code_sha256": digest(code), "config_sha256": digest(config),
            "data_sha256": corpus_sha256,
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "environment_sha256": digest(env)}


def log(msg: str, *, tag: str = "run") -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_model(model_dir: str, *, dtype: str = "float16", device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"loaded {model_dir} dtype={dtype} device={device} in {time.time()-t0:.1f}s", tag="model")
    return model, tok


#: Keys that describe a job attempt or its configuration but do not prove it
#: produced a result. A record holding only these keys (e.g.
#: ``{"job_id": ..., "status": "ok", "layer": 2, "steps": 50000}``) is a
#: partial write and must be retried on resume.
_NON_RESULT_KEYS = frozenset({
    "job_id", "status", "status_reason", "reason", "seconds", "attempt",
    "profile", "probe_only",
    # job identity / configuration echoed by the legacy writers
    "family", "variant", "method", "prompt_idx", "layer", "steps", "lr",
    "distance", "top_k", "tf32", "n_positions", "seed", "level", "metric",
    "vocab", "attacker_known_key", "alpha_e", "cipher_type", "corpus_sha256",
    "num_train_pairs", "n_train_pairs", "n_train_seqs", "epochs", "beam_width",
    "alphabet", "cipher_len", "architecture", "output_path", "matrix_shape",
    "matrices", "rows", "knowledge", "working_vocab", "seq_len", "chance",
    "max_global_symbols", "attacked_symbols", "chance_symbols",
    "cap_requested", "cap_used", "cap_reduced", "cap_reason",
    "memory_budget_bytes", "model", "model_path",
})


def _status_is_ok(status) -> bool:
    """Accept ``None``, ``"ok"`` and the enum's ``str()`` form (``AttackStatus.OK``)."""
    if status is None:
        return True
    return str(status).rsplit(".", 1)[-1].strip().lower() == "ok"


def status_value(status) -> str:
    """Plain status string (``AttackStatus.FAILED`` -> ``"failed"``)."""
    return str(getattr(status, "value", status))


def is_success_record(rec: dict, *, required_fields=()) -> bool:
    """Return whether *rec* proves its job finished successfully.

    "Finished" is not the same as "a line was appended": the old loader treated
    any record with a ``job_id`` as done, so a failure row (``status=failed``)
    or a partial row without result fields silently suppressed the retry. A
    record counts only when its status is missing/ok **and** it carries result
    data. Callers whose jobs have family-specific result fields pass them as
    *required_fields* so a truncated record is retried instead of resumed.
    """
    if not isinstance(rec, dict) or not rec.get("job_id"):
        return False
    if not _status_is_ok(rec.get("status")):
        return False
    if required_fields:
        return all(field in rec for field in required_fields)
    return bool(set(rec) - _NON_RESULT_KEYS)


def _bump(stats: dict | None, key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def load_records(path: Path, *, stats: dict | None = None) -> list[dict]:
    """Parse a JSONL artifact tolerantly, in file order.

    Malformed lines (e.g. a JSONL append interrupted mid-record) and non-object
    JSON values are skipped rather than raising, and are counted in *stats*
    (``skipped_lines``, ``decode_errors``). Resuming, summarising or auditing a
    log must never abort because its final line was cut off.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Keep the readable records; undecodable bytes become U+FFFD and their
        # line is skipped by the JSON parse below.
        text = path.read_bytes().decode("utf-8", errors="replace")
        _bump(stats, "decode_errors")
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            _bump(stats, "skipped_lines")
            continue
        if not isinstance(rec, dict):
            _bump(stats, "skipped_lines")
            continue
        records.append(rec)
    return records


def load_done(path: Path, *, required_fields=(), stats: dict | None = None,
              predicate=None) -> set[str]:
    """Job ids with at least one successful record in *path*.

    Failed and partial records stay out of the set, so the caller retries them;
    a later success for the same ``job_id`` marks the job done. Malformed lines
    (e.g. a JSONL append interrupted mid-record) are skipped, which leaves that
    job pending instead of aborting the whole resume. Pass *stats* to receive
    counters (``skipped_lines``, ``decode_errors``); *predicate* optionally adds
    a value-level requirement on top of ``required_fields`` (e.g. a finite loss).
    """
    out = set()
    for rec in load_records(path, stats=stats):
        if not is_success_record(rec, required_fields=required_fields):
            continue
        if predicate is not None and not predicate(rec):
            continue
        out.add(rec["job_id"])
    return out


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def mean_std(xs):
    import numpy as np

    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    return {
        "mean": float(np.mean(xs)),
        "std": float(np.std(xs)),
        "min": float(np.min(xs)),
        "max": float(np.max(xs)),
        "n": len(xs),
    }
