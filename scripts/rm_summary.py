"""Index legacy closed-loop JSONL records without merging incomparable runs.

Legacy job IDs do not reliably encode model, data, layer and budget. Every
readable record, including retries, remains a separate row; this diagnostic
index is not a protocol-compliant aggregate or a formal result table.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

from rm_common import OUTDIR, load_records, mean_std

METRICS = ("asr1", "asr5", "ttrsr", "mapping_top1", "mapping_top10",
           "mapping_asr10", "mapping_asr100", "z1_top1", "z1_top10",
           "mean_true_rank", "ser", "bleu4", "chance", "train_accuracy")
FILES = ["vma_full.jsonl", "vma_osnip_full.jsonl", "vma_weight_full.jsonl",
         "ia_full.jsonl", "isa_full.jsonl", "ima_full.jsonl", "ima_variants.jsonl",
         "nn_full.jsonl", "tfma_full.jsonl", "sda_full.jsonl", "sda2023_full.jsonl"]


def load(name: str, *, stats: dict | None = None) -> list[dict]:
    """Read one artifact with the same tolerant JSONL parsing as the resume path."""
    return load_records(OUTDIR / name, stats=stats)


def key(r: dict) -> str:
    parts = [str(r.get("variant") or r.get("level") or "?")]
    if r.get("metric"):
        parts.append(r["metric"])
    if r.get("layer") is not None and r.get("family") in ("isa", "nn_knn"):
        parts.append(f"L{r['layer']}")
    if r.get("attacker_known_key") is not None:
        parts.append("known-key" if r["attacker_known_key"] else "secret-key")
    if r.get("alpha_e") is not None:
        parts.append(f"alpha_e={r['alpha_e']}")
    if r.get("cipher_type"):
        parts.append(r["cipher_type"])
    if r.get("family") == "sda":
        # The training config and corpus hash are part of the job identity:
        # different budgets/corpora must be separate rows, not one mean.
        if r.get("num_train_pairs") is not None:
            parts.append(f"n_train={r['num_train_pairs']}")
        if r.get("epochs") is not None:
            parts.append(f"epochs={r['epochs']}")
        if r.get("beam_width") is not None:
            parts.append(f"beam={r['beam_width']}")
        if r.get("corpus_sha256"):
            parts.append(f"corpus={str(r['corpus_sha256'])[:16]}")
    if r.get("family") == "tfma":
        # Same rule for TFMA: the model/tokenizer, query cap, top-k and corpus
        # hash are part of the job identity, so old and new rows are not averaged.
        if r.get("model"):
            parts.append(f"model={r['model']}")
        if r.get("max_global_symbols") is not None:
            parts.append(f"max_symbols={r['max_global_symbols']}")
        if r.get("top_k") is not None:
            parts.append(f"top_k={r['top_k']}")
        if r.get("cap_used") is not None:
            parts.append(f"cap={r['cap_used']}")
        if r.get("memory_budget_bytes") is not None:
            parts.append(f"budget={r['memory_budget_bytes']}")
        if r.get("corpus_sha256"):
            parts.append(f"corpus={str(r['corpus_sha256'])[:16]}")
    if r.get("level") and r.get("family") == "nn_knn":
        parts[0] = r["level"]
    return " / ".join(parts)


def _md_cell(value) -> str:
    """Render one markdown-table cell (lists such as non_ok_status are flattened)."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)
    return str(value)


def main() -> int:
    summary: dict = {}
    read_stats: dict[str, dict] = {}
    lines = ["# Closed-loop legacy record index", "",
             "Each readable record is shown separately; no cross-run mean or formal acceptance is inferred.", ""]
    for f in FILES:
        stats: dict = {}
        recs = load(f, stats=stats)
        if stats:
            read_stats[f] = stats
        if not recs:
            continue
        groups: dict[str, list[dict]] = defaultdict(list)
        for index, r in enumerate(recs, start=1):
            # Record ordinal is intentionally unique even when legacy job IDs
            # collide or a retry omits identity fields.
            groups[f"{key(r)} / record={index}"].append((index, r))
        block = {}
        for g, entries in sorted(groups.items()):
            index, record = entries[0]
            rs = [record]
            row = {"n_jobs": 1, "source_record_index": index,
                   "source_job_id": record.get("job_id"),
                   "aggregation": "none; legacy record identity is incomplete"}
            for field in ("model", "model_path", "layer", "steps", "epochs",
                          "num_train_pairs", "beam_width", "n_positions",
                          "numerator", "denominator", "corpus_sha256"):
                if field in record:
                    row[field] = record[field]
            for m in METRICS:
                vals = [r.get(m) for r in rs if r.get("status") in (None, "ok")
                        and isinstance(r.get(m), (int, float)) and not isinstance(r.get(m), bool)
                        and math.isfinite(r[m])]
                if vals:
                    ms = mean_std(vals)
                    row[m] = {"mean": ms["mean"], "std": ms["std"], "n": ms["n"]}
            for m in ("verified", "exact", "z1_exact"):
                if any(m in r for r in rs):
                    row[m] = sum(bool(r.get(m)) for r in rs)
            if any(r.get("status") not in (None, "ok") for r in rs):
                row["non_ok_status"] = [r["status"] for r in rs
                                         if r.get("status") not in (None, "ok")]
            block[g] = row
        summary[f] = block
        lines.append(f"## {f.replace('.jsonl','')}")
        lines.append("")
        if stats:
            lines.append(f"Malformed lines skipped: {stats.get('skipped_lines', 0)}"
                         f" (decode errors: {stats.get('decode_errors', 0)})")
            lines.append("")
        cols = [m for m in METRICS if any(m in v for v in block.values())]
        # Every non-metric row key (verified / exact / z1_exact / non_ok_status)
        # gets a header cell too, so each row has the same number of columns.
        extra_cols: list[str] = []
        for v in block.values():
            for k in v:
                if k != "n_jobs" and k not in METRICS and k not in extra_cols:
                    extra_cols.append(k)
        header_cells = ["setting", "jobs"] + cols + extra_cols
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * len(header_cells))
        for g, v in block.items():
            cells = [(f"{v[c]['mean']:.4f}" + (f"±{v[c]['std']:.4f}" if v[c]["std"] else ""))
                     if c in v else "-" for c in cols]
            extra = [_md_cell(v[c]) if c in v else "-" for c in extra_cols]
            row_cells = [str(g), str(v["n_jobs"])] + cells + extra
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")
    summary["_read_stats"] = read_stats
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
