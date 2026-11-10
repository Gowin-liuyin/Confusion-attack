"""Compare planned units with result files without certifying the result contents."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obfuscation_attacks.evaluation.paper_matrix import iter_coverage, validate_matrix  # noqa: E402
from obfuscation_attacks.evaluation.public_guard import assert_public_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing comparison")
    if not args.results.is_dir():
        raise ValueError("results directory does not exist")
    validated = validate_matrix(args.matrix)
    paths = sorted(args.results.rglob("result.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    seen: set[str] = set()
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        def write(record):
            assert_public_payload(record, label="matrix comparison")
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            return line

        digest.update(write({"kind": "coverage_header", "matrix": str(args.matrix),
                             "matrix_content_sha256": validated["preceding_lines_sha256"],
                             "scope": "record presence only; all matching results require independent audit"}).encode())
        for row in iter_coverage(args.matrix, paths):
            seen.add(row["plan_spec_id"])
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            digest.update(write({"kind": "coverage", **row}).encode())
        unbound = 0
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                spec_id = payload.get("plan_spec_id") if isinstance(payload, dict) else None
            except (OSError, UnicodeError, json.JSONDecodeError):
                spec_id = None
            if spec_id not in seen:
                unbound += 1
                digest.update(write({"kind": "unbound_result", "path": str(path),
                                     "reason": "no matching planned identity or unreadable result"}).encode())
        write({"kind": "coverage_end", "counts": counts, "unbound_results": unbound,
               "preceding_lines_sha256": digest.hexdigest()})
    print(f"Compared {validated['spec_count']} planned units; {unbound} results were unbound")


if __name__ == "__main__":
    main()
